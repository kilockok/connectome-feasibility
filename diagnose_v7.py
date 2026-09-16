"""v7 Stage 4: frozen-encoder a probes + calibration + residual attribution."""
import csv
import json
from pathlib import Path
import numpy as np
import torch
from run_v7 import ROOT, setup_cfg, datasets
from latent_data import sample_indices, windows
from latent_probe_v2 import fit_ridge, apply_probe
from models.residual_v7 import build_v7, base_pre
from lif_adapt_v7 import AdaptationLIFSimulator, adaptation_reference

SEEDS = (1234, 1235, 1236, 1237, 1238)


@torch.no_grad()
def features_and_targets(kind, model, data, count=2048, seed=9191):
    bi, ti = sample_indices(data, count, seed)
    feats, a_trues, corrs, bases, ys = [], [], [], [], []
    for off in range(0, count, 32):
        b, t = bi[off:off + 32], ti[off:off + 32]
        x, y = windows(data, b, t, model.k)
        a_in = data['a'][b, t] if model.oracle else None
        f = model.features(x, a_in)
        out = model(x, a=a_in)
        feats.append(torch.cat((f.mean(1), f.std(1, unbiased=False)), -1).cpu())
        a_trues.append(data['a'][b, t].mean(1).cpu())
        corrs.append(out['corr'][..., 0].mean(1).cpu())
        base = base_pre(x[:, -1], model.cfg, model.W, model.i_bias)
        bases.append(base['v_base'].mean(1).cpu())
        ys.append(y[..., 0].mean(1).cpu())
    return (torch.cat(feats), torch.cat(a_trues), torch.cat(corrs), torch.cat(bases), torch.cat(ys))


def calib(pred, true):
    pred, true = pred.double(), true.double()
    den = (true - true.mean()).square().sum()
    r2 = float(1 - (pred - true).square().sum() / den.clamp(min=1e-12))
    slope = float(((pred - pred.mean()) * (true - true.mean())).sum()
                  / (pred - pred.mean()).square().sum().clamp(min=1e-12))
    bias = float((pred - true).mean())
    mae = float((pred - true).abs().mean())
    rmse = float((pred - true).square().mean().sqrt())
    return dict(r2=r2, slope=slope, bias=bias, mae=mae, rmse=rmse)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', nargs='+', type=int, default=list(SEEDS))
    args = p.parse_args()
    torch.set_num_threads(2)
    cfg = setup_cfg()
    from lif_adapt_v7 import AdaptConfig
    conn, data = datasets('adapt', cfg, AdaptConfig())
    rows = []
    for seed in args.seeds:
        for kind in ('k1', 'k2', 'set', 'ordered'):
            summary = json.loads((ROOT / 'natural' / 'training' / f'{kind}_seed{seed}.json').read_text())
            model = build_v7(kind, conn, cfg).cuda()
            model.load_state_dict(torch.load(summary['checkpoint'], map_location='cuda',
                                             weights_only=False)['state_dict'])
            model = model.eval()
            tr = features_and_targets(kind, model, data['train'], 2048)
            va = features_and_targets(kind, model, data['val'], 1024)
            te = features_and_targets(kind, model, data['test_seen'], 1024)
            # a probe (population-mean a; per-neuron a is high-dim, mean kept as the scalar target)
            m = fit_ridge((tr[0], tr[1][:, None]), (va[0], va[1][:, None]))
            pred = apply_probe(m, te[0])[:, 0]
            c = calib(pred, te[1])
            rows.append(dict(seed=seed, model=kind, target='a_mean', **c))
            # model's own corr head vs true adaptation contribution (next-step residual
            # on population mean): true contribution ~ -c*a; model corr should track it
            true_contrib = -0.5 * te[1]
            c2 = calib(te[2], true_contrib)
            rows.append(dict(seed=seed, model=kind, target='corr_vs_true_contrib', **c2))
            # residual prediction: model v vs base v error decomposition on population mean
            base_err = (te[3] - te[4]).abs().mean()
            rows.append(dict(seed=seed, model=kind, target='base_err_meanV',
                             r2=None, slope=None, bias=None, mae=float(base_err), rmse=None))
            print('diag done', kind, seed, flush=True)
            del model
            torch.cuda.empty_cache()
    with (ROOT / 'metrics_diagnostics.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print('ROWS', len(rows), flush=True)


if __name__ == '__main__':
    main()
