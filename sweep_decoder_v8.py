"""v8 history sweep (inference masking) + STP effective-state decoders."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
from run_v8 import ROOT, datasets
from run_v7 import setup_cfg
from latent_data import sample_indices, windows
from eval_latent import calibrate_threshold, summarize
from models.residual_v8 import build_v8
from latent_probe_v2 import fit_ridge, apply_probe
from calibrate_latent import write_csv

SEEDS = (1234, 1235, 1236, 1237, 1238)
KS = (1, 2, 4, 8, 16, 32)


@torch.no_grad()
def eval_keff(model, data, threshold, k_eff, count=1024, seed=8001):
    bi, ti = sample_indices(data, count, seed)
    outs, ys = [], []
    for st in range(0, count, 32):
        b, t = bi[st:st + 32], ti[st:st + 32]
        x, y = windows(data, b, t, model.k)
        if k_eff < x.shape[1]:
            x = x[:, -k_eff:]
        sc = data['stp_current'][b, t] if model.oracle else None
        outs.append({k: v.cpu() for k, v in model(x, stp_current=sc).items() if k in ('v', 's_logits', 'r')})
        ys.append(y.cpu())
    o = {k: torch.cat([a[k] for a in outs]) for k in outs[0]}
    return summarize(o, torch.cat(ys), threshold)


@torch.no_grad()
def decoder_features(model, data, count=2048, seed=9191):
    """Frozen representation -> effective STP factor on active probe edges.

    Target: per-window mean effective gain g of edges that fire at the window
    endpoint (the future-relevant STP sufficient statistic at the moment of
    use), plus the current-step postsyn STP current. Both are legal targets.
    """
    bi, ti = sample_indices(data, count, seed)
    feats, g_evt, g_cur = [], [], []
    for st in range(0, count, 32):
        b, t = bi[st:st + 32], ti[st:st + 32]
        x, _ = windows(data, b, t, model.k)
        sc = data['stp_current'][b, t] if model.oracle else None
        f = model.features(x, sc)
        feats.append(torch.cat((f.mean(1), f.std(1, unbiased=False)), -1).cpu())
        g = data['g_path'][b, t]
        s = data['states'][b, t, :, 1]
        # effective gain on edges that fire at t (event edges): mean g over firing edges
        firing = (s > .5).cpu()
        g = g.cpu()
        g_sel = (g * firing[..., None]).sum((1, 2)) / firing.sum(-1).clamp(min=1)
        g_evt.append(g_sel.cpu())
        g_cur.append(data['stp_current'][b, t].mean(-1).cpu())
    return torch.cat(feats), torch.cat(g_evt), torch.cat(g_cur)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--part', choices=['sweep', 'decoder'], required=True)
    args = p.parse_args()
    torch.set_num_threads(2)
    cfg = setup_cfg()
    conn, norm, data_all = datasets('stp', cfg)
    if args.part == 'sweep':
        rows = []
        for seed in SEEDS:
            for kind in ('set', 'ordered', 'event_simple', 'event_rich'):
                summary = json.loads((ROOT / 'natural' / 'training' / f'{kind}_seed{seed}.json').read_text())
                model = build_v8(kind, conn, cfg).cuda()
                model.load_state_dict(torch.load(summary['checkpoint'], map_location='cuda',
                                                 weights_only=False)['state_dict'])
                model = model.eval()
                th = 0.9
                for k_eff in KS:
                    m = eval_keff(model, data_all['test_seen'], th, k_eff)
                    rows.append(dict(seed=seed, model=kind, k_eff=k_eff, **m))
                print('sweep done', kind, seed, flush=True)
                del model
                torch.cuda.empty_cache()
        write_csv(ROOT / 'metrics' / 'history_sweep.csv', rows)
        print('ROWS', len(rows), flush=True)
    else:
        rows = []
        for seed in SEEDS:
            for kind in ('k1', 'k2', 'set', 'ordered', 'event_rich'):
                summary = json.loads((ROOT / 'natural' / 'training' / f'{kind}_seed{seed}.json').read_text())
                model = build_v8(kind, conn, cfg).cuda()
                model.load_state_dict(torch.load(summary['checkpoint'], map_location='cuda',
                                                 weights_only=False)['state_dict'])
                model = model.eval()
                tr = decoder_features(model, data_all['train'])
                va = decoder_features(model, data_all['val'])
                te = decoder_features(model, data_all['test_seen'])
                for name, j in (('g_event', 1), ('g_current', 2)):
                    m = fit_ridge((tr[0], tr[j][:, None]), (va[0], va[j][:, None]))
                    pred = apply_probe(m, te[0])[:, 0]
                    true = te[j]
                    pred, true = pred.double(), true.double()
                    den = (true - true.mean()).square().sum()
                    rows.append(dict(seed=seed, model=kind, target=name,
                                     r2=float(1 - (pred - true).square().sum() / den.clamp(min=1e-12)),
                                     mae=float((pred - true).abs().mean()),
                                     corr=float(((pred - pred.mean()) * (true - true.mean())).sum()
                                                / ((pred - pred.mean()).norm() * (true - true.mean()).norm()).clamp(min=1e-12))))
                print('decoder done', kind, seed, flush=True)
                del model
                torch.cuda.empty_cache()
        write_csv(ROOT / 'metrics' / 'decoder.csv', rows)
        print('ROWS', len(rows), flush=True)


if __name__ == '__main__':
    main()
