"""v7 natural held-out evaluation on shared samples (+coverage, stratification)."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
from run_v7 import ROOT, setup_cfg, datasets, LABELS
from latent_data import sample_indices, windows
from eval_latent import calibrate_threshold, summarize
from calibrate_latent import write_csv
from models.residual_v7 import build_v7, base_pre
from lif_adapt_v7 import AdaptationLIFSimulator, adaptation_reference

SEEDS = (1234, 1235, 1236, 1237, 1238)


@torch.no_grad()
def eval_model(kind, summary, conn, cfg, data, sim, threshold=None, count=1024):
    model = build_v7(kind, conn, cfg).cuda()
    model.load_state_dict(torch.load(summary['checkpoint'], map_location='cuda',
                                     weights_only=False)['state_dict'])
    model = model.eval()
    rows = {}
    for sp in ('test_seen', 'test_ood'):
        bi, ti = sample_indices(data[sp], count, 8001)
        outs, ys, cs = [], [], []
        for off in range(0, len(bi), 32):
            b, t = bi[off:off + 32], ti[off:off + 32]
            x, y = windows(data[sp], b, t, model.k)
            a = data[sp]['a'][b, t] if model.oracle else None
            out = model(x, a=a)
            outs.append({k: v.cpu() for k, v in out.items() if k in ('v', 's_logits', 'r', 'corr')})
            ys.append(y.cpu())
        o = {k: torch.cat([a2[k] for a2 in outs]) for k in outs[0]}
        y = torch.cat(ys)
        th = threshold if threshold is not None else calibrate_threshold(o, y)
        m = summarize(o, y, th)
        # base-LIF error on the same windows (formula predictor, no learning)
        xw, _ = windows(data[sp], bi, ti, 1)
        base = base_pre(xw[:, -1].cuda(), cfg, model.W, model.i_bias)
        bv = base['v_base'].cpu()
        m['base_v_rmse'] = float((bv - y[..., 0]).square().mean().sqrt())
        m['model_v_rmse'] = m['v_rmse']
        m['improve_over_base'] = m['base_v_rmse'] - m['v_rmse']
        # reset / non-reset stratification
        refr = y[..., 2] > 0
        m['v_rmse_reset'] = float((o['v'] - y[..., 0])[refr].square().mean().sqrt()) if refr.any() else None
        m['v_rmse_nonreset'] = float((o['v'] - y[..., 0])[~refr].square().mean().sqrt()) if (~refr).any() else None
        m['base_v_rmse_nonreset'] = float((bv - y[..., 0])[~refr].square().mean().sqrt()) if (~refr).any() else None
        m['coverage'] = len(y)
        # mechanistic reference a recovery on these windows (declared)
        ref = adaptation_reference(data[sp], sim)
        ta = data[sp]['a']
        m['ref_a_mae'] = float((ref - ta).abs().mean())
        rows[sp] = m
        rows[sp]['threshold'] = th
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', nargs='+', type=int, default=list(SEEDS))
    args = p.parse_args()
    torch.set_num_threads(2)
    cfg = setup_cfg()
    from lif_adapt_v7 import AdaptConfig
    conn, data = datasets('adapt', cfg, AdaptConfig())
    sim = AdaptationLIFSimulator(conn, cfg, torch.device('cuda'), AdaptConfig())
    thresholds = {}
    rows = []
    for seed in args.seeds:
        for kind in LABELS:
            summary = json.loads((ROOT / 'natural' / 'training' / f'{kind}_seed{seed}.json').read_text())
            r = eval_model(kind, summary, conn, cfg, data, sim, threshold=thresholds.get(kind))
            thresholds[kind] = r['test_seen']['threshold']
            for sp, m in r.items():
                rows.append(dict(seed=seed, model=kind, split=sp,
                                 **{k: v for k, v in m.items() if k != 'threshold'}))
        print('natural eval done seed', seed, flush=True)
    write_csv(ROOT / 'metrics_natural.csv', rows)
    print('ROWS', len(rows), flush=True)


if __name__ == '__main__':
    main()
