"""v8 natural held-out evaluation."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from run_v8 import ROOT, datasets, LABELS
from run_v7 import setup_cfg
from latent_data import sample_indices, windows
from eval_latent import calibrate_threshold, summarize
from models.residual_v8 import build_v8
from models.residual_v7 import base_pre
from calibrate_latent import write_csv

SEEDS = (1234, 1235, 1236, 1237, 1238)


@torch.no_grad()
def eval_model(kind, summary, conn, cfg, data, threshold=None, count=1024):
    model = build_v8(kind, conn, cfg).cuda()
    model.load_state_dict(torch.load(summary['checkpoint'], map_location='cuda',
                                     weights_only=False)['state_dict'])
    model = model.eval()
    rows = {}
    for sp in ('test_seen', 'test_ood'):
        bi, ti = sample_indices(data[sp], count, 8001)
        outs, ys = [], []
        for off in range(0, len(bi), 32):
            b, t = bi[off:off + 32], ti[off:off + 32]
            x, y = windows(data[sp], b, t, model.k)
            sc = data[sp]['stp_current'][b, t] if model.oracle else None
            out = model(x, stp_current=sc)
            outs.append({k: v.cpu() for k, v in out.items() if k in ('v', 's_logits', 'r')})
            ys.append(y.cpu())
        o = {k: torch.cat([a[k] for a in outs]) for k in outs[0]}
        y = torch.cat(ys)
        th = threshold if threshold is not None else calibrate_threshold(o, y)
        m = summarize(o, y, th)
        xw, _ = windows(data[sp], bi, ti, 1)
        base = base_pre(xw[:, -1].cuda(), cfg, model.W, model.i_bias)
        bv = base['v_base'].cpu()
        bs = base['logit_base'].cpu()
        m['base_v_rmse'] = float((bv - y[..., 0]).square().mean().sqrt())
        m['improve_over_base'] = m['base_v_rmse'] - m['v_rmse']
        b_out = dict(v=bv, s_logits=bs, r=torch.zeros_like(bs))
        bm = summarize(b_out, y, th)
        m['base_spike_f1'] = bm['spike_f1']
        m['coverage'] = len(y)
        rows[sp] = m
        rows[sp]['threshold'] = th
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', nargs='+', type=int, default=list(SEEDS))
    args = p.parse_args()
    torch.set_num_threads(2)
    cfg = setup_cfg()
    conn, norm, data = datasets('stp', cfg)
    thresholds = {}
    rows = []
    for seed in args.seeds:
        for kind in LABELS:
            summary = json.loads((ROOT / 'natural' / 'training' / f'{kind}_seed{seed}.json').read_text())
            r = eval_model(kind, summary, conn, cfg, data, threshold=thresholds.get(kind))
            thresholds[kind] = r['test_seen']['threshold']
            for sp, m in r.items():
                rows.append(dict(seed=seed, model=kind, split=sp,
                                 **{k: v for k, v in m.items() if k != 'threshold'}))
        print('v8 natural eval done seed', seed, flush=True)
    write_csv(ROOT / 'metrics' / 'natural.csv', rows)
    print('ROWS', len(rows), flush=True)


if __name__ == '__main__':
    main()
