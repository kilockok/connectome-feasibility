"""Stage 5: z observability per condition (current / unordered / ordered).

E0: current-step population stats -> z (ridge)
E1: unordered window statistics (the StatsHistory feature set) -> z (ridge)
E4: ordered per-step population sequence -> GRU -> z
Targets z_pos and z_vel; R2 / MAE / Pearson on held-out windows.
"""
import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
from run_latent_v4 import ROOT, base_data, obs_subset, subset_connectome, slice_data
from run_latent_v2 import setup as setup_v2
from latent_data import sample_indices, windows
from models.history_set_encoder import stats_features
from diagnose_latent_v2 import ridge_fit, ridge_apply, metrics, step_features, GRUEstimator, train_torch

K = 32


def current_features(x):
    pop = x[:, -1].mean(1)
    rate = x[:, -1, :, 1].mean(1, keepdim=True)
    return torch.cat((pop, rate), -1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tags', nargs='+', required=True)
    args = p.parse_args()
    torch.set_num_threads(2)
    cfg, lc, _ = setup_v2('hidden')
    out = ROOT / 'observability'
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for tag in args.tags:
        if 'obs' in tag:
            import re
            m = re.match(r'n(\d+)obs(\d+)_(\w+)', tag)
            n, n_obs, protocol = int(m.group(1)), int(m.group(2)), m.group(3)
            data, _ = base_data(n, cfg, lc)
            conn = Connectome.generate(replace(cfg, n_neurons=n))
            obs = torch.load(ROOT / 'data' / f'obs_{tag}.pt')
            data = slice_data(data, obs.cuda())
        else:
            n = int(tag[1:])
            data, _ = base_data(n, cfg, lc)
        dev = torch.device('cuda')
        splits = {}
        for sp in ('train', 'val', 'test_seen', 'test_ood'):
            bi, ti = sample_indices(data[sp], 2048 if sp == 'train' else 1024, 4242)
            x, _ = windows(data[sp], bi, ti, K)
            z = data[sp]['z'][bi, ti]
            splits[sp] = (x, z)
        estimators = {}
        for name, feat_fn in (('E0_current', current_features), ('E1_unordered', stats_features)):
            f = {sp: feat_fn(v[0]).cpu() for sp, v in splits.items()}
            m = ridge_fit(f['train'], splits['train'][1].cpu(), f['val'], splits['val'][1].cpu())
            estimators[name] = {sp: metrics(ridge_apply(m, f[sp]), z[1].cpu()) for sp, z in splits.items() if sp != 'train'}
        fseq = {sp: step_features(v[0]) for sp, v in splits.items()}
        m4 = train_torch(GRUEstimator(fseq['train'].shape[2]).to(dev),
                         fseq['train'], splits['train'][1], fseq['val'], splits['val'][1])
        estimators['E4_ordered_gru'] = {sp: metrics(m4(fseq[sp]).cpu(), z[1].cpu())
                                        for sp, z in splits.items() if sp != 'train'}
        for name, per in estimators.items():
            for sp, mm in per.items():
                for target, mmm in mm.items():
                    rows.append(dict(tag=tag, estimator=name, split=sp, target=target, **mmm))
        print('obs done', tag,
                  'E0 z_pos seen=%.3f' % estimators['E0_current']['test_seen']['z_pos']['r2'],
                  'E1=%.3f' % estimators['E1_unordered']['test_seen']['z_pos']['r2'],
                  'E4=%.3f' % estimators['E4_ordered_gru']['test_seen']['z_pos']['r2'], flush=True)
    with (out / 'z_observability.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print('OBSERVABILITY ROWS', len(rows), flush=True)


if __name__ == '__main__':
    from connectome import Connectome
    main()
