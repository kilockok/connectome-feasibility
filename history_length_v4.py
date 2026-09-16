"""Stage 6: history-length curves via architecture-consistent inference masking.

SetHistory: drop older tokens from the set (exact for a permutation-invariant
model; only fewer set elements are aggregated). GlobalTemporal: zero older
tokens (v3-validated occlusion). K_eff in {1,2,4,8,16,32}.
"""
import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
from run_latent_v4 import ROOT, base_data, subset_connectome, slice_data
from run_latent_v2 import setup as setup_v2
from models.history_set_encoder import load_model_v4
from latent_data import sample_indices, windows
from eval_latent import summarize

KS = (1, 2, 4, 8, 16, 32)


@torch.no_grad()
def eval_keff(model, data, threshold, k_eff, count=1024, seed=8001):
    bi, ti = sample_indices(data, count, seed)
    outs, ys = [], []
    for start in range(0, count, 32):
        b, t = bi[start:start + 32], ti[start:start + 32]
        x, y = windows(data, b, t, model.k)
        if k_eff < x.shape[1]:
            x = x[:, -k_eff:]
        outs.append({k: v.cpu() for k, v in model(x).items()})
        ys.append(y.cpu())
    out = {k: torch.cat([o[k] for o in outs]) for k in outs[0]}
    return summarize(out, torch.cat(ys), threshold)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tags', nargs='+', required=True)
    args = p.parse_args()
    torch.set_num_threads(2)
    cfg, lc, _ = setup_v2('hidden')
    rows = []
    for tag in args.tags:
        if 'obs' in tag:
            import re
            m = re.match(r'n(\d+)obs(\d+)_(\w+)', tag)
            n, n_obs, protocol = int(m.group(1)), int(m.group(2)), m.group(3)
            data, _ = base_data(n, cfg, lc)
            obs = torch.load(ROOT / 'data' / f'obs_{tag}.pt')
            data = slice_data(data, obs.cuda())
            conn = subset_connectome(Connectome.generate(replace(cfg, n_neurons=n)), obs)
        else:
            n = int(tag[1:])
            data, _ = base_data(n, cfg, lc)
            conn = Connectome.generate(replace(cfg, n_neurons=n))
        for label in ('set_k32', 'global_k32'):
            for path in sorted((ROOT / tag / 'training').glob(f'{label}_seed*.json')):
                summary = json.loads(path.read_text())
                model, blob = load_model_v4(summary, conn, torch.device('cuda'))
                for k_eff in KS:
                    m = eval_keff(model, data['test_seen'], blob['threshold'], k_eff)
                    rows.append(dict(tag=tag, label=label, seed=summary['seed'], k_eff=k_eff, **m))
                print('k-sweep done', tag, label, summary['seed'], flush=True)
                del model
                torch.cuda.empty_cache()
    out = ROOT / 'history_length'
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'history_length.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print('K-SWEEP ROWS', len(rows), flush=True)


if __name__ == '__main__':
    from connectome import Connectome
    main()
