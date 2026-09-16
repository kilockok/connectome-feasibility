"""Stage 5: temporal information profile of the trained ordered GlobalTemporal K32.

Correct inference-time masking (never weight truncation):
  occlusion K_eff in {1,2,4,8,16,32}: zero tokens older than the last K_eff
  reverse: flip the order of tokens[:-1]
  block shuffle b in {2,4,8}: permute tokens[:-1] within consecutive blocks
  preserve_last4: shuffle tokens[:-5], keep the last 4 in order
  zero_old: zero tokens[:-1] entirely
Metrics: seen one-step pooled F1 + V RMSE vs fully ordered.
"""
import json
import numpy as np
import torch
from run_latent_v3 import ROOT, datasets_v3
from run_latent_v2 import setup as setup_v2
from train_latent_v2 import load_model_v2
from analyze_latent_v3 import onestep_full
import analyze_latent_v3 as A


def permute_control(x, mode, generator, param=None):
    if mode == 'ordered':
        return x
    if mode == 'occlude':
        k_eff = param
        if k_eff >= x.shape[1]:
            return x
        x = x.clone()
        x[:, :-k_eff] = 0
        return x
    if mode == 'reverse':
        return torch.cat((x[:, :-1].flip(1), x[:, -1:]), 1)
    if mode == 'block':
        b = param
        old = x[:, :-1].clone()
        n = old.shape[1]
        for start in range(0, n, b):
            end = min(start + b, n)
            perm = torch.randperm(end - start, generator=generator).to(x.device)
            old[:, start:end] = old[:, start + perm]
        return torch.cat((old, x[:, -1:]), 1)
    if mode == 'preserve_last4':
        n = x.shape[1]
        perm = torch.randperm(n - 5, generator=generator).to(x.device)
        return torch.cat((x[:, perm], x[:, -5:]), 1)
    if mode == 'zero_old':
        x = x.clone()
        x[:, :-1] = 0
        return x
    raise ValueError(mode)


@torch.no_grad()
def onestep_control(model, data, threshold, mode, param=None, count=1024, seed=8001):
    from latent_data import sample_indices, windows
    bi, ti = sample_indices(data, count, seed)
    sg = torch.Generator().manual_seed(seed + 19)
    outs, tgts = [], []
    for start in range(0, count, 32):
        b, t = bi[start:start + 32], ti[start:start + 32]
        x, y = windows(data, b, t, model.k)
        x = permute_control(x, mode, sg, param)
        outs.append({k: v.cpu() for k, v in model(x).items()})
        tgts.append(y.cpu())
    out = {k: torch.cat([o[k] for o in outs]) for k in outs[0]}
    y = torch.cat(tgts)
    from eval_latent import summarize
    return summarize(out, y, threshold)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', nargs='+', type=int, default=[1234, 1235, 1236, 1237, 1238])
    args = p.parse_args()
    torch.set_num_threads(2)
    cfg, lc, conn = setup_v2('hidden')
    data = datasets_v3('hidden')['test_seen']
    out = ROOT / 'temporal_order'
    out.mkdir(parents=True, exist_ok=True)
    conditions = [('ordered', None)] + [('occlude', k) for k in (1, 2, 4, 8, 16)] + \
                 [('reverse', None), ('block', 2), ('block', 4), ('block', 8),
                  ('preserve_last4', None), ('zero_old', None)]
    rows = []
    for seed in args.seeds:
        for label in ('global_k32', 'gshuffle'):
            summary = json.loads((ROOT / 'replication' / 'hidden' / 'training'
                                  / f'{label}_seed{seed}.json').read_text())
            model, blob = load_model_v2(summary, conn, torch.device('cuda'))
            th = blob['threshold']
            for mode, param in conditions:
                m = onestep_control(model, data, th, mode, param)
                rows.append(dict(model=label, seed=seed, mode=mode, param=param if param is not None else '', **m))
            print('order-profile done', label, seed, flush=True)
            del model
            torch.cuda.empty_cache()
    from calibrate_latent import write_csv
    write_csv(out / 'temporal_order_metrics.csv', rows)
    print('TEMPORAL ORDER ROWS', len(rows), flush=True)


if __name__ == '__main__':
    main()
