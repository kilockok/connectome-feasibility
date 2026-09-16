"""v8 natural alias benchmark: matched current observables, separated edge event history.

Pairs (A,t),(B,s) from test_seen where the SAME presynaptic neuron j spikes at
both (probe event), current observables are close, but j's event history in
the last ~32 steps differs strongly. The STP state of edges out of j differs,
so the teacher's response to the probe differs. K1/K2 see the probe event but
not the earlier activity.
"""
import csv
import json
from pathlib import Path
import numpy as np
import torch
from run_v8 import ROOT, datasets
from run_v7 import setup_cfg
from latent_data import windows
from models.residual_v8 import build_v8
from calibrate_latent import write_csv

K = 32
SEEDS = (1234, 1235, 1236, 1237, 1238)


@torch.no_grad()
def find_pairs(data, n_pairs=64, seed=0):
    s = data['states'][..., 1]
    v = data['states'][..., 0]
    r = data['states'][..., 2]
    u = data['stimulus']
    T = s.shape[1]
    rng = np.random.default_rng(seed)
    cand = []
    for traj in range(len(s)):
        counts = s[traj].sum(-1).cumsum(0)
        for t in range(32, T - 2, 2):
            if int(counts[t] - counts[t - 32]) >= 6:
                cand.append((traj, t))
    dev = s.device
    idx = torch.tensor([[a, b] for a, b in cand], device=dev)
    vv = v[idx[:, 0], idx[:, 1]]
    ss = s[idx[:, 0], idx[:, 1]]
    rr = r[idx[:, 0], idx[:, 1]]
    uu = u[idx[:, 0], idx[:, 1]]
    f = torch.cat((vv, ss, .5 * rr, uu), -1)
    f = (f - f.mean(0)) / f.std(0).clamp(min=1e-6)
    d = torch.cdist(f, f)
    hist = s[idx[:, 0], max(0, K - 33):, :][:, :K, :] if K else None
    # event history distance: per-pair difference in spike count of the shared probe neuron
    pairs = []
    flat = d.flatten()
    xth = flat[torch.randint(0, flat.numel(), (2_000_000,), device=flat.device)].quantile(.05)
    ok = d <= xth
    co = ok.nonzero()
    for a, b in co.cpu().tolist():
        (ta, t), (tb, s_) = cand[a], cand[b]
        if ta == tb:
            continue
        shared = (ss[a] > .5) & (ss[b] > .5)
        if not shared.any():
            continue
        j = int(shared.nonzero()[0])
        ha = int(s[ta, t - 31:t, j].sum())
        hb = int(s[tb, s_ - 31:s_, j].sum())
        if abs(ha - hb) >= 3:
            pairs.append(dict(A=(ta, t), B=(tb, s_), j=j, d_x=float(d[a, b]),
                              v_rmse=float((v[ta, t] - v[tb, s_]).square().mean().sqrt()),
                              s_disagree=float((s[ta, t] != s[tb, s_]).float().mean()),
                              u_rmse=float((u[ta, t] - u[tb, s_]).square().mean().sqrt()),
                              hist_A=ha, hist_B=hb))
        if len(pairs) >= n_pairs:
            break
    return pairs


@torch.no_grad()
def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', nargs='+', type=int, default=list(SEEDS))
    args = p.parse_args()
    torch.set_num_threads(2)
    cfg = setup_cfg()
    conn, norm, data_all = datasets('stp', cfg)
    data = {k: v for k, v in data_all['test_seen'].items()}
    pairs = find_pairs(data)
    print('pairs:', len(pairs))
    if not pairs:
        raise RuntimeError('no alias pairs found')
    write_csv(ROOT / 'audit' / 'alias_pairs.csv', pairs)
    rows = []
    for seed in args.seeds:
        for kind in ('k1', 'k2', 'set', 'ordered', 'event_simple', 'event_rich', 'oracle'):
            summary = json.loads((ROOT / 'natural' / 'training' / f'{kind}_seed{seed}.json').read_text())
            model = build_v8(kind, conn, cfg).cuda()
            model.load_state_dict(torch.load(summary['checkpoint'], map_location='cuda',
                                             weights_only=False)['state_dict'])
            model = model.eval()
            for i, pr in enumerate(pairs):
                (ta, t), (tb, s_) = pr['A'], pr['B']
                outs = {}
                for tag, (tr, tt) in (('A', (ta, t)), ('B', (tb, s_))):
                    x, y = windows(data, torch.tensor([tr], device='cuda'), torch.tensor([tt], device='cuda'), model.k)
                    sc = data['stp_current'][tr, tt][None] if model.oracle else None
                    out = model(x, stp_current=sc)
                    outs[tag] = (out['v'][0].cpu(), y[0, :, 0].cpu())
                (vA, yA), (vB, yB) = outs['A'], outs['B']
                eAA = float((vA - yA).square().mean())
                eAB = float((vA - yB).square().mean())
                eBA = float((vB - yA).square().mean())
                eBB = float((vB - yB).square().mean())
                postsyn = (conn.dense_weight('cpu')[pr['j']] != 0).nonzero().flatten()
                dp_true = (yA[postsyn] - yB[postsyn]).float()
                dp_pred = (vA[postsyn] - vB[postsyn]).float()
                n = dp_true.norm().clamp(min=1e-9)
                proj = float((dp_pred * dp_true).sum() / n)
                cos = float(torch.nn.functional.cosine_similarity(dp_pred, dp_true, dim=0)) if float(n) > 1e-9 else None
                rows.append(dict(seed=seed, model=kind, pair=i,
                                 correct=float((eAA + eBB) < (eAB + eBA)),
                                 margin=float((eAB + eBA) - (eAA + eBB)),
                                 err_a=eAA, err_b=eBB,
                                 future_gap=float((yA - yB).square().mean().sqrt()),
                                 resp_diff_proj=proj, resp_diff_cos=cos,
                                 resp_diff_true=float(n),
                                 resp_diff_pred=float(dp_pred.norm())))
            print('alias done', kind, seed, flush=True)
            del model
            torch.cuda.empty_cache()
    write_csv(ROOT / 'metrics' / 'alias.csv', rows)
    print('ROWS', len(rows), flush=True)


if __name__ == '__main__':
    main()
