"""Counterfactual matched-history experiment (v3 core).

Keep the current observable state (and stimulus) fixed; replace ONLY the
historical context t-K+1..t-1 with a donor's history. If the ordered model
uses temporal history to infer hidden dynamical information, predictions
should shift systematically toward the donor's future - strongly for
opposite-velocity donors, weakly for same-velocity donors, and without
consistent direction for random donors. The shuffled-history model must show
a weaker effect. z is used ONLY by the experimenter to select matched pairs.
"""
import json
from pathlib import Path
import numpy as np
import torch
from run_latent_v3 import ROOT, datasets_v3
from run_latent_v2 import setup as setup_v2
from train_latent_v2 import load_model_v2
from eval_latent import state_from_output
from calibrate_latent import write_csv

H = 8
K = 32


def token_slice(data, traj, a, b):
    """Tokens [a..b] inclusive: [V,S,R,U] per step."""
    s = data['states'][traj, a:b + 1]
    u = data['stimulus'][traj, a:b + 1]
    return torch.cat((s, u.unsqueeze(-1)), -1)


def candidate_pool(data, t_lo=32, t_hi=None, stride=2, min_spikes=8):
    """Candidates with an ACTIVE recent window (population spike count in
    [t-31..t] >= min_spikes), so matched pairs are informative states."""
    T = data['stimulus'].shape[1]
    if t_hi is None:
        t_hi = T - H - 1
    rows = []
    w = data['states'][..., 1]
    for traj in range(len(data['states'])):
        counts = w[traj].sum(-1).cumsum(0)
        for t in range(t_lo, t_hi + 1, stride):
            if int(counts[t] - counts[t - 32]) >= min_spikes:
                rows.append((traj, t))
    return rows


@torch.no_grad()
def build_features(data, pool):
    dev = data['states'].device
    idx = torch.tensor([[tr, t] for tr, t in pool], device=dev)
    v = data['states'][idx[:, 0], idx[:, 1], :, 0]
    s = data['states'][idx[:, 0], idx[:, 1], :, 1]
    r = data['states'][idx[:, 0], idx[:, 1], :, 2]
    u = data['stimulus'][idx[:, 0], idx[:, 1]]
    zp = data['z'][idx[:, 0], idx[:, 1], 0:1]
    zv = data['z'][idx[:, 0], idx[:, 1], 1:2]
    f = torch.cat((v, s, .5 * r, u, 2.0 * zp), -1)
    mean, std = f.mean(0), f.std(0).clamp(min=1e-6)
    return (f - mean) / std, zv.squeeze(-1), zp.squeeze(-1)


@torch.no_grad()
def history_signatures(data, pool, k=K):
    """Per-candidate signature of the recent window: what the history implies."""
    dev = data['states'].device
    sigs = []
    for traj, t in pool:
        x = token_slice(data, traj, t - k + 1, t - 1)
        sigs.append(torch.cat((x.mean(0), x.std(0, unbiased=False), x[-1] - x[0]), -1)[None])
    sig = torch.cat(sigs)
    mean, std = sig.mean(0), sig.std(0).clamp(min=1e-6)
    return (sig - mean) / std


@torch.no_grad()
def find_matched_pairs(data, n_pairs=96, seed=0, same_sign=False, vel_floor_frac=.5,
                       hist_quantile=.6):
    """Closest CURRENT-state matches with opposite (or same) z_vel sign whose
    HISTORIES diverge (above the hist_quantile of candidate history distances)."""
    pool = candidate_pool(data)
    f, zv, zp = build_features(data, pool)
    vel_floor = vel_floor_frac * float(zv.std())
    d = torch.cdist(f, f)
    tr = torch.tensor([p[0] for p in pool], device=zv.device)
    same_traj = tr[:, None] == tr[None, :]
    product = zv[:, None] * zv[None, :]
    opp = (product > 0) if same_sign else (product < 0)
    ok = opp & (~same_traj) & (zv.abs()[:, None] >= vel_floor) & (zv.abs()[None, :] >= vel_floor)
    d = d.masked_fill(~ok, float('inf'))
    # Two-stage: current-match candidates, then require divergent RAW histories
    # (spike-pattern disagreement in the 31-step window above floor).
    cand = (d < 2.0) & ok
    idxs = cand.nonzero()
    dev = d.device
    hdis = []
    for a, b in idxs.cpu().tolist():
        ta, t = pool[a]; tb, s = pool[b]
        sa = data['states'][ta, t - 31:t, :, 1]
        sb = data['states'][tb, s - 31:s, :, 1]
        hdis.append(float((sa != sb).float().mean()))
    hdis = torch.tensor(hdis, device=dev)
    floor = hdis.quantile(hist_quantile)
    keep = hdis > floor
    d2 = torch.full_like(d, float('inf'))
    sel = idxs[keep]
    d2[sel[:, 0], sel[:, 1]] = d[sel[:, 0], sel[:, 1]]
    print(f'history disagreement floor(p{int(hist_quantile * 100)})={float(floor):.4f} '
          f'candidates={len(idxs)} kept={int(keep.sum())}', flush=True)
    flat = d2.flatten()
    order = torch.argsort(flat)
    m = d2.shape[0]
    chosen, used = [], set()
    for oi in order.cpu().tolist():
        a, b = oi // m, oi % m
        if a in used or b in used or float(flat[oi]) == float('inf'):
            continue
        chosen.append((a, b, float(flat[oi])))
        used.add(a); used.add(b)
        if len(chosen) >= n_pairs:
            break
    dev = data['states'].device
    pairs = []
    for a, b, dist in chosen:
        ta, tb = pool[a], pool[b]
        va, vb = data['states'][ta[0], ta[1], :, 0], data['states'][tb[0], tb[1], :, 0]
        sa, sb = data['states'][ta[0], ta[1], :, 1], data['states'][tb[0], tb[1], :, 1]
        ua, ub = data['stimulus'][ta[0], ta[1]], data['stimulus'][tb[0], tb[1]]
        pairs.append(dict(A=ta, B=tb, distance=dist,
                          v_rmse=float((va - vb).square().mean().sqrt()),
                          s_disagree=float((sa != sb).float().mean()),
                          u_rmse=float((ua - ub).square().mean().sqrt()),
                          dz_pos=float(data['z'][ta[0], ta[1], 0] - data['z'][tb[0], tb[1], 0]),
                          z_vel_A=float(data['z'][ta[0], ta[1], 1]),
                          z_vel_B=float(data['z'][tb[0], tb[1], 1])))
    return pairs


def make_swap(data, pair, k=K):
    (ta, t), (tb, s) = pair['A'], pair['B']
    xA = token_slice(data, ta, t - k + 1, t)[None]
    xB = token_slice(data, tb, s - k + 1, s)[None]
    donorB = token_slice(data, tb, s - k + 1, s - 1)[None]
    donorA = token_slice(data, ta, t - k + 1, t - 1)[None]
    xA_sw = torch.cat((donorB, xA[:, -1:]), 1)
    xB_sw = torch.cat((donorA, xB[:, -1:]), 1)
    assert torch.equal(xA_sw[:, -1], xA[:, -1]) and torch.equal(xB_sw[:, -1], xB[:, -1])
    return xA, xA_sw, xB, xB_sw


@torch.no_grad()
def rollout_from(model, x0, data, traj, t_end, horizon, threshold):
    """Autoregressive from the given window with the receiver's true future stimulus."""
    cfg = model_cfg = model._cfg
    x = x0.clone()
    vs, ps = [], []
    for h in range(1, horizon + 1):
        out = model(x)
        state = state_from_output(out, cfg, threshold)
        vs.append(state[..., 0].cpu())
        ps.append(out['s_logits'].sigmoid().cpu())
        if h < horizon:
            stim = data['stimulus'][traj, t_end + h]
            x = torch.cat((x[:, 1:], torch.cat((state, stim[None, :, None]), -1)[:, None]), 1)
    return torch.stack(vs, 1), torch.stack(ps, 1)


@torch.no_grad()
def swap_experiment(model, data, pairs, control='opposite', seed=0, n_pairs=64):
    """Returns per-pair per-h projection/cosine of the swap-induced shift."""
    rng = np.random.default_rng(seed)
    rows = []
    T = data['stimulus'].shape[1]
    for i, pair in enumerate(pairs[:n_pairs]):
        if control == 'random':
            # unmatched donor: random trajectory/time with valid future
            ta, t = pair['A']
            while True:
                tb2 = int(rng.integers(0, len(data['states'])))
                s2 = int(rng.integers(K - 1, T - H - 1))
                if tb2 != ta:
                    break
            pair = dict(pair, B=(tb2, s2))
        (ta, t), (tb, s) = pair['A'], pair['B']
        xA, xA_sw, xB, xB_sw = make_swap(data, pair)
        vA, pA = rollout_from(model, xA, data, ta, t, H, model._th)
        vA_sw, pA_sw = rollout_from(model, xA_sw, data, ta, t, H, model._th)
        vB, pB = rollout_from(model, xB, data, tb, s, H, model._th)
        vB_sw, pB_sw = rollout_from(model, xB_sw, data, tb, s, H, model._th)
        for h in (1, 2, 4, 8):
            dt_v = (data['states'][tb, s + h, :, 0] - data['states'][ta, t + h, :, 0]).cpu()
            dt_p = (data['states'][tb, s + h, :, 1] - data['states'][ta, t + h, :, 1]).cpu()
            for tag, dm_v, dm_p in (('A', vA_sw[0, h - 1] - vA[0, h - 1], pA_sw[0, h - 1] - pA[0, h - 1]),
                                    ('B', vB_sw[0, h - 1] - vB[0, h - 1], pB_sw[0, h - 1] - pB[0, h - 1])):
                sign = 1. if tag == 'A' else -1.  # B swaps toward A's future
                dv, dp = sign * dt_v, sign * dt_p
                def proj_cos(dm, dt):
                    dm, dt = dm.flatten().float(), dt.flatten().float()
                    n = dt.norm().clamp(min=1e-9)
                    return float((dm * dt).sum() / n), float(torch.nn.functional.cosine_similarity(dm, dt, dim=0))
                pv, cv = proj_cos(dm_v, dv)
                pp, cp = proj_cos(dm_p, dp)
                rows.append(dict(pair=i, side=tag, h=h, control=control,
                                 proj_v=pv, cos_v=cv, proj_p=pp, cos_p=cp,
                                 shift_v=float(dm_v.norm()), shift_p=float(dm_p.norm()),
                                 true_v=float(dv.norm()), true_p=float(dp.norm())))
    return rows


def attach_runtime(model, cfg, threshold):
    model._cfg = cfg
    model._th = threshold
    return model


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', nargs='+', type=int, default=[1234, 1235, 1236, 1237, 1238])
    p.add_argument('--pairs', type=int, default=96)
    args = p.parse_args()
    torch.set_num_threads(2)
    cfg, lc, conn = setup_v2('hidden')
    data = datasets_v3('hidden')['test_seen']
    out = ROOT / 'counterfactual'
    out.mkdir(parents=True, exist_ok=True)
    pairs_opp = find_matched_pairs(data, n_pairs=args.pairs, same_sign=False)
    pairs_same = find_matched_pairs(data, n_pairs=args.pairs, same_sign=True)
    import csv
    with (out / 'matched_pairs.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(pairs_opp[0]))
        w.writeheader(); w.writerows(pairs_opp)
    err = dict(n=len(pairs_opp),
               v_rmse=float(np.mean([p['v_rmse'] for p in pairs_opp])),
               s_disagree=float(np.mean([p['s_disagree'] for p in pairs_opp])),
               u_rmse=float(np.mean([p['u_rmse'] for p in pairs_opp])),
               abs_dz_pos=float(np.mean([abs(p['dz_pos']) for p in pairs_opp])),
               z_pos_std=float(data['z'][..., 0].std()))
    (out / 'match_quality.json').write_text(json.dumps(err, indent=2))
    print('match quality:', json.dumps(err), flush=True)
    all_rows = []
    for seed in args.seeds:
        for label, conds in (('global_k32', ('opposite', 'same', 'random')), ('gshuffle', ('opposite',))):
            summary = json.loads((ROOT / 'replication' / 'hidden' / 'training'
                                  / f'{label}_seed{seed}.json').read_text())
            model, blob = load_model_v2(summary, conn, torch.device('cuda'))
            model = attach_runtime(model.eval(), cfg, blob['threshold'])
            for cond in conds:
                pairs = pairs_opp if cond != 'same' else pairs_same
                rows = swap_experiment(model, data, pairs, control=cond, seed=seed)
                for r in rows:
                    r.update(model=label, seed=seed)
                all_rows.extend(rows)
            print('swap done', label, seed, flush=True)
            del model
            torch.cuda.empty_cache()
    write_csv(out / 'swap_metrics.csv', all_rows)
    print('COUNTERFACTUAL ROWS', len(all_rows), flush=True)


if __name__ == '__main__':
    main()
