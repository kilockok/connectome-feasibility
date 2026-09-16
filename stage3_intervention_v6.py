"""Stage 3 evaluation: intervention tracking for M1/M2/M3/M4 under the d protocol."""
import csv
import json
from pathlib import Path
import numpy as np
import torch
from run_latent_v2 import setup as setup_v2
from alias_eval_v5 import load_checked
from latent_data import windows
from lif_latent_v2 import HiddenStateLIFSimulatorV2
from reaudit_v6 import build_branches, TAU, DS, N
from estimators_v6 import base_residual_terms
from stage3_v6 import GainChannelModel, PlainResidualModel

ROOT = Path('results/latent_state_v6')
SEEDS = (1234, 1235, 1236, 1237, 1238)


@torch.no_grad()
def model_q_and_v(model, sim, data, traj, t):
    dev = next(model.parameters()).device
    x, _ = windows(data, torch.tensor([traj], device=dev), torch.tensor([t], device=dev), model.k)
    out, q = model(x)
    return out, (float(q[0]) if q is not None else None)


@torch.no_grad()
def scalar_q_series(sim, data, mask_th=3.0689072608947754, window=16):
    e, b, free = base_residual_terms(sim, data)
    B, T, N = e.shape
    lam = 1e-3 * (b * b)[(b * b) > 0].mean()
    q = torch.full((B, T), float('nan'), device=e.device)
    for t in range(1, T):
        s0 = max(0, t - window)
        bb = b[:, s0:t] * free[:, s0:t]
        num = (bb * e[:, s0:t]).sum((1, 2))
        den = (bb * bb).sum((1, 2))
        ok = den > mask_th
        q[:, t] = torch.where(ok, num / (den + lam), torch.full_like(num, float('nan')))
    return q


@torch.no_grad()
def proj_gain(model, sim, data, traj, t, threshold):
    dev = next(model.parameters()).device
    x, _ = windows(data, torch.tensor([traj], device=dev), torch.tensor([t], device=dev), model.k)
    res = model(x)
    out = res[0] if isinstance(res, tuple) else res
    state = data['states'][traj, t]
    u = data['stimulus'][traj, t]
    xb = sim.step(state[None], u[None], torch.zeros(1, 2, device=dev), data['silence'])[0]
    xt = sim.step(state[None], u[None], data['z'][traj, t][None], data['silence'])[0]
    isyn = state[:, 1] @ sim.W
    free = (state[:, 2] <= 0) & (xb[:, 1] <= .5) & (xt[:, 1] <= .5)
    if float((isyn[free] * isyn[free]).sum()) < 10.0:
        return None
    dh = out['v'][0] - xb[:, 0]
    ge = float((dh[free] * isyn[free]).sum() / (isyn[free] * isyn[free]).sum().clamp(min=1e-12)) / sim.cfg.alpha + 1.0
    return ge


@torch.no_grad()
def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', nargs='+', type=int, default=list(SEEDS))
    args = p.parse_args()
    torch.set_num_threads(2)
    cfg, lc, conn = setup_v2('hidden')
    sim = HiddenStateLIFSimulatorV2(conn, cfg, torch.device('cuda'), lc)
    branches = {k: build_branches(sim, cfg, k) for k in ('phase_jump', 'regime', 'sham')}
    rows = []
    for seed in args.seeds:
        m1, th1 = load_checked('global_k32', seed, conn, torch.device('cuda'))
        m2 = PlainResidualModel(conn, cfg).cuda()
        m2.load_state_dict(torch.load(ROOT / 'checkpoints' / f'm2_plain_seed{seed}' / 'best_val.pt',
                                      map_location='cuda', weights_only=False)['state_dict'])
        m2 = m2.eval()
        m3 = GainChannelModel(conn, cfg).cuda()
        m3.load_state_dict(torch.load(ROOT / 'checkpoints' / f'm3_gain_seed{seed}' / 'best_val.pt',
                                      map_location='cuda', weights_only=False)['state_dict'])
        m3 = m3.eval()
        for kind, (ctrl, inter) in branches.items():
            qs4 = {name: scalar_q_series(sim, d) for name, d in (('ctrl', ctrl), ('int', inter))}
            for traj in range(N):
                for d in DS:
                    t = TAU + d
                    tq_c = float(sim.gain(ctrl['z'][traj:traj + 1, t]))
                    tq_i = float(sim.gain(inter['z'][traj:traj + 1, t]))
                    row = dict(seed=seed, condition=kind, traj=traj, d=d,
                               true_q_ctrl=tq_c - 1, true_q_int=tq_i - 1)
                    # M1 implied gain (info-masked projection)
                    g1c = proj_gain(m1, sim, ctrl, traj, t, th1)
                    g1i = proj_gain(m1, sim, inter, traj, t, th1)
                    row['m1_ge_ctrl'] = g1c
                    row['m1_ge_int'] = g1i
                    # M2 implied gain
                    g2c = proj_gain(m2, sim, ctrl, traj, t, th1)
                    g2i = proj_gain(m2, sim, inter, traj, t, th1)
                    row['m2_ge_ctrl'] = g2c
                    row['m2_ge_int'] = g2i
                    # M3 own q_hat
                    _, q3c = model_q_and_v(m3, sim, ctrl, traj, t)
                    _, q3i = model_q_and_v(m3, sim, inter, traj, t)
                    row['m3_q_ctrl'] = q3c
                    row['m3_q_int'] = q3i
                    # M4 scalar estimator
                    row['m4_q_ctrl'] = float(qs4['ctrl'][traj, t]) if not torch.isnan(qs4['ctrl'][traj, t]) else None
                    row['m4_q_int'] = float(qs4['int'][traj, t]) if not torch.isnan(qs4['int'][traj, t]) else None
                    rows.append(row)
        print('stage3 intervention done seed', seed, flush=True)
        del m1, m2, m3
        torch.cuda.empty_cache()
    with (ROOT / 'metrics_stage3_intervention.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print('ROWS', len(rows), flush=True)


if __name__ == '__main__':
    main()
