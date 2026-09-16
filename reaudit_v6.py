"""Stage 1: Gate C reaudit under the frozen d/h protocol (no training)."""
import csv
import json
from pathlib import Path
import numpy as np
import torch
from run_latent_v2 import setup as setup_v2
from alias_eval_v5 import load_checked
from latent_data import windows
from lif_latent_v2 import HiddenStateLIFSimulatorV2
from eval_latent import state_from_output

ROOT = Path('results/latent_state_v6')
TAU = 128
DS = (0, 1, 2, 4, 8, 16, 32)
HS = (1, 4, 8)
N = 32
SEEDS80 = list(range(80_000_000, 80_000_000 + N))


def build_branches(sim, cfg, kind):
    """Control + intervention branches sharing stimulus and exogenous noise."""
    ctrl = sim.generate(SEEDS80, 'test_seen')
    if kind == 'phase_jump':
        inter = sim.generate(SEEDS80, 'test_seen', (TAU, 'phase_jump', 1.5))
    elif kind == 'regime':
        parts = []
        zr = ctrl['z'][:, TAU]
        for i in range(N):
            parts.append(sim.generate([SEEDS80[i]], 'test_seen',
                                      (TAU, 'regime', float(-zr[i, 0]), float(-zr[i, 1]))))
        inter = {k: torch.cat([p[k] for p in parts]) for k in parts[0]}
    elif kind == 'sham':
        inter = sim.generate(SEEDS80, 'test_seen', (TAU, 'vel_flip'))
    else:
        raise ValueError(kind)
    return ctrl, inter


@torch.no_grad()
def implied_gain(model, sim, data, traj, t, threshold):
    """Model's implied gain from its V prediction at (traj, t), projected on I_syn."""
    dev = next(model.parameters()).device
    x, _ = windows(data, torch.tensor([traj], device=dev), torch.tensor([t], device=dev), model.k)
    z = data['z'][traj, t][None] if model.oracle else None
    out = model(x, z=z)
    state = data['states'][traj, t]
    u = data['stimulus'][traj, t]
    xb = sim.step(state[None], u[None], torch.zeros(1, 2, device=dev), data['silence'])[0]
    isyn = state[:, 1] @ sim.W
    dh = out['v'][0] - xb[:, 0]
    den = (isyn * isyn).sum().clamp(min=1e-12)
    ge = float((dh * isyn).sum() / den) / sim.cfg.alpha + 1.0
    return ge, out, float(den)


@torch.no_grad()
def short_horizon(model, sim, data, traj, t, h, threshold):
    """Predict V at t+h with context frozen at t; true future stimulus per contract."""
    dev = next(model.parameters()).device
    x, _ = windows(data, torch.tensor([traj], device=dev), torch.tensor([t], device=dev), model.k)
    cfg = sim.cfg
    for step in range(1, h + 1):
        z = data['z'][traj, t + step - 1][None] if model.oracle else None
        out = model(x, z=z)
        state = state_from_output(out, cfg, threshold)
        if step < h:
            stim = data['stimulus'][traj, t + step]
            x = torch.cat((x[:, 1:], torch.cat((state, stim[None, :, None]), -1)[:, None]), 1)
    return out['v'][0].cpu()


@torch.no_grad()
def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', nargs='+', type=int, default=[1234])
    args = p.parse_args()
    torch.set_num_threads(2)
    cfg, lc, conn = setup_v2('hidden')
    sim = HiddenStateLIFSimulatorV2(conn, cfg, torch.device('cuda'), lc)
    branches = {k: build_branches(sim, cfg, k) for k in ('phase_jump', 'regime', 'sham')}
    rows = []
    for seed in args.seeds:
        models = {lab: load_checked(lab, seed, conn, torch.device('cuda'))
                  for lab in ('gnn_k1', 'set_k32', 'global_k32', 'oracle')}
        for lab, (model, threshold) in models.items():
            for kind, (ctrl, inter) in branches.items():
                # d=0 sanity: identical inputs must give identical predictions
                g0c, o0c, _ = implied_gain(model, sim, ctrl, 0, TAU, threshold)
                g0i, o0i, _ = implied_gain(model, sim, inter, 0, TAU, threshold)
                dv = float((o0c['v'] - o0i['v']).abs().max())
                if dv > 1e-6 and kind != 'sham':
                    print(f'WARN d=0 mismatch {lab} {kind} {dv:.2e}', flush=True)
                for traj in range(N):
                    for d in DS:
                        gc, _, den_c = implied_gain(model, sim, ctrl, traj, TAU + d, threshold)
                        gi, _, _ = implied_gain(model, sim, inter, traj, TAU + d, threshold)
                        qc = float(sim.gain(ctrl['z'][traj:traj + 1, TAU + d]))
                        qi = float(sim.gain(inter['z'][traj:traj + 1, TAU + d]))
                        row = dict(seed=seed, model=lab, condition=kind, traj=traj, d=d,
                                   pred_gain_ctrl=gc, pred_gain_int=gi,
                                   true_gain_ctrl=qc, true_gain_int=qi,
                                   true_dq=qi - qc, pred_dq=gi - gc,
                                   isyn_power=float(den_c))
                        if d in (0, 1, 2, 4, 8, 16, 32):
                            for h in HS:
                                vi = short_horizon(model, sim, inter, traj, TAU + d, h, threshold)
                                ti = inter['states'][traj, TAU + d + h, :, 0].cpu()
                                row[f'v_err_int_h{h}'] = float((vi - ti).square().mean().sqrt())
                                vc = short_horizon(model, sim, ctrl, traj, TAU + d, h, threshold)
                                tc = ctrl['states'][traj, TAU + d + h, :, 0].cpu()
                                row[f'v_err_ctrl_h{h}'] = float((vc - tc).square().mean().sqrt())
                        rows.append(row)
            print('reaudit done', lab, seed, flush=True)
        for lab in list(models):
            del models[lab]
        torch.cuda.empty_cache()
    with (ROOT / 'metrics_reaudit.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print('ROWS', len(rows), flush=True)


if __name__ == '__main__':
    main()
