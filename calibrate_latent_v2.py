"""Stage 7: teacher-only calibration sweep for the second-order latent.

Screens (alpha, beta, omega, sigma) on activity health and mechanistic branch
divergence before ANY model is trained:

  A) z_pos causal strength: identical (X, U), gain(z_pos=+v) vs gain(z_pos=-v)
     -> next-state RMSE / spike disagreement.
  B) z_vel order relevance: identical (X, z_pos, future U, z_vel=+v vs -v)
     -> branch divergence after h steps (the instantaneous gain is equal, so
        h=1 is ~0 by construction; later steps must diverge).

Selection (fixed before seeing results): among candidates with mean late
activity in (0.001, 0.15), fewer than half wholly inactive late trajectories,
and branch RMSE A > 0.001, pick the largest branch RMSE A.
"""
import json
from dataclasses import replace
from pathlib import Path
import torch
from config import Config
from connectome import Connectome
from lif_latent_v2 import HiddenStateLIFSimulatorV2, LatentV2Config

ROOT = Path('results/latent_state_v2')
SEEDS = list(range(70_000_000, 70_000_000 + 16))


def base_config():
    sel = json.loads((Path('results/latent_state_v1/teacher_calibration/selection.json')).read_text())
    cfg = Config(**sel['config'])
    return replace(cfg, n_train_traj=512, n_val_traj=64, n_test_seen_traj=64, n_test_traj=64)


@torch.no_grad()
def branch_divergence(sim, data, v=1.0, h_max=32, n=8):
    """Mechanistic branches from shared observable states."""
    c = sim.cfg
    states = data['states'][:n]
    u = data['stimulus'][:n]
    t0 = 64
    x = states[:, t0]
    # A) z_pos = +v vs -v held constant for 8 steps (pure gain effect).
    za = torch.tensor([v, 0.0], device=x.device).expand(n, 2)
    zb = torch.tensor([-v, 0.0], device=x.device).expand(n, 2)
    xa, xb = x, x
    for h in range(1, 9):
        xa = sim.step(xa, u[:, t0 + h], za)
        xb = sim.step(xb, u[:, t0 + h], zb)
    a_rmse = (xa[..., 0] - xb[..., 0]).square().mean().sqrt()
    a_dis = (xa[..., 1] != xb[..., 1]).float().mean()
    # B) z_vel = +v vs -v, same z_pos=0: equal instantaneous gain, later divergence.
    va = torch.tensor([0.0, v], device=x.device).expand(n, 2)
    vb = torch.tensor([0.0, -v], device=x.device).expand(n, 2)
    xa, xb, divs = x, x, {}
    za, zb = va, vb
    for h in range(1, h_max + 1):
        za, zb = sim.z_step(za, torch.zeros(n, device=x.device)), sim.z_step(zb, torch.zeros(n, device=x.device))
        xa = sim.step(xa, u[:, t0 + h], za)
        xb = sim.step(xb, u[:, t0 + h], zb)
        if h in (1, 2, 4, 8, 16, 32):
            divs[h] = float((xa[..., 0] - xb[..., 0]).square().mean().sqrt())
    return dict(a_rmse=float(a_rmse), a_spike_disagreement=float(a_dis), b_divergence=divs)


@torch.no_grad()
def candidate_row(alpha, beta, omega, sigma, cfg, conn, dev):
    lc = LatentV2Config(alpha=alpha, beta=beta, omega=omega, sigma=sigma)
    sim = HiddenStateLIFSimulatorV2(conn, cfg, dev, lc)
    d = sim.generate(SEEDS, 'train')
    z, s = d['z'], d['states'][..., 1]
    late = s[:, -100:]
    rate = float(late.mean())
    inactive = float((late.mean((1, 2)) == 0).float().mean())
    p = z[..., 0].flatten()
    ac8 = float(torch.corrcoef(torch.stack((p[:-8], p[8:])))[0, 1])
    br = branch_divergence(sim, d)
    return dict(alpha=alpha, beta=beta, omega=omega, sigma=sigma,
                z_pos_std=float(z[..., 0].std()), z_vel_std=float(z[..., 1].std()),
                z_pos_autocorr8=ac8, late_activity=rate, inactive_fraction=inactive,
                branch_rmse=br['a_rmse'], branch_spike_disagreement=br['a_spike_disagreement'],
                vel_branch_rmse_h8=br['b_divergence'][8], vel_branch_rmse_h32=br['b_divergence'][32])


def main():
    torch.set_num_threads(2)
    dev = torch.device('cuda')
    cfg, conn = base_config(), None
    conn = Connectome.generate(cfg)
    out = ROOT / 'teacher_calibration'
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for alpha in (0.4, 0.7, 1.0):
        for beta in (0.98, 0.99):
            for omega in (0.06, 0.09, 0.12):
                for sigma in (0.008, 0.015):
                    if omega ** 2 >= 1 - beta:
                        continue
                    row = candidate_row(alpha, beta, omega, sigma, cfg, conn, dev)
                    rows.append(row)
                    print('a=%.1f b=%.2f w=%.2f s=%.3f | late=%.4f inact=%.2f branch=%.4f velB32=%.4f'
                          % (alpha, beta, omega, sigma, row['late_activity'], row['inactive_fraction'],
                             row['branch_rmse'], row['vel_branch_rmse_h32']), flush=True)
    import csv
    with (out / 'hidden_state_sweep.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    eligible = [r for r in rows if 0.001 < r['late_activity'] < 0.15
                and r['inactive_fraction'] < 0.5 and r['branch_rmse'] > 0.001]
    chosen = max(eligible, key=lambda r: r['branch_rmse']) if eligible else None
    selection = dict(status='selected' if chosen else 'no_eligible_candidate', latent=chosen,
                     rule='max branch RMSE among activity-eligible; calibration seeds only',
                     limit='Branch divergence is necessary, not sufficient; oracle ceiling is Stage 8.')
    (out / 'selection.json').write_text(json.dumps(selection, indent=2))
    print(json.dumps(selection, indent=2))


if __name__ == '__main__':
    main()
