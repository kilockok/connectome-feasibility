"""Stage 6: is the Ordered representation a latent tracker or an error compensator?

6.1 same latent path, different observable perturbation history (injected
stimulus noise; z is autonomous in this teacher so the latent path is
identical by construction). 6.2 matched noise pattern, different latent
paths. Conditional decoding of z and of the injected error.
"""
import csv
import json
from pathlib import Path
import numpy as np
import torch
from run_latent_v2 import setup as setup_v2
from alias_eval_v5 import load_checked
from latent_data import windows, sample_indices
from lif_latent_v2 import HiddenStateLIFSimulatorV2
from dataset import sample_traj_params, build_stimulus
from latent_probe_v2 import fit_ridge, apply_probe

ROOT = Path('results/latent_state_v5')
SEEDS = (1234, 1235, 1236, 1237, 1238)
K = 32


@torch.no_grad()
def gen_with_noise(sim, cfg, seeds, split, noise_seed, noise_std=0.5):
    """Legal trajectories with injected stimulus noise; z path unchanged."""
    c = cfg
    ng = torch.Generator().manual_seed(noise_seed)
    stimuli, initial, silenced, paths = [], [], [], []
    for seed in seeds:
        g = torch.Generator().manual_seed(int(seed))
        p = sample_traj_params(g, c, split)
        u = build_stimulus(p, c)
        noise = torch.randn(u.shape, generator=ng) * noise_std
        keep = torch.rand(u.shape, generator=ng) < 0.5
        u = u + noise * keep
        stimuli.append(u)
        initial.append(torch.stack((torch.rand(c.n_neurons, generator=g) * c.v_th,
                                    torch.zeros(c.n_neurons), torch.zeros(c.n_neurons)), -1))
        silenced.append(p['silenced'])
        paths.append(sim.z_path(seed, None))
    u_clean = torch.stack([build_stimulus(sample_traj_params(torch.Generator().manual_seed(int(s)), c, split), c)
                           for s in seeds]).to(sim.device)
    u = torch.stack(stimuli).to(sim.device)
    x0 = torch.stack(initial).to(sim.device)
    sil = torch.stack(silenced).to(sim.device)
    z = torch.stack(paths).to(sim.device)
    states = [x0]
    for t in range(c.T):
        states.append(sim.step(states[-1], u[:, t], z[:, t], sil))
    return dict(states=torch.stack(states, 1), stimulus=u, z=z, silence=sil,
                noise=(u - u_clean), noise_seed=noise_seed)


@torch.no_grad()
def reprs(model, data, t_end=200, stride=4):
    dev = next(model.parameters()).device
    hs, zs, es = [], [], []
    for traj in range(len(data['states'])):
        for t in range(K - 1, t_end, stride):
            b = torch.tensor([traj], device=dev)
            tt = torch.tensor([t], device=dev)
            x, _ = windows(data, b, tt, K)
            _, h = model.encode(x)
            hs.append(h[0].cpu())
            zs.append(data['z'][traj, t].cpu())
            es.append(data['noise'][traj, t - K + 1:t + 1].abs().mean().cpu())
    return torch.stack(hs), torch.stack(zs), torch.stack(es)


def r2_of(pred, true):
    pred, true = pred.double(), true.double()
    den = (true - true.mean()).square().sum()
    return float(1 - (pred - true).square().sum() / den.clamp(min=1e-12))


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', nargs='+', type=int, default=list(SEEDS))
    args = p.parse_args()
    torch.set_num_threads(2)
    cfg, lc, conn = setup_v2('hidden')
    sim = HiddenStateLIFSimulatorV2(conn, cfg, torch.device('cuda'), lc)
    base_seeds = list(range(90_000_000, 90_000_000 + 24))
    alt_seeds = list(range(91_000_000, 91_000_000 + 24))
    # 6.1: same z path (same trajectory seeds), different injected noise
    dA = gen_with_noise(sim, cfg, base_seeds, 'test_seen', noise_seed=1)
    dB = gen_with_noise(sim, cfg, base_seeds, 'test_seen', noise_seed=2)
    # 6.2: same noise pattern, different z paths
    dC = gen_with_noise(sim, cfg, alt_seeds, 'test_seen', noise_seed=2)
    rows = []
    for seed in args.seeds:
        model, th = load_checked('global_k32', seed, conn, torch.device('cuda'))
        hA, zA, eA = reprs(model, dA)
        hB, zB, eB = reprs(model, dB)
        hC, zC, eC = reprs(model, dC)
        # z decode stability under perturbation (train on A, test on A and B)
        ntr = int(len(hA) * .6)
        m = fit_ridge((hA[:ntr], zA[:ntr, 0:1]), (hA[ntr:ntr + len(hA) // 5], zA[ntr:ntr + len(hA) // 5, 0:1]))
        z_r2_A = r2_of(apply_probe(m, hA[ntr + len(hA) // 5:])[:, 0], zA[ntr + len(hA) // 5:, 0])
        z_r2_B = r2_of(apply_probe(m, hB[ntr + len(hB) // 5:])[:, 0], zB[ntr + len(hB) // 5:, 0])
        # representation distance: same latent (A,B) vs different latent (A,C)
        rep_dist_same_latent = float((hA - hB).norm(dim=-1).mean())
        rep_dist_diff_latent = float((hA - hC).norm(dim=-1).mean())
        # error decoding: window noise label = mean |noise| is not recoverable from sim;
        # use the noise_seed identity as the error label via a binary classifier on A vs B inputs
        # conditional decoding: z from h controlling for injected-noise proxy (stimulus window power)
        def residualize(y, c):
            mm = fit_ridge((c[:, None], y[:, None]), (c[:, None], y[:, None]))
            return y - apply_probe(mm, c[:, None])[:, 0]
        z_resid = residualize(zA[:, 0], eA)
        e_resid = residualize(eA, zA[:, 0])
        m2 = fit_ridge((hA[:ntr], z_resid[:ntr, None]), (hA[ntr:], z_resid[ntr:, None]))
        z_given_e = r2_of(apply_probe(m2, hA[ntr:])[:, 0], z_resid[ntr:])
        m3 = fit_ridge((hA[:ntr], e_resid[:ntr, None]), (hA[ntr:], e_resid[ntr:, None]))
        e_given_z = r2_of(apply_probe(m3, hA[ntr:])[:, 0], e_resid[ntr:])
        rows.append(dict(seed=seed, z_r2_clean=z_r2_A, z_r2_perturbed=z_r2_B,
                         rep_dist_same_latent=rep_dist_same_latent,
                         rep_dist_diff_latent=rep_dist_diff_latent,
                         z_r2_given_error=z_given_e, error_r2_given_z=e_given_z))
        print('error-vs-latent done seed', seed, flush=True)
        del model
        torch.cuda.empty_cache()
    with (ROOT / 'table5_error_vs_latent.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print('ROWS', len(rows), flush=True)


if __name__ == '__main__':
    main()
