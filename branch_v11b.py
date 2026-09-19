"""v11b Part A4: controlled branch experiments under TEACHER dynamics.

Replicates teachers_v9.generate() exactly (same per-seed RNG layout), keeping
the hidden state at the branch point t*. Variants from the same full teacher
state (visible + hidden):
  cont   : teacher continuation (reference)
  flip   : flip ONE neuron's spike at t* (with consistent V/R and hidden
           updates: adapt a and STP (u,x) see the flipped spike)
  corr   : apply ONE residual correction to vn at t* (no spike change)
Deviation vs continuation over H'=32 steps: V RMSE path, spike agreement,
new spike errors introduced. Branch neuron: max |I_syn| contributor at t*
(plus a fixed random control neuron per trajectory, seed-fixed).

This answers: does one spike error propagate? does one correction reduce
the deviation? does correction introduce new spike errors? - under teacher
dynamics, NOT under the model's own rollout.
"""
import csv
import json
from pathlib import Path
import numpy as np
import torch
from connectome import Connectome
from run_v7 import setup_cfg
from dataset import sample_traj_params, build_stimulus
from teachers_v9 import MechanismLIFSimulator
from protocol_v9 import ROOT as ROOT9, FAMILIES, SELECTED, spec_for

ROOT = Path('results/latent_state_v11b')
TSTAR = 112
HBR = 32
NTRAJ = 8


@torch.no_grad()
def run_teacher_with_hidden(sim, cfg, seeds, split, dev):
    """Exact replica of MechanismLIFSimulator.generate(), but returns the
    per-step state list AND the hidden state trajectory needed for branching
    (adapt a path; STP (u,x) at each t; gain/ou are precomputed paths)."""
    c = cfg
    spec = sim.spec
    stimuli, initial, silenced = [], [], []
    zpaths, a0s, oupaths = [], [], []
    for seed in seeds:
        g = torch.Generator().manual_seed(int(seed))
        p = sample_traj_params(g, c, split)
        u = build_stimulus(p, c)
        stimuli.append(u)
        initial.append(torch.stack((torch.rand(c.n_neurons, generator=g) * c.v_th,
                                    torch.zeros(c.n_neurons), torch.zeros(c.n_neurons)), -1))
        silenced.append(p['silenced'])
        if spec.gain is not None:
            zpaths.append(sim.z_init_path(seed, spec.gain))
        if spec.adapt is not None:
            a0s.append(sim.a_init(seed, spec.adapt))
        if spec.ou is not None:
            oupaths.append(sim.ou_path(seed, spec.ou))
    u = torch.stack(stimuli).to(dev)
    x = torch.stack(initial).to(dev)
    sil = torch.stack(silenced).to(dev)
    B = len(seeds)
    z = torch.stack(zpaths).to(dev) if spec.gain is not None else None
    a = torch.stack(a0s).to(dev) if spec.adapt is not None else None
    ou = torch.stack(oupaths).to(dev) if spec.ou is not None else None
    if spec.stp is not None:
        uu = sim.U.clone().expand(B, -1, -1).clone()
        xx = torch.ones(B, sim.N, sim.N, device=dev)
        stp_state = (uu, xx)
    else:
        stp_state = None
    states = [x]
    hiddens = []
    for t in range(c.T):
        zt = z[:, t] if z is not None else None
        out = ou[:, t] if ou is not None else None
        x, (_, a, stp_state, _) = sim.step(x, u[:, t], (zt, a, stp_state, out), sil)
        states.append(x)
        hiddens.append((a.clone() if a is not None else None,
                        (stp_state[0].clone(), stp_state[1].clone()) if stp_state is not None else None))
    return dict(states=torch.stack(states, 1), stimulus=u, silence=sil,
                hiddens=hiddens, z=z, ou=ou)


@torch.no_grad()
def continue_from(sim, cfg, x, a, stp_state, z, ou, u, sil, t0, H, dev):
    """Continue teacher stepping from full state at time t0 (state AFTER step
    t0-1 => next step index t0) for H steps. z/ou are precomputed paths."""
    spec = sim.spec
    out_states = []
    for t in range(t0, min(t0 + H, cfg.T)):
        zt = z[:, t] if z is not None else None
        out = ou[:, t] if ou is not None else None
        x, (_, a, stp_state, _) = sim.step(x, u[:, t], (zt, a, stp_state, out), sil)
        out_states.append(x)
    return torch.stack(out_states, 1) if out_states else torch.zeros(x.shape[0], 0, *x.shape[1:], device=dev)


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    cfg = setup_cfg()
    conn = Connectome.generate(cfg)
    dev = torch.device('cuda')
    W = conn.dense_weight(dev)
    rows = []
    rng = np.random.default_rng(777)
    for fam in FAMILIES:
        pid = SELECTED[fam]['splitB'][0]
        sim = MechanismLIFSimulator(conn, cfg, dev, spec_for(fam, pid))
        seeds = [cfg.traj_seed('test_seen', 64 + i) for i in range(NTRAJ)]
        d = run_teacher_with_hidden(sim, cfg, seeds, 'test_seen', dev)
        states, u, sil = d['states'], d['stimulus'], d['silence']
        B = len(seeds)
        # branch neuron: max |I_syn| contributor at t* (pre-registered rule),
        # control: fixed random neuron per trajectory
        s_prev = states[:, TSTAR - 1, :, 1]
        isyn = s_prev @ W
        j_main = isyn.abs().argmax(1).cpu().numpy()
        j_ctrl = torch.tensor(rng.integers(0, cfg.n_neurons, B))
        for variant in ('flip_main', 'flip_ctrl', 'corr_main'):
            devs_vs, devs_ss, newerr = [], [], []
            for b in range(B):
                xb = states[b:b + 1, TSTAR].clone()
                ab = d['hiddens'][TSTAR - 1][0][b:b + 1].clone() if d['hiddens'][TSTAR - 1][0] is not None else None
                stpb = tuple(v[b:b + 1].clone() for v in d['hiddens'][TSTAR - 1][1]) if d['hiddens'][TSTAR - 1][1] is not None else None
                zb = d['z'][b:b + 1] if d['z'] is not None else None
                oub = d['ou'][b:b + 1] if d['ou'] is not None else None
                ub = u[b:b + 1]; silb = sil[b:b + 1]
                # reference continuation from identical state
                ref = continue_from(sim, cfg, xb, ab, stpb, zb, oub, ub, silb, TSTAR, HBR, dev)
                # variant: modify state at TSTAR then continue
                xv = xb.clone()
                if variant.startswith('flip'):
                    j = int(j_main[b]) if variant == 'flip_main' else int(j_ctrl[b])
                    was = xv[0, j, 1].item() > 0.5
                    xv[0, j, 1] = 0.0 if was else 1.0
                    if was:   # un-spike: restore a plausible sub-threshold V
                        xv[0, j, 0] = 0.5 * cfg.v_th
                        xv[0, j, 2] = 0.0
                    else:     # force spike: reset + refractory
                        xv[0, j, 0] = cfg.v_reset
                        xv[0, j, 2] = 1.0
                else:
                    j = int(j_main[b])
                    xv[0, j, 0] = xv[0, j, 0] + 0.3 * cfg.v_th   # one residual-sized V correction
                out = continue_from(sim, cfg, xv, ab, stpb, zb, oub, ub, silb, TSTAR, HBR, dev)
                dv = (out[..., 0] - ref[..., 0]).square().mean(2).sqrt()      # [1,H]
                ds = (out[..., 1] != ref[..., 1]).float().mean(2)             # spike mismatch frac
                devs_vs.append(dv[0]); devs_ss.append(ds[0])
                newerr.append(int(ds[0].sum()))
            for h in range(len(devs_vs[0])):
                rows.append(dict(family=fam, variant=variant, h=h,
                                 v_rmse=float(torch.stack(devs_vs)[:, h].mean()),
                                 spike_mismatch=float(torch.stack(devs_ss)[:, h].mean())))
            print(fam, variant, 'mean new spike errors/traj:',
                  float(np.mean(newerr)), flush=True)
        del sim
        torch.cuda.empty_cache()
    outdir = ROOT / 'results' / 'rollout'
    outdir.mkdir(parents=True, exist_ok=True)
    with (outdir / 'branch.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print('BRANCH COMPLETE', len(rows), flush=True)


if __name__ == '__main__':
    main()
