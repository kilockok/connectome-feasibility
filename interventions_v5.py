"""Stage 5: latent-intervention response alignment + observable-perturbation control."""
import csv
import json
from pathlib import Path
import numpy as np
import torch
from run_latent_v3 import datasets_v3
from run_latent_v2 import setup as setup_v2
from alias_eval_v5 import load_checked
from latent_data import windows, history_control
from lif_latent_v2 import HiddenStateLIFSimulatorV2
from eval_latent import first_sustained

ROOT = Path('results/latent_state_v5')
SEEDS = (1234, 1235, 1236, 1237, 1238)
AT = 128
N = 32


@torch.no_grad()
def response_curve(model, sim, data, z_fn, n=N):
    """Representation and predicted-residual responses along a trajectory."""
    dev = next(model.parameters()).device
    rows = []
    isyn = data['states'][..., 1] @ sim.W  # [B,T+1,N]
    for delay in range(-8, 33):
        end = AT + delay
        b = torch.arange(len(data['states']), device=dev)
        t = torch.full((len(b),), end, device=dev)
        x, _ = windows(data, b, t, model.k)
        out = model(x)
        _, h = model.encode(x)
        z = data['z'][:, end]
        # true residual at this step (teacher - base on the true state)
        xt = sim.step(data['states'][:, end], data['stimulus'][:, end], z, data['silence'])
        xb = sim.step(data['states'][:, end], data['stimulus'][:, end], torch.zeros_like(z), data['silence'])
        res = (xt - xb)[..., 0]
        free = (data['states'][:, end, :, 2] <= 0) & (xb[..., 1] <= .5) & (xt[..., 1] <= .5)
        # predicted residual: model V minus base branch, projected onto I_syn
        ii = isyn[:, end]
        dh = out['v'] - xb[..., 0]
        aLIF = sim.cfg.alpha
        ge = ((dh * ii).sum(-1) / (ii * ii).sum(-1).clamp(min=1e-12)) / aLIF + 1.0
        rows.append(dict(delay=delay,
                         true_z_pos=z[:, 0].cpu(), true_z_vel=z[:, 1].cpu(),
                         true_gain=sim.gain(z).cpu(),
                         repr=h.cpu(), pred_gain=ge.cpu(),
                         res_v=res.cpu(), free=free.cpu()))
    return rows


@torch.no_grad()
def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', nargs='+', type=int, default=list(SEEDS))
    args = p.parse_args()
    torch.set_num_threads(2)
    cfg, lc, conn = setup_v2('hidden')
    dev = torch.device('cuda')
    sim = HiddenStateLIFSimulatorV2(conn, cfg, dev, lc)
    conditions = {
        'phase_jump': (AT, 'phase_jump', 1.5),
        'regime': None,  # per-trajectory reflection
        'vel_flip': (AT, 'vel_flip'),
        'obs_bump': 'observable',
    }
    datas = {}
    for name, spec in conditions.items():
        if name == 'obs_bump':
            d0 = sim.generate(list(range(80_000_000, 80_000_000 + N)), 'test_seen')
            # observable perturbation: +0.5 V to all neurons at AT, z unchanged
            d = {k: v.clone() for k, v in d0.items()}
            d['states'][:, AT, :, 0] += 0.5
            # re-simulate from AT onward with teacher to keep dynamics legal
            states = [d['states'][:, AT]]
            for t in range(AT, cfg.T):
                states.append(sim.step(states[-1], d['stimulus'][:, t], d['z'][:, t], d['silence']))
            d['states'] = torch.cat((d['states'][:, :AT + 1], torch.stack(states[1:], 1)), 1)
            datas[name] = d
        elif name == 'regime':
            parts = []
            zr = None
            for i in range(N):
                d0 = sim.generate([80_000_000 + i], 'test_seen')
                z0 = d0['z'][0, AT]
                parts.append(sim.generate([80_000_000 + i], 'test_seen',
                                          (AT, 'regime', float(-z0[0]), float(-z0[1]))))
            datas[name] = {k: torch.cat([di[k] for di in parts]) for k in parts[0]}
        else:
            datas[name] = sim.generate(list(range(80_000_000, 80_000_000 + N)), 'test_seen', spec)
    rows_out = []
    for seed in args.seeds:
        models = {lab: load_checked(lab, seed, conn, dev)[0] for lab in ('global_k32', 'set_k32', 'deriv')}
        for cond, d in datas.items():
            curves = {lab: response_curve(m, sim, d, None) for lab, m in models.items()}
            for delay in range(-8, 33):
                zpos = curves['global_k32'][delay + 8]['true_z_pos']
                gain = curves['global_k32'][delay + 8]['true_gain']
                row = dict(seed=seed, condition=cond, delay=delay,
                           true_z_pos=float(zpos.mean()), true_gain=float(gain.mean()))
                for lab in models:
                    c = curves[lab][delay + 8]
                    row[f'{lab}_pred_gain'] = float(c['pred_gain'].mean())
                    row[f'{lab}_repr_norm'] = float(c['repr'].float().norm(dim=-1).mean())
                rows_out.append(row)
        print('interventions done seed', seed, flush=True)
    with (ROOT / 'table4_interventions.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0]))
        w.writeheader(); w.writerows(rows_out)
    print('ROWS', len(rows_out), flush=True)


if __name__ == '__main__':
    main()
