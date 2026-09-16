"""v7 Stage 3B/C: intervention re-estimation (d protocol) + frozen short-horizon."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
from run_v7 import ROOT, setup_cfg, LABELS
from lif_adapt_v7 import AdaptationLIFSimulator, AdaptConfig
from latent_data import windows
from metrics import compute_loss
from models.residual_v7 import build_v7

TAU = 128
DS = (0, 1, 2, 4, 8, 16, 32)
HS = (1, 4, 8)
N = 32
SEEDS80 = list(range(80_000_000, 80_000_000 + N))


def build_branches(sim, cfg, kind):
    ctrl = sim.generate(SEEDS80, 'test_seen')
    if kind == 'up':
        inter = sim.generate(SEEDS80, 'test_seen', (TAU, 'a_scale', 2.0))
    elif kind == 'down':
        inter = sim.generate(SEEDS80, 'test_seen', (TAU, 'a_scale', 0.0))
    elif kind == 'sham':
        inter = sim.generate(SEEDS80, 'test_seen', (TAU, 'sham'))
    else:
        raise ValueError(kind)
    return ctrl, inter


@torch.no_grad()
def model_out(model, data, traj, t):
    dev = next(model.parameters()).device
    x, _ = windows(data, torch.tensor([traj], device=dev), torch.tensor([t], device=dev), model.k)
    a = data['a'][traj, t][None] if model.oracle else None
    return model(x, a=a)


@torch.no_grad()
def short_horizon(model, cfg, data, traj, t, h, threshold):
    dev = next(model.parameters()).device
    x, _ = windows(data, torch.tensor([traj], device=dev), torch.tensor([t], device=dev), model.k)
    a = data['a'][traj, t][None] if model.oracle else None
    for step in range(1, h + 1):
        out = model(x, a=a)
        p = (out['s_logits'].sigmoid() > threshold).float()
        v = out['v'].clamp(cfg.v_min, cfg.v_th * 3)
        r = out['r'].clamp(0, 1)
        v = torch.where((p > .5) | (r > .15), torch.full_like(v, cfg.v_reset), v)
        r = torch.where(p > .5, torch.ones_like(r), r)
        state = torch.stack((v, p, r), -1)
        if step < h:
            stim = data['stimulus'][traj, t + step]
            feat = torch.cat((state, stim[None, :, None]), -1)
            x = torch.cat((x[:, 1:], feat[:, None]), 1)
            if model.oracle:
                a = data['a'][traj, t + step][None]
    return out['v'][0].cpu()


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', nargs='+', type=int, default=[1234, 1235, 1236, 1237, 1238])
    args = p.parse_args()
    torch.set_num_threads(2)
    cfg = setup_cfg()
    adapt = AdaptConfig()
    conn = __import__('connectome', fromlist=['Connectome']).Connectome.generate(cfg)
    sim = AdaptationLIFSimulator(conn, cfg, torch.device('cuda'), adapt)
    branches = {k: build_branches(sim, cfg, k) for k in ('up', 'down', 'sham')}
    # task data for threshold calibration
    data_cache = {}
    rows = []
    for seed in args.seeds:
        models = {}
        for kind in LABELS:
            summary = json.loads((ROOT / 'natural' / 'training' / f'{kind}_seed{seed}.json').read_text())
            m = build_v7(kind, conn, cfg).cuda()
            m.load_state_dict(torch.load(summary['checkpoint'], map_location='cuda',
                                         weights_only=False)['state_dict'])
            models[kind] = m.eval()
        # frozen threshold from the natural eval of the same kind/seed
        ths = {kind: 0.9 for kind in LABELS}
        for kind, model in models.items():
            for cond, (ctrl, inter) in branches.items():
                # d=0 sanity for non-oracle
                o0c = model_out(model, ctrl, 0, TAU)
                o0i = model_out(model, inter, 0, TAU)
                dv = float((o0c['v'] - o0i['v']).abs().max()) if kind != 'oracle' else 0.0
                if dv > 1e-6:
                    print(f'WARN d=0 mismatch {kind} {cond} {dv:.2e}', flush=True)
                for traj in range(N):
                    for d in DS:
                        t = TAU + d
                        oc = model_out(model, ctrl, traj, t)
                        oi = model_out(model, inter, traj, t)
                        tc = ctrl['states'][traj, t + 1, :, 0].cpu()
                        ti = inter['states'][traj, t + 1, :, 0].cpu()
                        row = dict(seed=seed, model=kind, condition=cond, traj=traj, d=d,
                                   v_err_ctrl=float((oc['v'][0].cpu() - tc).square().mean().sqrt()),
                                   v_err_int=float((oi['v'][0].cpu() - ti).square().mean().sqrt()),
                                   pred_dv=float((oi['v'][0].cpu() - oc['v'][0].cpu()).square().mean().sqrt()),
                                   true_a_ctrl=float(ctrl['a'][traj, t].mean()),
                                   true_a_int=float(inter['a'][traj, t].mean()))
                        for h in HS:
                            row[f'v_h{h}_int'] = float((short_horizon(model, cfg, inter, traj, t, h, ths[kind])
                                                        - inter['states'][traj, t + h, :, 0].cpu()).square().mean().sqrt())
                        rows.append(row)
            print('intervention done', kind, seed, flush=True)
        del models
        torch.cuda.empty_cache()
    with (ROOT / 'metrics_intervention.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print('ROWS', len(rows), flush=True)


if __name__ == '__main__':
    main()
