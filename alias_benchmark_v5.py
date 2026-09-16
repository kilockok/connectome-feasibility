"""Stage 1: state-aliasing benchmark construction."""
import csv
import json
from pathlib import Path
import numpy as np
import torch
from run_latent_v3 import datasets_v3
from run_latent_v2 import setup as setup_v2
from lif_latent_v2 import HiddenStateLIFSimulatorV2

ROOT = Path('results/latent_state_v5')
K = 32
H = 8


def token_slice(data, traj, a, b):
    s = data['states'][traj, a:b + 1]
    u = data['stimulus'][traj, a:b + 1]
    return torch.cat((s, u.unsqueeze(-1)), -1)


@torch.no_grad()
def main():
    cfg, lc, conn = setup_v2('hidden')
    data = datasets_v3('hidden')['test_seen']
    sim = HiddenStateLIFSimulatorV2(conn, cfg, torch.device('cuda'), lc)
    T = data['stimulus'].shape[1]
    # candidate pool with activity and future room
    pool = []
    w = data['states'][..., 1]
    for traj in range(len(data['states'])):
        counts = w[traj].sum(-1).cumsum(0)
        for t in range(32, T - H - 1, 2):
            if int(counts[t] - counts[t - 32]) >= 8 and int(counts[t] - counts[t - 1]) >= 1:
                pool.append((traj, t))
    dev = data['states'].device
    idx = torch.tensor([[a, b] for a, b in pool], device=dev)
    v = data['states'][idx[:, 0], idx[:, 1], :, 0]
    s = data['states'][idx[:, 0], idx[:, 1], :, 1]
    r = data['states'][idx[:, 0], idx[:, 1], :, 2]
    u = data['stimulus'][idx[:, 0], idx[:, 1]]
    zp = data['z'][idx[:, 0], idx[:, 1], 0]
    zv = data['z'][idx[:, 0], idx[:, 1], 1]
    f = torch.cat((v, s, .5 * r, u), -1)
    f = (f - f.mean(0)) / f.std(0).clamp(min=1e-6)
    d = torch.cdist(f, f)
    dzp = (zp[:, None] - zp[None, :]).abs()
    tr = idx[:, 0]
    same_traj = tr[:, None] == tr[None, :]
    tiers = dict(strict=(.05, .75), medium=(.10, .60), loose=(.20, .50))
    out_rows = []
    for tier, (qx, qz) in tiers.items():
        xth = d.flatten().quantile(qx)
        zth = dzp.flatten().quantile(qz)
        ok = (d <= xth) & (dzp >= zth) & (~same_traj)
        cand = ok.nonzero()
        order = torch.argsort(d[cand[:, 0], cand[:, 1]])
        used, chosen = set(), []
        for oi in order.cpu().tolist():
            a, b = cand[oi].cpu().tolist()
            if a in used or b in used:
                continue
            chosen.append((a, b))
            used.add(a); used.add(b)
            if len(chosen) >= 48:
                break
        out_rows.extend((tier, a, b) for a, b in chosen)
    # negative matched pairs: close x AND close z_pos
    xth = d.flatten().quantile(.05)
    zlo = dzp.flatten().quantile(.25)
    ok = (d <= xth) & (dzp <= zlo) & (~same_traj)
    cand = ok.nonzero()
    order = torch.argsort(d[cand[:, 0], cand[:, 1]])
    used, chosen = set(), []
    for oi in order.cpu().tolist():
        a, b = cand[oi].cpu().tolist()
        if a in used or b in used:
            continue
        chosen.append((a, b))
        used.add(a); used.add(b)
        if len(chosen) >= 48:
            break
    out_rows.extend(('negative', a, b) for a, b in chosen)

    rows = []
    for tier, ai, bi in out_rows:
        (ta, t), (tb, s) = pool[ai], pool[bi]
        xa, xb = data['states'][ta, t], data['states'][tb, s]
        ua, ub = data['stimulus'][ta, t], data['stimulus'][tb, s]
        za, zb = data['z'][ta, t], data['z'][tb, s]
        sil = data['silence']
        Da = sim.step(xa[None], ua[None], za[None], sil)[0]
        Ba = sim.step(xa[None], ua[None], torch.zeros_like(za[None]), sil)[0]
        Db = sim.step(xb[None], ub[None], zb[None], sil)[0]
        Bb = sim.step(xb[None], ub[None], torch.zeros_like(zb[None]), sil)[0]
        resA, resB = (Da - Ba), (Db - Bb)
        rows.append(dict(
            tier=tier, A=f'{ta}:{t}', B=f'{tb}:{s}',
            d_x=float(torch.nn.functional.mse_loss(f[ai], f[bi]).sqrt()),
            dz_pos=float(za[0] - zb[0]), dz_vel=float(za[1] - zb[1]),
            dz_vel_sign_match=int(torch.sign(za[1]) == torch.sign(zb[1])),
            v_rmse=float((xa[:, 0] - xb[:, 0]).square().mean().sqrt()),
            s_disagree=float((xa[:, 1] != xb[:, 1]).float().mean()),
            u_rmse=float((ua - ub).square().mean().sqrt()),
            residual_v_rmse=float((resA[:, 0] - resB[:, 0]).square().mean().sqrt()),
            future1_v_rmse=float((Da[:, 0] - Db[:, 0]).square().mean().sqrt()),
            gain_a=float(sim.gain(za[None])), gain_b=float(sim.gain(zb[None]))))
    (ROOT / 'alias_pairs').mkdir(exist_ok=True)
    with (ROOT / 'alias_pairs_summary.csv').open('w', newline='') as fh:
        wcsv = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wcsv.writeheader(); wcsv.writerows(rows)
    torch.save(dict(pool=pool, rows=rows), ROOT / 'alias_pairs' / 'pairs.pt')
    for tier in ('strict', 'medium', 'loose', 'negative'):
        sel = [r for r in rows if r['tier'] == tier]
        print(tier, 'n=%d dx=%.4f |dz_pos|=%.3f resV=%.4f fut1V=%.4f' % (
            len(sel), np.mean([r['d_x'] for r in sel]), np.mean([abs(r['dz_pos']) for r in sel]),
            np.mean([r['residual_v_rmse'] for r in sel]), np.mean([r['future1_v_rmse'] for r in sel])))


if __name__ == '__main__':
    main()
