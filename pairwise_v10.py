"""v10 Stage 7: GAIN-vs-STP pairwise benchmark + factorial intervention design.

Part 1: D(GAIN,STP; d) per library entry on design contexts (gain+stp
families), z/SKL between the two candidates' theta-marginalized rollouts.
Part 2 (Figure 7): factorial 2x2x2 teacher experiment -
  factor A: global current (low / high via EXC8 pre-excitation)
  factor B: edge history (fresh / loaded via J preload)
  factor C: delay (short 4 / long 32)
  response = postsyn V AUC after the probe (baseline-subtracted),
  R(M, A, B, C) mean over 16 testB contexts per family (teacher truth).
"""
import csv
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from connectome import Connectome
from run_v7 import setup_cfg
from protocol_v9 import ROOT as ROOT9, FAMILIES, SELECTED, spec_for
from teachers_v9 import MechanismLIFSimulator
from common_v10 import CANDIDATES
from library_v10 import build_stim, response_window, cost as dcost
from intervention_v9 import out_neighbors, J, TAU
from design_v10 import passive_fits, estates_for, rollouts_for_entry
from library_v10 import EXC8

ROOT = Path('results/latent_state_v10')
NCTX = 8


@torch.no_grad()
def part1(cfg, conn, dev, store, entries, Inb, g_amp):
    """GAIN-STP pairwise z per entry on design contexts."""
    blob = store
    rows = []
    for fam in ('gain', 'stp'):
        d = blob[f'{fam}/train']
        idxs = torch.linspace(0, len(d['states']) - 1, NCTX).long()
        states = d['states'][idxs].to(dev)
        stim = d['stimulus'][idxs].to(dev)
        fits = passive_fits(states, stim, cfg, conn, dev)
        est = estates_for(states, stim, cfg, conn, fits)
        for e in entries:
            es = torch.stack([build_stim(e, cfg, Inb, g_amp)] * len(idxs)).to(dev)
            stim_seg = stim[:, TAU:] + es[:, TAU:]
            preds = rollouts_for_entry(states, stim_seg, cfg, conn, fits, est, response_window(e))
            mg, ms = preds['gain'][0], preds['stp'][0]
            mu_base = preds['null'][0]
            resp = ((mg - mu_base).abs() + (ms - mu_base).abs()).mean(1)
            topn = resp.mean(0).topk(min(32, resp.shape[1])).indices
            mi, mj = mg[:, :, topn], ms[:, :, topn]
            si2 = fits['gain']['s2'][:, None, None] + preds['gain'][1][:, :, topn] + 1e-6
            sj2 = fits['stp']['s2'][:, None, None] + preds['stp'][1][:, :, topn] + 1e-6
            z = ((mi - mj).abs() / (si2 + sj2).sqrt()).mean((1, 2))
            kl = 0.5 * (si2 / sj2 + (mi - mj) ** 2 / sj2 - 1 + torch.log(sj2 / si2)) \
               + 0.5 * (sj2 / si2 + (mi - mj) ** 2 / si2 - 1 + torch.log(si2 / sj2))
            skl = (0.5 * kl).clamp(min=0).mean((1, 2))
            rows.append(dict(family_context=fam, id=e['id'], ifamily=e['family'],
                             gain_stp_z=float(z.mean()), gain_stp_skl=float(skl.mean()),
                             cost=dcost(e, 24)))
        del states, stim, fits, est
        torch.cuda.empty_cache()
    path = ROOT / 'metrics' / 'pairwise_gain_stp.csv'
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    # figure 3b: gain-stp skl ranking
    agg = {}
    for r in rows:
        agg.setdefault(r['id'], []).append(r['gain_stp_skl'])
    ids = sorted(agg, key=lambda k: -np.mean(agg[k]))
    fam_of = {r['id']: r['ifamily'] for r in rows}
    colors = dict(zip(sorted(set(fam_of.values())), plt.cm.tab10.colors))
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.bar(range(len(ids)), [np.mean(agg[k]) for k in ids],
           color=[colors[fam_of[k]] for k in ids])
    ax.set_xticks(range(len(ids)), ids, rotation=75, ha='right', fontsize=6)
    ax.set_ylabel('symmetric KL (GAIN vs STP predictions)')
    ax.set_title('v10 Figure 3: GAIN-vs-STP intervention landscape (design contexts)')
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[f]) for f in colors]
    ax.legend(handles, list(colors), fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT / 'figures' / 'fig3_gain_stp.png', dpi=140)
    print('PART1 COMPLETE', flush=True)
    return rows


@torch.no_grad()
def part2(cfg, conn, dev, store, Inb):
    """Factorial 2x2x2 teacher response tensor."""
    from dataset import sample_traj_params, build_stimulus
    res = {}
    for fam in FAMILIES + ('null',):
        if fam == 'null':
            spec = __import__('teachers_v9', fromlist=['MechSpec']).MechSpec(name='null')
            pid = None
        else:
            pid = SELECTED[fam]['splitB'][0]
            spec = spec_for(fam, pid)
        sim = MechanismLIFSimulator(conn, cfg, dev, spec)
        seeds = [cfg.traj_seed('test_seen', 700 + i) for i in range(16)]
        for A in (0, 1):
            for Bf in (0, 1):
                for C in (4, 32):
                    u = torch.zeros(cfg.T, cfg.n_neurons)
                    t_probe = TAU + (C if Bf else 0) + (3 if A else 0)
                    if Bf:
                        u[TAU:TAU + 2, J] = 6.0
                    if A:
                        u[TAU + (C if Bf else 0):TAU + (C if Bf else 0) + 3, EXC8] = 5.0
                    u[t_probe:t_probe + 2, J] = 6.0
                    dd = sim.generate(seeds, 'test_seen', extra_stim=u.expand(16, -1, -1))
                    post = dd['states'][:, t_probe + 2:t_probe + 10, :, 0][:, :, Inb].mean((1, 2))
                    pre = dd['states'][:, t_probe - 8:t_probe, :, 0][:, :, Inb].mean((1, 2))
                    res[(fam, A, Bf, C)] = float((post - pre).mean())
        del sim
        torch.cuda.empty_cache()
        print('factorial teacher', fam, flush=True)
    # csv + figure 7
    rows = [dict(family=f, A=a, B=b, C=c, response=v)
            for (f, a, b, c), v in res.items()]
    path = ROOT / 'metrics' / 'factorial.csv'
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6), sharey=True)
    for ax, fam in zip(axes, FAMILIES + ('null',)):
        M = np.zeros((2, 2, 2))
        for A in (0, 1):
            for Bf in (0, 1):
                for Ci, C in enumerate((4, 32)):
                    M[A, Bf, Ci] = res[(fam, A, Bf, Ci)]
        im = ax.imshow(M.reshape(4, 2), cmap='RdBu_r',
                       vmin=-np.abs(M).max(), vmax=np.abs(M).max())
        ax.set_xticks((0, 1), ('C=4', 'C=32'), fontsize=8)
        ax.set_yticks((0, 1, 2, 3), ('A0B0', 'A0B1', 'A1B0', 'A1B1'), fontsize=8)
        ax.set_title(fam, fontsize=9)
    fig.suptitle('v10 Figure 7: factorial mechanism response tensor (postsyn V AUC, teacher)')
    fig.tight_layout()
    fig.savefig(ROOT / 'figures' / 'fig7_factorial.png', dpi=140)
    print('PART2 COMPLETE', flush=True)


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    cfg = setup_cfg()
    conn = Connectome.generate(cfg)
    dev = torch.device('cuda')
    blob = torch.load(ROOT9 / 'data' / 'v9_data.pt', map_location='cpu', weights_only=False)
    lib = json.loads((ROOT / 'protocol' / 'intervention_library.json').read_text())
    entries = lib['entries']
    Inb_full = out_neighbors(conn, J)
    Wcpu = conn.dense_weight('cpu')
    strength = Wcpu[J].abs().sum(0)[Inb_full]
    Inb = Inb_full[strength.argsort(descending=True)].numpy()
    g_amp = float(lib['g_amp'])
    part1(cfg, conn, dev, blob['store'], entries, Inb, g_amp)
    part2(cfg, conn, dev, blob['store'], Inb)


if __name__ == '__main__':
    main()
