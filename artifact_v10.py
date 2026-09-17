"""v10 Stage 1-2: artifact audit (Gate A prerequisite).

A. NULL equivalence: NULL teacher vs HARD base predictor must match to
   machine precision (bitwise); base_pre (soft sigmoid) residual shown for
   contrast - that soft path is the v9 artifact floor.
B. Reset-conditioned residual decomposition (free / refractory / reset /
   post-reset transitions) - where does the base_pre artifact live?
C. Mechanism residual decomposition vs HARD base (artifact-free): all-step,
   event-step, information-bearing-step, free-step RMS per family.
"""
import csv
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from connectome import Connectome
from run_v7 import setup_cfg
from protocol_v9 import ROOT as ROOT9, FAMILIES
from teachers_v9 import MechanismLIFSimulator, MechSpec
from models.residual_v7 import base_pre

ROOT = Path('results/latent_state_v10')


@torch.no_grad()
def hard_base_step(sim, x, u, sil):
    out, _ = sim.step(x, u, (None, None, None, None), sil)
    return out


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    cfg = setup_cfg()
    conn = Connectome.generate(cfg)
    dev = torch.device('cuda')
    sim = MechanismLIFSimulator(conn, cfg, dev, MechSpec(name='null'))
    W = conn.dense_weight(dev)
    ib = conn.i_bias.to(dev) if conn.i_bias is not None else torch.zeros(cfg.n_neurons, device=dev)
    blob = torch.load(ROOT9 / 'data' / 'v9_data.pt', map_location='cpu', weights_only=False)
    store = blob['store']
    rows = []

    # ---------- A. NULL equivalence ----------
    d = store['null/test']
    states, stim, sil = d['states'].to(dev), d['stimulus'].to(dev), d['silence'].to(dev)
    preds = []
    for t in range(states.shape[1] - 1):
        preds.append(hard_base_step(sim, states[:, t], stim[:, t], sil))
    hb = torch.stack(preds, 1)                      # [B,T,N,3]
    tgt = states[:, 1:]
    dv = (hb[..., 0] - tgt[..., 0]).abs()
    ds = (hb[..., 1] - tgt[..., 1]).abs()
    rows.append(dict(section='A', metric='null_hardbase_v_maxabs', value=float(dv.max())))
    rows.append(dict(section='A', metric='null_hardbase_v_rms', value=float(dv.square().mean().sqrt())))
    rows.append(dict(section='A', metric='null_hardbase_spike_disagree', value=float(ds.mean())))
    # base_pre soft path on the same NULL data (the v9 artifact)
    xf = torch.cat((states[:, :-1], stim.unsqueeze(-1)), -1)
    B, T, N, _ = xf.shape
    soft = base_pre(xf.reshape(B * T, N, 4), cfg, W, ib)
    r_soft = (soft['v_base'].reshape(B, T, N) - tgt[..., 0])
    rows.append(dict(section='A', metric='null_basepre_soft_v_rms', value=float(r_soft.square().mean().sqrt())))
    rows.append(dict(section='A', metric='null_basepre_soft_v_maxabs', value=float(r_soft.abs().max())))

    # ---------- B. reset-conditioned decomposition (base_pre artifact) ----------
    refr = xf[..., 2] > 0
    vn_pre = (xf.reshape(B * T, N, 4)[..., 0] + 0)  # placeholder
    fire_base = soft['logit_base'].reshape(B, T, N) > 0
    fire_true = tgt[..., 1] > 0.5
    free = (~refr) & (~fire_true)
    pre_reset = refr
    reset = fire_true
    post_reset = torch.zeros_like(free)
    post_reset[:, 1:] = reset[:, :-1]
    for name, mask in (('free', free), ('pre_reset', pre_reset),
                       ('reset', reset), ('post_reset', post_reset)):
        if mask.any():
            rows.append(dict(section='B', metric=f'basepre_soft_rms_{name}',
                             value=float(r_soft[mask].square().mean().sqrt())))
            rows.append(dict(section='B', metric=f'basepre_soft_frac_{name}',
                             value=float(mask.float().mean())))

    # ---------- C. mechanism residual vs HARD base ----------
    for fam in FAMILIES:
        d = store[f'{fam}/testA']
        states, stim, sil = d['states'].to(dev), d['stimulus'].to(dev), d['silence'].to(dev)
        preds = []
        for t in range(states.shape[1] - 1):
            preds.append(hard_base_step(sim, states[:, t], stim[:, t], sil))
        hb = torch.stack(preds, 1)
        tgt = states[:, 1:]
        e = tgt[..., 0] - hb[..., 0]
        refr = states[:, :-1, :, 2] > 0
        fire_true = tgt[..., 1] > 0.5
        free = (~refr) & (~fire_true)
        isyn = torch.einsum('btj,ji->bti', states[:, :-1, :, 1], W)
        info = free & (isyn.abs() > isyn.abs().median())
        event = fire_true | (states[:, :-1, :, 1] > 0.5)
        for name, mask in (('all', torch.ones_like(free)), ('free', free),
                           ('event', event), ('info_bearing', info)):
            rows.append(dict(section='C', metric=f'{fam}_hardresid_rms_{name}',
                             value=float(e[mask].square().mean().sqrt())))
        rows.append(dict(section='C', metric=f'{fam}_hardresid_mae_free',
                         value=float(e[free].abs().mean())))
        # same decomposition for the soft base_pre residual (contaminated)
        xf = torch.cat((states[:, :-1], stim.unsqueeze(-1)), -1)
        Bf, Tf = xf.shape[0], xf.shape[1]
        soft = base_pre(xf.reshape(Bf * Tf, N, 4), cfg, W, ib)
        rs = (soft['v_base'].reshape(Bf, Tf, N) - tgt[..., 0])
        rows.append(dict(section='C', metric=f'{fam}_softresid_rms_all',
                         value=float(rs.square().mean().sqrt())))
        rows.append(dict(section='C', metric=f'{fam}_softresid_rms_free',
                         value=float(rs[free].square().mean().sqrt())))
        del states, stim, hb, preds
        torch.cuda.empty_cache()

    path = ROOT / 'metrics' / 'artifact.csv'
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['section', 'metric', 'value'])
        w.writeheader(); w.writerows(rows)
    for r in rows:
        print(r['section'], r['metric'], f"{r['value']:.6f}", flush=True)

    # ---------- Figure 1: artifact decomposition ----------
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    m = {r['metric']: r['value'] for r in rows}
    axes[0].bar(['hard base\n(NULL)', 'soft base_pre\n(NULL)'],
                [m['null_hardbase_v_rms'], m['null_basepre_soft_v_rms']],
                color=['tab:green', 'tab:red'])
    axes[0].set_ylabel('V residual RMS')
    axes[0].set_title('A: NULL reference path')
    cats = ['free', 'pre_reset', 'reset', 'post_reset']
    axes[1].bar(cats, [m[f'basepre_soft_rms_{c}'] for c in cats], color='tab:purple')
    axes[1].set_title('B: base_pre artifact by transition type (NULL)')
    x = np.arange(4)
    for j, fam in enumerate(FAMILIES):
        axes[2].bar(x + j * 0.25 - 0.25,
                    [m[f'{fam}_hardresid_rms_all'], m[f'{fam}_hardresid_rms_free'],
                     m[f'{fam}_hardresid_rms_info_bearing'], m[f'{fam}_hardresid_rms_event']],
                    0.25, label=fam)
    axes[2].axhline(m['null_basepre_soft_v_rms'], color='tab:red', ls='--',
                    label='soft artifact floor')
    axes[2].set_xticks(x, ['all', 'free', 'info-bearing', 'event'])
    axes[2].set_title('C: mechanism residual vs HARD base')
    axes[2].legend(fontsize=8)
    fig.suptitle('v10 Figure 1: artifact decomposition')
    fig.tight_layout()
    fig.savefig(ROOT / 'figures' / 'fig1_artifact.png', dpi=140)
    print('FIG1 SAVED')


if __name__ == '__main__':
    main()
