"""v6 figures + conclusion assembly."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('results/latent_state_v6')
(ROOT / 'figures').mkdir(exist_ok=True)
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
DS = (0, 1, 2, 4, 8, 16, 32)


def load(name):
    return list(csv.DictReader(open(ROOT / name)))


def f(x):
    return float(x)


inv = load('metrics_stage3_intervention.csv')
re = load('metrics_reaudit.csv')

# Fig 1: observation divergence propagation (true branch difference) + model q error
fig, ax = plt.subplots(figsize=(8, 4.5))
for cond, c in (('phase_jump', '#ca715b'), ('regime', '#247d90')):
    ys = [np.mean([f(r['true_q_int']) - f(r['true_q_ctrl']) for r in inv if r['condition'] == cond and int(r['d']) == d]) for d in DS]
    ax.plot(DS, ys, marker='o', color=c, label=f'true Δq ({cond})')
ax.axhline(0, color='gray', ls='--')
ax.set(xlabel='d (new observations absorbed after tau)', ylabel='true branch gain difference',
       title='Intervention effect on the observable dynamics (effect propagation, not recoverability)')
ax.legend(); ax.grid(alpha=.2)
fig.tight_layout(); fig.savefig(ROOT / 'figures/fig1_divergence_propagation.png', dpi=160); plt.close(fig)

# Fig 2: |q error| by d per estimator (phase_jump)
fig, ax = plt.subplots(figsize=(8, 4.5))
for lab, col, off, c in (('M1 v5-Ordered (I_syn proj)', 'm1_ge_int', 1.0, '#247d90'),
                         ('M3 gain-head (own q_hat)', 'm3_q_int', 0.0, '#ca715b'),
                         ('M4 scalar estimator (no training)', 'm4_q_int', 0.0, '#3a9a57')):
    ys, ns = [], []
    for d in DS:
        sel = [r for r in inv if r['condition'] == 'phase_jump' and int(r['d']) == d and r[col] not in ('', 'None')]
        ys.append(np.mean([abs((f(r[col]) - off) - f(r['true_q_int'])) for r in sel]))
        ns.append(len(sel))
    ax.plot(DS, ys, marker='o', color=c, label=lab)
sel = [r for r in inv if r['condition'] == 'sham' and r['m4_q_int'] not in ('', 'None')]
sh = [np.mean([abs(f(r['m4_q_int']) - f(r['true_q_int'])) for r in sel if int(r['d']) == d]) for d in DS]
ax.plot(DS, sh, ls=':', color='#8b8b8b', label='M4 on sham (natural-error reference)')
ax.set(xlabel='d', ylabel='mean |q error| (info-bearing points)',
       title='Gain re-estimation after phase_jump (+1.5 to z_pos)')
ax.legend(fontsize=8); ax.grid(alpha=.2)
fig.tight_layout(); fig.savefig(ROOT / 'figures/fig2_q_error_by_d.png', dpi=160); plt.close(fig)

# Fig 3: single-trajectory responses (phase_jump, M4 vs M3 vs true)
fig, ax = plt.subplots(figsize=(8, 4.5))
for traj in (0, 5, 11):
    sel = [r for r in inv if r['condition'] == 'phase_jump' and int(r['traj']) == traj]
    tq = [np.mean([f(r['true_q_int']) for r in sel if int(r['d']) == d]) for d in DS]
    m4 = [np.nanmean([f(r['m4_q_int']) if r['m4_q_int'] not in ('', 'None') else np.nan
                      for r in sel if int(r['d']) == d]) for d in DS]
    m3 = [np.mean([f(r['m3_q_int']) for r in sel if int(r['d']) == d]) for d in DS]
    ax.plot(DS, tq, color='black', lw=1.5)
    ax.plot(DS, m4, marker='o', ms=3, label=f'M4 (traj {traj})' if traj == 0 else None)
    ax.plot(DS, m3, marker='s', ms=3, ls='--', label=f'M3 (traj {traj})' if traj == 0 else None)
ax.set(xlabel='d', ylabel='q', title='Per-trajectory gain re-estimation after phase_jump (3 example trajectories)')
ax.legend(fontsize=8); ax.grid(alpha=.2)
fig.tight_layout(); fig.savefig(ROOT / 'figures/fig3_single_traj_response.png', dpi=160); plt.close(fig)

# Fig 4: short-horizon V RMSE at fixed d (old models vs M2/M3 on phase_jump)
fig, ax = plt.subplots(figsize=(8, 4.5))
hs = (1, 4, 8)
for m, c in (('gnn_k1', '#666666'), ('set_k32', '#869aba'), ('global_k32', '#247d90'), ('oracle', '#3a9a57')):
    ys = [np.mean([f(r[f'v_err_int_h{h}']) for r in re if r['model'] == m and r['condition'] == 'phase_jump'
                   and r[f'v_err_int_h{h}'] not in ('', 'None') and int(r['d']) == 8]) for h in hs]
    ax.plot(hs, ys, marker='o', label=m, color=c)
ax.set(xlabel='horizon h (context frozen at tau+8)', ylabel='V RMSE (phase_jump branch)',
       title='Short-horizon prediction after intervention (d=8, old models)')
ax.legend(); ax.grid(alpha=.2)
fig.tight_layout(); fig.savefig(ROOT / 'figures/fig4_short_horizon.png', dpi=160); plt.close(fig)
print('FIGS DONE')
