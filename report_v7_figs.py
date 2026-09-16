"""v7 figures."""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('results/latent_state_v7')
(ROOT / 'figures').mkdir(exist_ok=True)
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
MODELS = ('k1', 'k2', 'set', 'ordered', 'oracle')
COLORS = dict(k1='#666666', k2='#869aba', set='#4d6a91', ordered='#247d90', oracle='#3a9a57')


def load(name):
    return list(csv.DictReader(open(ROOT / name)))


def f(x):
    return float(x)


nat = [r for r in load('metrics_natural.csv') if r['split'] == 'test_seen']
inv = load('metrics_intervention.csv')
diag = load('metrics_diagnostics.csv')

# Fig 1: natural improvement over base
fig, ax = plt.subplots(figsize=(7.5, 4))
xs = np.arange(len(MODELS))
im = [[f(r['improve_over_base']) for r in nat if r['model'] == m] for m in MODELS]
ax.bar(xs, [np.mean(v) for v in im], yerr=[np.std(v, ddof=1) for v in im], capsize=4,
       color=[COLORS[m] for m in MODELS])
ax.axhline(0, color='gray', ls='--')
ax.set(xticks=xs, xticklabels=MODELS, ylabel='V RMSE improvement over base LIF',
       title='Natural held-out prediction: learned correction over base (5 paired seeds)')
ax.grid(alpha=.2, axis='y')
fig.tight_layout(); fig.savefig(ROOT / 'figures/fig1_natural_improvement.png', dpi=160); plt.close(fig)

# Fig 2: intervention d-curves
fig, axs = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
DS = (0, 1, 2, 4, 8, 16, 32)
for ax, cond in zip(axs, ('up', 'down', 'sham')):
    for m in MODELS:
        ys = [np.mean([f(r['v_err_int']) for r in inv if r['model'] == m and r['condition'] == cond and int(r['d']) == d]) for d in DS]
        ax.plot(DS, ys, marker='o', ms=3, label=m, color=COLORS[m])
    ax.set(xlabel='d (new observations after tau)', ylabel='one-step V RMSE' if ax is axs[0] else None,
           title=f'{cond} (a-scale x{2.0 if cond == "up" else 0.0 if cond == "down" else 1.0})')
    ax.grid(alpha=.2)
axs[0].legend(fontsize=8)
fig.tight_layout(); fig.savefig(ROOT / 'figures/fig2_intervention_dcurves.png', dpi=160); plt.close(fig)

# Fig 3: short-horizon at d=8 (up)
fig, ax = plt.subplots(figsize=(7.5, 4))
hs = (1, 4, 8)
w = .15
xs = np.arange(len(hs))
for j, m in enumerate(MODELS):
    ys = [np.mean([f(r[f'v_h{h}_int']) for r in inv if r['model'] == m and r['condition'] == 'up' and int(r['d']) == 8]) for h in hs]
    ax.bar(xs + (j - 2) * w, ys, w, label=m, color=COLORS[m])
ax.set(xticks=xs, xticklabels=[f'h={h}' for h in hs], ylabel='V RMSE',
       title='Short-horizon prediction after intervention (context frozen at tau+8)')
ax.legend(fontsize=8); ax.grid(alpha=.2, axis='y')
fig.tight_layout(); fig.savefig(ROOT / 'figures/fig3_short_horizon.png', dpi=160); plt.close(fig)

# Fig 4: probe calibration (a_mean + corr slope)
fig, axs = plt.subplots(1, 2, figsize=(11, 4))
models4 = ('k1', 'k2', 'set', 'ordered')
xs = np.arange(len(models4))
for j, (tgt, lab, c) in enumerate((('a_mean', 'a_mean R2', '#247d90'),)):
    ys = [np.mean([f(r['r2']) for r in diag if r['model'] == m and r['target'] == tgt]) for m in models4]
    axs[0].bar(xs, ys, color=c)
    axs[0].set(ylim=(0.9, 1.005))
sel = [r for r in diag if r['target'] == 'corr_vs_true_contrib']
ys2 = [np.mean([f(r['r2']) for r in sel if r['model'] == m]) for m in models4]
axs[1].bar(xs, ys2, color='#ca715b')
axs[0].set(xticks=xs, xticklabels=models4, title='adaptation decoding (probe, held-out)')
axs[1].set(xticks=xs, xticklabels=models4, title='corr head vs true contribution (R2, calibration weak)')
for ax in axs:
    ax.grid(alpha=.2, axis='y')
fig.tight_layout(); fig.savefig(ROOT / 'figures/fig4_probe_calibration.png', dpi=160); plt.close(fig)
print('FIGS DONE')
