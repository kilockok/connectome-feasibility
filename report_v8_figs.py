"""v8 figures."""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('results/latent_state_v8')
MODELS = ('k1', 'k2', 'set', 'ordered', 'event_simple', 'event_rich', 'oracle')
COLORS = dict(k1='#666666', k2='#869aba', set='#4d6a91', ordered='#247d90',
              event_simple='#a08b62', event_rich='#8b8b8b', oracle='#3a9a57')
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})


def load(name):
    return list(csv.DictReader(open(ROOT / 'metrics' / name)))


def f(x):
    return float(x)


nat = [r for r in load('natural.csv') if r['split'] == 'test_seen']
swp = load('history_sweep.csv')

# Fig 1: heldout improvement
fig, ax = plt.subplots(figsize=(8, 4))
xs = np.arange(len(MODELS))
im = [[f(r['improve_over_base']) for r in nat if r['model'] == m] for m in MODELS]
ax.bar(xs, [np.mean(v) for v in im], yerr=[np.std(v, ddof=1) for v in im], capsize=4,
       color=[COLORS[m] for m in MODELS])
ax.set(xticks=xs, xticklabels=MODELS, ylabel='V RMSE improvement over base LIF',
       title='STP teacher: natural held-out improvement (5 paired seeds)')
ax.tick_params(axis='x', rotation=20); ax.grid(alpha=.2, axis='y')
fig.tight_layout(); fig.savefig(ROOT / 'figures' / 'heldout_improvement.png', dpi=160); plt.close(fig)

# Fig 2: history length curves
fig, ax = plt.subplots(figsize=(7, 4.5))
KS = (1, 2, 4, 8, 16, 32)
for m in ('ordered', 'set', 'event_simple', 'event_rich'):
    ys = [np.mean([f(r['spike_f1']) for r in swp if r['model'] == m and int(r['k_eff']) == k]) for k in KS]
    ax.plot(KS, ys, marker='o', label=m, color=COLORS[m])
ax.set(xscale='log', xticks=KS, xticklabels=KS, xlabel='K_eff (history kept)',
       ylabel='Seen spike F1 (inference masking)',
       title='History-length curves vs STP timescales (tau_rec 4-24, tau_fac 2-24)')
for tau in (4, 8, 24):
    ax.axvline(tau, color='gray', ls=':', alpha=.5)
ax.legend(fontsize=8); ax.grid(alpha=.2)
fig.tight_layout(); fig.savefig(ROOT / 'figures' / 'history_length.png', dpi=160); plt.close(fig)

# Fig 3: recovery curve (true g difference burst - sparse per cluster)
fig, ax = plt.subplots(figsize=(7, 4.5))
rec = {
    'depression': [(-0.026, 0.26), (-0.014, 0.25), (-0.017, 0.25), (-0.012, 0.25), (-0.012, 0.25), (-0.006, 0.25)],
    'facilitation': [(0.166, 2.87), (0.154, 2.92), (0.132, 2.91), (0.113, 2.94), (0.048, 2.98), (0.013, 3.02)],
    'mixed': [(-0.020, 0.87), (-0.011, 0.86), (-0.013, 0.86), (-0.007, 0.86), (-0.013, 0.85), (-0.007, 0.85)],
}
DS = (0, 2, 4, 8, 16, 32)
for name, vals in rec.items():
    ax.plot(DS, [v[0] for v in vals], marker='o', label=f'{name} (Δg)')
ax.axhline(0, color='gray', ls='--')
ax.set(xlabel='probe delay d after burst', ylabel='true effective-gain difference (burst - sparse)',
       title='STP recovery at the probe edge (teacher ground truth)')
ax.legend(); ax.grid(alpha=.2)
fig.tight_layout(); fig.savefig(ROOT / 'figures' / 'recovery_curve.png', dpi=160); plt.close(fig)

# Fig 4: model comparison ordered vs controls (paired deltas)
fig, ax = plt.subplots(figsize=(8, 4))
pairs = [('ordered', 'k2'), ('ordered', 'set'), ('ordered', 'event_simple'), ('ordered', 'event_rich')]
xs = np.arange(len(pairs))
for i, (a, b) in enumerate(pairs):
    d = [f(x['v_rmse']) - f(y['v_rmse']) for x, y in zip(
        [r for r in nat if r['model'] == a], [r for r in nat if r['model'] == b])]
    d = [-x for x in d]  # ordered advantage = control - ordered
    ax.bar([i], [np.mean(d)], yerr=[np.std(d, ddof=1)], capsize=5,
           color='#247d90' if np.mean(d) > 0 else '#ca715b')
    for j, v in enumerate(d):
        ax.plot([i - .15 + .075 * j], [v], 'k.', ms=4, alpha=.6)
ax.axhline(0, color='gray', ls='--')
ax.set(xticks=xs, xticklabels=[f'ordered - {b}' for a, b in pairs],
       ylabel='V RMSE advantage (positive = ordered better)',
       title='Ordered vs each control, per-seed paired (5 seeds shown as dots)')
ax.grid(alpha=.2, axis='y')
fig.tight_layout(); fig.savefig(ROOT / 'figures' / 'model_comparison.png', dpi=160); plt.close(fig)

# Fig 5: cross-mechanism summary
fig, ax = plt.subplots(figsize=(8, 4))
mechs = ['temporal gain\n(v2-v6)', 'adaptation\n(v7)', 'STP\n(v8)']
adv = [0.045, 0.001, 0.0038]
ax.bar(range(3), adv, color=['#247d90', '#869aba', '#4d6a91'])
ax.set(xticks=range(3), xticklabels=mechs, ylabel='ordered - K2 (V RMSE or F1 advantage, approx)',
       title='Cross-mechanism: ordered-history advantage vs local observability')
ax.grid(alpha=.2, axis='y')
fig.tight_layout(); fig.savefig(ROOT / 'figures' / 'cross_mechanism.png', dpi=160); plt.close(fig)
print('FIGS DONE')
