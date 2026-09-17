"""v10 result aggregation + figures 4/5/6/8."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('results/latent_state_v10')
FAMS = ('gain', 'adapt', 'stp')
POL_ORDER = ['passive', 'random', 'heuristic_stp', 'heuristic_gain',
             'optimized_global', 'optimized_adaptive', 'oracle', 'v9hand']

# ---------------- single probe ----------------
sp = list(csv.DictReader((ROOT / 'metrics' / 'single_probe.csv').open()))
print('single_probe rows', len(sp))
agg = {}
for r in sp:
    key = (r['cohort'], r['family'], r['policy'])
    agg.setdefault(key, []).append((int(r['correct']), float(r['entropy']), float(r['conf_true']), r['seed']))
summary = {}
for key, v in agg.items():
    seeds = sorted({s for *_, s in [(c, e, cf, s) for c, e, cf, s in v]})
    per_seed = {}
    for c, e, cf, s in v:
        per_seed.setdefault(s, []).append((c, cf))
    seed_acc = [np.mean([c for c, _ in per_seed[s]]) for s in sorted(per_seed)]
    seed_ent = []
    for s in sorted(per_seed):
        es = [e for _, e, _, ss in [(vv[0], vv[1], vv[2], vv[3]) for vv in v] if ss == s]
        seed_ent.append(np.mean(es))
    summary[key] = dict(acc_mean=np.mean(seed_acc), acc_std=np.std(seed_acc),
                        acc_min=np.min(seed_acc), n_seeds=len(seed_acc),
                        ent_mean=np.mean(seed_ent))
print(f"{'cohort':6s} {'family':6s} {'policy':20s} {'acc':>16s}")
for cohort in ('testA', 'testB', 'testC'):
    for pol in POL_ORDER:
        for fam in FAMS:
            k = (cohort, fam, pol)
            if k in summary:
                s = summary[k]
                print(f"{cohort:6s} {fam:6s} {pol:20s} {s['acc_mean']:.3f}±{s['acc_std']:.3f} min={s['acc_min']:.3f} n={s['n_seeds']}")
# overall (all families)
print()
for cohort in ('testA', 'testB', 'testC'):
    for pol in POL_ORDER:
        accs = []
        for fam in FAMS:
            k = (cohort, fam, pol)
            if k in summary:
                accs.append(summary[k]['acc_mean'])
        if accs:
            print(f"{cohort:6s} ALL    {pol:20s} {np.mean(accs):.3f}")
np.save(ROOT / 'data' / 'single_probe_summary.npy', summary, allow_pickle=True)

# Figure 4: policy accuracy per cohort
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)
for ax, cohort in zip(axes, ('testA', 'testB', 'testC')):
    x = np.arange(len(POL_ORDER))
    for j, fam in enumerate(FAMS):
        vals = [summary.get((cohort, fam, pol), dict(acc_mean=np.nan))['acc_mean'] for pol in POL_ORDER]
        ax.bar(x + j * 0.27 - 0.27, vals, 0.27, label=fam)
    ax.axhline(0.25, color='gray', ls=':')
    ax.set_xticks(x, [p.replace('_', '\n') for p in POL_ORDER], fontsize=7)
    ax.set_title(cohort)
    ax.set_ylim(0, 1)
    if cohort == 'testA':
        ax.legend(fontsize=8)
axes[0].set_ylabel('accuracy@1probe')
fig.suptitle('v10 Figure 4: single-probe candidate identification by policy')
fig.tight_layout()
fig.savefig(ROOT / 'figures' / 'fig4_policies.png', dpi=140)
print('fig4 saved')

# ---------------- sequential ----------------
sq_path = ROOT / 'metrics' / 'sequential.csv'
if sq_path.exists():
    sq = list(csv.DictReader(sq_path.open()))
    print('sequential rows', len(sq))
    agg_s = {}
    for r in sq:
        key = (r['family'], r['policy'], int(r['budget']))
        agg_s.setdefault(key, []).append((int(r['correct']), float(r['entropy']), float(r['conf_true']), r['seed']))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for pol, mk in (('optimized', 'o-'), ('random', 's--')):
        for fam in FAMS:
            bs = sorted({k[2] for k in agg_s if k[1] == pol and k[0] == fam})
            acc = [np.mean([v[0] for v in agg_s[(fam, pol, b)]]) for b in bs]
            axes[0].plot(bs, acc, mk, label=f'{pol}:{fam}', ms=4)
            ent = [np.mean([v[1] for v in agg_s[(fam, pol, b)]]) for b in bs]
            axes[1].plot(bs, ent, mk, label=f'{pol}:{fam}', ms=4)
    axes[0].axhline(0.25, color='gray', ls=':')
    axes[0].set_xlabel('probe budget'); axes[0].set_ylabel('accuracy')
    axes[0].set_title('v10 Figure 5: accuracy vs probe budget (testB)')
    axes[1].set_xlabel('probe budget'); axes[1].set_ylabel('posterior entropy')
    axes[1].set_title('v10 Figure 6: posterior entropy vs budget')
    axes[0].legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(ROOT / 'figures' / 'fig56_sequential.png', dpi=140)
    print('fig56 saved')

# Figure 8: parameter OOD (from single probe, key policies)
fig, ax = plt.subplots(figsize=(7.5, 4.5))
pols = ['passive', 'random', 'heuristic_stp', 'optimized_adaptive', 'oracle']
x = np.arange(len(pols))
for j, cohort in enumerate(('testA', 'testB', 'testC')):
    vals = []
    for pol in pols:
        accs = [summary[(cohort, fam, pol)]['acc_mean'] for fam in FAMS if (cohort, fam, pol) in summary]
        vals.append(np.mean(accs) if accs else np.nan)
    ax.plot(x + j * 0.22 - 0.22, vals, 'o-', label={'testA': 'seen', 'testB': 'held-out', 'testC': 'extrapolated'}[cohort])
ax.axhline(0.25, color='gray', ls=':')
ax.set_xticks(x, [p.replace('_', '\n') for p in pols], fontsize=8)
ax.set_ylabel('accuracy (all families)')
ax.set_title('v10 Figure 8: held-out parameter generalization')
ax.legend()
fig.tight_layout()
fig.savefig(ROOT / 'figures' / 'fig8_param_ood.png', dpi=140)
print('fig8 saved')
