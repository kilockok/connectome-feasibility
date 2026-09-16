"""v4 final assembly: figures 1-9, falsification review, conclusion.md (22 answers)."""
import csv
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('results/latent_state_v4')
N_TAGS = ['n100', 'n250', 'n500', 'n1000']
OBS_TAGS = ['n1000obs50_degree', 'n1000obs100_degree', 'n1000obs250_degree',
            'n1000obs500_degree', 'n1000obs1000_degree']
NVAL = dict(n100=100, n250=250, n500=500, n1000=1000)
OVAL = {t: int(t.split('obs')[1].split('_')[0]) for t in OBS_TAGS}


def load(name):
    return list(csv.DictReader(open(ROOT / name)))


def f(x):
    return float(x)


def cell(rows, tag, key):
    if tag == 'n1000obs1000_degree':
        tag = 'n1000'  # obs-all is exactly the N=1000 condition
    v = [f(r[key]) for r in rows if r['tag'] == tag and r.get(key) not in (None, '', 'None')]
    return (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.) if v else (None, None)


def main():
    dec = load('tables/decomposition.csv')
    obs = load('observability/z_observability.csv') if (ROOT / 'observability/z_observability.csv').exists() else []
    ksw = load('history_length/history_length.csv') if (ROOT / 'history_length/history_length.csv').exists() else []
    (ROOT / 'figures').mkdir(exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})

    # Fig 1: N vs temporal decomposition
    fig, ax = plt.subplots(figsize=(7, 4.5))
    xs = [NVAL[t] for t in N_TAGS]
    for key, color, lab in (('unordered_gain', '#869aba', 'unordered_gain (Set-K1)'),
                            ('order_gain', '#247d90', 'order_gain (Ordered-Set)'),
                            ('temporal_gain', '#4d6a91', 'temporal_gain (Ordered-K1)')):
        ys = [cell(dec, t, key)[0] for t in N_TAGS]
        es = [cell(dec, t, key)[1] for t in N_TAGS]
        ax.errorbar(xs, ys, yerr=es, marker='o', label=lab, color=color, capsize=3)
    ax.axhline(0, color='gray', ls='--')
    ax.set(xscale='log', xticks=xs, xticklabels=xs, xlabel='N (teacher = model N)',
           ylabel='Paired F1 gain', title='Temporal decomposition vs population size')
    ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(ROOT / 'figures/fig1_n_decomposition.png', dpi=160); plt.close(fig)

    # Fig 2: N vs order_share
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ys = [cell(dec, t, 'order_share')[0] for t in N_TAGS]
    es = [cell(dec, t, 'order_share')[1] for t in N_TAGS]
    ax.errorbar(xs, ys, yerr=es, marker='o', color='#ca715b', capsize=3)
    ax.set(xscale='log', xticks=xs, xticklabels=xs, xlabel='N',
           ylabel='order_gain / temporal_gain', title='Order share of the temporal advantage')
    ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(ROOT / 'figures/fig2_n_order_share.png', dpi=160); plt.close(fig)

    # Fig 3: N_obs vs gains
    fig, ax = plt.subplots(figsize=(7, 4.5))
    xo = [OVAL[t] for t in OBS_TAGS]
    for key, color, lab in (('unordered_gain', '#869aba', 'unordered_gain (Set-K1)'),
                            ('order_gain', '#247d90', 'order_gain (Ordered-Set)'),
                            ('temporal_gain', '#4d6a91', 'temporal_gain (Ordered-K1)'),
                            ('deriv_gain', '#8b8b8b', 'deriv_gain (Deriv-K1)')):
        ys = [cell(dec, t, key)[0] for t in OBS_TAGS]
        es = [cell(dec, t, key)[1] for t in OBS_TAGS]
        ax.errorbar(xo, ys, yerr=es, marker='o', label=lab, color=color, capsize=3)
    ax.axhline(0, color='gray', ls='--')
    ax.set(xscale='log', xticks=xo, xticklabels=xo, xlabel='N_obs (observed neurons, teacher N=1000)',
           ylabel='Paired F1 gain', title='Observability sweep: gains vs N_obs')
    ax.legend(fontsize=8); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(ROOT / 'figures/fig3_nobs_gains.png', dpi=160); plt.close(fig)

    # Fig 4: N_obs vs order_share
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ys = [cell(dec, t, 'order_share')[0] for t in OBS_TAGS]
    es = [cell(dec, t, 'order_share')[1] for t in OBS_TAGS]
    ax.errorbar(xo, ys, yerr=es, marker='o', color='#ca715b', capsize=3)
    ax.set(xscale='log', xticks=xo, xticklabels=xo, xlabel='N_obs',
           ylabel='order_gain / temporal_gain', title='Order share vs observability')
    ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(ROOT / 'figures/fig4_nobs_order_share.png', dpi=160); plt.close(fig)

    # Fig 5: z observability vs N_obs
    if obs:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for est, color, lab in (('E0_current', '#8b8b8b', 'current stats'),
                                ('E1_unordered', '#869aba', 'unordered window stats'),
                                ('E4_ordered_gru', '#247d90', 'ordered GRU')):
            ys = []
            for t in OBS_TAGS:
                v = [f(r['r2']) for r in obs if r['tag'] == t and r['estimator'] == est
                     and r['target'] == 'z_pos' and r['split'] == 'test_seen']
                ys.append(np.mean(v) if v else np.nan)
            ax.plot(xo, ys, marker='o', label=lab, color=color)
        ax.set(xscale='log', xticks=xo, xticklabels=xo, xlabel='N_obs',
               ylabel='z_pos R2 (held-out)', title='Latent observability vs N_obs')
        ax.legend(); ax.grid(alpha=.2)
        fig.tight_layout(); fig.savefig(ROOT / 'figures/fig5_nobs_z_observability.png', dpi=160); plt.close(fig)

        # Fig 6: unordered z observability vs order_gain scatter
        fig, ax = plt.subplots(figsize=(6, 5))
        pts = []
        for t in OBS_TAGS + N_TAGS:
            og = cell(dec, t, 'order_gain')[0]
            v = [f(r['r2']) for r in obs if r['tag'] == t and r['estimator'] == 'E1_unordered'
                 and r['target'] == 'z_pos' and r['split'] == 'test_seen']
            if og is not None and v:
                pts.append((float(np.mean(v)), og, t))
        if pts:
            xs6 = [p[0] for p in pts]; ys6 = [p[1] for p in pts]
            ax.scatter(xs6, ys6, s=40, color='#247d90')
            for x6, y6, t in pts:
                ax.annotate(t.replace('n1000obs', 'obs').replace('_degree', '').replace('n', 'N'), (x6, y6),
                            fontsize=7, xytext=(4, 4), textcoords='offset points')
            if len(pts) >= 3:
                a, b = np.polyfit(xs6, ys6, 1)
                xl = np.linspace(min(xs6), max(xs6), 20)
                ax.plot(xl, a * xl + b, ls=':', color='#ca715b',
                        label=f'slope={a:+.3f}, r={np.corrcoef(xs6, ys6)[0, 1]:+.2f}')
                ax.legend()
            ax.set(xlabel='Unordered z_pos observability (R2)', ylabel='order_gain (Ordered-Set)',
                   title='Does unordered z-observability predict lower order gain?')
            ax.grid(alpha=.2)
            fig.tight_layout(); fig.savefig(ROOT / 'figures/fig6_obs_vs_ordergain.png', dpi=160); plt.close(fig)

    # Fig 7: history length curves
    if ksw:
        fig, axs = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
        for ax, label in zip(axs, ('set_k32', 'global_k32')):
            for t, color in zip(('n1000obs50_degree', 'n1000obs100_degree', 'n1000obs500_degree', 'n1000'),
                                ('#ca715b', '#869aba', '#4d6a91', '#8b8b8b')):
                xs7, ys7 = [], []
                for k in (1, 2, 4, 8, 16, 32):
                    v = [f(r['spike_f1']) for r in ksw if r['tag'] == t and r['label'] == label
                         and int(r['k_eff']) == k]
                    if v:
                        xs7.append(k); ys7.append(np.mean(v))
                ax.plot(xs7, ys7, marker='o', label=t.replace('n1000obs', 'obs').replace('_degree', ''), color=color)
            ax.set(xscale='log', xticks=[1, 2, 4, 8, 16, 32], xticklabels=[1, 2, 4, 8, 16, 32],
                   xlabel='K_eff', ylabel='Seen F1', title=f'{label} history-length curves')
            ax.legend(fontsize=8); ax.grid(alpha=.2)
        fig.tight_layout(); fig.savefig(ROOT / 'figures/fig7_history_length.png', dpi=160); plt.close(fig)

    # Fig 8: oracle-recovery fractions at N=100 (model ladder)
    fig, ax = plt.subplots(figsize=(8, 4))
    labs = ['gnn_k1', 'stats_k32', 'deriv', 'set_k32', 'gshuffle', 'global_k32', 'oracle']
    tag0 = 'n100'
    og = cell(dec, tag0, 'oracle_gain')[0]
    k1 = cell(dec, tag0, 'P_K1')[0]
    pmap = {'gnn_k1': 'P_K1', 'stats_k32': 'P_stats', 'deriv': 'P_deriv', 'set_k32': 'P_set',
            'gshuffle': 'P_shuffled', 'global_k32': 'P_order', 'oracle': 'P_oracle'}
    vals = []
    for l in labs:
        m = cell(dec, tag0, pmap[l])
        vals.append(((m[0] - k1) / og if m[0] is not None and og else None))
    ax.bar(labs, [v if v is not None else 0 for v in vals],
           color=['#666666', '#869aba', '#8b8b8b', '#4d6a91', '#ca715b', '#247d90', '#3a9a57'])
    ax.set(ylabel='Fraction of oracle_gain recovered (N=100)', title='Model ladder: share of the oracle advantage')
    ax.tick_params(axis='x', rotation=25); ax.grid(alpha=.2, axis='y')
    fig.tight_layout(); fig.savefig(ROOT / 'figures/fig8_ladder_recovery.png', dpi=160); plt.close(fig)

    # Fig 9: derivative baseline vs ordered
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    width = .35
    xs9 = np.arange(len(OBS_TAGS))
    dv = [cell(dec, t, 'deriv_gain')[0] for t in OBS_TAGS]
    og9 = [cell(dec, t, 'order_gain')[0] for t in OBS_TAGS]
    ax.bar(xs9 - width / 2, dv, width, label='deriv_gain (Deriv-K1)', color='#8b8b8b')
    ax.bar(xs9 + width / 2, og9, width, label='order_gain (Ordered-Set)', color='#247d90')
    ax.set(xticks=xs9, xticklabels=[OVAL[t] for t in OBS_TAGS], xlabel='N_obs',
           ylabel='Paired F1 gain', title='Derivative baseline vs order-specific gain')
    ax.legend(); ax.grid(alpha=.2, axis='y')
    fig.tight_layout(); fig.savefig(ROOT / 'figures/fig9_deriv_vs_ordered.png', dpi=160); plt.close(fig)
    print('FIGS DONE')


if __name__ == '__main__':
    torch.set_num_threads(2)
    main()
