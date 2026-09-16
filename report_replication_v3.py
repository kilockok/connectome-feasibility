"""Stage 1-2 replication gate: paired seeds, effect sizes, Gate A/B decision."""
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from calibrate_latent import write_csv

ROOT = Path('results/latent_state_v3')
SEEDS = (1234, 1235, 1236, 1237, 1238)
HIDDEN = ['gnn_k1', 'global_k32', 'gshuffle', 'wide', 'oracle']


def load_entries(regime):
    out = {}
    for p in (ROOT / 'eval' / 'entries').glob(f'{regime}_*.json'):
        e = json.loads(p.read_text())
        out[(e['label'], e['seed'])] = e
    return out


def f1(e, sp='test_seen'):
    return e['one_step'][sp]['pooled']['spike_f1']


def bootstrap_paired_delta(entry, reference, sp='test_seen', n=2000):
    """Whole-trajectory resampling, shared indices across the pair."""
    def load(e):
        return torch.load(ROOT / 'eval' / 'entries' / f"{e['regime']}_{e['label']}_{e['seed']}.pt",
                          weights_only=False)[sp]
    a, b = load(entry), load(reference)
    assert torch.equal(a['target'], b['target'])
    rng = np.random.default_rng(1123)
    bi = torch.randint(64, (len(a['target']),), generator=torch.Generator().manual_seed(8001))
    stats = []
    for e, d in ((entry, a), (reference, b)):
        p = (d['out']['s_logits'].sigmoid() > e['threshold']).float()
        t = d['target'][..., 1]
        v = torch.stack(((p * t).sum(-1), (p * (1 - t)).sum(-1), ((1 - p) * t).sum(-1),
                          (d['out']['v'] - d['target'][..., 0]).square().mean(-1), torch.ones(len(p))), -1)
        stats.append(torch.zeros(64, 5).index_add_(0, bi, v).numpy())
    idx = rng.integers(0, 64, (n, 64))
    def ev(s):
        sm = s[idx].sum(1)
        return 2 * sm[:, 0] / np.maximum(2 * sm[:, 0] + sm[:, 1] + sm[:, 2], 1), np.sqrt(sm[:, 3] / sm[:, 4])
    af, av = ev(stats[0]); bf, bv = ev(stats[1])
    d = af - bf
    dv = av - bv
    return dict(lo=float(np.quantile(d, .025)), hi=float(np.quantile(d, .975)),
                v_lo=float(np.quantile(dv, .025)), v_hi=float(np.quantile(dv, .975)))


def main():
    entries = load_entries('hidden')
    markov = load_entries('markov')
    seeds = sorted({s for (l, s) in entries})
    rows = []
    for s in seeds:
        k1 = entries.get(('gnn_k1', s))
        if k1 is None:
            continue
        row = dict(seed=s)
        for lab, key in (('oracle', 'oracle_gain'), ('global_k32', 'temporal_gain'),
                         ('gshuffle', 'shuffle_gain'), ('wide', 'width_gain')):
            e = entries.get((lab, s))
            if e is None:
                continue
            row[key] = f1(e) - f1(k1)
            row[f'{key}_macro'] = (e['one_step']['test_seen']['macro_f1']
                                   - k1['one_step']['test_seen']['macro_f1'])
        if ('global_k32', s) in entries and ('gshuffle', s) in entries:
            og = f1(entries[('global_k32', s)]) - f1(entries[('gshuffle', s)])
            row['order_gain'] = og
            row['order_gain_macro'] = (entries[('global_k32', s)]['one_step']['test_seen']['macro_f1']
                                       - entries[('gshuffle', s)]['one_step']['test_seen']['macro_f1'])
            ci = bootstrap_paired_delta(entries[('global_k32', s)], entries[('gshuffle', s)])
            row['order_gain_lo'], row['order_gain_hi'] = ci['lo'], ci['hi']
        if 'oracle_gain' in row and 'temporal_gain' in row and abs(row['oracle_gain']) > 1e-9:
            row['recovery_fraction'] = row['temporal_gain'] / row['oracle_gain']
        rows.append(row)
    (ROOT / 'replication').mkdir(parents=True, exist_ok=True)
    write_csv(ROOT / 'replication' / 'paired_deltas.csv', rows)

    def col(key):
        return np.array([r[key] for r in rows if key in r], float)
    summary = {}
    for key in ('oracle_gain', 'temporal_gain', 'order_gain', 'width_gain', 'recovery_fraction',
                'order_gain_macro', 'temporal_gain_macro'):
        v = col(key)
        if len(v):
            summary[key] = dict(n=len(v), mean=float(v.mean()), std=float(v.std(ddof=1)) if len(v) > 1 else 0.,
                                min=float(v.min()), max=float(v.max()),
                                same_direction=int((v > 0).sum()) if key != 'width_gain' else int((v != 0).sum()),
                                effect_dz=float(v.mean() / v.std(ddof=1)) if len(v) > 1 and v.std(ddof=1) > 0 else None,
                                ci95_t=[float(v.mean() - 1.96 * v.std(ddof=1) / np.sqrt(len(v))),
                                        float(v.mean() + 1.96 * v.std(ddof=1) / np.sqrt(len(v)))] if len(v) > 1 else None)
    og = col('order_gain')
    ci_lo = [r['order_gain_lo'] for r in rows if 'order_gain_lo' in r]
    ci_hi = [r['order_gain_hi'] for r in rows if 'order_gain_hi' in r]
    # Markov control gate
    mk = []
    for s in sorted({s for (l, s) in markov}):
        if ('global_k32', s) in markov and ('gnn_k1', s) in markov:
            mk.append(f1(markov[('global_k32', s)]) - f1(markov[('gnn_k1', s)]))
    mk = np.array(mk, float)
    gates = dict(
        n_seeds=len(og),
        order_gain=summary.get('order_gain'),
        per_seed_bootstrap_ci=[[float(l), float(h)] for l, h in zip(ci_lo, ci_hi)],
        gate_A=bool(len(og) >= 3 and og.mean() > 0
                    and (np.mean(ci_lo) > 0 or (og > 0).sum() >= max(3, len(og) - 1))),
        gate_A_basis='mean order_gain>0 AND (mean per-seed bootstrap CI lower>0 OR >=n-1 of n seeds positive)',
        temporal_gain=summary.get('temporal_gain'),
        oracle_gain=summary.get('oracle_gain'),
        recovery_fraction=summary.get('recovery_fraction'),
        width_gain=summary.get('width_gain'),
        markov_temporal_gain=dict(n=len(mk), mean=float(mk.mean()) if len(mk) else None,
                                  std=float(mk.std(ddof=1)) if len(mk) > 1 else None,
                                  per_seed=mk.tolist()),
        gate_B=bool(len(mk) >= 3 and abs(mk.mean()) < 0.01),
    )
    (ROOT / 'replication' / 'gates.json').write_text(json.dumps(gates, indent=2))

    # Figure 1: paired ordered-vs-shuffled across seeds
    fig, ax = plt.subplots(figsize=(7, 4))
    xs = np.arange(len(rows))
    vals = [r.get('order_gain', np.nan) for r in rows]
    los = [r.get('order_gain_lo', np.nan) for r in rows]
    his = [r.get('order_gain_hi', np.nan) for r in rows]
    ax.errorbar(xs, vals, yerr=[np.array(vals) - np.array(los), np.array(his) - np.array(vals)],
                marker='o', capsize=4, ls='none', color='#247d90', label='order_gain (bootstrap 95% CI)')
    ax.axhline(0, color='gray', ls='--')
    ax.axhline(np.nanmean(vals), color='#ca715b', ls=':', label=f'mean {np.nanmean(vals):+.4f}')
    ax.set(xticks=xs, xticklabels=[str(r['seed']) for r in rows], xlabel='Seed',
           ylabel='ordered - shuffled F1', title='Paired order gain per seed (whole-trajectory bootstrap)')
    ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(ROOT / 'figures' / 'fig1_paired_order_gain.png', dpi=160); plt.close(fig)

    # Figure 2: gains + recovery fraction
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    labs = ['oracle_gain', 'temporal_gain', 'order_gain', 'width_gain']
    axs[0].bar(labs, [summary.get(l, {}).get('mean', np.nan) for l in labs],
               yerr=[summary.get(l, {}).get('std', np.nan) for l in labs], capsize=4,
               color=['#3a9a57', '#247d90', '#4d6a91', '#8b8b8b'])
    axs[0].set(ylabel='Paired F1 delta (mean ± SD)', title='Hidden teacher, per-seed paired gains')
    axs[0].tick_params(axis='x', rotation=20)
    rf = col('recovery_fraction')
    axs[1].plot(range(len(rf)), rf, marker='o', ls='none', color='#247d90')
    axs[1].axhline(float(rf.mean()) if len(rf) else 0, color='#ca715b', ls=':',
                   label=f'mean {float(rf.mean()):.3f}' if len(rf) else '')
    axs[1].set(xticks=range(len(rf)), xticklabels=[str(r['seed']) for r in rows if 'recovery_fraction' in r],
               xlabel='Seed', ylabel='recovery_fraction', title='temporal_gain / oracle_gain per seed')
    axs[1].legend(); axs[1].grid(alpha=.2)
    fig.tight_layout(); fig.savefig(ROOT / 'figures' / 'fig2_gains_recovery.png', dpi=160); plt.close(fig)
    print(json.dumps(gates, indent=2))


if __name__ == '__main__':
    torch.set_num_threads(2)
    main()
