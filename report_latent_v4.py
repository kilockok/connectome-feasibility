"""Stage 8: v4 unified report — decomposition tables, figures, falsification, conclusion."""
import csv
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from calibrate_latent import write_csv

ROOT = Path('results/latent_state_v4')
V3 = Path('results/latent_state_v3')
CORE = ['gnn_k1', 'set_k32', 'stats_k32', 'deriv', 'global_k32', 'gshuffle', 'wide', 'oracle']


def entry_path(tag, label, seed):
    """v4 tags live in v4/eval; reused v3 conditions map to v3/eval."""
    v3_map = {'n100': 'hidden', 'n1000': 'n1000'}
    if tag in v3_map and (V3 / 'eval' / 'entries' / f'{v3_map[tag]}_{label}_{seed}.json').exists() \
            and not (ROOT / 'eval' / 'entries' / f'{tag}_{label}_{seed}.json').exists():
        return V3 / 'eval' / 'entries' / f'{v3_map[tag]}_{label}_{seed}.json', v3_map[tag]
    return ROOT / 'eval' / 'entries' / f'{tag}_{label}_{seed}.json', tag


def load_entry(tag, label, seed):
    p, eff = entry_path(tag, label, seed)
    if not p.exists():
        return None
    e = json.loads(p.read_text())
    e['_tag'] = tag
    e['_eff'] = eff
    return e


def f1(e):
    return e['one_step']['test_seen']['pooled']['spike_f1']


def vrmse(e):
    return e['one_step']['test_seen']['pooled']['v_rmse']


def bootstrap_delta(tag, entry, reference, n=2000):
    """Whole-trajectory resampling, shared indices across the pair."""
    def load(e):
        p, _ = entry_path(e['_tag'], e['label'], e['seed'])
        return torch.load(p.with_suffix('.pt'), weights_only=False)['test_seen']
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
        return 2 * sm[:, 0] / np.maximum(2 * sm[:, 0] + sm[:, 1] + sm[:, 2], 1)
    d = ev(stats[0]) - ev(stats[1])
    return dict(mean=float(d.mean()), lo=float(np.quantile(d, .025)), hi=float(np.quantile(d, .975)))


def gains_for(tag, seeds=(1234, 1235, 1236)):
    """Per-seed paired decomposition for one condition."""
    rows = []
    for s in seeds:
        k1 = load_entry(tag, 'gnn_k1', s)
        st = load_entry(tag, 'set_k32', s)
        od = load_entry(tag, 'global_k32', s)
        oc = load_entry(tag, 'oracle', s)
        if not all((k1, st, od, oc)):
            continue
        row = dict(tag=tag, seed=s, P_K1=f1(k1), P_set=f1(st), P_order=f1(od), P_oracle=f1(oc))
        stx = load_entry(tag, 'stats_k32', s)
        dv = load_entry(tag, 'deriv', s)
        sh = load_entry(tag, 'gshuffle', s)
        row['P_stats'] = f1(stx) if stx else None
        row['P_deriv'] = f1(dv) if dv else None
        row['P_shuffled'] = f1(sh) if sh else None
        row['temporal_gain'] = row['P_order'] - row['P_K1']
        row['unordered_gain'] = row['P_set'] - row['P_K1']
        row['order_gain'] = row['P_order'] - row['P_set']
        row['oracle_gain'] = row['P_oracle'] - row['P_K1']
        row['stats_gain'] = (row['P_stats'] - row['P_K1']) if stx else None
        row['deriv_gain'] = (row['P_deriv'] - row['P_K1']) if dv else None
        row['shuffled_gain'] = (row['P_shuffled'] - row['P_K1']) if sh else None
        row['recovery_fraction'] = row['temporal_gain'] / row['oracle_gain']
        row['unordered_fraction'] = row['unordered_gain'] / row['oracle_gain']
        row['order_fraction'] = row['order_gain'] / row['oracle_gain']
        row['order_share'] = row['order_gain'] / row['temporal_gain'] if abs(row['temporal_gain']) > 1e-9 else None
        for key, ref in (('unordered_gain', st), ('order_gain', od)):
            ref_e = k1 if key == 'unordered_gain' else st
            ci = bootstrap_delta(tag, od if key == 'order_gain' else st, ref_e)
            row[f'{key}_lo'], row[f'{key}_hi'] = ci['lo'], ci['hi']
        rows.append(row)
    return rows


def main():
    (ROOT / 'tables').mkdir(parents=True, exist_ok=True)
    (ROOT / 'figures').mkdir(parents=True, exist_ok=True)
    n_tags = ['n100', 'n250', 'n500', 'n1000']
    obs_tags = ['n1000obs50_degree', 'n1000obs100_degree', 'n1000obs250_degree',
                'n1000obs500_degree']
    all_rows = []
    for tag in n_tags + obs_tags:
        all_rows.extend(gains_for(tag))
    write_csv(ROOT / 'tables' / 'decomposition.csv', all_rows)
    # Table 1: N scaling
    t1 = [r for r in all_rows if r['tag'] in n_tags]
    write_csv(ROOT / 'tables' / 'table1_n_scaling.csv', t1)
    # Table 2: N_obs sweep
    t2 = [r for r in all_rows if r['tag'] in obs_tags]
    for r in gains_for('n1000'):
        t2.append(dict(r, tag='n1000obs1000_degree'))
    write_csv(ROOT / 'tables' / 'table2_n_obs.csv', t2)
    all_rows.extend(t2[-len(gains_for('n1000')):] if gains_for('n1000') else [])

    def agg(rows, key):
        v = np.array([r[key] for r in rows if r.get(key) is not None], float)
        return (float(v.mean()), float(v.std(ddof=1)), [round(x, 4) for x in v]) if len(v) else None

    summary = {}
    for tag in n_tags + obs_tags:
        rs = [r for r in all_rows if r['tag'] == tag]
        if not rs:
            continue
        summary[tag] = {k: agg(rs, k) for k in
                        ('P_K1', 'P_set', 'P_stats', 'P_deriv', 'P_order', 'P_oracle', 'P_shuffled',
                         'temporal_gain', 'unordered_gain', 'order_gain', 'oracle_gain', 'stats_gain',
                         'deriv_gain', 'shuffled_gain', 'recovery_fraction', 'unordered_fraction',
                         'order_fraction', 'order_share')}
        summary[tag]['unordered_gain_ci'] = [[r['unordered_gain_lo'], r['unordered_gain_hi']] for r in rs]
        summary[tag]['order_gain_ci'] = [[r['order_gain_lo'], r['order_gain_hi']] for r in rs]
    (ROOT / 'tables' / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps({t: {k: (round(v[0], 4) if v else None) for k, v in s.items() if isinstance(v, tuple) and k in ('unordered_gain', 'order_gain', 'order_share', 'temporal_gain', 'oracle_gain')} for t, s in summary.items()}, indent=2))


if __name__ == '__main__':
    torch.set_num_threads(2)
    main()
