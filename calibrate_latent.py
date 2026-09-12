"""Small preregistered teacher-only calibration; never inspects test outcomes."""
import csv
import json
from dataclasses import asdict, replace
from pathlib import Path
import torch
from config import get_config
from connectome import Connectome
from lif_latent import HiddenStateLIFSimulator, LatentConfig


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def calibrate(out, n=100):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = replace(get_config('small'), n_neurons=n, T=256, K=32, silence_prob=0.)
    conn = Connectome.generate(cfg)
    candidates = [LatentConfig(.99, .1, .10), LatentConfig(.99, .2, .10),
                  LatentConfig(.99, .3, .10), LatentConfig(.98, .3, .10),
                  LatentConfig(.995, .3, .10), LatentConfig(.99, .3, .15)]
    rows, activity, observable = [], [], []
    for i, lc in enumerate(candidates):
        sim = HiddenStateLIFSimulator(conn, cfg, torch.device('cuda'), lc)
        # Dedicated calibration seeds; not reused in train/validation/test.
        d = sim.generate(list(range(70_000_000, 70_000_032)), 'train')
        s = d['states'][:, 1:, :, 1]
        late = s[:, 32:]
        for j in range(len(s)):
            activity.append(dict(candidate=i, trajectory=j, rate=float(s[j].mean()),
                                 late_rate=float(late[j].mean()), z_std=float(d['z'][j].std())))
        x = d['states'][:, 32:-1:8].reshape(-1, n, 3)
        u = d['stimulus'][:, 32::8].reshape(-1, n)
        lo = sim.step(x, u, torch.full((len(x),), -1., device='cuda'))
        hi = sim.step(x, u, torch.full((len(x),), 1., device='cuda'))
        difference = (lo-hi).square().mean((1,2)).sqrt()
        for j in range(len(x)):
            observable.append(dict(candidate=i, pair=j, observable_distance=0.,
                                   hidden_distance=2., next_state_rmse=float(difference[j]),
                                   spike_disagreement=float((lo[j,:,1]!=hi[j,:,1]).float().mean())))
        row = dict(candidate=i, **asdict(lc), rate=float(s.mean()), late_rate=float(late.mean()),
                   inactive_fraction=float((late.sum((1,2))==0).float().mean()),
                   next_state_rmse=float(difference.mean()),
                   observable_pairs_changed=float((difference>1e-6).float().mean()))
        row['eligible'] = (.001 < row['late_rate'] < .15 and row['next_state_rmse'] > .001
                           and row['inactive_fraction'] < .5)
        rows.append(row)
        print(row, flush=True)
    write_csv(out/'hidden_state_sweep.csv', rows)
    write_csv(out/'activity_stats.csv', activity)
    write_csv(out/'observability_analysis.csv', observable)
    eligible = [r for r in rows if r['eligible']]
    if not eligible:
        (out/'selection.json').write_text(json.dumps(dict(status='no_eligible_regime', rows=rows), indent=2))
        raise RuntimeError('No stable observable teacher regime; inspect calibration before training')
    # Maximize measurable teacher effect among activity-eligible settings; no model/test tuning.
    selected = max(eligible, key=lambda r:r['next_state_rmse'])
    result = dict(status='selected', config=asdict(cfg),
                  latent={k:selected[k] for k in ('rho','alpha','sigma_z')},
                  rule='max paired next-state RMSE among teacher activity-eligible candidates; calibration seeds only',
                  limitation='Branch observability is necessary, not sufficient evidence that history adds information.')
    (out/'selection.json').write_text(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    torch.set_num_threads(2)
    calibrate('results/latent_state_v1/teacher_calibration')
