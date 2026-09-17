"""v9 Stage 1d: low-effect pool for tighter matching (disagree overlap)."""
import csv
import json
from dataclasses import asdict
from pathlib import Path
import torch
from connectome import Connectome
from lif import LIFSimulator
from teachers_v9 import (MechanismLIFSimulator, MechSpec, GainParams, AdaptParams,
                         STPParams, OUParams)
from run_v7 import setup_cfg
from effect_v9 import effect_stats
import dataset

SPECS = (
    [('gain', MechSpec(name=f'gain_a{a}_w{w}', gain=GainParams(alpha=a, omega=w)))
     for a in (0.04, 0.06, 0.08) for w in (0.05, 0.08, 0.12)] +
    [('adapt', MechSpec(name=f'adapt_c{c}_b{b}_t{t}', adapt=AdaptParams(beta=b, tau_a=t, c=c)))
     for (c, b) in ((0.15, 0.2), (0.25, 0.15), (0.15, 0.35), (0.25, 0.25), (0.35, 0.15))
     for t in (8.0, 20.0, 40.0)] +
    [('stp', MechSpec(name=f'stp_s{s}_m{m}', stp=STPParams(strength=s, tau_scale=m)))
     for s in (0.012, 0.015, 0.025) for m in (0.5, 1.0, 2.0)] +
    [('ou', MechSpec(name=f'ou_g{g}_t{t}', ou=OUParams(sigma=g, tau=t)))
     for g in (0.02, 0.03) for t in (8.0, 32.0)]
)


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    cfg = setup_cfg()
    conn = Connectome.generate(cfg)
    dev = torch.device('cuda')
    base = LIFSimulator(conn, cfg, dev)
    seeds = [cfg.seed + 9_000_000 + i for i in range(8)]
    d_base = dataset.generate_batch(seeds, 'train', base, cfg)
    rows = []
    for family, spec in SPECS:
        sim = MechanismLIFSimulator(conn, cfg, dev, spec)
        d = sim.generate(seeds, 'train')
        stats = effect_stats(d, d_base)
        row = dict(family=family, name=spec.name,
                   params=json.dumps({k: asdict(getattr(spec, k)) for k in ('gain', 'adapt', 'stp', 'ou')
                                      if getattr(spec, k) is not None}))
        row.update(stats)
        rows.append(row)
        print(family, spec.name, f"resRMS={stats['residual_v_rms']:.4f}",
              f"disagree={stats['spike_disagree']:.4f}",
              f"f1drop={stats['f1_drop']:.4f}", f"rate={stats['rate_mech']:.4f}",
              f"rchg={stats['rate_change']:.4f}", flush=True)
        del sim, d
        torch.cuda.empty_cache()
    path = Path('results/latent_state_v9/metrics/effect_pool.csv')
    with path.open('a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writerows(rows)
    print('LOW POOL COMPLETE')


if __name__ == '__main__':
    main()
