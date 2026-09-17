"""v9 Stage 1b: extend the pool toward milder ADAPT / lower GAIN,STP so the
effect strata can overlap (Gate A matching)."""
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

ROOT = Path('results/latent_state_v9')
M = 8
POOL = {
    'gain': [MechSpec(name=f'gain_a{a}_w{w}', gain=GainParams(alpha=a, omega=w))
             for a in (0.08, 0.1) for w in (0.05, 0.08, 0.12)],
    'adapt': ([MechSpec(name=f'adapt_c{c}_b{b}_t{t}', adapt=AdaptParams(beta=b, tau_a=t, c=c))
               for c in (0.15, 0.25) for b in (0.3, 0.5, 0.8) for t in (8.0, 20.0, 40.0)] +
              [MechSpec(name='adapt_c0.35_b0.8_t8.0', adapt=AdaptParams(beta=0.8, tau_a=8.0, c=0.35)),
               MechSpec(name='adapt_c0.35_b0.5_t20.0', adapt=AdaptParams(beta=0.5, tau_a=20.0, c=0.35))]),
    'stp': [MechSpec(name=f'stp_s{s}_m{m}', stp=STPParams(strength=s, tau_scale=m))
            for s in (0.02, 0.03) for m in (0.5, 1.0, 2.0)],
}


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    cfg = setup_cfg()
    conn = Connectome.generate(cfg)
    dev = torch.device('cuda')
    base = LIFSimulator(conn, cfg, dev)
    seeds = [cfg.seed + 9_000_000 + i for i in range(M)]
    d_base = dataset.generate_batch(seeds, 'train', base, cfg)
    rows = []
    for family, specs in POOL.items():
        for spec in specs:
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
                  f"f1drop={stats['f1_drop']:.4f}", f"rate={stats['rate_mech']:.4f}", flush=True)
            del sim, d
            torch.cuda.empty_cache()
    path = ROOT / 'metrics' / 'effect_pool.csv'
    with path.open('a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writerows(rows)
    print('POOL EXTENDED ->', path, flush=True)


if __name__ == '__main__':
    main()
