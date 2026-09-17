"""v9 Stage 1c: measure splitB/splitC selection candidates."""
import csv
import json
from dataclasses import asdict
from pathlib import Path
import torch
from connectome import Connectome
from lif import LIFSimulator
from teachers_v9 import (MechanismLIFSimulator, MechSpec, GainParams, AdaptParams,
                         STPParams)
from run_v7 import setup_cfg
from effect_v9 import effect_stats
import dataset

SPECS = [
    ('gain', MechSpec(name='gain_a0.22_w0.10', gain=GainParams(alpha=0.22, omega=0.10))),
    ('gain', MechSpec(name='gain_a0.45_w0.16', gain=GainParams(alpha=0.45, omega=0.16))),
    ('adapt', MechSpec(name='adapt_c0.25_b0.6_t14.0', adapt=AdaptParams(beta=0.6, tau_a=14.0, c=0.25))),
    ('adapt', MechSpec(name='adapt_c0.5_b0.8_t60.0', adapt=AdaptParams(beta=0.8, tau_a=60.0, c=0.5))),
    ('adapt', MechSpec(name='adapt_c0.5_b0.8_t40.0', adapt=AdaptParams(beta=0.8, tau_a=40.0, c=0.5))),
    ('stp', MechSpec(name='stp_s0.075_m1.5', stp=STPParams(strength=0.075, tau_scale=1.5))),
    ('stp', MechSpec(name='stp_s0.15_m4.0', stp=STPParams(strength=0.15, tau_scale=4.0))),
    ('stp', MechSpec(name='stp_s0.15_m0.35', stp=STPParams(strength=0.15, tau_scale=0.35))),
]


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
    print('CANDIDATES MEASURED')


if __name__ == '__main__':
    main()
