"""v9 Stage 1: candidate-parameter pool effect measurement.

For each candidate config of each mechanism family, generate M trajectories
on shared dedicated seeds (offset 9e6) plus the base LIF on the same seeds,
and record observable effect statistics vs base. The selection step then
chooses an overlapping effect stratum (Gate A).
"""
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

ROOT = Path('results/latent_state_v9')
M = 8
POOL = {
    'gain': [MechSpec(name=f'gain_a{a}_w{w}', gain=GainParams(alpha=a, omega=w))
             for a in (0.15, 0.3, 0.45, 0.6) for w in (0.05, 0.08, 0.12)],
    'adapt': [MechSpec(name=f'adapt_b{b}_t{t}', adapt=AdaptParams(beta=b, tau_a=t))
              for b in (0.15, 0.3, 0.5, 0.8) for t in (8.0, 20.0, 40.0)],
    'stp': [MechSpec(name=f'stp_s{s}_m{m}', stp=STPParams(strength=s, tau_scale=m))
            for s in (0.05, 0.1, 0.2, 0.35, 0.5, 0.75) for m in (0.5, 1.0, 2.0)],
    'ou': [MechSpec(name=f'ou_g{g}_t{t}', ou=OUParams(sigma=g, tau=t))
           for g in (0.05, 0.1, 0.2, 0.35) for t in (8.0, 32.0)],
}


@torch.no_grad()
def effect_stats(d_mech, d_base):
    vt = d_mech['states'][:, 1:, :, 0]
    vb = d_base['states'][:, :, :, 0]          # dataset.generate_batch: [B,T]
    st_ = d_mech['states'][:, 1:, :, 1]
    sb = d_base['states'][:, :, :, 1]
    res = vt - vb
    tp = ((sb > .5) & (st_ > .5)).sum().float()
    fp = ((sb > .5) & (st_ <= .5)).sum().float()
    fn = ((sb <= .5) & (st_ > .5)).sum().float()
    prec, rec = tp / (tp + fp).clamp(min=1), tp / (tp + fn).clamp(min=1)
    f1_base_as_pred = 2 * prec * rec / (prec + rec).clamp(min=1e-9)
    return dict(residual_v_rms=float(res.square().mean().sqrt()),
                residual_v_std=float(res.std()),
                spike_disagree=float((st_ - sb).abs().mean()),
                f1_drop=float(1 - f1_base_as_pred),
                rate_mech=float(st_.mean()), rate_base=float(sb.mean()),
                rate_change=abs(float(st_.mean()) - float(sb.mean())))


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    cfg = setup_cfg()
    conn = Connectome.generate(cfg)
    dev = torch.device('cuda')
    base = LIFSimulator(conn, cfg, dev)
    seeds = [cfg.seed + 9_000_000 + i for i in range(M)]
    import dataset
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
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print('POOL COMPLETE ->', path, flush=True)


if __name__ == '__main__':
    main()


