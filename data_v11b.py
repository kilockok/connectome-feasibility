"""v11b data: testD - the v11b FINAL held-out evaluation set.

Rationale (spec Part B): testB was used to diagnose v11's failure, so v11b
trains/selects on train/val ONLY and evaluates finally on testD: FRESH
interpolated parameter points (interior of the v9 train ranges, distinct
from testB/testC) with FRESH trajectory seeds (test_seen idx 224..287,
unused by v9; null idx 288..351). Same stimulus protocol as testB
(split='test_seen'). Parameters were fixed here BEFORE any v11b training -
not selected from any result region.

  gainD : alpha=0.05, omega=0.10     (train corners (0.04,0.05),(0.04,0.12),(0.08,0.08); testB was (0.06,0.08))
  adaptD: beta=0.28, tau_a=12, c=0.18 (train beta[0.15,0.35] tau[8,40] c[0.15,0.25]; testB was (0.25,14,0.2))
  stpD  : s=0.022, m=0.75            (train s[0.015,0.025] m[0.5,2.0]; testB was (0.02,1.5))
"""
from pathlib import Path
import torch
from connectome import Connectome
from teachers_v9 import MechanismLIFSimulator, MechSpec, GainParams, AdaptParams, STPParams
from run_v7 import setup_cfg
from train_latent import atomic_save

ROOT = Path('results/latent_state_v11b')
TESTD = dict(
    gain=('gainD', GainParams(alpha=0.05, omega=0.10)),
    adapt=('adaptD', AdaptParams(beta=0.28, tau_a=12.0, c=0.18)),
    stp=('stpD', STPParams(strength=0.022, tau_scale=0.75)),
)
IDX0 = 224
NULL_IDX0 = 288
COUNT = 64


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    cfg = setup_cfg()
    conn = Connectome.generate(cfg)
    dev = torch.device('cuda')
    path = ROOT / 'configs' / 'testd_data.pt'
    if path.exists():
        print('cache exists:', path)
        return
    store = {}
    for fam, (name, p) in TESTD.items():
        sim = MechanismLIFSimulator(conn, cfg, dev, MechSpec(name=name, **{fam: p}))
        d = sim.generate([cfg.traj_seed('test_seen', IDX0 + j) for j in range(COUNT)], 'test_seen')
        store[f'{fam}/testD'] = dict(states=d['states'].cpu(), stimulus=d['stimulus'].cpu(),
                                     silence=d['silence'].cpu(),
                                     hidden_summary=d['hidden_summary'].cpu())
        print('gen', name, COUNT, 'rate', float(store[f'{fam}/testD']['states'][..., 1].mean()),
              flush=True)
        del sim
        torch.cuda.empty_cache()
    sim = MechanismLIFSimulator(conn, cfg, dev, MechSpec(name='null'))
    d = sim.generate([cfg.traj_seed('test_seen', NULL_IDX0 + j) for j in range(COUNT)], 'test_seen')
    store['null/testD'] = dict(states=d['states'].cpu(), stimulus=d['stimulus'].cpu(),
                               silence=d['silence'].cpu(),
                               hidden_summary=d['hidden_summary'].cpu())
    print('gen null testD', COUNT, 'rate', float(store['null/testD']['states'][..., 1].mean()),
          flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_save(dict(store=store, params={f: str(p) for f, (n, p) in TESTD.items()},
                     idx0=IDX0, null_idx0=NULL_IDX0, count=COUNT), path)
    print('TESTD COMPLETE ->', path, flush=True)


if __name__ == '__main__':
    main()
