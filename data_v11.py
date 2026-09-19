"""v11 data: reuse the v9 effect-matched pool + generate the NULL-teacher
training split (artifact control, G2). No new teacher physics; NULL uses the
same train/val trajectory seeds as the mechanism families (paired stimuli).

OOD datasets (OU / mixtures / v10-library stimulus patterns) are generated in
data_v11_ood (Stage 3) to keep Stage 1 lean.
"""
from pathlib import Path
import torch
from connectome import Connectome
from teachers_v9 import MechanismLIFSimulator, MechSpec
from protocol_v9 import ROOT as ROOT9, FAMILIES
from train_latent import atomic_save

ROOT = Path('results/latent_state_v11')
GPU_KEYS = ('states', 'stimulus', 'silence')


def load_pool(cfg):
    """v9 mixed mechanism pool (gain/adapt/stp, mechanism-blind) + null test."""
    blob = torch.load(ROOT9 / 'data' / 'v9_data.pt', map_location='cpu', weights_only=False)
    store = blob['store']
    pool = {}
    for split in ('train', 'val', 'testA', 'testB', 'testC'):
        parts, mech = [], []
        for fi, fam in enumerate(FAMILIES):
            d = store[f'{fam}/{split}']
            parts.append(d)
            mech += [fi] * len(d['states'])
        pool[split] = {k: torch.cat([p[k] for p in parts]) for k in GPU_KEYS}
        pool[split]['mech'] = torch.tensor(mech)
    pool['null'] = {k: v for k, v in store['null/test'].items() if k in GPU_KEYS}
    pool['null']['mech'] = torch.full((len(pool['null']['states']),), -1)
    return pool


def null_train(cfg, conn, n_train=512, n_val=64):
    """NULL-teacher train/val trajectories (same seeds => paired stimuli with
    the mechanism families' train/val splits). Cached; pure base dynamics."""
    path = ROOT / 'data' / 'null_train.pt'
    if path.exists():
        return torch.load(path, map_location='cpu', weights_only=False)
    (ROOT / 'data').mkdir(parents=True, exist_ok=True)
    sim = MechanismLIFSimulator(conn, cfg, torch.device('cuda'), MechSpec(name='null'))
    out = {}
    for split, count, idx0 in (('train', n_train, 0), ('val', n_val, 0)):
        chunks = []
        for i in range(0, count, 64):
            seeds = [cfg.traj_seed('train' if split == 'train' else 'val', idx0 + j)
                     for j in range(i, min(i + 64, count))]
            d = sim.generate(seeds, 'train' if split == 'train' else 'val')
            chunks.append({k: v.cpu() for k, v in d.items() if k in GPU_KEYS})
        out[split] = {k: torch.cat([c[k] for c in chunks]) for k in GPU_KEYS}
        print('v11 null data', split, count,
              'rate', float(out[split]['states'][..., 1].mean()), flush=True)
    atomic_save(out, path)
    return out


if __name__ == '__main__':
    from run_v7 import setup_cfg
    cfg = setup_cfg()
    conn = Connectome.generate(cfg)
    null_train(cfg, conn)
    print('null train data ready')
