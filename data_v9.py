"""v9 dataset generation: effect-matched candidate dataset (Stage 1 output).

Layout per family (gain/adapt/stp):
  train : 3 configs x 170 trajs (split train, idx 0..509)
  val   : 3 configs x 21      (split val,   idx 0..62)
  testA : 3 configs x 21      (split test_seen, idx 0..62)   held-out trajectories
  testB : 64                  (split test_seen, idx 64..127) held-out parameters
  testC : 32                  (split test_seen, idx 128..159) extrapolation
  null  : 64 base-LIF         (split test_seen, idx 160..223)
Every tensor stays on CPU in the cache; loaders move slices to GPU.
"""
import json
from dataclasses import asdict
from pathlib import Path
import torch
from connectome import Connectome
from teachers_v9 import MechanismLIFSimulator, MechSpec
from run_v7 import setup_cfg
from train_latent import atomic_save
from protocol_v9 import (ROOT, SELECTED, LAYOUT, TESTB_IDX0, TESTC_IDX0, NULL_IDX0,
                         spec_for, FAMILIES)

PROTOCOL = 'v9_effect_matched_v1'


def gen_range(sim, cfg, split, idx0, count, batch=64):
    chunks = []
    for i in range(idx0, idx0 + count, batch):
        ids = list(range(i, min(i + batch, idx0 + count)))
        d = sim.generate([cfg.traj_seed(split, j) for j in ids], split)
        chunks.append(dict(states=d['states'].cpu(), stimulus=d['stimulus'].cpu(),
                           silence=d['silence'].cpu(), hidden_summary=d['hidden_summary'].cpu()))
    return {k: torch.cat([c[k] for c in chunks]) for k in chunks[0]}


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    cfg = setup_cfg()
    conn = Connectome.generate(cfg)
    dev = torch.device('cuda')
    key = dict(config=asdict(cfg), protocol=PROTOCOL,
               selected={f: [n for n, _ in SELECTED[f]['train']] +
                         [SELECTED[f]['splitB'][0], SELECTED[f]['splitC'][0]] for f in FAMILIES})
    path = ROOT / 'data' / 'v9_data.pt'
    if path.exists():
        print('cache exists:', path)
        return
    store = {}
    L = LAYOUT
    for fam in FAMILIES:
        sel = SELECTED[fam]
        parts = {'train': [], 'val': [], 'testA': []}
        cfg_ids = {'train': [], 'val': [], 'testA': []}
        for ci, (pid, p) in enumerate(sel['train']):
            sim = MechanismLIFSimulator(conn, cfg, dev, spec_for(fam, pid))
            for split, count in (('train', L['train_per_config']), ('val', L['val_per_config']),
                                 ('testA', L['testA_per_config'])):
                idx0 = {'train': ci * L['train_per_config'],
                        'val': ci * L['val_per_config'],
                        'testA': ci * L['testA_per_config']}[split]
                sp = {'train': 'train', 'val': 'val', 'testA': 'test_seen'}[split]
                parts[split].append(gen_range(sim, cfg, sp, idx0, count))
                cfg_ids[split] += [ci] * count
                print('gen', fam, pid, split, count, flush=True)
            del sim
            torch.cuda.empty_cache()
        for split in parts:
            store[f'{fam}/{split}'] = {k: torch.cat([c[k] for c in parts[split]])
                                       for k in parts[split][0]}
            store[f'{fam}/{split}']['config_id'] = torch.tensor(cfg_ids[split])
        for key, split, idx0, count in (('splitB', 'testB', TESTB_IDX0, L['testB']),
                                        ('splitC', 'testC', TESTC_IDX0, L['testC'])):
            pid = sel[key][0]
            sim = MechanismLIFSimulator(conn, cfg, dev, spec_for(fam, pid))
            store[f'{fam}/{split}'] = gen_range(sim, cfg, 'test_seen', idx0, count)
            store[f'{fam}/{split}']['config_id'] = torch.full((count,), -1)
            print('gen', fam, pid, split, count, flush=True)
            del sim
            torch.cuda.empty_cache()
    sim = MechanismLIFSimulator(conn, cfg, dev, MechSpec(name='null'))
    store['null/test'] = gen_range(sim, cfg, 'test_seen', NULL_IDX0, L['null_test'])
    store['null/test']['config_id'] = torch.zeros(L['null_test'], dtype=torch.long)
    print('gen null test', L['null_test'], flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_save(dict(key=key, store=store), path)
    rates = {k: float(v['states'][..., 1].mean()) for k, v in store.items()}
    print('rates', json.dumps(rates, indent=1), flush=True)
    print('V9 DATA COMPLETE ->', path, flush=True)


if __name__ == '__main__':
    main()
