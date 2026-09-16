"""latent_state_v3: causal validation of the v2 positive result.

Stage 1 replication trains only the informative subset (M0 gnn_k1,
M1 global_k32, M2 gshuffle, M3 wide, M4 oracle) at 5+ paired seeds on the
SAME teacher and the SAME trajectory pools as v2 (v2 data caches are loaded
read-only; v2 artifacts are never modified).
"""
import argparse
import json
from pathlib import Path
import torch
from run_latent_v2 import setup as setup_v2
from train_latent_v2 import train_one_v2

ROOT = Path('results/latent_state_v3')
CKPTS = ROOT / 'checkpoints'
REPLICATION_HIDDEN = ['gnn_k1', 'global_k32', 'gshuffle', 'wide', 'oracle']
REPLICATION_MARKOV = ['gnn_k1', 'global_k32']


def datasets_v3(regime):
    """Read-only load of the identical v2 cache (same teacher, same splits)."""
    path = Path('results/latent_state_v2/data') / f'{regime}.pt'
    blob = torch.load(path, map_location='cpu', weights_only=False)
    return {sp: {k: v.to('cuda') for k, v in d.items()} for sp, d in blob['data'].items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=['hidden', 'markov'], required=True)
    p.add_argument('--labels', nargs='+')
    p.add_argument('--seeds', nargs='+', type=int, default=[1234, 1235, 1236, 1237, 1238])
    p.add_argument('--epochs', type=int, default=24)
    p.add_argument('--steps', type=int, default=48)
    args = p.parse_args()
    torch.set_num_threads(2)
    cfg, lc, conn = setup_v2(args.stage)
    data = datasets_v3(args.stage)
    labels = args.labels or (REPLICATION_MARKOV if args.stage == 'markov' else REPLICATION_HIDDEN)
    out = ROOT / 'replication' / args.stage
    out.mkdir(parents=True, exist_ok=True)
    snapshot = dict(seeds=args.seeds, labels=labels, epochs=args.epochs, steps=args.steps, batch=16,
                    training='teacher forced; identical budget per label/seed',
                    data='read-only v2 cache (same teacher, same trajectory pools)',
                    protocol='latent_state_v3_replication')
    (out / 'config_snapshot.json').write_text(json.dumps(snapshot, indent=2))
    for seed in args.seeds:
        for label in labels:
            train_one_v2(label, seed, conn, cfg, lc, data, out, CKPTS / 'replication' / args.stage,
                         args.epochs, args.steps)
    print('V3 REPLICATION COMPLETE', args.stage, flush=True)


if __name__ == '__main__':
    main()
