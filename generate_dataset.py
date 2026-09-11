"""Pre-generate the fixed evaluation datasets (cached to results/cache/).

    python generate_dataset.py [--scale small|full] [--inspect 100]

The training split is generated on the fly during training from a fixed
seed pool; this script materialises the val / test_seen / test_ood splits
(identical for every model) and optionally saves a small inspection file.
"""
from __future__ import annotations

import argparse

import torch

from config import get_config, CACHE_DIR, add_common_args
from connectome import get_connectome
from dataset import generate_batch, load_or_generate, traj_count
from device import get_device
from lif import LIFSimulator


def main():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--inspect", type=int, default=0,
                        help="also save N train-protocol trajectories for inspection")
    args = parser.parse_args()

    cfg = get_config(args.scale)
    torch.manual_seed(cfg.seed)
    device = get_device(override=args.device)
    conn = get_connectome(cfg, device)
    print(f"[connectome] {conn.summary()}")
    sim = LIFSimulator(conn, cfg, device)

    for split in ("val", "test_seen", "test_ood"):
        d = load_or_generate(split, sim, cfg)
        S = d["states"][..., 1]
        stim_neurons = sorted({int(i) for m in d["meta"] for i in m["stim_ids"]})
        lo, hi = cfg.stim_range(split)
        print(f"[{split}] n={traj_count(cfg, split)} rate={S.mean():.4f} "
              f"stim neurons in [{min(stim_neurons)}, {max(stim_neurons)}] "
              f"(protocol [{lo}, {hi}))")

    if args.inspect > 0:
        seeds = [cfg.traj_seed("train", i) for i in range(args.inspect)]
        d = generate_batch(seeds, "train", sim, cfg)
        out = {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in d.items()}
        path = CACHE_DIR / f"inspect_train_N{cfg.n_neurons}_n{args.inspect}.pt"
        torch.save(out, path)
        S = d["states"][..., 1]
        print(f"[inspect] {args.inspect} train trajectories -> {path} "
              f"(rate={S.mean():.4f})")


if __name__ == "__main__":
    main()
