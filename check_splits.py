"""Sanity check 4: strict trajectory-level split, verified empirically.

    python check_splits.py [--scale full]

1. The trajectory seed pools of train/val/test/test_seen must be disjoint
   (they are disjoint by construction via split offsets; asserted here).
2. No two splits may contain the same trajectory. Since a trajectory is
   fully determined by its seed, disjoint seeds suffice in theory; we also
   regenerate the *stimulus* of every eval trajectory plus the full train
   pool and assert no cross-split duplicate actually occurs.
3. Windows never cross trajectory boundaries (make_windows slices t0..t0+K
   within one trajectory; target index t0+K <= T-1 by construction).

Windowing itself cannot leak across splits: train windows come from
generate_batch(seeds from the train pool) and val/test windows from
load_or_generate on disjoint seed pools.
"""
from __future__ import annotations

import hashlib

import torch

from config import get_config, add_common_args
from dataset import build_stimulus, sample_traj_params, traj_count


def stim_hash(params, cfg) -> str:
    s = build_stimulus(params, cfg)
    return hashlib.sha256(s.numpy().tobytes()).hexdigest()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    args = parser.parse_args()
    cfg = get_config(args.scale)

    splits = ("train", "val", "test_ood", "test_seen")
    pools = {sp: {cfg.traj_seed(sp, i) for i in range(traj_count(cfg, sp))}
             for sp in splits}

    print("[1] seed pool sizes:",
          {sp: len(p) for sp, p in pools.items()})
    for i, a in enumerate(splits):
        for b in splits[i + 1:]:
            inter = pools[a] & pools[b]
            assert not inter, f"seed overlap {a}/{b}: {len(inter)}"
    print("[1] PASS: all seed pools pairwise disjoint")

    # [2] empirical duplicate scan on actual stimulus tensors
    hashes = {}
    n_dup = 0
    for sp in splits:
        for i in range(traj_count(cfg, sp)):
            g = torch.Generator().manual_seed(cfg.traj_seed(sp, i))
            p = sample_traj_params(g, cfg, sp)
            h = stim_hash(p, cfg)
            if h in hashes:
                n_dup += 1
                print(f"    DUPLICATE: {sp}#{i} == {hashes[h]}")
            else:
                hashes[h] = sp
    assert n_dup == 0, f"{n_dup} duplicate stimuli across splits"
    print(f"[2] PASS: {len(hashes)} unique stimuli, no cross-split duplicate")

    # [3] stimulus ranges are spatially disjoint where required (check 5)
    tr_lo, tr_hi = cfg.stim_range("train")
    od_lo, od_hi = cfg.stim_range("test_ood")
    assert od_lo >= tr_hi, "test_ood stimulus range overlaps train range"
    print(f"[3] PASS: test_ood stimulus range [{od_lo},{od_hi}) is fully "
          f"outside train range [{tr_lo},{tr_hi}) — neurons never directly "
          f"stimulated during training (check 5 holds by construction)")

    print("all split checks passed")


if __name__ == "__main__":
    main()
