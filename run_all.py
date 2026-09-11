"""Run the full pipeline for one scale:
sanity -> dataset -> train all models -> evaluate + plots.

    python run_all.py [--scale small|full] [--models gru transformer connectome gnn]
"""
from __future__ import annotations

import argparse
import subprocess
import sys

from config import add_common_args

DEFAULT_MODELS = ["gru", "transformer", "connectome", "gnn"]


def run(cmd: list[str]):
    print(f"\n{'=' * 70}\n$ {' '.join(cmd)}\n{'=' * 70}", flush=True)
    r = subprocess.run([sys.executable] + cmd)
    if r.returncode != 0:
        raise SystemExit(f"command failed: {cmd}")


def main():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--unroll", type=int, default=0,
                        help="also run phase-2 unrolled fine-tuning with U steps")
    parser.add_argument("--unroll-epochs", type=int, default=10)
    args = parser.parse_args()

    run(["test_lif.py", "--scale", args.scale])
    run(["generate_dataset.py", "--scale", args.scale, "--inspect", "100"])
    if not args.skip_train:
        for m in args.models:
            run(["train.py", "--model", m, "--scale", args.scale])
        if args.unroll > 0:
            for m in args.models:
                run(["train.py", "--model", m, "--scale", args.scale,
                     "--unroll", str(args.unroll),
                     "--unroll-epochs", str(args.unroll_epochs)])
    run(["evaluate.py", "--scale", args.scale, "--models", *args.models])


if __name__ == "__main__":
    main()
