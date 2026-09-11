"""Phase-2 experiment-matrix orchestrator.

Runs the GNN A-G ablation (A = phase-1 checkpoint, eval-only), the surrogate
sweep, transfers the winning recipe to the connectome transformer, and finally
kicks off one unified rollout_eval pass.

    python run_rollout_v2.py --scale full --model gnn --experiments A B C D E F G
    python run_rollout_v2.py --scale full --model gnn --experiments G --surrogate-sweep
    python run_rollout_v2.py --scale full --model connectome --experiments <winner>
    python run_rollout_v2.py --scale full --model gnn --eval-only
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from config import add_common_args, get_config
from rollout_config import CHECKPOINT_DIR, ROLLOUT2_DIR, add_rollout_args

ROOT = Path(__file__).resolve().parent


def _run(cmd: list[str]) -> int:
    print(f"[run] {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=ROOT)
    return r.returncode


def main():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    add_rollout_args(parser)
    parser.add_argument("--experiments", nargs="*", default=None,
                        help="subset of A-G (default: all)")
    parser.add_argument("--surrogate-sweep", action="store_true",
                        help="run last listed experiment with all 3 surrogates")
    parser.add_argument("--eval-only", action="store_true",
                        help="skip training; run rollout_eval over existing finals")
    args = parser.parse_args()
    cfg = get_config(args.scale)
    if args.seed is not None:
        cfg.seed = args.seed

    exps = [e.upper() for e in (args.experiments or list("ABCDEFG"))]
    base = [sys.executable, "rollout_train.py", "--model", args.model,
            "--scale", args.scale]
    if args.seed is not None:
        base += ["--seed", str(args.seed)]
    if getattr(args, "smoke", False):
        base += ["--smoke"]
    if getattr(args, "grad_ckpt", None):
        base += ["--grad-ckpt", args.grad_ckpt]

    trained_exps, extra_entries = [], []

    if not args.eval_only:
        sweep = []
        if args.surrogate_sweep:
            sweep = ["ste", "sigmoid", "fast_sigmoid"]
        for exp in exps:
            if exp == "A":
                continue                    # phase-1 checkpoint, eval-only
            if sweep:
                for sg in sweep:
                    code = _run(base + ["--experiment", exp,
                                        "--surrogate", sg])
                    if code != 0:
                        raise SystemExit(
                            f"experiment {exp}/{sg} failed (exit {code})")
                    extra_entries.append(
                        (f"{exp}-{sg}",
                         CHECKPOINT_DIR /
                         f"ckpt_{args.model}_{args.scale}_rollout_v2_"
                         f"{exp}_seed{cfg.seed}.pt"))
                    # later sweep runs would overwrite the same final name;
                    # stash each under a surrogate-specific copy
                    dst = CHECKPOINT_DIR / \
                        (f"ckpt_{args.model}_{args.scale}_rollout_v2_"
                         f"{exp}-{sg}_seed{cfg.seed}.pt")
                    src = CHECKPOINT_DIR / \
                        (f"ckpt_{args.model}_{args.scale}_rollout_v2_"
                         f"{exp}_seed{cfg.seed}.pt")
                    if src.exists():
                        dst.write_bytes(src.read_bytes())
                    extra_entries[-1] = (f"{exp}-{sg}", dst)
            else:
                code = _run(base + ["--experiment", exp])
                if code != 0:
                    raise SystemExit(f"experiment {exp} failed (exit {code})")
                trained_exps.append(exp)

    # ---- unified evaluation -------------------------------------------
    eval_cmd = [sys.executable, "rollout_eval.py", "--scale", args.scale,
                "--model", args.model, "--experiments"] + exps
    for label, path in extra_entries:
        if path.exists():
            eval_cmd += ["--entry", f"{label}={path}"]
    code = _run(eval_cmd)
    if code != 0:
        raise SystemExit(f"rollout_eval failed (exit {code})")
    print(f"[done] all outputs in {ROLLOUT2_DIR}")


if __name__ == "__main__":
    main()
