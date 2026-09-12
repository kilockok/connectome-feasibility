"""Phase-2/3 experiment orchestrator (see PHASE2.md, run_rollout_v2.py).

    python run_rollout_v2.py --scale full --model gnn --experiments B C D E F G
    python run_rollout_v2.py --scale small --matrix v3 --model gnn_temporal \
        --experiments D E F
    python run_rollout_v2.py --scale full --model gnn --experiments G \
        --surrogate-sweep          # best-experiment x {ste, sigmoid, fast_sigmoid}
    python run_rollout_v2.py --scale small --matrix v3 --model gnn_temporal \
        --experiments D E F --eval-only

Each experiment runs as a SUBPROCESS (`rollout_train.py`) for GPU memory
hygiene, sequentially. Afterwards one evaluation pass runs over all
produced checkpoints: rollout_eval.py for matrix v2, temporal_eval.py for
matrix v3. `--smoke` propagates to every subprocess.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from config import get_config, add_common_args
from rollout_config import (EXPERIMENTS, EXPERIMENTS_V3, RolloutConfig,
                            final_ckpt_path, phase1_ckpt_path, summary_path)

MATRICES = {"v2": EXPERIMENTS, "v3": EXPERIMENTS_V3}
SURROGATES = ("ste", "sigmoid", "fast_sigmoid")


def run(cmd: list[str], log_prefix: str) -> int:
    print(f"\n=== [run] {' '.join(cmd)}")
    t0 = __import__("time").time()
    p = subprocess.run(cmd)
    dt = __import__("time").time() - t0
    print(f"=== [done:{p.returncode}] {log_prefix} ({dt / 60:.1f} min)")
    return p.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--model", default="gnn",
                        choices=["gnn", "connectome", "gnn_temporal",
                                 "gnn_wide"])
    parser.add_argument("--matrix", default="v2", choices=list(MATRICES))
    parser.add_argument("--experiments", nargs="*", default=None,
                        help="letters within the matrix (v2: A-G, v3: D-F)")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--surrogate-sweep", action="store_true",
                        help="after the main runs: repeat the LAST listed "
                             "experiment once per surrogate mode")
    parser.add_argument("--eval-only", action="store_true",
                        help="skip training, run only the unified eval pass")
    parser.add_argument("--skip-eval", action="store_true")
    # forwarded trainer options
    parser.add_argument("--k-hist", type=int, default=None)
    parser.add_argument("--t-layers", type=int, default=None)
    parser.add_argument("--t-heads", type=int, default=None)
    parser.add_argument("--pos", default=None, choices=["learned", "sincos"])
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--stages", default=None)
    parser.add_argument("--tbptt", type=int, default=None)
    parser.add_argument("--grad-ckpt", default=None,
                        choices=["auto", "on", "off"])
    parser.add_argument("--buffer-capacity", type=int, default=None)
    args = parser.parse_args()

    cfg = get_config(args.scale)
    if args.seed is not None:
        cfg.seed = args.seed
    experiments = MATRICES[args.matrix]
    exps = [e.upper() for e in (args.experiments or list(experiments))]
    for e in exps:
        if e not in experiments:
            raise SystemExit(f"unknown experiment {e!r} for matrix "
                             f"{args.matrix} (choices: {list(experiments)})")

    def trainer_cmd(exp: str, extra: list[str]) -> list[str]:
        cmd = [sys.executable, "rollout_train.py", "--scale", args.scale,
               "--model", args.model, "--matrix", args.matrix,
               "--experiment", exp]
        if args.seed is not None:
            cmd += ["--seed", str(args.seed)]
        for flag, val in (("--k-hist", args.k_hist),
                          ("--t-layers", args.t_layers),
                          ("--t-heads", args.t_heads),
                          ("--pos", args.pos),
                          ("--dropout", args.dropout),
                          ("--stages", args.stages),
                          ("--tbptt", args.tbptt),
                          ("--grad-ckpt", args.grad_ckpt),
                          ("--buffer-capacity", args.buffer_capacity)):
            if val is not None:
                cmd += [flag, str(val)]
        if args.causal:
            cmd += ["--causal"]
        if args.smoke:
            cmd += ["--smoke"]
        if args.device != "auto":
            cmd += ["--device", args.device]
        return cmd + extra

    if not args.eval_only:
        for e in exps:
            if experiments[e].get("eval_only"):
                print(f"[skip] {e}: eval-only experiment")
                continue
            rc_ = RolloutConfig(base=cfg, experiment=e, model=args.model,
                                scale=args.scale, version=args.matrix,
                                **experiments[e])
            p1 = phase1_ckpt_path(cfg, args.model)
            if not p1.exists():
                raise SystemExit(f"[fatal] phase-1 checkpoint missing: {p1}")
            if final_ckpt_path(rc_).exists() and not args.smoke:
                print(f"[skip] {e}: final checkpoint exists "
                      f"({final_ckpt_path(rc_).name}); delete to retrain")
                continue
            code = run(trainer_cmd(e, []), f"{args.matrix}/{e}")
            if code != 0:
                raise SystemExit(f"experiment {e} failed (exit {code})")
        if args.surrogate_sweep:
            last = exps[-1]
            for s in SURROGATES:
                code = run(trainer_cmd(last, ["--surrogate", s]),
                           f"{args.matrix}/{last}/surr={s}")
                if code != 0:
                    raise SystemExit(f"surrogate sweep {s} failed")

    if args.skip_eval:
        return

    # ---- unified eval over produced finals --------------------------------
    entries = []
    for e in exps:
        rc_ = RolloutConfig(base=cfg, experiment=e, model=args.model,
                            scale=args.scale, version=args.matrix,
                            **experiments[e])
        path = phase1_ckpt_path(cfg, args.model) \
            if experiments[e].get("eval_only") else final_ckpt_path(rc_)
        if path.exists():
            entries.append((e, path))
        elif not experiments[e].get("eval_only"):
            print(f"[eval] {e}: no final checkpoint, skipped")
    if not entries:
        print("[eval] nothing to evaluate")
        return
    if args.matrix == "v2":
        cmd = [sys.executable, "rollout_eval.py", "--scale", args.scale] + \
              [a for e, p in entries for a in ("--entry", f"{e}={p}")]
    else:
        cmd = [sys.executable, "temporal_eval.py", "--scale", args.scale] + \
              [a for e, p in entries for a in ("--entry", f"{e}={p}")]
    if args.seed is not None:
        cmd += ["--seed", str(args.seed)]
    if args.device != "auto":
        cmd += ["--device", args.device]
    run(cmd, f"eval/{args.matrix}")


if __name__ == "__main__":
    main()
