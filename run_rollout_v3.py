"""Rollout-v3 (v3r matrix) orchestrator: E0 -> E1 -> E2 -> E3, gated.

    python run_rollout_v3.py --scale full
    python run_rollout_v3.py --scale full --experiments E0 E1
    python run_rollout_v3.py --scale small --smoke     # pipeline sanity
    python run_rollout_v3.py --scale full --eval-only  # unified eval only

Each experiment runs as a SUBPROCESS (`rollout_train.py`) for GPU memory
hygiene.  Gates (user spec: "不要未经评估自动进入更长 horizon"):

  E0  runs once the phase-1 checkpoint exists.
  E1  requires E0 finished (final ckpt + summary).
  E2  additionally requires, from E1's summary (or E0's if E1 was skipped):
        calibrated one-step F1 of the last stage >= 0.98  AND
        best val rollout score > phase-1 reference score.
  E3  same gate checked against E2's summary.
  On gate failure the remaining experiments are SKIPPED (not failed) and the
  unified eval still runs over whatever exists.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from config import get_config, add_common_args
from rollout_config import (EXPERIMENTS_V3R, RolloutConfig, final_ckpt_path,
                            phase1_ckpt_path, summary_path)

ORDER = ["E0", "E1", "E2", "E3"]


def run(cmd: list[str], log_prefix: str) -> int:
    print(f"\n=== [run] {' '.join(cmd)}", flush=True)
    t0 = time.time()
    p = subprocess.run(cmd)
    print(f"=== [done:{p.returncode}] {log_prefix} "
          f"({(time.time() - t0) / 60:.1f} min)", flush=True)
    return p.returncode


def read_summary(rc: RolloutConfig) -> dict | None:
    sp = summary_path(rc)
    if not sp.exists():
        return None
    with open(sp) as f:
        return json.load(f)


def gate_ok(prev_rc: RolloutConfig) -> tuple[bool, str]:
    """Spec gate before advancing to a longer-horizon experiment."""
    s = read_summary(prev_rc)
    if s is None:
        return False, f"no summary at {summary_path(prev_rc)}"
    if not final_ckpt_path(prev_rc).exists():
        return False, "final checkpoint missing"
    stages = s.get("stages") or []
    if not stages:
        return False, "no stage summaries"
    f1_cal = stages[-1].get("calibrated_f1", 0.0)
    best = max((st.get("best_score", -1e9) for st in stages), default=-1e9)
    ref = s.get("ref_rollout_score", -1e9)
    if f1_cal < 0.98:
        return False, f"calibrated one-step F1 {f1_cal:.3f} < 0.98"
    if best <= ref:
        return False, f"best rollout score {best:.3f} <= phase-1 ref {ref:.3f}"
    return True, (f"F1={f1_cal:.3f}, best score={best:.3f} > ref {ref:.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--experiments", nargs="*", default=None,
                        help="subset of E0-E3 (default: all, in order)")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-gates", action="store_true",
                        help="run every listed experiment regardless of gates")
    parser.add_argument("--tangent", action="store_true",
                        help="forwarded: enable tangent loss for all runs")
    args = parser.parse_args()

    cfg = get_config(args.scale)
    if args.seed is not None:
        cfg.seed = args.seed

    def mk_rc(exp: str) -> RolloutConfig:
        return RolloutConfig(base=cfg, experiment=exp, model="gnn",
                             scale=args.scale, version="v3r",
                             **EXPERIMENTS_V3R[exp])

    exps = [e.upper() for e in (args.experiments or ORDER)]
    for e in exps:
        if e not in EXPERIMENTS_V3R:
            raise SystemExit(f"unknown v3r experiment {e!r} "
                             f"(choices: {ORDER})")

    p1 = phase1_ckpt_path(cfg, "gnn")
    if not args.eval_only and not p1.exists():
        raise SystemExit(f"[fatal] phase-1 checkpoint missing: {p1}\n"
                         f"train it first: train.py --model gnn "
                         f"--scale {args.scale}")

    if not args.eval_only:
        done, skipped = [], []
        for i, e in enumerate(exps):
            rc = mk_rc(e)
            fcp = final_ckpt_path(rc)
            if fcp.exists() and not args.smoke:
                print(f"[skip] {e}: final checkpoint exists ({fcp.name})")
                done.append(e)
                continue
            if i > 0 and not args.skip_gates:
                # gate on the nearest previously-finished experiment
                prev = None
                for pe in reversed(exps[:i]):
                    prc = mk_rc(pe)
                    if final_ckpt_path(prc).exists() or pe in done:
                        prev = prc
                        break
                if prev is not None:
                    ok, why = gate_ok(prev)
                    print(f"[gate] {e} <- {prev.experiment}: "
                          f"{'PASS' if ok else 'BLOCK'} ({why})")
                    if not ok:
                        skipped.append(e)
                        continue
            cmd = [sys.executable, "-u", "rollout_train.py", "--scale",
                   args.scale,
                   "--model", "gnn", "--matrix", "v3r", "--experiment", e]
            if args.smoke:
                cmd += ["--smoke"]
            if args.tangent:
                cmd += ["--tangent"]
            if args.device != "auto":
                cmd += ["--device", args.device]
            code = run(cmd, f"v3r/{e}")
            if code != 0:
                raise SystemExit(f"experiment {e} failed (exit {code})")
            done.append(e)
        if skipped:
            print(f"\n[gate] skipped (conditions not met): {skipped}")

    if args.skip_eval:
        return
    entries = []
    for e in exps:
        rc = mk_rc(e)
        p = final_ckpt_path(rc)
        if p.exists():
            entries.append((e, p))
        else:
            print(f"[eval] {e}: no final checkpoint, skipped")
    if p1.exists():
        entries.insert(0, ("phase1", p1))
    if not entries:
        print("[eval] nothing to evaluate")
        return
    eval_script = Path("rollout_eval_v3.py")
    if not eval_script.exists():
        print("[eval] rollout_eval_v3.py not found yet; skipping eval pass")
        return
    cmd = [sys.executable, "-u", "rollout_eval_v3.py", "--scale", args.scale]
    cmd += [a for e, p in entries for a in ("--entry", f"{e}={p}")]
    cmd += ["--reinject"]
    if args.seed is not None:
        cmd += ["--seed", str(args.seed)]
    if args.device != "auto":
        cmd += ["--device", args.device]
    run(cmd, "eval/v3r")


if __name__ == "__main__":
    main()
