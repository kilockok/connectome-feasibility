"""Rollout-v4 orchestrator: 2x2 ablation A0-A3 (DAgger x Tangent), gated
follow-ups, unified eval (spec §29/§36).

    python run_rollout_v4.py --scale full            # A0-A3 + effects + eval
    python run_rollout_v4.py --scale full --experiments A0 A2
    python run_rollout_v4.py --scale small --smoke   # pipeline sanity
    python run_rollout_v4.py --scale full --eval-only

STEP 2: A0-A3 run as SUBPROCESSES (rollout_train.py --matrix v4) — no gates
between them; they ARE the ablation.  STEP 3: main effects / interaction
(§9) are computed from each summary's best val score (§17) and printed as
the intermediate conclusion, then the unified eval runs over phase1 + A0-A3
(written to results/rollout_v4/eval_unified/).  S0-S2 (STEP 4-6) and the
temporal hybrid (STEP 7-9) are NOT launched here — they are decided from the
2x2 outcome per spec §29.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from config import get_config, add_common_args
from rollout_config import (EXPERIMENTS_V4, ROLLOUT4_DIR, RolloutConfig,
                            final_ckpt_path, phase1_ckpt_path, summary_path)

ORDER = ["A0", "A1", "A2", "A3"]


def run(cmd: list[str], log_prefix: str) -> int:
    print(f"\n=== [run] {' '.join(cmd)}", flush=True)
    t0 = time.time()
    p = subprocess.run(cmd)
    print(f"=== [done:{p.returncode}] {log_prefix} "
          f"({(time.time() - t0) / 60:.1f} min)", flush=True)
    return p.returncode


def best_val_score(rc: RolloutConfig) -> float | None:
    sp = summary_path(rc)
    if not sp.exists():
        return None
    with open(sp) as f:
        s = json.load(f)
    stages = s.get("stages") or []
    vals = [st.get("best_score") for st in stages
            if st.get("best_score") is not None]
    return max(vals) if vals else None


def effects_2x2(scores: dict[str, float]) -> dict[str, float]:
    """Spec §9 main effects + interaction on the §17 val score."""
    m = lambda k: scores.get(k)          # noqa: E731
    return {
        "dagger": ((m("A1") or 0.0) + (m("A3") or 0.0)
                   - (m("A0") or 0.0) - (m("A2") or 0.0)) / 2.0,
        "tangent": ((m("A2") or 0.0) + (m("A3") or 0.0)
                    - (m("A0") or 0.0) - (m("A1") or 0.0)) / 2.0,
        "interaction": ((m("A3") or 0.0) - (m("A2") or 0.0)
                        - (m("A1") or 0.0) + (m("A0") or 0.0)),
    }


def print_intermediate(scores: dict[str, float]) -> None:
    """Spec §36: 中间结论 after the 2x2 (val-side only; test comes later)."""
    print("\n" + "=" * 68)
    print("[2x2] best val score (§17) per cell:")
    for k in ORDER:
        v = scores.get(k)
        print(f"  {k}: {v:.4f}" if v is not None else f"  {k}: MISSING")
    eff = effects_2x2(scores)
    print(f"[2x2] main effect DAgger   = {eff['dagger']:+.4f}")
    print(f"[2x2] main effect Tangent  = {eff['tangent']:+.4f}")
    print(f"[2x2] interaction (A3-A2-A1+A0) = {eff['interaction']:+.4f}")
    a3, a0 = scores.get("A3"), scores.get("A0")
    if a3 is not None and a0 is not None:
        print(f"[2x2] synergy check: A3-A0 = {a3 - a0:+.4f} "
              f"(vs sum of mains {eff['dagger'] + eff['tangent']:+.4f})")
    print("=" * 68, flush=True)
    out = ROLLOUT4_DIR / "ablation_2x2"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "effects_val.json", "w") as f:
        json.dump({"scores": scores, "effects": eff}, f, indent=2)


def entry_model(exp: str) -> str:
    """Architecture per entry family (spec §20-26): T* = GNN+Temporal
    Transformer, G* = param-matched wide GNN control, A*/S* = plain GNN."""
    e = exp.upper()
    if e.startswith("T"):
        return "gnn_temporal"
    if e.startswith("G"):
        return "gnn_wide"
    return "gnn"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--experiments", nargs="*", default=None,
                        help="subset of A0-A3 (default: all, in order)")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    cfg = get_config(args.scale)
    if args.seed is not None:
        cfg.seed = args.seed

    def mk_rc(exp: str) -> RolloutConfig:
        return RolloutConfig(base=cfg, experiment=exp, model=entry_model(exp),
                             scale=args.scale, version="v4",
                             **EXPERIMENTS_V4[exp])

    exps = [e.upper() for e in (args.experiments or ORDER)]
    for e in exps:
        if e not in EXPERIMENTS_V4:
            raise SystemExit(f"unknown v4 experiment {e!r} (choices: {ORDER})")

    p1 = phase1_ckpt_path(cfg, "gnn")
    if not args.eval_only and not p1.exists():
        raise SystemExit(f"[fatal] phase-1 checkpoint missing: {p1}\n"
                         f"train it first: train.py --model gnn "
                         f"--scale {args.scale}")
    # per-architecture phase-1 checkpoints (T/G entries need their own)
    if not args.eval_only:
        for e in exps:
            m = entry_model(e)
            pm = phase1_ckpt_path(cfg, m)
            if not pm.exists():
                raise SystemExit(f"[fatal] phase-1 checkpoint for {m} "
                                 f"missing: {pm}\ntrain it first: "
                                 f"train.py --model {m} --scale {args.scale}")

    scores: dict[str, float] = {}
    if not args.eval_only:
        for e in exps:
            rc = mk_rc(e)
            fcp = final_ckpt_path(rc)
            if fcp.exists() and not args.smoke:
                print(f"[skip] {e}: final checkpoint exists ({fcp.name})")
            else:
                cmd = [sys.executable, "-u", "rollout_train.py", "--scale",
                       args.scale, "--model", entry_model(e), "--matrix",
                       "v4", "--experiment", e]
                if args.smoke:
                    cmd += ["--smoke"]
                if args.device != "auto":
                    cmd += ["--device", args.device]
                code = run(cmd, f"v4/{e}")
                if code != 0:
                    raise SystemExit(f"experiment {e} failed (exit {code})")
            v = best_val_score(rc)
            if v is not None:
                scores[e] = v
        if len(scores) == 4:
            print_intermediate(scores)
        else:
            print(f"[2x2] incomplete ({sorted(scores)}); "
                  f"effects computed when all four cells exist")

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
        print("[eval] rollout_eval_v3.py not found; skipping eval pass")
        return
    cmd = [sys.executable, "-u", "rollout_eval_v3.py", "--scale", args.scale]
    cmd += [a for e, p in entries for a in ("--entry", f"{e}={p}")]
    cmd += ["--reinject", "--out", str(ROLLOUT4_DIR / "eval_unified")]
    if args.seed is not None:
        cmd += ["--seed", str(args.seed)]
    if args.device != "auto":
        cmd += ["--device", args.device]
    run(cmd, "eval/v4")


if __name__ == "__main__":
    main()
