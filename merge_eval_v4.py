"""Merge two rollout_eval_v3.py payloads into the final v4 eval_unified.

The original 19-entry run died mid-S5 (accidental server shutdown) and the
script only persists results at the end, so completed entries were lost.
Each entry is deterministic across runs (torch.manual_seed(cfg.seed+3407)
is re-set per entry before that entry runs), so merging the local midterm
payload (phase1, A0-A3) with a resumed run covering the remaining entries
is exact.

Usage: python merge_eval_v4.py MIDTERM_JSON RESUME_JSON OUT_DIR
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rollout_eval_v3 import (make_figures, write_metrics_csv,
                             write_summary_tables)

ORDER = ["phase1", "A0", "A1", "A2", "A3",
         "S1", "S2", "S3", "S4", "S5", "S6", "S7",
         "T2", "T3", "G1",
         "p1temp", "p1temp_k16", "p1temp_k1", "p1wide"]


def load(path: str):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    for e in d["entries"].values():
        for split, hm in e.get("rollout", {}).items():
            e["rollout"][split] = {int(h): m for h, m in hm.items()}
    return d


def main():
    a = load(sys.argv[1])   # midterm: phase1, A0-A3
    b = load(sys.argv[2])   # resume: S1..p1wide
    for k in ("scale", "seed", "n_neurons", "T", "K", "H", "horizons",
              "n_traj", "reinject_H"):
        assert a[k] == b[k], f"protocol mismatch {k}: {a[k]!r} vs {b[k]!r}"
    entries = {}
    for lab in ORDER:
        src = a["entries"].get(lab) or b["entries"].get(lab)
        assert src is not None, f"entry missing from both payloads: {lab}"
        entries[lab] = src
    extra = (set(a["entries"]) | set(b["entries"])) - set(ORDER)
    assert not extra, f"unmapped entries: {extra}"

    out = Path(sys.argv[3])
    fig_dir = out / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    payload = {k: a[k] for k in ("scale", "seed", "n_neurons", "T", "K", "H",
                                 "horizons", "n_traj", "reinject_H")}
    payload["entries"] = entries
    (out / "metrics_unified.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[save] {out / 'metrics_unified.json'} ({len(entries)} entries)")
    write_metrics_csv(entries, ORDER, out / "metrics_unified.csv")
    t12 = write_summary_tables(entries, ORDER, a["reinject_H"],
                               out / "summary_tables.md")
    make_figures(entries, ORDER, a["H"], a["reinject_H"], fig_dir)
    print("[done]")


if __name__ == "__main__":
    main()
