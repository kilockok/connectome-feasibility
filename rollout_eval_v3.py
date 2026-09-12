"""Unified rollout-v3 evaluation: the v3r DAgger+TBPTT matrix vs phase-1.

Evaluates the rollout-v3 DAgger+TBPTT matrix checkpoints (experiments E0..E3
of matrix "v3r" in rollout_config.MATRICES) against the phase-1 one-step GNN
baseline under one protocol:

  * spike threshold tuned on VAL ONLY (evaluate.tune_threshold), then frozen
    for all test evaluation (same OOD protocol as rollout_eval.py)
  * closed-loop autoregressive rollout from t0=0 over H = min(max horizon,
    T-K) steps on BOTH test_seen and test_ood splits
  * per-horizon metrics, per trajectory first and then averaged (the shared
    aggregation convention of rollout_eval.py), plus per-trajectory std
  * effective prediction horizon per entry x split: first horizon where
    spike F1 drops below 0.9 / 0.7, where the rate ratio leaves
    [0.8, 1.25], where mean state cosine drops below 0.95, plus a silent
    collapse diagnostic (rate ratio < 0.2 sustained for >= 5 steps)
  * optional reinjection retest (--reinject): re-anchor the rollout to ground
    truth every K in {1,5,10,20,50} steps plus K=inf, report spike F1@100 and
    F1 vs steps-since-anchor buckets (reinjection.rollout_reinject machinery)

CLI:
    python rollout_eval_v3.py --scale full \
        --entry phase1=results/checkpoints/ckpt_gnn_full_seed1234.pt \
        --entry E3=results/checkpoints/ckpt_gnn_full_rollout_v3r_E3_seed1234.pt \
        --reinject
    python rollout_eval_v3.py --scale small \
        --entry phase1=results/checkpoints/ckpt_gnn_small_seed1234.pt \
        --n-traj 8 --horizons 1,10,25 --out results/rollout_v3/eval_test

With no --entry, defaults to: phase-1 = phase1_ckpt_path(cfg, --model), plus
every v3r experiment E0..E3 whose final_ckpt_path exists on disk. Missing
checkpoints are skipped with a printed notice.

Writes into --out (default results/rollout_v3/eval_unified):
  metrics_unified.json   nested entry -> split -> horizon -> metrics, plus
                         effective horizons and reinjection results
  metrics_unified.csv    flat long-form: entry, split, horizon, metric, value
  summary_tables.md      the three tables (rollout F1 / effective horizon /
                         reinjection)
  figures/fig1..fig9     effective horizon, F1 / rate ratio / V RMSE / pop
                         corr / state cosine vs horizon, silent collapse,
                         reinjection by K and by steps-since-anchor
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch

from config import get_config, add_common_args
from connectome import get_connectome
from dataset import load_or_generate
from device import get_device
from evaluate import tune_threshold, onestep_eval
from lif import LIFSimulator
from models import build_model
from models.mechanistic import maybe_wrap
from reinjection import rollout_reinject
from rollout import rollout
from rollout_config import (MATRICES, ROLLOUT3R_DIR, RolloutConfig,
                            final_ckpt_path, phase1_ckpt_path)
from rollout_eval import (_at_horizon, _downsample_series, _fmt_cell,
                          _nanmean, _population_groups, _sanitise,
                          attractor_diagnostics, rollout_series)

SPLITS = ("test_seen", "test_ood")
SPLIT_TAGS = {"test_seen": "seen", "test_ood": "ood"}
V3R_EXPERIMENTS = ("E0", "E1", "E2", "E3")
ROLLOUT_CHUNK = 16            # trajectories per rollout batch (memory guard)
REINJECT_KS = (1, 5, 10, 20, 50, 10_000)   # 10_000 == infinity (no reinject)
REINJECT_H = 100
BUCKET_RANGES = ((1, 5), (6, 10), (11, 20), (21, 50), (51, 10 ** 9))
BUCKET_LABELS = ("1-5", "6-10", "11-20", "21-50", "51+")
K_LABELS = ("1", "5", "10", "20", "50", "inf")
SEED_OFFSET = 3407            # fixed offset: reseed identically per entry
ATTRACTOR_FLAGS = ("dead", "saturated", "collapsed_variance", "periodic",
                   "fixed_subset")


# ----------------------------------------------------------------------
# small numeric helpers
def _nanstd(x: torch.Tensor) -> torch.Tensor:
    """Std over finite entries (ddof=1); NaN if fewer than two finite."""
    finite = torch.isfinite(x)
    if finite.sum() < 2:
        return torch.tensor(float("nan"))
    return x[finite].std()


def _flags_str(attractor: dict) -> str:
    """Compact 'flag(seen), flag(ood)' summary of attractor diagnostics."""
    flags = []
    for split in SPLITS:
        a = attractor.get(split, {})
        for f in ATTRACTOR_FLAGS:
            if a.get(f):
                flags.append(f"{f}({SPLIT_TAGS[split]})")
    return ", ".join(flags) if flags else "no"


def _empty_cache(device) -> None:
    if device.type == "xpu":
        torch.xpu.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def _entry_colors(labels):
    """tab10 per entry; identical to rollout_eval._entry_colors."""
    cmap = plt.get_cmap("tab10")
    colors, j = {}, 0
    for lab in labels:
        colors[lab] = cmap(j % 10)
        j += 1
    return colors


# ----------------------------------------------------------------------
# metric computation (pred/true [B, H, N, 3] CPU tensors)
@torch.no_grad()
def horizon_metrics_v3(pred: torch.Tensor, true: torch.Tensor,
                       h: int) -> dict:
    """Rollout metrics over the window [0, h), per trajectory then averaged.

    Every metric is reported as the mean over trajectories plus the
    per-trajectory std ('<metric>_std'). Keys: spike_f1, spike_precision,
    spike_recall, v_rmse, firing_rate_pred, firing_rate_true, rate_ratio
    (= (pred+1e-9)/(true+1e-9) per trajectory), pop_corr (uncentred cosine
    between per-neuron spike-rate vectors, = rollout_metrics pop_similarity),
    active_overlap (Jaccard of neurons active at all in the window),
    state_cosine (flattened [N,3] state at step h-1).
    """
    p, t = pred[:, :h], true[:, :h]
    ps, ts = p[..., 1], t[..., 1]
    pv, tv = p[..., 0], t[..., 0]
    B = p.shape[0]

    tp = (ps * ts).sum(dim=(1, 2))
    fp = (ps * (1 - ts)).sum(dim=(1, 2))
    fn = ((1 - ps) * ts).sum(dim=(1, 2))
    prec = tp / (tp + fp).clamp(min=1.0)
    rec = tp / (tp + fn).clamp(min=1.0)
    f1 = 2 * prec * rec / (prec + rec).clamp(min=1e-9)

    v_rmse = ((pv - tv) ** 2).mean(dim=(1, 2)).sqrt()

    rate_p = ps.mean(dim=(1, 2))                        # [B]
    rate_t = ts.mean(dim=(1, 2))
    rate_ratio = (rate_p + 1e-9) / (rate_t + 1e-9)

    rp, rt = ps.mean(dim=1), ts.mean(dim=1)             # [B, N] per-neuron rates
    num = (rp * rt).sum(dim=1)
    den = rp.norm(dim=1) * rt.norm(dim=1)
    pop = torch.where(den > 1e-12, num / den.clamp(min=1e-12),
                      torch.full_like(num, float("nan")))

    ap, at = ps.sum(dim=1) > 0, ts.sum(dim=1) > 0
    inter = (ap & at).sum(dim=1).float()
    union = (ap | at).sum(dim=1).float()
    overlap = torch.where(union > 0, inter / union.clamp(min=1),
                          torch.full_like(inter, float("nan")))

    state_cos = torch.nn.functional.cosine_similarity(
        p[:, h - 1].reshape(B, -1), t[:, h - 1].reshape(B, -1), dim=1)

    out = {}
    for name, x in (("spike_f1", f1), ("spike_precision", prec),
                    ("spike_recall", rec), ("v_rmse", v_rmse),
                    ("firing_rate_pred", rate_p), ("firing_rate_true", rate_t),
                    ("rate_ratio", rate_ratio), ("pop_corr", pop),
                    ("active_overlap", overlap), ("state_cosine", state_cos)):
        out[name] = float(_nanmean(x))
        out[name + "_std"] = float(_nanstd(x))
    return out


def effective_horizons(metrics: dict, series: dict, max_h: int) -> dict:
    """Effective prediction horizon from the per-horizon rollout series.

    H_F1_0.9 / H_F1_0.7: first horizon with mean spike F1 below the level
    (else max evaluated horizon). H_rate_0.8: first horizon where the rate
    ratio leaves [0.8, 1.25]. H_state: first horizon where the mean
    per-trajectory state cosine drops below 0.95. Silent collapse: first
    step where the per-timestep rate ratio stays < 0.2 for >= 5 consecutive
    steps (else None) plus a boolean flag.
    """
    hs = sorted(int(h) for h in metrics)

    def first_below(key: str, thr: float) -> int:
        for h in hs:
            v = metrics[h].get(key)
            if v is not None and math.isfinite(v) and v < thr:
                return h
        return max_h

    h_rate = max_h
    for h in hs:
        r = metrics[h].get("rate_ratio")
        if r is not None and math.isfinite(r) and (r < 0.8 or r > 1.25):
            h_rate = h
            break

    ratios = [(p + 1e-9) / (t + 1e-9)
              for p, t in zip(series["rate_pred"], series["rate_true"])]
    step = None
    for i in range(len(ratios) - 4):
        if all(r < 0.2 for r in ratios[i:i + 5]):
            step = i + 1                                   # 1-indexed step
            break

    return {"H_F1_0.9": first_below("spike_f1", 0.9),
            "H_F1_0.7": first_below("spike_f1", 0.7),
            "H_rate_0.8": h_rate,
            "H_state": first_below("state_cosine", 0.95),
            "silent_collapse_step": step,
            "silent_collapse": step is not None}


# ----------------------------------------------------------------------
# rollout driver — trajectory chunks keep peak memory bounded
@torch.no_grad()
def run_entry_rollout(model, data: dict, cfg, device, H: int, horizons,
                      threshold: float, n_traj: int,
                      chunk: int = ROLLOUT_CHUNK):
    """Closed-loop rollout from t0=0 (hard reset, silence masks from the
    data). Returns (metrics per horizon, time series, attractor diag)."""
    K = cfg.K
    n = min(n_traj, data["states"].shape[0])
    states, stim = data["states"][:n], data["stimulus"][:n]
    sil = data.get("silence")
    preds = []
    for i0 in range(0, n, chunk):
        sl = slice(i0, min(i0 + chunk, n))
        context = torch.cat([states[sl, :K],
                             stim[sl, :K].unsqueeze(-1)], dim=-1).to(device)
        fut = stim[sl, K:K + H].to(device)
        sm = sil[sl].to(device) if sil is not None else None
        p = rollout(model, context, fut, cfg, spike_threshold=threshold,
                    silence_mask=sm)
        preds.append(p.float().cpu())
    pred = torch.cat(preds)
    true = states[:, K:K + H].float().cpu()

    metrics = {h: horizon_metrics_v3(pred, true, h)
               for h in horizons if h <= H}
    series = rollout_series(pred, true)
    attractor = attractor_diagnostics(pred, H)
    return metrics, series, attractor


# ----------------------------------------------------------------------
# reinjection retest (reinjection.py machinery, chunked over trajectories)
@torch.no_grad()
def run_reinjection(model, data: dict, cfg, device, threshold: float,
                    n_traj: int, H_re: int,
                    chunk: int = ROLLOUT_CHUNK) -> dict:
    """Re-anchor the rollout to ground truth every K steps (K=inf: free
    rollout). Returns per K: overall spike F1@H_re (per trajectory, then
    averaged) and pooled spike F1 per steps-since-anchor bucket aggregated
    over all anchor resets.
    """
    K = cfg.K
    n = min(n_traj, data["states"].shape[0])
    states, stim = data["states"][:n], data["stimulus"][:n]
    sil = data.get("silence")
    true_s = states[:, K:K + H_re, :, 1].float().cpu()  # [B, H_re, N]

    out = {}
    for k_re in REINJECT_KS:
        preds = []
        for i0 in range(0, n, chunk):
            sl = slice(i0, min(i0 + chunk, n))
            st = states[sl].to(device)
            sm = sil[sl].to(device) if sil is not None else None
            p = rollout_reinject(model, st, stim[sl].to(device), cfg, k_re,
                                 threshold, silence_mask=sm)
            preds.append(p[:, :H_re, :, 1].float().cpu())
        ps = torch.cat(preds)                           # [B, H_re, N]

        # overall F1@H_re: per trajectory, then averaged
        tp = (ps * true_s).sum(dim=(1, 2))
        fp = (ps * (1 - true_s)).sum(dim=(1, 2))
        fn = ((1 - ps) * true_s).sum(dim=(1, 2))
        prec = tp / (tp + fp).clamp(min=1.0)
        rec = tp / (tp + fn).clamp(min=1.0)
        f1 = 2 * prec * rec / (prec + rec).clamp(min=1e-9)
        f1_h = float(_nanmean(f1))

        # steps-since-anchor distance per step (1 = first step after anchor)
        H = ps.shape[1]
        if k_re >= 10_000:
            dist = torch.arange(1, H + 1)
        else:
            dist = (torch.arange(H) % k_re) + 1
        buckets = {}
        for (lo, hi), lbl in zip(BUCKET_RANGES, BUCKET_LABELS):
            idx = [s for s in range(H) if lo <= int(dist[s]) <= hi]
            if not idx:
                buckets[lbl] = None
                continue
            sub_p, sub_t = ps[:, idx], true_s[:, idx]
            tp = float((sub_p * sub_t).sum())
            fp = float((sub_p * (1 - sub_t)).sum())
            fn = float(((1 - sub_p) * sub_t).sum())
            buckets[lbl] = {"f1": 2 * tp / max(2 * tp + fp + fn, 1.0),
                            "n_samples": len(idx) * ps.shape[0]}
        lbl = "inf" if k_re >= 10_000 else str(k_re)
        out[lbl] = {"f1_at_H": f1_h, "buckets": buckets}
        print(f"    reinject K={lbl:>4s}: F1@{H_re}={f1_h:.3f} "
              f"buckets=" + " ".join(
                  f"{l}={buckets[l]['f1']:.3f}" if buckets[l] else f"{l}=—"
                  for l in BUCKET_LABELS))
    return out


# ----------------------------------------------------------------------
# entries / model loading
def _parse_override(spec: str) -> dict:
    """'k=v;k=v' -> {k: typed v} (int/float/bool/str coercion)."""
    out = {}
    for pair in spec.split(";"):
        if not pair.strip():
            continue
        k, _, v = pair.partition("=")
        v = v.strip()
        if v.lower() in ("true", "false"):
            v = v.lower() == "true"
        else:
            try:
                v = int(v)
            except ValueError:
                try:
                    v = float(v)
                except ValueError:
                    pass
        out[k.strip()] = v
    return out


def resolve_entries(args, cfg) -> list[tuple[str, Path, dict]]:
    """Explicit --entry LABEL=PATH[|k=v;k=v] entries (optional model_kwargs
    override for eval-time ablations, e.g. k_hist=1 on the same weights), or
    defaults: phase-1 + every v3r final that exists on disk."""
    entries: list[tuple[str, Path, dict]] = []
    if args.entry:
        for spec in args.entry:
            if "=" not in spec:
                raise SystemExit(f"--entry expects LABEL=PATH, got {spec!r}")
            label, rest = spec.split("=", 1)
            overrides = {}
            if "|" in rest:
                rest, ov = rest.split("|", 1)
                overrides = _parse_override(ov)
            p = Path(rest)
            if not p.exists():
                print(f"[skip] entry {label!r}: checkpoint not found: {p}")
                continue
            entries.append((label.strip(), p, overrides))
    else:
        p1 = phase1_ckpt_path(cfg, args.model)
        if p1.exists():
            entries.append(("phase1", p1, {}))
        else:
            print(f"[skip] default phase-1 entry missing: {p1}")
        for exp in V3R_EXPERIMENTS:
            rc = RolloutConfig(base=cfg, experiment=exp, model=args.model,
                               scale=args.scale, version="v3r",
                               **MATRICES["v3r"][exp])
            p = final_ckpt_path(rc)
            if p.exists():
                entries.append((exp, p, {}))
            else:
                print(f"[skip] v3r/{exp}: no final checkpoint ({p.name})")
    if not entries:
        raise SystemExit("no usable entries after skips: pass --entry "
                         "LABEL=PATH or train the v3r matrix first")
    labels = [l for l, _, _ in entries]
    if len(set(labels)) != len(labels):
        raise SystemExit(f"duplicate entry labels: {labels}")
    return entries


def load_entry_model(path: Path, cfg, conn, device, scale: str,
                     overrides: dict | None = None):
    """Rebuild a model from a checkpoint blob. v3r finals get their exact
    RolloutConfig back (MATRICES['v3r'][exp], version='v3r', like
    run_rollout_v2.py constructs it); every blob's model_kwargs is passed to
    build_model; mechanistic blobs are wrapped via maybe_wrap before the
    state_dict load. Returns (model, blob, rc_or_None)."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    name = blob.get("model")
    if name is None:
        raise SystemExit(f"{path}: checkpoint blob has no 'model' key")
    exp = str(blob.get("experiment", "")).upper()
    version = str(blob.get("version", ""))
    rc = None
    if version == "v3r" and exp in MATRICES["v3r"]:
        rc = RolloutConfig(base=cfg, experiment=exp, model=name, scale=scale,
                           version="v3r", **MATRICES["v3r"][exp])
    if version == "v4" and exp in MATRICES["v4"]:
        rc = RolloutConfig(base=cfg, experiment=exp, model=name, scale=scale,
                           version="v4", **MATRICES["v4"][exp])
    kwargs = dict(blob.get("model_kwargs") or {})
    kwargs.update(overrides or {})          # eval-time ablation overrides
    model = build_model(name, cfg, conn, device, kwargs)
    mechanistic = bool(blob.get("mechanistic", False))
    if mechanistic:
        wrap_rc = rc if rc is not None \
            else RolloutConfig(base=cfg, mechanistic=True)
        model = maybe_wrap(model, cfg, wrap_rc)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    print(f"[load] {path.name} (model={name}, experiment={exp or '?'}, "
          f"version={version or '?'}, mechanistic={mechanistic}, "
          f"epoch={blob.get('epoch', '?')})")
    return model, blob, rc


# ----------------------------------------------------------------------
# tables
def _table1_lines(results: dict, labels: list[str]) -> list[str]:
    """Table 1: rollout spike F1 by horizon + silent-collapse step."""
    lines = ["## Table 1 - rollout spike F1 by horizon (per entry x split)",
             "",
             "| Entry | Split | One-step val F1 | F1@10 | F1@25 | F1@50 | "
             "F1@100 | F1@200 | Silent-collapse step |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for lab in labels:
        e = results[lab]
        one = e.get("onestep_val", {}).get("spike_f1")
        for split in SPLITS:
            cells = [lab, SPLIT_TAGS[split], _fmt_cell(one, False)]
            for target in (10, 25, 50, 100, 200):
                v, capped = _at_horizon(e["rollout"].get(split), target,
                                        "spike_f1")
                cells.append(_fmt_cell(v, capped))
            sc = e["effective_horizon"][split].get("silent_collapse_step")
            cells.append(str(sc) if sc is not None else "none")
            lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("* horizon capped by T-K: value shown is at the longest "
                 "evaluated horizon (* = capped).")
    return lines


def _table2_lines(results: dict, labels: list[str]) -> list[str]:
    """Table 2: effective prediction horizons."""
    lines = ["## Table 2 - effective prediction horizon (per entry x split)",
             "",
             "| Entry | Split | H_F1_0.9 | H_F1_0.7 | H_rate_0.8 | H_state | "
             "rate_ratio@50 | pop corr@50 |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for lab in labels:
        e = results[lab]
        for split in SPLITS:
            eh = e["effective_horizon"][split]
            rr, _ = _at_horizon(e["rollout"].get(split), 50, "rate_ratio")
            pc, _ = _at_horizon(e["rollout"].get(split), 50, "pop_corr")
            cells = [lab, SPLIT_TAGS[split],
                     str(eh["H_F1_0.9"]), str(eh["H_F1_0.7"]),
                     str(eh["H_rate_0.8"]), str(eh["H_state"]),
                     _fmt_cell(rr, False), _fmt_cell(pc, False)]
            lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("H_X = first evaluated horizon crossing the threshold (max "
                 "evaluated horizon if never crossed); rate_ratio = "
                 "(pred+1e-9)/(true+1e-9); pop corr = per-neuron rate "
                 "cosine (pop_similarity).")
    return lines


def _table3_lines(results: dict, labels: list[str], H_re: int) -> list[str]:
    """Table 3: reinjection retest — F1@H_re and steps-since-anchor buckets."""
    lines = [f"## Table 3 - reinjection retest (F1@{H_re} and spike F1 by "
             "steps-since-anchor bucket)", "",
             "| Entry | Split | K | F1@" + str(H_re) + " | F1 1-5 | F1 6-10 | "
             "F1 11-20 | F1 21-50 | F1 51+ |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for lab in labels:
        for split in SPLITS:
            per_k = results[lab].get("reinject", {}).get(split)
            if not per_k:
                continue
            for k in K_LABELS:
                d = per_k.get(k)
                if not d:
                    continue
                cells = [lab, SPLIT_TAGS[split], k,
                         _fmt_cell(d["f1_at_H"], False)]
                for lbl in BUCKET_LABELS:
                    b = d["buckets"].get(lbl)
                    cells.append(_fmt_cell(b["f1"], False) if b else "—")
                lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("K = steps between ground-truth re-anchoring (inf = free "
                 "rollout); bucket F1 pooled over all anchor resets and "
                 "trajectories.")
    return lines


def write_summary_tables(results: dict, labels: list[str], H_re: int | None,
                         path: Path) -> list[str]:
    """Write summary_tables.md (exactly three tables) and return the
    Table-1/Table-2 blocks for the compact console print."""
    t1 = _table1_lines(results, labels)
    t2 = _table2_lines(results, labels)
    lines = ["# Rollout v3 (v3r DAgger+TBPTT matrix) — summary tables", ""]
    lines += t1 + [""] + t2
    if H_re is not None and any(results[l].get("reinject") for l in labels):
        lines += [""] + _table3_lines(results, labels, H_re)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[save] {path}")
    return t1 + [""] + t2


def write_metrics_csv(results: dict, labels: list[str],
                      path: Path) -> None:
    """Flat long-form CSV: entry, split, horizon, metric, value."""
    rows = []
    for lab in labels:
        e = results[lab]
        for k, v in e.get("onestep_val", {}).items():
            rows.append({"entry": lab, "split": "val", "horizon": 1,
                         "metric": k, "value": v})
        for split, hm in e.get("rollout", {}).items():
            for h, m in hm.items():
                for k, v in m.items():
                    rows.append({"entry": lab, "split": split,
                                 "horizon": int(h), "metric": k, "value": v})
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"[save] {path}")


# ----------------------------------------------------------------------
# figures
def _fig_metric_panels(results, labels, colors, metric, ylabel, fname,
                       fig_dir, logy: bool = False, hlines=()):
    """Two side-by-side panels (test_seen / test_ood), one line per entry."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, split in zip(axes, SPLITS):
        plotted = False
        for lab in labels:
            m = results[lab].get("rollout", {}).get(split)
            if not m:
                continue
            hs = sorted(int(h) for h in m)
            pairs = [(h, m[h][metric]) for h in hs
                     if m[h].get(metric) is not None
                     and math.isfinite(m[h][metric])]
            if not pairs:
                continue
            ax.plot([p[0] for p in pairs], [p[1] for p in pairs],
                    marker="o", ms=3, c=colors[lab], label=lab)
            plotted = True
        ax.set_xscale("log")
        ax.set_xlabel("horizon (steps)")
        ax.set_ylabel(ylabel)
        ax.set_title(SPLIT_TAGS[split])
        ax.grid(alpha=0.3)
        if logy:
            ax.set_yscale("log")
        for y in hlines:
            ax.axhline(y, c="r", ls="--", lw=1)
        if plotted:
            ax.legend(fontsize=8)
    fig.suptitle(f"{ylabel} vs horizon")
    fig.tight_layout()
    fig.savefig(fig_dir / fname, dpi=150)
    plt.close(fig)


def _fig_effective_horizon(results, labels, H, fig_dir):
    """fig1: grouped bar of the three effective-horizon definitions."""
    keys = ("H_F1_0.9", "H_F1_0.7", "H_rate_0.8")
    fig, axes = plt.subplots(1, 2, figsize=(max(9.0, 1.5 * len(labels)), 4.5))
    x = list(range(len(labels)))
    w = 0.8 / len(keys)
    for ax, split in zip(axes, SPLITS):
        for j, key in enumerate(keys):
            vals = [results[lab]["effective_horizon"][split][key]
                    for lab in labels]
            ax.bar([xi + (j - (len(keys) - 1) / 2) * w for xi in x], vals, w,
                   label=key)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("effective horizon (steps)")
        ax.set_title(SPLIT_TAGS[split])
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=8)
    fig.suptitle(f"effective prediction horizon (max evaluated h={H})")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig1_effective_horizon_bar.png", dpi=150)
    plt.close(fig)


def _fig_silent_collapse(results, labels, H, fig_dir):
    """fig6: silent-collapse step per entry ('none' -> max_h+10)."""
    fig, axes = plt.subplots(1, 2, figsize=(max(8.0, 1.3 * len(labels)), 4.5))
    for ax, split in zip(axes, SPLITS):
        vals, nones = [], []
        for lab in labels:
            sc = results[lab]["effective_horizon"][split] \
                .get("silent_collapse_step")
            vals.append(sc if sc is not None else H + 10)
            nones.append(sc is None)
        bars = ax.bar(labels, vals, color="tab:red")
        for b, is_none in zip(bars, nones):
            if is_none:
                b.set_alpha(0.35)
                ax.annotate("none", (b.get_x() + b.get_width() / 2,
                                     b.get_height()),
                            ha="center", va="bottom", fontsize=8)
        ax.axhline(H, c="k", ls=":", lw=1)
        ax.set_ylabel("silent-collapse step")
        ax.set_title(SPLIT_TAGS[split])
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle(f"silent collapse (rate ratio < 0.2 for >= 5 steps; faded = "
                 f"none, bar at max h+10; dotted = max h={H})")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig6_silent_collapse.png", dpi=150)
    plt.close(fig)


def _fig_reinjection_by_k(results, labels, colors, H_re, fig_dir):
    """fig7: F1@H_re vs reinjection period K, one panel per split."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    x = list(range(len(K_LABELS)))
    for ax, split in zip(axes, SPLITS):
        plotted = False
        for lab in labels:
            per_k = results[lab].get("reinject", {}).get(split)
            if not per_k:
                continue
            y = [per_k[k]["f1_at_H"] if k in per_k else float("nan")
                 for k in K_LABELS]
            ax.plot(x, y, marker="o", ms=4, c=colors[lab], label=lab)
            plotted = True
        ax.set_xticks(x)
        ax.set_xticklabels(K_LABELS)
        ax.set_xlabel("K (steps between truth re-anchoring)")
        ax.set_ylabel(f"spike F1@{H_re}")
        ax.set_ylim(0, 1.05)
        ax.set_title(SPLIT_TAGS[split])
        ax.grid(alpha=0.3)
        if plotted:
            ax.legend(fontsize=8)
    fig.suptitle("reinjection retest: F1@100 vs reinjection period")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig7_reinjection_f1_by_k.png", dpi=150)
    plt.close(fig)


def _fig_reinjection_distance(results, labels, colors, fig_dir):
    """fig8: F1 vs steps-since-anchor buckets per K; entries E3 and phase-1
    highlighted (fallback: first two entries)."""
    hl = [l for l in ("E3", "phase1") if l in labels] or labels[:2]
    styles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 1))]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    x = list(range(len(BUCKET_LABELS)))
    for ax, split in zip(axes, SPLITS):
        plotted = False
        for lab in hl:
            per_k = results[lab].get("reinject", {}).get(split)
            if not per_k:
                continue
            c = colors[lab] if lab != "phase1" else "#000000"
            for j, k in enumerate(K_LABELS):
                d = per_k.get(k)
                if not d:
                    continue
                y = [d["buckets"][lbl]["f1"]
                     if d["buckets"].get(lbl) else float("nan")
                     for lbl in BUCKET_LABELS]
                ax.plot(x, y, ls=styles[j % len(styles)], marker="o", ms=3,
                        c=c, label=f"{lab} K={k}")
                plotted = True
        ax.set_xticks(x)
        ax.set_xticklabels(BUCKET_LABELS)
        ax.set_xlabel("steps since last ground-truth anchor")
        ax.set_ylabel("spike F1 (pooled)")
        ax.set_ylim(0, 1.05)
        ax.set_title(SPLIT_TAGS[split])
        ax.grid(alpha=0.3)
        if plotted:
            ax.legend(fontsize=7)
    fig.suptitle("reinjection retest: F1 vs steps-since-anchor (E3 vs phase-1)")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig8_reinjection_f1_vs_distance.png", dpi=150)
    plt.close(fig)


def make_figures(results, labels, H, H_re, fig_dir):
    colors = _entry_colors(labels)
    _fig_effective_horizon(results, labels, H, fig_dir)
    _fig_metric_panels(results, labels, colors, "spike_f1", "spike F1",
                       "fig2_f1_vs_horizon.png", fig_dir)
    _fig_metric_panels(results, labels, colors, "rate_ratio", "rate ratio "
                       "(pred+1e-9)/(true+1e-9)",
                       "fig3_rate_ratio_vs_horizon.png", fig_dir,
                       logy=True, hlines=(0.8, 1.25, 0.2))
    _fig_metric_panels(results, labels, colors, "v_rmse", "V RMSE",
                       "fig4_v_rmse_vs_horizon.png", fig_dir)
    _fig_metric_panels(results, labels, colors, "pop_corr", "population rate "
                       "corr (cosine)", "fig5_pop_corr_vs_horizon.png", fig_dir)
    _fig_silent_collapse(results, labels, H, fig_dir)
    if H_re is not None and any(results[l].get("reinject") for l in labels):
        _fig_reinjection_by_k(results, labels, colors, H_re, fig_dir)
        _fig_reinjection_distance(results, labels, colors, fig_dir)
    _fig_metric_panels(results, labels, colors, "state_cosine", "state cosine",
                       "fig9_state_cosine_vs_horizon.png", fig_dir)
    print(f"[plots] -> {fig_dir}")


# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--entry", action="append", default=None,
                        metavar="LABEL=PATH",
                        help="checkpoint entry (repeatable); default: "
                             "phase-1 + existing v3r E0..E3 finals")
    parser.add_argument("--model", default="gnn",
                        choices=["gnn", "connectome", "gnn_temporal",
                                 "gnn_wide"],
                        help="model family for the default phase-1 entry")
    parser.add_argument("--n-traj", type=int, default=None,
                        help="rollout trajectories per split "
                             "(default: cfg.n_rollout_traj)")
    parser.add_argument("--horizons", default="1,2,5,10,20,25,50,100,200",
                        help="comma-separated horizons (capped by T-K)")
    parser.add_argument("--reinject", action="store_true",
                        help="run the reinjection retest (K sweep)")
    parser.add_argument("--out", default=str(ROLLOUT3R_DIR / "eval_unified"),
                        help="output directory")
    args = parser.parse_args()

    cfg = get_config(args.scale)
    if args.seed is not None:
        cfg = replace(cfg, seed=args.seed)
    device = get_device(override=args.device)
    conn = get_connectome(cfg, device)
    sim = LIFSimulator(conn, cfg, device)

    out_dir = Path(args.out)
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    n_traj = args.n_traj or cfg.n_rollout_traj
    req = sorted({int(h) for h in args.horizons.split(",")
                  if h.strip() and int(h) >= 1})
    if not req:
        raise SystemExit("no valid horizons requested")
    H = min(max(req), cfg.T - cfg.K)
    horizons = [h for h in req if h <= H]
    dropped = [h for h in req if h > H]
    if dropped:
        print(f"[setup] horizons > T-K={cfg.T - cfg.K} dropped: {dropped}")
    H_re = min(REINJECT_H, cfg.T - cfg.K) if args.reinject else None

    entries = resolve_entries(args, cfg)
    print(f"[setup] scale={args.scale} N={cfg.n_neurons} T={cfg.T} K={cfg.K} "
          f"H={H} horizons={horizons} n_traj={n_traj} "
          f"entries={[l for l, _, _ in entries]}")

    print("[data] loading fixed eval splits ...")
    data = {split: load_or_generate(split, sim, cfg)
            for split in ("val", "test_seen", "test_ood")}

    results: dict = {}
    labels: list[str] = []
    for label, path, overrides in entries:
        torch.manual_seed(cfg.seed + SEED_OFFSET)   # identical per entry
        model, blob, _rc = load_entry_model(path, cfg, conn, device,
                                            args.scale, overrides)

        # threshold tuned on VAL only, frozen for all test evaluation
        th = tune_threshold(model, data["val"], cfg, device, n_windows=512)
        val_one, _ = onestep_eval(model, data["val"], cfg, device, th)
        print(f"[threshold] {label}: {th:.2f} "
              f"(val one-step F1={val_one['spike_f1']:.3f})")

        entry = {"checkpoint": str(path), "model": blob.get("model"),
                 "mechanistic": bool(blob.get("mechanistic", False)),
                 "experiment": blob.get("experiment"),
                 "version": blob.get("version"),
                 "threshold": th, "onestep_val": val_one,
                 "rollout": {}, "effective_horizon": {}, "attractor": {},
                 "series": {}, "reinject": {}}
        for split in SPLITS:
            m, series, attr = run_entry_rollout(
                model, data[split], cfg, device, H, horizons, th, n_traj)
            entry["rollout"][split] = m
            entry["series"][split] = series
            entry["attractor"][split] = attr
            entry["effective_horizon"][split] = effective_horizons(
                m, series, max(horizons))
            eh = entry["effective_horizon"][split]
            hmax = max(m)
            print(f"[rollout] {label:12s} {SPLIT_TAGS[split]:4s} h={hmax}: "
                  f"f1={m[hmax]['spike_f1']:.3f} "
                  f"v_rmse={m[hmax]['v_rmse']:.3f} "
                  f"pop_corr={m[hmax]['pop_corr']:.3f} "
                  f"H_F1_0.9={eh['H_F1_0.9']} H_rate={eh['H_rate_0.8']} "
                  f"silent={eh['silent_collapse_step']}")
            print(f"[attractor] {label:12s} {SPLIT_TAGS[split]:4s} "
                  f"flags={_flags_str({split: attr})} "
                  f"rate={attr['mean_rate']:.5f}")
        if H_re is not None:
            for split in SPLITS:
                print(f"[reinject] {label:12s} {SPLIT_TAGS[split]:4s}")
                entry["reinject"][split] = run_reinjection(
                    model, data[split], cfg, device, th, n_traj, H_re)
        results[label] = entry
        labels.append(label)
        del model
        _empty_cache(device)

    # ---------------- outputs --------------------------------------------
    payload = {"scale": args.scale, "seed": cfg.seed, "n_neurons": cfg.n_neurons,
               "T": cfg.T, "K": cfg.K, "H": H, "horizons": horizons,
               "n_traj": n_traj, "reinject_H": H_re, "entries": {}}
    for lab in labels:
        e = dict(results[lab])
        e["series"] = {split: _downsample_series(e["series"][split])
                       for split in SPLITS}
        payload["entries"][lab] = e
    json_path = out_dir / "metrics_unified.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(_sanitise(payload), f, indent=2)
    print(f"[save] {json_path}")

    write_metrics_csv(results, labels, out_dir / "metrics_unified.csv")
    t12 = write_summary_tables(results, labels, H_re,
                               out_dir / "summary_tables.md")
    print("\n".join(t12))                    # compact console print
    make_figures(results, labels, H, H_re, fig_dir)


if __name__ == "__main__":
    main()
