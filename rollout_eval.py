"""Unified phase-2 rollout evaluation (see PHASE2.md, section rollout_eval.py).

Evaluates phase-1 and phase-2 checkpoints under one protocol:
  * thresholds tuned on VAL only (one-step + rollout), frozen for test splits
  * one-step eval on test_seen / test_ood (calibrated threshold)
  * closed-loop rollout from t0=0 over H = min(max horizon, T-K) steps on
    test_seen / test_ood, metrics at every requested horizon
  * naive (x[t+1]=x[t]) baseline entry always appended
  * attractor diagnostics over the tail of each rollout

Aggregation convention (consistent across ALL rollout metrics): every metric
is computed PER TRAJECTORY first and then averaged with equal weight over
trajectories (this includes spike F1 — it is the mean of per-trajectory F1,
not a pooled-count F1).

CLI:
    python rollout_eval.py --scale full \
        --entry phase1=results/checkpoints/ckpt_gnn_full_seed1234.pt \
        --entry G=results/checkpoints/ckpt_gnn_full_rollout_v2_G_seed1234.pt
    python rollout_eval.py --scale full --model gnn --experiments A B C
    python rollout_eval.py --scale small \
        --entry phase1=results/checkpoints/ckpt_gnn_small_seed1234.pt \
        --n-traj 8 --horizons 1,10,25 --out-dir results/rollout_v2_evaltest

Writes into --out-dir (default results/rollout_v2): metrics.json,
metrics.csv, summary_tables.md and figures/*.png. Existing
lif_sensitivity_*.png / reinjection_vs_error.png are never touched.
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
import torch.nn.functional as F

from config import get_config, add_common_args
from connectome import get_connectome
from dataset import load_or_generate
from device import get_device
from evaluate import tune_threshold, tune_rollout_threshold, onestep_eval
from lif import LIFSimulator
from models import build_model
from rollout import rollout, naive_rollout
from rollout_config import (ROLLOUT2_DIR, EXPERIMENTS, RolloutConfig,
                            phase1_ckpt_path, final_ckpt_path)

SPLITS = ("test_seen", "test_ood")
SPLIT_TAGS = {"test_seen": "seen", "test_ood": "ood"}
N_POP_GROUPS = 10            # regional activity correlation groups
N_RATE_BINS = 20             # rate-histogram distance bins
ROLLOUT_CHUNK = 16           # trajectories per rollout batch (memory guard)
ATTRACTOR_FLAGS = ("dead", "saturated", "collapsed_variance", "periodic",
                   "fixed_subset")


# ----------------------------------------------------------------------
# small numeric helpers
def _nanmean(x: torch.Tensor, dim: int | None = None) -> torch.Tensor:
    """Mean over finite entries; NaN where no finite entry exists."""
    finite = torch.isfinite(x)
    if dim is None:
        if not finite.any():
            return torch.tensor(float("nan"))
        return x[finite].mean()
    cnt = finite.sum(dim)
    s = torch.where(finite, x, torch.zeros_like(x)).sum(dim)
    return torch.where(cnt > 0, s / cnt.clamp(min=1),
                       torch.full_like(s, float("nan")))


@torch.no_grad()
def _pearson_rows(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Pearson correlation along dim=1 per row; NaN on zero-variance rows."""
    a = a - a.mean(dim=1, keepdim=True)
    b = b - b.mean(dim=1, keepdim=True)
    den = a.norm(dim=1) * b.norm(dim=1)
    num = (a * b).sum(dim=1)
    out = num / den.clamp(min=1e-12)
    return torch.where(den > 1e-12, out, torch.full_like(out, float("nan")))


def _population_groups(n_neurons: int, n_groups: int, device) -> torch.Tensor:
    """[G, N] 0/1 membership of contiguous chunks. Uses losses.population_groups
    when available (phase-2 module), otherwise a local equivalent."""
    try:
        from losses import population_groups
        return population_groups(n_neurons, n_groups, device)
    except ImportError:
        g = torch.zeros(n_groups, n_neurons, device=device)
        for i in range(n_groups):
            g[i, i * n_neurons // n_groups:(i + 1) * n_neurons // n_groups] = 1.0
        return g


def _sanitise(o):
    """Recursively make a structure JSON-safe: NaN/inf -> None."""
    if isinstance(o, dict):
        return {str(k): _sanitise(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_sanitise(v) for v in o]
    if isinstance(o, bool) or o is None or isinstance(o, (int, str)):
        return o
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if torch.is_tensor(o):
        return _sanitise(o.tolist() if o.numel() > 1 else o.item())
    return str(o)


# ----------------------------------------------------------------------
# metric computation (pred/true [B, H, N, 3] CPU tensors)
@torch.no_grad()
def horizon_metrics(pred: torch.Tensor, true: torch.Tensor, h: int,
                    groups: torch.Tensor | None) -> dict:
    """Rollout metrics over the window [0, h). Per trajectory, then averaged."""
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

    rp, rt = ps.mean(dim=1), ts.mean(dim=1)              # [B, N] per-neuron rates
    pop_corr = _pearson_rows(rp, rt)
    pop_mae = (rp - rt).abs().mean(dim=1)

    ap, at = ps.sum(dim=1) > 0, ts.sum(dim=1) > 0
    jaccard = ((ap & at).sum(dim=1).float()
               / (ap | at).sum(dim=1).clamp(min=1).float())

    state_cos = F.cosine_similarity(p[:, h - 1].reshape(B, -1),
                                    t[:, h - 1].reshape(B, -1), dim=1)

    if groups is not None and h >= 2:
        G = groups.shape[0]
        gsize = groups.sum(dim=1).clamp(min=1.0)                 # [G]
        gp = torch.einsum("gn,bhn->bhg", groups, ps) / gsize     # [B, h, G]
        gt = torch.einsum("gn,bhn->bhg", groups, ts) / gsize
        rc = _pearson_rows(gp.transpose(1, 2).reshape(B * G, h),
                           gt.transpose(1, 2).reshape(B * G, h))
        regional = _nanmean(rc.reshape(B, G), dim=1)
    else:
        regional = torch.full((B,), float("nan"))

    hp = torch.stack([torch.histc(rp[b], bins=N_RATE_BINS, min=0.0, max=1.0)
                      for b in range(B)])
    ht = torch.stack([torch.histc(rt[b], bins=N_RATE_BINS, min=0.0, max=1.0)
                      for b in range(B)])
    hp = hp / hp.sum(dim=1, keepdim=True).clamp(min=1.0)
    ht = ht / ht.sum(dim=1, keepdim=True).clamp(min=1.0)
    hist_l1 = (hp - ht).abs().sum(dim=1)

    def m(x):
        return float(_nanmean(x))

    return {"spike_f1": m(f1), "spike_precision": m(prec), "spike_recall": m(rec),
            "v_rmse": m(v_rmse), "pop_rate_corr": m(pop_corr),
            "pop_rate_mae": m(pop_mae), "active_jaccard": m(jaccard),
            "state_cosine": m(state_cos), "regional_corr": m(regional),
            "rate_hist_l1": m(hist_l1),
            "spike_rate_pred": float(ps.mean()), "spike_rate_true": float(ts.mean())}


@torch.no_grad()
def rollout_series(pred: torch.Tensor, true: torch.Tensor) -> dict:
    """Per-timestep series (t = 1..H), averaged over trajectories, plus the
    per-trajectory predicted rate(t) kept for attractor diagnostics."""
    ps, ts = pred[..., 1], true[..., 1]
    pv, tv = pred[..., 0], true[..., 0]
    rate_p, rate_t = ps.mean(dim=2), ts.mean(dim=2)        # [B, H]
    act_p = (ps > 0.5).sum(dim=2).float()
    act_t = (ts > 0.5).sum(dim=2).float()
    return {"rate_pred": rate_p.mean(dim=0).tolist(),
            "rate_true": rate_t.mean(dim=0).tolist(),
            "active_pred": act_p.mean(dim=0).tolist(),
            "active_true": act_t.mean(dim=0).tolist(),
            "vvar_pred": pv.var(dim=2).mean(dim=0).tolist(),
            "vvar_true": tv.var(dim=2).mean(dim=0).tolist(),
            "rate_pred_traj": rate_p.tolist()}


@torch.no_grad()
def attractor_diagnostics(pred: torch.Tensor, H: int) -> dict:
    """Trivial-attractor flags over the tail (last 50 steps, or last H//2 if
    H < 100) of the predicted rollout, using the trajectory-averaged mean
    population rate series. Booleans plus the raw numbers."""
    W = min(50 if H >= 100 else max(H // 2, 2), H)
    ps = pred[:, H - W:, :, 1]                              # [B, W, N]
    rate = ps.mean(dim=(0, 2))                              # [W]
    mean_rate = float(rate.mean())
    rate_var = float(rate.var(unbiased=False)) if W > 1 else 0.0

    rc = rate - rate.mean()
    denom = float((rc * rc).sum())
    max_ac, max_lag = float("nan"), 0
    if denom > 1e-12 and W >= 3:                            # zero-variance guard
        best = -1.0
        for lag in range(2, min(50, W - 1) + 1):
            ac = float((rc[:-lag] * rc[lag:]).sum()) / denom
            if ac > best:
                best, max_lag = ac, lag
        max_ac = best

    counts = ps.sum(dim=(0, 1))                             # [N]
    total = float(counts.sum())
    k = max(1, counts.numel() // 20)
    share = (float(counts.sort(descending=True).values[:k].sum()) / total
             if total > 0 else 0.0)

    return {"window": W, "mean_rate": mean_rate, "rate_var": rate_var,
            "max_autocorr": max_ac, "argmax_lag": max_lag,
            "top5pct_spike_share": share,
            "dead": bool(mean_rate < 1e-4),
            "saturated": bool(mean_rate > 0.5),
            "collapsed_variance": bool(rate_var < 1e-8),
            "periodic": bool(math.isfinite(max_ac) and max_ac > 0.9),
            "fixed_subset": bool(share > 0.95)}


# ----------------------------------------------------------------------
# rollout driver — trajectory chunks keep peak memory bounded
@torch.no_grad()
def run_entry_rollout(model, data: dict, cfg, device, H: int, horizons,
                      threshold: float, n_traj: int,
                      chunk: int = ROLLOUT_CHUNK):
    """Closed-loop rollout from t0=0 (hard reset, silence masks from the
    data). model=None selects the naive baseline. Returns
    (metrics per horizon, time series, attractor diagnostics)."""
    K = cfg.K
    n = min(n_traj, data["states"].shape[0])
    states, stim = data["states"][:n], data["stimulus"][:n]
    sil = data.get("silence")
    preds = []
    for i0 in range(0, n, chunk):
        sl = slice(i0, min(i0 + chunk, n))
        context = torch.cat([states[sl, :K],
                             stim[sl, :K].unsqueeze(-1)], dim=-1).to(device)
        if model is None:
            p = naive_rollout(context, H, cfg)
        else:
            fut = stim[sl, K:K + H].to(device)
            sm = sil[sl].to(device) if sil is not None else None
            p = rollout(model, context, fut, cfg, spike_threshold=threshold,
                        silence_mask=sm)
        preds.append(p.float().cpu())
    pred = torch.cat(preds)
    true = states[:, K:K + H].float().cpu()

    groups = _population_groups(cfg.n_neurons, N_POP_GROUPS, "cpu")
    metrics = {h: horizon_metrics(pred, true, h, groups)
               for h in horizons if h <= H}
    series = rollout_series(pred, true)
    attractor = attractor_diagnostics(pred, H)
    return metrics, series, attractor


# ----------------------------------------------------------------------
# entries / model loading
def resolve_entries(args, cfg) -> list[tuple[str, Path]]:
    """(--entry LABEL=PATH) and/or (--model M --experiments A B C -> naming
    convention: A = phase-1 checkpoint, B..G = rollout_config.final_ckpt_path).
    """
    entries: list[tuple[str, Path]] = []
    for spec in args.entry or []:
        if "=" not in spec:
            raise SystemExit(f"--entry expects LABEL=PATH, got {spec!r}")
        label, path = spec.split("=", 1)
        entries.append((label.strip(), Path(path)))
    for exp in (args.experiments or []):
        exp = exp.upper()
        if exp not in EXPERIMENTS:
            raise SystemExit(f"unknown experiment {exp!r}")
        if exp == "A":
            path = phase1_ckpt_path(cfg, args.model)
        else:
            rc = RolloutConfig(base=cfg, experiment=exp, model=args.model,
                               scale=args.scale, **EXPERIMENTS[exp])
            path = final_ckpt_path(rc)
        entries.append((exp, path))
    if not entries:
        raise SystemExit("no entries: pass --entry LABEL=PATH or "
                         "--model M --experiments A B C")
    labels = [l for l, _ in entries]
    if len(set(labels)) != len(labels):
        raise SystemExit(f"duplicate entry labels: {labels}")
    for label, path in entries:
        if not path.exists():
            raise SystemExit(f"entry {label!r}: checkpoint not found: {path}")
    return entries


def load_entry_model(path: Path, cfg, conn, device):
    """Rebuild a model from a checkpoint blob. blob['model'] names the
    architecture; blob['mechanistic'] (absent in phase-1) requests the
    MechanisticWrapper via models/mechanistic.py (imported lazily)."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    name = blob.get("model")
    if name is None:
        raise SystemExit(f"{path}: checkpoint blob has no 'model' key")
    mechanistic = bool(blob.get("mechanistic", False))
    model = build_model(name, cfg, conn, device)
    if mechanistic:
        try:
            from models.mechanistic import maybe_wrap
        except ImportError as e:
            raise SystemExit(
                f"{path}: mechanistic=True but models/mechanistic.py is not "
                f"available ({e}); cannot rebuild this entry")
        rc = RolloutConfig(base=cfg, mechanistic=True)
        model = maybe_wrap(model, cfg, rc)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    print(f"[load] {path.name} (model={name}, mechanistic={mechanistic}, "
          f"epoch={blob.get('epoch', '?')})")
    return model, blob


def _empty_cache(device):
    if device.type == "xpu":
        torch.xpu.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


# ----------------------------------------------------------------------
# table / figure helpers
def _at_horizon(m: dict | None, target: int, key: str):
    """(value, capped) at exactly `target`, else at the longest evaluated
    horizon below it (capped=True), else (None, False)."""
    if not m:
        return None, False
    hs = sorted(int(h) for h in m)
    if target in hs:
        return m[target][key], False
    lower = [h for h in hs if h < target]
    if lower:
        return m[lower[-1]][key], True
    return None, False


def _fmt_cell(v, capped: bool) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "—"
    return f"{v:.3f}" + ("*" if capped else "")


def _flags_str(attractor: dict) -> str:
    flags = []
    for split in SPLITS:
        a = attractor.get(split, {})
        for f in ATTRACTOR_FLAGS:
            if a.get(f):
                flags.append(f"{f}({SPLIT_TAGS[split]})")
    return ", ".join(flags) if flags else "no"


def _entry_colors(labels):
    cmap = plt.get_cmap("tab10")
    colors, j = {}, 0
    for lab in labels:
        if lab == "naive":
            colors[lab] = "#000000"
        else:
            colors[lab] = cmap(j % 10)
            j += 1
    return colors


def _downsample_series(series: dict, cap: int = 256) -> dict:
    H = len(series["rate_pred"])
    stride = max(1, math.ceil(H / cap))
    idx = list(range(0, H, stride))
    out = {"t": [i + 1 for i in idx]}
    for k, v in series.items():
        if k == "rate_pred_traj":
            out[k] = [[row[i] for i in idx] for row in v]
        else:
            out[k] = [v[i] for i in idx]
    return out


# ----------------------------------------------------------------------
def write_summary_tables(results: dict, labels: list[str], H: int,
                         path: Path) -> None:
    lines = ["# Rollout v2 — summary tables", ""]
    lines.append("| Method | One-step OOD F1 | h10 F1 | h50 F1 | h100 F1 | "
                 "h200 F1 | PopCorr@200 |")
    lines.append("|---|---|---|---|---|---|---|")
    for lab in labels:
        e = results[lab]
        one = e.get("onestep", {}).get("test_ood", {}).get("spike_f1")
        ood = e.get("rollout", {}).get("test_ood")
        cells = [lab, _fmt_cell(one, False)]
        for target, key in ((10, "spike_f1"), (50, "spike_f1"),
                            (100, "spike_f1"), (200, "spike_f1"),
                            (200, "pop_rate_corr")):
            v, capped = _at_horizon(ood, target, key)
            cells.append(_fmt_cell(v, capped))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("* horizon capped by T-K: value shown is at the longest "
                 "evaluated horizon.")
    lines.append("")
    lines.append("| Method | Seen rollout | OOD rollout | Trivial attractor? |")
    lines.append("|---|---|---|---|")
    for lab in labels:
        e = results[lab]
        vs, _ = _at_horizon(e.get("rollout", {}).get("test_seen"), H, "spike_f1")
        vo, _ = _at_horizon(e.get("rollout", {}).get("test_ood"), H, "spike_f1")
        lines.append("| " + " | ".join(
            [lab, _fmt_cell(vs, False), _fmt_cell(vo, False),
             _flags_str(e.get("attractor", {}))]) + " |")
    lines.append("")
    lines.append("Seen/OOD rollout = spike F1 at the longest evaluated "
                 f"horizon (h={H}).")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[save] {path}")


def write_metrics_csv(results: dict, labels: list[str], H: int,
                      path: Path) -> None:
    rows = []
    for lab in labels:
        e = results[lab]
        for split, m in e.get("onestep", {}).items():
            for k, v in m.items():
                rows.append({"entry": lab, "eval": "onestep", "split": split,
                             "horizon": 1, "metric": k, "value": v})
        for split, horizons_m in e.get("rollout", {}).items():
            for h, m in horizons_m.items():
                for k, v in m.items():
                    rows.append({"entry": lab, "eval": "rollout",
                                 "split": split, "horizon": int(h),
                                 "metric": k, "value": v})
        for split, a in e.get("attractor", {}).items():
            for k, v in a.items():
                rows.append({"entry": lab, "eval": "attractor", "split": split,
                             "horizon": H, "metric": k,
                             "value": float(v) if isinstance(v, bool) else v})
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"[save] {path}")


# ----------------------------------------------------------------------
# figures
def _fig_metric_vs_horizon(results, labels, colors, metric, ylabel, fname,
                           fig_dir):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for lab in labels:
        for split, ls, tag in (("test_seen", "-", "seen"),
                               ("test_ood", "--", "OOD")):
            m = results[lab].get("rollout", {}).get(split)
            if not m:
                continue
            hs = sorted(int(h) for h in m)
            ax.plot(hs, [m[h][metric] for h in hs], ls=ls, marker="o", ms=3,
                    c=colors[lab], label=f"{lab} ({tag})")
    ax.set_xscale("log")
    ax.set_xlabel("horizon (steps)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} vs horizon (solid=seen, dashed=OOD)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / fname, dpi=130)
    plt.close(fig)


def _fig_attractor(results, labels, colors, H, fig_dir):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    t = list(range(1, H + 1))
    ref = results[labels[0]]["series"]

    for col, split, ttl in ((0, "test_seen", "seen"), (1, "test_ood", "OOD")):
        ax = axes[0, col]
        for lab in labels:
            ax.plot(t, results[lab]["series"][split]["rate_pred"],
                    c=colors[lab], lw=1.0, label=lab)
        ax.plot(t, ref[split]["rate_true"], c="#555555", lw=2.0, ls=":",
                label="true")
        ax.set_title(f"population spike rate(t) — {ttl}")
        ax.set_xlabel("rollout step")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    ax = axes[0, 2]
    for lab in labels:
        ax.plot(t, results[lab]["series"]["test_ood"]["vvar_pred"],
                c=colors[lab], lw=1.0, label=lab)
    ax.plot(t, ref["test_ood"]["vvar_true"], c="#555555", lw=2.0, ls=":",
            label="true")
    ax.set_title("V variance across neurons(t) — OOD")
    ax.set_xlabel("rollout step")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    for lab in labels:
        ax.plot(t, results[lab]["series"]["test_ood"]["active_pred"],
                c=colors[lab], lw=1.0, label=lab)
    ax.plot(t, ref["test_ood"]["active_true"], c="#555555", lw=2.0, ls=":",
            label="true")
    ax.set_title("active neuron count(t) — OOD")
    ax.set_xlabel("rollout step")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    x = range(len(labels))
    w = 0.35
    for j, (split, tag) in enumerate((("test_seen", "seen"),
                                      ("test_ood", "ood"))):
        vals = []
        for lab in labels:
            v = results[lab]["attractor"][split].get("max_autocorr")
            vals.append(v if v is not None and math.isfinite(v) else 0.0)
        ax.bar([xi + (j - 0.5) * w for xi in x], vals, w,
               color=[colors[l] for l in labels],
               alpha=0.55 if split == "test_seen" else 1.0, label=tag)
    ax.axhline(0.9, c="r", ls="--", lw=1)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_title("max autocorr of rate(t), lags 2..50 (faint=seen)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1, 2]
    ax.axis("off")
    txt = ["triggered attractor flags", ""]
    for lab in labels:
        for split in SPLITS:
            a = results[lab]["attractor"][split]
            flags = [f for f in ATTRACTOR_FLAGS if a.get(f)] or ["none"]
            txt.append(f"{lab} [{SPLIT_TAGS[split]}]: {', '.join(flags)}")
    ax.text(0.0, 1.0, "\n".join(txt), va="top", fontsize=9,
            family="monospace")

    fig.suptitle("attractor diagnostics")
    fig.tight_layout()
    fig.savefig(fig_dir / "attractor_diagnostics.png", dpi=130)
    plt.close(fig)


def _fig_tradeoff(results, labels, colors, fig_dir):
    fig, ax = plt.subplots(figsize=(7, 5))
    plotted = False
    for lab in labels:
        if lab == "naive":
            continue                                  # no one-step threshold
        one = results[lab].get("onestep", {}).get("test_ood", {}).get("spike_f1")
        if one is None:
            continue
        ood = results[lab].get("rollout", {}).get("test_ood")
        f1_50, cap50 = _at_horizon(ood, 50, "spike_f1")
        pc_200, cap200 = _at_horizon(ood, 200, "pop_rate_corr")
        if f1_50 is not None:
            ax.scatter([one], [f1_50], marker="o", c=[colors[lab]], s=50,
                       zorder=3)
            ax.annotate(f"{lab}{'*' if cap50 else ''}", (one, f1_50),
                        fontsize=8, textcoords="offset points", xytext=(5, 4))
        if pc_200 is not None:
            ax.scatter([one], [pc_200], marker="s", c=[colors[lab]], s=50,
                       zorder=3)
            ax.annotate(f"{lab}{'*' if cap200 else ''}", (one, pc_200),
                        fontsize=8, textcoords="offset points", xytext=(5, -10))
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.scatter([], [], marker="o", c="#777777", label="rollout OOD F1@50")
    ax.scatter([], [], marker="s", c="#777777", label="pop_rate_corr@200")
    ax.set_xlabel("one-step OOD spike F1")
    ax.set_ylabel("rollout metric")
    ax.set_title("one-step vs rollout tradeoff (* = horizon capped)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "one_step_vs_rollout_tradeoff.png", dpi=130)
    plt.close(fig)


def _fig_ablation(results, labels, colors, H, horizons, fig_dir):
    groups = []
    for target in (10, 50):
        if target in horizons:
            groups.append((f"h{target}", target))
    groups.append((f"h{H} (longest)", H))
    fig, ax = plt.subplots(figsize=(max(6.0, 1.4 * len(labels)), 4))
    x = range(len(labels))
    w = 0.8 / len(groups)
    for gi, (gname, target) in enumerate(groups):
        vals = []
        for lab in labels:
            m = results[lab].get("rollout", {}).get("test_ood")
            v, _ = _at_horizon(m, target, "spike_f1")
            vals.append(v if v is not None else float("nan"))
        ax.bar([xi + (gi - (len(groups) - 1) / 2) * w for xi in x], vals, w,
               label=gname)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("OOD spike F1")
    ax.set_title("method ablation: OOD spike F1 by horizon")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "method_ablation.png", dpi=130)
    plt.close(fig)


def make_figures(results, labels, H, horizons, fig_dir):
    colors = _entry_colors(labels)
    _fig_metric_vs_horizon(results, labels, colors, "spike_f1", "spike F1",
                           "rollout_spike_f1.png", fig_dir)
    _fig_metric_vs_horizon(results, labels, colors, "v_rmse", "V RMSE",
                           "rollout_v_rmse.png", fig_dir)
    _fig_metric_vs_horizon(results, labels, colors, "pop_rate_corr",
                           "population rate corr", "rollout_population_corr.png",
                           fig_dir)
    _fig_metric_vs_horizon(results, labels, colors, "pop_rate_mae",
                           "population rate MAE", "rollout_rate_mae.png",
                           fig_dir)
    _fig_attractor(results, labels, colors, H, fig_dir)
    _fig_tradeoff(results, labels, colors, fig_dir)
    _fig_ablation(results, labels, colors, H, horizons, fig_dir)
    print(f"[plots] -> {fig_dir}")


# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--entry", action="append", default=None,
                        metavar="LABEL=PATH",
                        help="checkpoint entry (repeatable)")
    parser.add_argument("--model", default="gnn", choices=["gnn", "connectome"],
                        help="model family for --experiments lookup")
    parser.add_argument("--experiments", nargs="*", default=None,
                        help="A..G; A = phase-1 ckpt, B..G = phase-2 finals")
    parser.add_argument("--n-traj", type=int, default=None,
                        help="rollout trajectories per split "
                             "(default: cfg.n_rollout_traj)")
    parser.add_argument("--horizons", default="1,2,5,10,20,25,50,100,200",
                        help="comma-separated horizons (capped by T-K)")
    parser.add_argument("--out-dir", default=str(ROLLOUT2_DIR))
    args = parser.parse_args()

    cfg = get_config(args.scale)
    if args.seed is not None:
        cfg = replace(cfg, seed=args.seed)
    torch.manual_seed(cfg.seed)
    device = get_device(override=args.device)
    conn = get_connectome(cfg, device)
    sim = LIFSimulator(conn, cfg, device)

    out_dir = Path(args.out_dir)
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

    entries = resolve_entries(args, cfg)
    print(f"[setup] scale={args.scale} N={cfg.n_neurons} T={cfg.T} K={cfg.K} "
          f"H={H} horizons={horizons} n_traj={n_traj} "
          f"entries={[l for l, _ in entries] + ['naive']}")

    print("[data] loading fixed eval splits ...")
    data = {split: load_or_generate(split, sim, cfg)
            for split in ("val", "test_seen", "test_ood")}

    results: dict = {}
    labels: list[str] = []
    for label, path in entries:
        model, blob = load_entry_model(path, cfg, conn, device)
        th_one = tune_threshold(model, data["val"], cfg, device, n_windows=512)
        print(f"[threshold/onestep] {label}: {th_one:.2f}")
        th_roll = tune_rollout_threshold(model, data["val"], cfg)
        print(f"[threshold/rollout] {label}: {th_roll:.2f}")

        entry = {"checkpoint": str(path), "model": blob.get("model"),
                 "mechanistic": bool(blob.get("mechanistic", False)),
                 "thresholds": {"onestep": th_one, "rollout": th_roll},
                 "onestep": {}, "rollout": {}, "attractor": {}, "series": {}}
        for split in SPLITS:
            r, _ = onestep_eval(model, data[split], cfg, device, th_one)
            entry["onestep"][split] = r
            print(f"[onestep] {label:12s} {split:9s} "
                  f"f1={r['spike_f1']:.3f} v_rmse={r['v_rmse']:.4f}")
        for split in SPLITS:
            m, series, attr = run_entry_rollout(
                model, data[split], cfg, device, H, horizons, th_roll, n_traj)
            entry["rollout"][split] = m
            entry["series"][split] = series
            entry["attractor"][split] = attr
            hmax = max(m)
            print(f"[rollout] {label:12s} {SPLIT_TAGS[split]:4s} h={hmax}: "
                  f"f1={m[hmax]['spike_f1']:.3f} "
                  f"v_rmse={m[hmax]['v_rmse']:.3f} "
                  f"pop_corr={m[hmax]['pop_rate_corr']:.3f}")
            print(f"[attractor] {label:12s} {SPLIT_TAGS[split]:4s} "
                  f"flags={_flags_str({split: attr})} "
                  f"rate={attr['mean_rate']:.5f} "
                  f"max_ac={attr['max_autocorr']:.3f}")
        results[label] = entry
        labels.append(label)
        del model
        _empty_cache(device)

    # ---------------- naive baseline -------------------------------------
    entry = {"checkpoint": None, "model": "naive", "mechanistic": False,
             "thresholds": None, "onestep": {}, "rollout": {},
             "attractor": {}, "series": {}}
    for split in SPLITS:
        m, series, attr = run_entry_rollout(
            None, data[split], cfg, device, H, horizons, 0.5, n_traj)
        entry["rollout"][split] = m
        entry["series"][split] = series
        entry["attractor"][split] = attr
        hmax = max(m)
        print(f"[rollout] {'naive':12s} {SPLIT_TAGS[split]:4s} h={hmax}: "
              f"f1={m[hmax]['spike_f1']:.3f} v_rmse={m[hmax]['v_rmse']:.3f} "
              f"pop_corr={m[hmax]['pop_rate_corr']:.3f}")
    results["naive"] = entry
    labels.append("naive")

    # ---------------- outputs --------------------------------------------
    payload = {"scale": args.scale, "seed": cfg.seed, "n_neurons": cfg.n_neurons,
               "T": cfg.T, "K": cfg.K, "H": H, "horizons": horizons,
               "n_traj": n_traj, "entries": {}}
    for lab in labels:
        e = dict(results[lab])
        e["series"] = {split: _downsample_series(e["series"][split])
                       for split in SPLITS}
        payload["entries"][lab] = e
    json_path = out_dir / "metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(_sanitise(payload), f, indent=2)
    print(f"[save] {json_path}")

    write_metrics_csv(results, labels, H, out_dir / "metrics.csv")
    write_summary_tables(results, labels, H, out_dir / "summary_tables.md")
    make_figures(results, labels, H, horizons, fig_dir)


if __name__ == "__main__":
    main()
