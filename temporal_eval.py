"""Phase-3 unified evaluation: GNN vs GNN+Temporal Transformer (spec 十七–
二十四).

    python temporal_eval.py --scale small \
        --entry A=results/checkpoints/ckpt_gnn_small_seed1234.pt \
        --entry C=results/checkpoints/ckpt_gnn_temporal_small_seed1234.pt \
        --entry F=results/checkpoints/ckpt_gnn_temporal_small_rollout_v3_F_seed1234.pt

Per entry the checkpoint blob carries everything needed to rebuild the exact
architecture ("model", "model_kwargs", "K", "mechanistic") — entries with
different window lengths K mix freely. Thresholds are tuned on VAL ONLY
(one-step via evaluate.tune_threshold, rollout via
evaluate.tune_rollout_threshold), then frozen for test_seen/test_ood.

Outputs (default results/rollout_v3/):
  metrics.json / metrics.csv     long-format metrics for every entry/split
  summary_tables.md              the three spec tables (二十三)
  figures/*.png                  the ten spec figures (二十四)

Analyses beyond the plain rollout metrics:
  * --shuffle-entries LABELS     one-step + rollout with the history window
                                 temporally permuted (last step kept fixed);
                                 the "does temporal order matter" control
                                 (spec 十三)
  * --reinject-entries LABELS    ground-truth reinjection K sweep
                                 (1,5,10,20,50,inf; spec 十八) via
                                 reinjection.rollout_reinject
  * --attention-entry LABEL      temporal attention curves for high-rate /
                                 low-rate / near-threshold / hub neurons
                                 (spec 十四; gnn_temporal entries only)
  * per-entry parameter count + analytic FLOPs/step estimate (spec 二十二)
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
from dataset import load_or_generate, make_windows
from device import get_device
from evaluate import tune_threshold, tune_rollout_threshold
from lif import LIFSimulator
from models import build_model, count_params
from models.mechanistic import maybe_wrap
from reinjection import rollout_reinject
from rollout import rollout, naive_rollout, rollout_metrics
from rollout_config import ROLLOUT3_DIR, RolloutConfig
from rollout_eval import (_at_horizon, _fmt_cell, _population_groups,
                          _sanitise, attractor_diagnostics, horizon_metrics,
                          rollout_series)

SPLITS = ("test_seen", "test_ood")
SPLIT_TAGS = {"test_seen": "seen", "test_ood": "ood"}
REINJECT_KS = (1, 5, 10, 20, 50, 10_000)
ROLLOUT_CHUNK = 16


# ----------------------------------------------------------------------
# entry loading
def load_entry(path: Path, cfg, conn, device):
    """Rebuild one entry from its checkpoint blob. Returns
    (model_or_None, entry_cfg, meta dict). model=None selects naive."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    name = blob.get("model")
    if name is None:
        raise SystemExit(f"{path}: checkpoint blob has no 'model' key")
    entry_cfg = replace(cfg, K=int(blob.get("K", cfg.K)))
    kwargs = dict(blob.get("model_kwargs") or {})
    model = build_model(name, entry_cfg, conn, device, kwargs)
    mechanistic = bool(blob.get("mechanistic", False))
    if mechanistic:
        model = maybe_wrap(model, entry_cfg,
                           RolloutConfig(base=entry_cfg, mechanistic=True))
    model.load_state_dict(blob["state_dict"])
    model.eval()
    meta = {"model": name, "model_kwargs": kwargs, "K": entry_cfg.K,
            "mechanistic": mechanistic, "params": count_params(model),
            "epoch": blob.get("epoch", "?")}
    print(f"[load] {path.name}: {name} {kwargs or ''} mech={mechanistic} "
          f"params={meta['params'] / 1e6:.3f}M")
    return model, entry_cfg, meta


# ----------------------------------------------------------------------
# FLOPs estimate (analytic MACs per forward, batch 1; FLOPs = 2 x MACs)
def estimate_macs(meta: dict, cfg_entry, conn) -> int:
    """Rough MACs/step. Linear: in*out per token. Attention block over
    length k per neuron: qkv 3kd^2 + attn 2k^2d + proj kd^2 + ffn 8kd^2.
    MessageRound: msg Nd^2 + contrib/agg 2Ed + update N(2d*2d + 2d*d)."""
    N, E = cfg_entry.n_neurons, conn.n_edges
    d = cfg_entry.d_model
    K = cfg_entry.K
    name = meta["model"]
    if name in ("gnn", "gnn_wide"):
        d = meta["model_kwargs"].get("d_model", cfg_entry.d_model)
        L = meta["model_kwargs"].get("gnn_layers", cfg_entry.gnn_layers)
        dt = cfg_entry.d_temporal
        temporal = N * (4 * dt + cfg_entry.temporal_layers
                        * (12 * K * dt * dt + 2 * K * K * dt))
        spatial = L * (N * d * d + 2 * E * d + 6 * N * d * d) + N * d
        return int(temporal + N * dt * d + spatial + N * d * 3)
    if name == "gnn_temporal":
        k = int(meta["model_kwargs"].get("k_hist", K))
        tL = int(meta["model_kwargs"].get("t_layers", 2))
        spatial_per_step = (N * 4 * d
                            + cfg_entry.gnn_layers
                            * (N * d * d + 2 * E * d + 6 * N * d * d)
                            + N * d)
        temporal = N * tL * (12 * k * d * d + 2 * k * k * d)
        return int(k * spatial_per_step + temporal + N * d * 3)
    return 0


# ----------------------------------------------------------------------
# rollout driver with per-entry cfg and optional history permutation
@torch.no_grad()
def entry_rollout(model, data: dict, cfg_e, device, H: int, horizons,
                  threshold: float, n_traj: int, shuffle_seed: int | None,
                  chunk: int = ROLLOUT_CHUNK):
    """Closed-loop rollout from t0=0 (rollout.py semantics). shuffle_seed
    set: permute history positions 0..K-2 of the initial context (the
    current state stays at the last position) — spec 十三 control."""
    K = cfg_e.K
    n = min(n_traj, data["states"].shape[0])
    states, stim = data["states"][:n], data["stimulus"][:n]
    sil = data.get("silence")
    perm = None
    if shuffle_seed is not None:
        g = torch.Generator().manual_seed(shuffle_seed)
        perm = torch.cat([torch.randperm(K - 1, generator=g),
                          torch.tensor([K - 1])])
    preds = []
    for i0 in range(0, n, chunk):
        sl = slice(i0, min(i0 + chunk, n))
        context = torch.cat([states[sl, :K],
                             stim[sl, :K].unsqueeze(-1)], dim=-1)
        if perm is not None:
            context = context[:, perm]
        context = context.to(device)
        if model is None:
            p = naive_rollout(context, H, cfg_e)
        else:
            fut = stim[sl, K:K + H].to(device)
            sm = sil[sl].to(device) if sil is not None else None
            p = rollout(model, context, fut, cfg_e, spike_threshold=threshold,
                        silence_mask=sm)
        preds.append(p.float().cpu())
    pred = torch.cat(preds)
    true = states[:, K:K + H].float().cpu()

    groups = _population_groups(cfg_e.n_neurons, 10, "cpu")
    metrics = {h: horizon_metrics(pred, true, h, groups)
               for h in horizons if h <= H}
    # dv RMSE: one-step V increments over the rollout window
    for h, m in metrics.items():
        pv, tv = pred[:, :h, :, 0], true[:, :h, :, 0]
        if h >= 2:
            dvp = pv[:, 1:] - pv[:, :-1]
            dvt = tv[:, 1:] - tv[:, :-1]
            m["dv_rmse"] = float(((dvp - dvt) ** 2).mean().sqrt())
        else:
            m["dv_rmse"] = None
    series = rollout_series(pred, true)
    attractor = attractor_diagnostics(pred, H)
    return metrics, series, attractor


@torch.no_grad()
def entry_onestep(model, data, cfg_e, device, threshold: float,
                  n_windows=1024, shuffle_seed: int | None = None) -> dict:
    """One-step metrics (+ dv_mse) on fixed windows; optional history
    permutation (positions 0..K-2, last step fixed)."""
    from metrics import compute_metrics
    g = torch.Generator().manual_seed(888)
    perm = None
    if shuffle_seed is not None:
        gp = torch.Generator().manual_seed(shuffle_seed)
        perm = torch.cat([torch.randperm(cfg_e.K - 1, generator=gp),
                          torch.tensor([cfg_e.K - 1])])
    states, stim = data["states"], data["stimulus"]
    per = max(1, n_windows // states.shape[0])
    agg: dict[str, float] = {}
    count = 0
    for i0 in range(0, states.shape[0], 32):
        sl = slice(i0, min(i0 + 32, states.shape[0]))
        for _ in range(per):
            x, y, _ = make_windows(states[sl], stim[sl], cfg_e.K, generator=g)
            if perm is not None:
                x = x[:, perm]
            x, y = x.to(device), y.to(device)
            out = model(x)
            m = compute_metrics(out, y, auroc=False, threshold=threshold)
            m["dv_mse"] = float(((out["v"] - x[:, -1, :, 0])
                                 - (y[..., 0] - x[:, -1, :, 0]))
                                .pow(2).mean())
            for k, v in m.items():
                agg[k] = agg.get(k, 0.0) + v
            count += 1
    return {k: v / count for k, v in agg.items()}


@torch.no_grad()
def entry_reinject(model, data, cfg_e, device, threshold: float, n_traj: int,
                   horizons):
    """Reinjection sweep (spec 十八): h-max spike F1 per anchor K."""
    n = min(n_traj, data["states"].shape[0])
    states = data["states"][:n].to(device)
    stim = data["stimulus"][:n].to(device)
    sil = data["silence"][:n].to(device) \
        if data.get("silence") is not None else None
    K = cfg_e.K
    true = states[:, K:]
    hs = [h for h in horizons if h <= true.shape[1]]
    per_k = {}
    for k_re in REINJECT_KS:
        pred = rollout_reinject(model, states, stim, cfg_e, k_re, threshold,
                                silence_mask=sil)
        m = rollout_metrics(pred, true, hs)
        lbl = "inf" if k_re >= 10_000 else str(k_re)
        per_k[lbl] = {str(h): mm for h, mm in m.items()}
        print(f"    reinject K={lbl:>4s}: "
              f"h{hs[-1]} f1={m[hs[-1]]['spike_f1']:.3f}")
    return per_k


# ----------------------------------------------------------------------
# temporal attention capture (spec 十四)
@torch.no_grad()
def capture_attention(model, data, cfg_e, device, conn, n_traj: int = 8):
    """Last-block head-averaged attention from the predicting position over
    the history window, for 4 neuron categories: high-rate, low-rate,
    near-threshold, hub. Returns {category: [k] curve}."""
    states, stim = data["states"][:n_traj], data["stimulus"][:n_traj]
    K = cfg_e.K
    t0 = K + 20                                   # inside active regime

    post = states[:, t0:t0 + 50]                     # activity window
    rate = post[..., 1].mean(dim=1).mean(dim=0)      # [N]
    vmean = post[..., 0].mean(dim=1).mean(dim=0)
    deg_out = torch.bincount(conn.edge_index[0].cpu(),
                             minlength=cfg_e.n_neurons).float()
    cats = {
        "high_firing": int(rate.argmax()),
        "low_firing": int(rate.argmin()),
        "near_threshold": int((vmean - cfg_e.v_th).abs().argmin()),
        "hub": int(deg_out.argmax()),
    }
    # mechanistic entries wrap the base model: attention lives in .base
    net = getattr(model, "base", model)
    idx = t0 + torch.arange(K)
    x = torch.cat([states[:, idx], stim[:, idx].unsqueeze(-1)], dim=-1)
    _out, w = net(x.to(device), return_attn=True)    # w [B, N, k]
    w = w.mean(dim=0).cpu()                          # [N, k]
    curves = {}
    for cat, nid in cats.items():
        curves[cat] = {"neuron": nid, "rate": float(rate[nid]),
                       "v_mean": float(vmean[nid]),
                       "deg_out": float(deg_out[nid]),
                       "attn": w[nid].tolist()}
    return curves


# ----------------------------------------------------------------------
# figures
def _colors(labels):
    cmap = plt.get_cmap("tab10")
    out, j = {}, 0
    for lab in labels:
        if lab == "naive":
            out[lab] = "#000000"
        else:
            out[lab] = cmap(j % 10)
            j += 1
    return out


def fig_one_step(results, labels, colors, fig_dir):
    fig, ax = plt.subplots(figsize=(max(6.0, 1.2 * len(labels)), 4))
    x = range(len(labels))
    w = 0.35
    for j, (split, tag) in enumerate((("test_seen", "seen"),
                                      ("test_ood", "OOD"))):
        vals = [results[l].get("onestep", {}).get(split, {})
                .get("spike_f1", float("nan")) for l in labels]
        ax.bar([xi + (j - 0.5) * w for xi in x], vals, w,
               color=[colors[l] for l in labels],
               alpha=0.55 if split == "test_seen" else 1.0, label=tag)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("one-step spike F1")
    ax.set_ylim(0, 1.05)
    ax.set_title("one-step spike F1 (faint=seen, solid=OOD)")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "gnn_vs_temporal_one_step.png", dpi=130)
    plt.close(fig)


def fig_metric_vs_horizon(results, labels, colors, metric, ylabel, fname,
                          fig_dir):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for lab in labels:
        for split, ls, tag in (("test_seen", "-", "seen"),
                               ("test_ood", "--", "OOD")):
            m = results[lab].get("rollout", {}).get(split)
            if not m:
                continue
            hs = sorted(int(h) for h in m)
            vals = [m[h].get(metric) for h in hs]
            pairs = [(h, v) for h, v in zip(hs, vals) if v is not None]
            if not pairs:
                continue
            ax.plot([p[0] for p in pairs], [p[1] for p in pairs], ls=ls,
                    marker="o", ms=3, c=colors[lab], label=f"{lab} ({tag})")
    ax.set_xscale("log")
    ax.set_xlabel("horizon (steps)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} vs horizon (solid=seen, dashed=OOD)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / fname, dpi=130)
    plt.close(fig)


def fig_reinjection(results, labels, colors, fig_dir):
    ks = ["1", "5", "10", "20", "50", "inf"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for lab in labels:
        per_k = results[lab].get("reinject", {}).get("test_ood")
        if not per_k:
            continue
        y = []
        for k in ks:
            m = per_k.get(k)
            hmax = max(int(h) for h in m) if m else None
            y.append(m[str(hmax)]["spike_f1"] if m else float("nan"))
        ax.plot(range(len(ks)), y, marker="o", c=colors[lab], label=lab)
    ax.set_xticks(range(len(ks)), ks)
    ax.set_xlabel("reinjection period K (steps between truth re-anchoring)")
    ax.set_ylabel("spike F1 at max horizon")
    ax.set_title("reinjection comparison (OOD)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "reinjection_comparison.png", dpi=130)
    plt.close(fig)


def fig_history_ablation(results, labels, colors, fig_dir):
    """Entries whose label encodes k (e.g. k1/k4/k8/k16): one-step OOD F1 +
    rollout F1@20/@50 vs k."""
    ks = []
    for lab in labels:
        if lab.startswith("k") and lab[1:].isdigit():
            ks.append((int(lab[1:]), lab))
    if not ks:
        return
    ks.sort()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = range(len(ks))
    w = 0.27
    series = [("one-step", None), ("rollout@20", 20), ("rollout@50", 50)]
    for j, (name, target) in enumerate(series):
        vals = []
        for _, lab in ks:
            if target is None:
                v = results[lab].get("onestep", {}).get("test_ood", {}) \
                    .get("spike_f1")
            else:
                m = results[lab].get("rollout", {}).get("test_ood")
                v, _ = _at_horizon(m, target, "spike_f1")
            vals.append(v if v is not None else float("nan"))
        ax.bar([xi + (j - 1) * w for xi in x], vals, w, label=name)
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"K={k}" for k, _ in ks], fontsize=9)
    ax.set_ylabel("OOD spike F1")
    ax.set_ylim(0, 1.05)
    ax.set_title("history-length ablation (one-step-trained gnn_temporal)")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "history_length_ablation.png", dpi=130)
    plt.close(fig)


def fig_shuffle_ablation(results, labels, colors, fig_dir):
    rows = []
    for lab in labels:
        sh = results[lab].get("shuffle")
        if not sh:
            continue
        for cond, key in (("correct", None), ("shuffled", "shuffled")):
            m1 = results[lab]["onestep"]["test_ood"]["spike_f1"] \
                if cond == "correct" else sh["onestep"]["test_ood"]["spike_f1"]
            if cond == "correct":
                m20, _ = _at_horizon(results[lab]["rollout"]["test_ood"],
                                     20, "spike_f1")
                m50, _ = _at_horizon(results[lab]["rollout"]["test_ood"],
                                     50, "spike_f1")
            else:
                m20, _ = _at_horizon(sh["rollout"]["test_ood"], 20,
                                     "spike_f1")
                m50, _ = _at_horizon(sh["rollout"]["test_ood"], 50,
                                     "spike_f1")
            rows.append((lab, cond, m1, m20 or 0.0, m50 or 0.0))
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(max(6.0, 1.6 * len(rows)), 4))
    x = range(len(rows))
    w = 0.27
    for j, name in enumerate(("one-step", "rollout@20", "rollout@50")):
        ax.bar([xi + (j - 1) * w for xi in x], [r[2 + j] for r in rows], w,
               label=name)
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{r[0]}\n{r[1]}" for r in rows], fontsize=8)
    ax.set_ylabel("OOD spike F1")
    ax.set_title("history-order ablation: correct vs shuffled history")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "shuffled_history_ablation.png", dpi=130)
    plt.close(fig)


def fig_attention(attn: dict, label: str, fig_dir):
    if not attn:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for cat, d in attn.items():
        k = len(d["attn"])
        lags = list(range(-k + 1, 1))
        ax.plot(lags, d["attn"], marker="o", ms=3,
                label=f"{cat} (n={d['neuron']}, rate={d['rate']:.3f})")
    ax.set_xlabel("lag (0 = current step)")
    ax.set_ylabel("attention weight")
    ax.set_title(f"temporal attention from the predicting position — {label}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "temporal_attention_examples.png", dpi=130)
    plt.close(fig)


def fig_param_matched(results, labels, colors, fig_dir):
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for lab in labels:
        meta = results[lab].get("meta", {})
        p = meta.get("params")
        one = results[lab].get("onestep", {}).get("test_ood", {}) \
            .get("spike_f1")
        m50, _ = _at_horizon(results[lab].get("rollout", {}).get("test_ood"),
                             50, "spike_f1")
        if p is None:
            continue
        if one is not None:
            ax.scatter([p / 1e6], [one], marker="o", c=[colors[lab]], s=50)
            ax.annotate(lab, (p / 1e6, one), fontsize=8,
                        textcoords="offset points", xytext=(5, 4))
        if m50 is not None:
            ax.scatter([p / 1e6], [m50], marker="s", c=[colors[lab]], s=50)
            ax.annotate(lab, (p / 1e6, m50), fontsize=8,
                        textcoords="offset points", xytext=(5, -10))
    ax.scatter([], [], marker="o", c="#777777", label="one-step OOD F1")
    ax.scatter([], [], marker="s", c="#777777", label="rollout OOD F1@50")
    ax.set_xlabel("parameters (M)")
    ax.set_ylabel("OOD spike F1")
    ax.set_title("parameter-matched comparison")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "parameter_matched_comparison.png", dpi=130)
    plt.close(fig)


def fig_tradeoff(results, labels, colors, H, fig_dir):
    fig, ax = plt.subplots(figsize=(7, 5))
    plotted = False
    for lab in labels:
        if lab == "naive":
            continue
        one = results[lab].get("onestep", {}).get("test_ood", {}) \
            .get("spike_f1")
        ood = results[lab].get("rollout", {}).get("test_ood")
        f50, c50 = _at_horizon(ood, 50, "spike_f1")
        pc, c200 = _at_horizon(ood, 200, "pop_rate_corr")
        if one is None:
            continue
        if f50 is not None:
            ax.scatter([one], [f50], marker="o", c=[colors[lab]], s=50)
            ax.annotate(f"{lab}{'*' if c50 else ''}", (one, f50), fontsize=8,
                        textcoords="offset points", xytext=(5, 4))
        if pc is not None:
            ax.scatter([one], [pc], marker="s", c=[colors[lab]], s=50)
            ax.annotate(f"{lab}{'*' if c200 else ''}", (one, pc), fontsize=8,
                        textcoords="offset points", xytext=(5, -10))
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


# ----------------------------------------------------------------------
# tables (spec 二十三)
def write_tables(results, labels, H, path: Path) -> None:
    lines = ["# Phase-3 — GNN vs GNN+Temporal Transformer", ""]
    lines.append("| Model | Params | FLOPs/step (est) | OOD one-step F1 | "
                 "h10 | h20 | h50 | h100 | h200 | PopCorr@200 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for lab in labels:
        e = results[lab]
        meta = e.get("meta", {})
        p = meta.get("params")
        fl = meta.get("macs")
        one = e.get("onestep", {}).get("test_ood", {}).get("spike_f1")
        ood = e.get("rollout", {}).get("test_ood")
        cells = [lab,
                 f"{p / 1e6:.3f}M" if p else "—",
                 f"{2 * fl / 1e9:.2f}G" if fl else "—",
                 _fmt_cell(one, False)]
        for target, key in ((10, "spike_f1"), (20, "spike_f1"),
                            (50, "spike_f1"), (100, "spike_f1"),
                            (200, "spike_f1"), (200, "pop_rate_corr")):
            v, capped = _at_horizon(ood, target, key)
            cells.append(_fmt_cell(v, capped))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("* horizon capped by T-K: value shown is at the longest "
                 "evaluated horizon.")
    lines.append("")
    lines.append("| Model | K=5 reinject | K=10 | K=20 | K=50 | Autonomous |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for lab in labels:
        per_k = results[lab].get("reinject", {}).get("test_ood")
        if not per_k:
            continue
        cells = [lab]
        for k in ("5", "10", "20", "50", "inf"):
            m = per_k.get(k)
            if m:
                hmax = max(int(h) for h in m)
                cells.append(_fmt_cell(m[str(hmax)]["spike_f1"], False))
            else:
                cells.append("—")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("| History | OOD F1 | Rollout@20 | Rollout@50 |")
    lines.append("|---|---:|---:|---:|")
    for lab in labels:
        e = results[lab]
        if "shuffle" not in e:
            continue
        one = e.get("onestep", {}).get("test_ood", {}).get("spike_f1")
        ood = e.get("rollout", {}).get("test_ood")
        r20, _ = _at_horizon(ood, 20, "spike_f1")
        r50, _ = _at_horizon(ood, 50, "spike_f1")
        lines.append("| " + " | ".join(
            [f"{lab} correct", _fmt_cell(one, False), _fmt_cell(r20, False),
             _fmt_cell(r50, False)]) + " |")
        sh = e["shuffle"]
        one_s = sh["onestep"]["test_ood"]["spike_f1"]
        r20s, _ = _at_horizon(sh["rollout"]["test_ood"], 20, "spike_f1")
        r50s, _ = _at_horizon(sh["rollout"]["test_ood"], 50, "spike_f1")
        lines.append("| " + " | ".join(
            [f"{lab} shuffled", _fmt_cell(one_s, False),
             _fmt_cell(r20s, False), _fmt_cell(r50s, False)]) + " |")
    for lab in labels:
        if lab == "k1":
            e = results[lab]
            one = e.get("onestep", {}).get("test_ood", {}).get("spike_f1")
            ood = e.get("rollout", {}).get("test_ood")
            r20, _ = _at_horizon(ood, 20, "spike_f1")
            r50, _ = _at_horizon(ood, 50, "spike_f1")
            lines.append("| " + " | ".join(
                ["K=1 (last-state-only)", _fmt_cell(one, False),
                 _fmt_cell(r20, False), _fmt_cell(r50, False)]) + " |")
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[save] {path}")


def write_csv(results, labels, H, path: Path) -> None:
    rows = []
    for lab in labels:
        e = results[lab]
        for split, m in e.get("onestep", {}).items():
            for k, v in m.items():
                rows.append({"entry": lab, "eval": "onestep", "split": split,
                             "horizon": 1, "metric": k, "value": v})
        for split, hm in e.get("rollout", {}).items():
            for h, m in hm.items():
                for k, v in m.items():
                    rows.append({"entry": lab, "eval": "rollout",
                                 "split": split, "horizon": int(h),
                                 "metric": k, "value": v})
        for split, per_k in e.get("reinject", {}).items():
            for k, hm in per_k.items():
                for h, m in hm.items():
                    rows.append({"entry": lab, "eval": f"reinject_K{k}",
                                 "split": split, "horizon": int(h),
                                 "metric": "spike_f1",
                                 "value": m["spike_f1"]})
        sh = e.get("shuffle")
        if sh:
            for k, v in sh["onestep"]["test_ood"].items():
                rows.append({"entry": lab, "eval": "shuffle_onestep",
                             "split": "test_ood", "horizon": 1,
                             "metric": k, "value": v})
            for h, m in sh["rollout"]["test_ood"].items():
                rows.append({"entry": lab, "eval": "shuffle_rollout",
                             "split": "test_ood", "horizon": int(h),
                             "metric": "spike_f1", "value": m["spike_f1"]})
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"[save] {path}")


# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--entry", action="append", default=None,
                        metavar="LABEL=PATH", help="checkpoint entry "
                        "(repeatable); labels k1/k4/... feed the history "
                        "ablation figure")
    parser.add_argument("--n-traj", type=int, default=None)
    parser.add_argument("--horizons", default="1,2,5,10,20,25,50,100,200")
    parser.add_argument("--shuffle-entries", nargs="*", default=None,
                        help="labels to also evaluate with shuffled history "
                             "(default: all gnn_temporal entries)")
    parser.add_argument("--reinject-entries", nargs="*", default=None,
                        help="labels for the reinjection sweep")
    parser.add_argument("--attention-entry", default=None,
                        help="label for temporal attention capture")
    parser.add_argument("--reinject-traj", type=int, default=16)
    parser.add_argument("--out-dir", default=str(ROLLOUT3_DIR))
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

    if not args.entry:
        raise SystemExit("no entries: pass --entry LABEL=PATH (repeatable)")
    entries = []
    for spec in args.entry:
        if "=" not in spec:
            raise SystemExit(f"--entry expects LABEL=PATH, got {spec!r}")
        label, path = spec.split("=", 1)
        p = Path(path)
        if not p.exists():
            raise SystemExit(f"entry {label!r}: not found: {p}")
        entries.append((label.strip(), p))
    labels = [l for l, _ in entries]
    if len(set(labels)) != len(labels):
        raise SystemExit(f"duplicate labels: {labels}")

    req = sorted({int(h) for h in args.horizons.split(",")
                  if h.strip() and int(h) >= 1})
    n_traj = args.n_traj or cfg.n_rollout_traj

    print("[data] loading fixed eval splits ...")
    data = {split: load_or_generate(split, sim, cfg)
            for split in ("val", "test_seen", "test_ood")}

    results: dict = {}
    for label, path in entries:
        model, cfg_e, meta = load_entry(path, cfg, conn, device)
        meta["macs"] = estimate_macs(meta, cfg_e, conn)
        H = min(max(req), cfg_e.T - cfg_e.K)
        horizons = [h for h in req if h <= H]
        th_one = tune_threshold(model, data["val"], cfg_e, device,
                                n_windows=512)
        th_roll = tune_rollout_threshold(model, data["val"], cfg_e)
        print(f"[threshold] {label}: onestep={th_one:.2f} "
              f"rollout={th_roll:.2f}")
        entry = {"meta": meta,
                 "thresholds": {"onestep": th_one, "rollout": th_roll},
                 "onestep": {}, "rollout": {}, "attractor": {},
                 "series": {}}
        for split in SPLITS:
            entry["onestep"][split] = entry_onestep(
                model, data[split], cfg_e, device, th_one)
            r = entry["onestep"][split]
            print(f"[onestep] {label:12s} {split:9s} f1={r['spike_f1']:.3f} "
                  f"v_rmse={r['v_rmse']:.4f} dv_mse={r['dv_mse']:.5f}")
        for split in SPLITS:
            m, series, attr = entry_rollout(
                model, data[split], cfg_e, device, H, horizons, th_roll,
                n_traj, shuffle_seed=None)
            entry["rollout"][split] = m
            entry["series"][split] = series
            entry["attractor"][split] = attr
            hmax = max(m)
            print(f"[rollout] {label:12s} {SPLIT_TAGS[split]:4s} h={hmax}: "
                  f"f1={m[hmax]['spike_f1']:.3f} "
                  f"v_rmse={m[hmax]['v_rmse']:.3f} "
                  f"pop_corr={m[hmax]['pop_rate_corr']:.3f}")

        shuffle_for = args.shuffle_entries
        if shuffle_for is None:
            shuffle_for = [label] if meta["model"] == "gnn_temporal" else []
        if label in shuffle_for:
            sh = {"onestep": {}, "rollout": {}}
            sh["onestep"]["test_ood"] = entry_onestep(
                model, data["test_ood"], cfg_e, device, th_one,
                shuffle_seed=999)
            m, _, _ = entry_rollout(model, data["test_ood"], cfg_e, device,
                                    H, horizons, th_roll, n_traj,
                                    shuffle_seed=999)
            sh["rollout"]["test_ood"] = m
            entry["shuffle"] = sh
            print(f"[shuffle] {label}: onestep f1="
                  f"{sh['onestep']['test_ood']['spike_f1']:.3f} "
                  f"(was {entry['onestep']['test_ood']['spike_f1']:.3f})")

        if args.reinject_entries and label in args.reinject_entries:
            entry["reinject"] = {"test_ood": entry_reinject(
                model, data["test_ood"], cfg_e, device, th_roll,
                args.reinject_traj, req)}
        if args.attention_entry == label and meta["model"] == "gnn_temporal":
            entry["attention"] = capture_attention(
                model, data["test_ood"], cfg_e, device, conn)
        results[label] = entry
        del model
        if device.type == "xpu":
            torch.xpu.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()

    # naive reference
    H = min(max(req), cfg.T - cfg.K)
    horizons = [h for h in req if h <= H]
    naive = {"meta": {"model": "naive", "params": 0, "macs": 0},
             "onestep": {}, "rollout": {}, "attractor": {}, "series": {}}
    for split in SPLITS:
        m, series, attr = entry_rollout(None, data[split], cfg, device, H,
                                        horizons, 0.5, n_traj, None)
        naive["rollout"][split] = m
        naive["series"][split] = series
        naive["attractor"][split] = attr
    results["naive"] = naive
    labels_all = labels + ["naive"]

    payload = {"scale": args.scale, "seed": cfg.seed,
               "n_neurons": cfg.n_neurons, "T": cfg.T,
               "horizons": req, "n_traj": n_traj,
               "entries": {l: results[l] for l in labels_all}}
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(_sanitise(payload), f, indent=2)
    print(f"[save] {out_dir / 'metrics.json'}")
    write_csv(results, labels_all, H, out_dir / "metrics.csv")
    write_tables(results, labels_all, H, out_dir / "summary_tables.md")

    colors = _colors(labels_all)
    fig_one_step(results, labels_all, colors, fig_dir)
    fig_metric_vs_horizon(results, labels_all, colors, "spike_f1",
                          "spike F1", "gnn_vs_temporal_rollout_f1.png",
                          fig_dir)
    fig_metric_vs_horizon(results, labels_all, colors, "v_rmse", "V RMSE",
                          "gnn_vs_temporal_v_rmse.png", fig_dir)
    fig_metric_vs_horizon(results, labels_all, colors, "pop_rate_corr",
                          "population rate corr",
                          "rollout_population_corr.png", fig_dir)
    fig_reinjection(results, labels_all, colors, fig_dir)
    fig_history_ablation(results, labels_all, colors, fig_dir)
    fig_shuffle_ablation(results, labels_all, colors, fig_dir)
    if args.attention_entry and args.attention_entry in results:
        fig_attention(results[args.attention_entry].get("attention"),
                      args.attention_entry, fig_dir)
    fig_param_matched(results, labels_all, colors, fig_dir)
    fig_tradeoff(results, labels_all, colors, H, fig_dir)
    print(f"[plots] -> {fig_dir}")


if __name__ == "__main__":
    main()
