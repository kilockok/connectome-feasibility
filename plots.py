"""Generate all result figures into results/figures/.

Reads: results/metrics_{scale}.json, results/history_*_{scale}.csv,
       results/cache/eval_examples_{scale}.pt
Called automatically at the end of evaluate.py, or standalone:
    python plots.py [--scale small|full]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from config import FIGURES_DIR, RESULTS_DIR, add_common_args

MODEL_ORDER = ["gru", "transformer", "connectome", "gnn", "naive"]
COLORS = {"gru": "#7f7f7f", "transformer": "#1f77b4",
          "connectome": "#d62728", "gnn": "#2ca02c", "naive": "#000000"}
LABELS = {"gru": "GRU", "transformer": "Vanilla TF",
          "connectome": "Connectome TF", "gnn": "GNN", "naive": "Naive"}


def _models_in(results) -> list[str]:
    return [m for m in MODEL_ORDER if m in results["models"]
            or m in results.get("perturbation", {})]


def fig_training_curves(scale):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    plotted = False
    for name in MODEL_ORDER[:-1]:
        p = RESULTS_DIR / f"history_{name}_{scale}.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        plotted = True
        axes[0].plot(df["epoch"], df["train_loss"], c=COLORS[name], ls="--",
                     alpha=0.5)
        axes[0].plot(df["epoch"], df["val_loss"], c=COLORS[name],
                     label=LABELS[name])
        axes[1].plot(df["epoch"], df["val_v_mse"], c=COLORS[name])
        axes[2].plot(df["epoch"], df["val_spike_f1"], c=COLORS[name])
    if not plotted:
        return
    naive_v = df["val_naive_v_mse"].iloc[-1]
    axes[1].axhline(naive_v, c="k", ls=":", label="naive")
    axes[0].set_title("loss (solid=val, dashed=train)")
    axes[1].set_title("val V MSE")
    axes[2].set_title("val spike F1")
    for ax in axes:
        ax.set_xlabel("epoch"); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"fig1_training_curves_{scale}.png", dpi=130)
    plt.close(fig)


def fig_onestep_scatter(scale, examples):
    sc = examples["scatter"]
    if not sc:
        return
    names = [n for n in MODEL_ORDER if n in sc]
    fig, axes = plt.subplots(1, len(names), figsize=(4.5 * len(names), 4.2),
                             sharex=True, sharey=True)
    if len(names) == 1:
        axes = [axes]
    for ax, name in zip(axes, names):
        d = sc[name]
        ax.scatter(d["v_true"][::5], d["v_pred"][::5], s=1, alpha=0.15)
        lim = [float(d["v_true"].min()), float(d["v_true"].max())]
        ax.plot(lim, lim, "r-", lw=1)
        ax.set_title(f"{LABELS[name]} (OOD)")
        ax.set_xlabel("true V"); ax.set_ylabel("pred V")
    fig.suptitle("one-step membrane potential prediction (unseen stimulus)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"fig2_onestep_scatter_{scale}.png", dpi=130)
    plt.close(fig)


def fig_spike_f1(scale, results):
    rows = []
    for name in results["models"]:
        m = results["models"][name].get("onestep", {}).get("test_ood")
        if m:
            rows.append((name, m["spike_precision"], m["spike_recall"],
                         m["spike_f1"], m.get("spike_auroc", np.nan)))
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(rows)); w = 0.25
    for i, (lab, j) in enumerate((("precision", 1), ("recall", 2), ("F1", 3))):
        ax.bar(x + (i - 1) * w, [r[j] for r in rows], w, label=lab)
    for i, r in enumerate(rows):
        ax.text(i + w, r[3] + 0.01, f"AUROC\n{r[4]:.3f}", ha="center",
                fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[r[0]] for r in rows])
    ax.set_ylim(0, 1.0); ax.legend()
    ax.set_title("spike prediction on unseen-stimulus (OOD) test")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"fig3_spike_f1_{scale}.png", dpi=130)
    plt.close(fig)


def _get_rollout(results, name, key):
    m = results["models"].get(name, {})
    if key in ("seen", "ood"):
        return m.get("rollout", {}).get(key) or m.get(f"rollout_{key}")
    return None


def fig_rollout_error(scale, results):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for col, key, ttl in ((0, "seen", "seen stimulus"),
                          (1, "ood", "unseen stimulus (OOD)")):
        for name in _models_in(results):
            m = _get_rollout(results, name, key)
            if not m:
                continue
            hs = sorted(int(h) for h in m)
            axes[0, col].plot(hs, [m[str(h)]["v_rmse"] for h in hs],
                              marker="o", c=COLORS[name], label=LABELS[name])
            axes[1, col].plot(hs, [m[str(h)]["spike_f1"] for h in hs],
                              marker="o", c=COLORS[name], label=LABELS[name])
        axes[0, col].set_title(f"V RMSE vs horizon — {ttl}")
        axes[1, col].set_title(f"spike F1 vs horizon — {ttl}")
        axes[1, col].set_xlabel("rollout steps")
    for ax in axes.flat:
        ax.set_xscale("log"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"fig4_rollout_error_{scale}.png", dpi=130)
    plt.close(fig)


def fig_seen_vs_unseen(scale, results):
    names = [n for n in MODEL_ORDER[:-1]
             if "onestep" in results["models"].get(n, {})]
    if not names:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    w = 0.35
    x = np.arange(len(names))
    for j, (split, lab) in enumerate((("test_seen", "seen"),
                                      ("test_ood", "OOD"))):
        vm = [results["models"][n]["onestep"][split]["v_mse"] for n in names]
        f1 = [results["models"][n]["onestep"][split]["spike_f1"] for n in names]
        axes[0].bar(x + (j - 0.5) * w, vm, w, label=lab)
        axes[1].bar(x + (j - 0.5) * w, f1, w, label=lab)
    axes[0].set_title("one-step V MSE"); axes[1].set_title("one-step spike F1")
    for ax in axes:
        ax.set_xticks(x); ax.set_xticklabels([LABELS[n] for n in names])
        ax.legend()
    fig.suptitle("seen vs unseen stimulus generalization")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"fig5_seen_vs_unseen_{scale}.png", dpi=130)
    plt.close(fig)


def fig_trajectory_compare(scale, examples, n_show=100):
    ro = examples["rollout"]
    if "naive" not in ro:
        return
    true = ro["naive"]["true"][0, :n_show]              # [h, N, 3]
    names = [n for n in MODEL_ORDER if n in ro]
    fig, axes = plt.subplots(len(names), 2, figsize=(13, 2.6 * len(names)))
    if len(names) == 1:
        axes = axes[None, :]
    for row, name in enumerate(names):
        pred = ro[name]["pred"][0, :n_show]
        axes[row, 0].imshow(true[..., 1].T, aspect="auto", cmap="gray_r",
                            interpolation="nearest")
        axes[row, 0].set_ylabel(f"GT\nneuron")
        axes[row, 0].set_title(f"ground truth spikes" if row == 0 else "")
        axes[row, 1].imshow(pred[..., 1].T, aspect="auto", cmap="gray_r",
                            interpolation="nearest")
        axes[row, 1].set_title(f"{LABELS[name]} rollout spikes"
                               if row == 0 else "")
        axes[row, 1].set_ylabel(LABELS[name])
    axes[0, 0].set_title("ground truth spikes (OOD trajectory)")
    for ax in axes[-1]:
        ax.set_xlabel("rollout step")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"fig6_trajectory_compare_{scale}.png", dpi=130)
    plt.close(fig)


def fig_population_rate(scale, examples, n_show=200):
    ro = examples["rollout"]
    if "naive" not in ro:
        return
    true = ro["naive"]["true"][0, :n_show, :, 1].mean(dim=1)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(true, c="k", lw=2, label="ground truth")
    for name in MODEL_ORDER:
        if name in ro and name != "naive":
            pred = ro[name]["pred"][0, :n_show, :, 1].mean(dim=1)
            ax.plot(pred, c=COLORS[name], alpha=0.8, label=LABELS[name])
    ax.set_xlabel("rollout step"); ax.set_ylabel("population firing rate")
    ax.set_title("population firing rate over time (OOD rollout)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"fig7_population_rate_{scale}.png", dpi=130)
    plt.close(fig)


def fig_lesion_response(scale, results, examples):
    pe = examples["perturbation"]
    pm = results.get("perturbation", {})
    if not pm:
        return
    names = [n for n in MODEL_ORDER[:-1] if n in pm]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    # left: response MSE bars (silencing), with zero reference
    kinds = [k for k in ("silence", "lesion") if any(k in pm.get(n, {}) for n in pm)]
    width = 0.8 / max(len(names) + 1, 1)
    x = np.arange(len(kinds))
    for j, name in enumerate(names + ["zero_reference"]):
        vals = [pm.get(name, {}).get(k, {}).get("resp_mse_v", np.nan)
                for k in kinds]
        axes[0].bar(x + (j - len(names) / 2) * width, vals, width,
                    label=LABELS.get(name, name),
                    color=COLORS.get(name, "#9467bd"))
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(kinds)
    axes[0].set_title("perturbation response MSE (V) — lower=better")
    axes[0].legend(fontsize=8)
    # right: spike-count delta scatter, GT vs model (silencing example)
    for name in names:
        if name not in pe:
            continue
        d = pe[name]
        axes[1].scatter(d["gt_count_delta"], d["md_count_delta"], s=8,
                        alpha=0.6, c=COLORS[name], label=LABELS[name])
    if pe:
        allgt = torch.cat([pe[n]["gt_count_delta"] for n in pe])
        lim = float(allgt.abs().max()) + 1e-6
        axes[1].plot([-lim, lim], [-lim, lim], "k-", lw=1)
    axes[1].set_xlabel("GT spike-count delta")
    axes[1].set_ylabel("model spike-count delta")
    axes[1].set_title(f"silencing response per neuron (x={list(pe.values())[0]['silence_x'] if pe else '?'})")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"fig8_lesion_response_{scale}.png", dpi=130)
    plt.close(fig)


def make_all(scale):
    metrics_path = RESULTS_DIR / f"metrics_{scale}.json"
    if not metrics_path.exists():
        print(f"[plots] {metrics_path} not found, skipping")
        return
    with open(metrics_path) as f:
        results = json.load(f)
    ex_path = RESULTS_DIR / "cache" / f"eval_examples_{scale}.pt"
    examples = torch.load(ex_path, map_location="cpu",
                          weights_only=False) if ex_path.exists() else {}
    examples.setdefault("scatter", {}); examples.setdefault("rollout", {})
    examples.setdefault("perturbation", {})

    fig_training_curves(scale)
    fig_onestep_scatter(scale, examples)
    fig_spike_f1(scale, results)
    fig_rollout_error(scale, results)
    fig_seen_vs_unseen(scale, results)
    fig_trajectory_compare(scale, examples)
    fig_population_rate(scale, examples)
    fig_lesion_response(scale, results, examples)
    print(f"[plots] figures -> {FIGURES_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    args = parser.parse_args()
    make_all(args.scale)
