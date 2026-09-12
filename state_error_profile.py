"""Rollout-v4 STEP 1: state error profiling (spec §2-4, §10-11).

NO TRAINING. Loads the phase-1 checkpoint and rolls it autonomously on fixed
val / test_seen trajectories, recording the per-neuron membrane error

    deltaV_i(t) = V_pred_i(t) - V_true_i(t)

per rollout step. Produces quantile tables (all neurons, threshold-near
subsets at |V_true-theta| in {1e-3, 3e-3, 1e-2}, spike-disagreement neurons),
error-growth curves with teacher-tolerance crossings (H_err_*), silent
collapse onset (spec §10) and recommended tangent sigmas (spec §11).

Outputs (results/rollout_v4/profiling/):
  profile.json            all statistics + recommended sigmas
  table_error_growth.md   the two spec tables
  fig01_state_error_growth.png
  fig02_error_vs_teacher_tolerance.png
  fig08_tangent_sigma_distribution.png

    python state_error_profile.py --scale full [--n-traj 64] [--H 50]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from config import get_config, add_common_args, RESULTS_DIR
from connectome import get_connectome
from dataset import load_or_generate
from device import get_device
from evaluate import tune_threshold
from lif import LIFSimulator
from models import build_model
from rollout import rollout

OUT_DIR = RESULTS_DIR / "rollout_v4" / "profiling"
STEPS = (1, 2, 3, 5, 8, 10, 12, 15, 20, 25, 50)
TOL_LINES = (1e-3, 3e-3, 1e-2)          # teacher tolerance bands (§4)
ERR_LEVELS = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2)
THRESH_NEAR = (1e-3, 3e-3, 1e-2)        # threshold-near bands (§2)
MIN_ACTIVITY = 1e-4                     # spec §10 minimum activity gate


def _quantiles(x: torch.Tensor, q=(0.5, 0.75, 0.9, 0.95, 0.99)):
    """Quantiles of a flat float tensor -> {q: value}."""
    x = x.flatten().float()
    out = {}
    for qq in q:
        out[f"p{int(qq * 100)}"] = float(torch.quantile(x, qq).item())
    return out


@torch.no_grad()
def profile_split(model, data, cfg, device, threshold, n_traj, H, chunk=16):
    """Autonomous rollout + full deltaV statistics for one split.

    Returns a dict with per-step stats (1-indexed steps up to H)."""
    K = cfg.K
    n = min(n_traj, data["states"].shape[0])
    states, stim = data["states"][:n], data["stimulus"][:n]
    sil = data.get("silence")

    v_pred = torch.empty(n, H, cfg.n_neurons)
    v_true = torch.empty(n, H, cfg.n_neurons)
    s_pred = torch.empty(n, H, cfg.n_neurons)
    s_true = torch.empty(n, H, cfg.n_neurons)
    for i0 in range(0, n, chunk):
        sl = slice(i0, min(i0 + chunk, n))
        context = torch.cat([states[sl, :K],
                             stim[sl, :K].unsqueeze(-1)], dim=-1).to(device)
        fut = stim[sl, K:K + H].to(device)
        sm = sil[sl].to(device) if sil is not None else None
        p = rollout(model, context, fut, cfg, spike_threshold=threshold,
                    silence_mask=sm)
        v_pred[i0:i0 + p.shape[0]] = p[:, :H, :, 0].cpu()
        s_pred[i0:i0 + p.shape[0]] = p[:, :H, :, 1].cpu()
        v_true[i0:i0 + p.shape[0]] = states[sl, K:K + H, :, 0].cpu()
        s_true[i0:i0 + p.shape[0]] = states[sl, K:K + H, :, 1].cpu()

    dV = v_pred - v_true                               # [B,H,N]
    absdV = dV.abs()
    steps_out = {}
    for t in range(1, H + 1):
        a = absdV[:, t - 1]                            # [B,N]
        vt = v_true[:, t - 1]
        st, sp = s_true[:, t - 1], s_pred[:, t - 1]
        row = {"rmse": float(dV[:, t - 1].pow(2).mean().sqrt().item()),
               "mean_abs": float(a.mean().item()),
               "max_abs": float(a.max().item()),
               **_quantiles(a)}
        # threshold-near subsets
        for eps in THRESH_NEAR:
            m = (vt - cfg.v_th).abs() < eps
            row[f"near{eps:g}_frac"] = float(m.float().mean().item())
            if bool(m.any()):
                q = _quantiles(a[m], (0.5, 0.9))
                row[f"near{eps:g}_p50"] = q["p50"]
                row[f"near{eps:g}_p90"] = q["p90"]
            else:
                row[f"near{eps:g}_p50"] = None
                row[f"near{eps:g}_p90"] = None
        # spike disagreement
        mism = (sp > 0.5) != (st > 0.5)
        row["spike_mismatch_frac"] = float(mism.float().mean().item())
        if bool(mism.any()):
            q = _quantiles(a[mism], (0.5, 0.9))
            row["mismatch_p50"] = q["p50"]
            row["mismatch_p90"] = q["p90"]
        else:
            row["mismatch_p50"] = None
            row["mismatch_p90"] = None
        # firing-rate ratio (population, per trajectory then mean); the
        # mean is taken over ACTIVE trajectories only (r_true > gate) so
        # quiescent steps do not explode the ratio
        rp = sp.mean(dim=1)                            # [B]
        rt = st.mean(dim=1)
        act = rt > MIN_ACTIVITY
        ratio_t = (rp + 1e-9) / (rt + 1e-9)
        row["rate_pred"] = float(rp.mean().item())
        row["rate_true"] = float(rt.mean().item())
        row["rate_ratio_mean"] = float(
            ratio_t[act].mean().item()) if bool(act.any()) else None
        row["n_active"] = int(act.sum().item())
        steps_out[t] = row

    # silent collapse onset (spec §10): r_true > MIN_ACTIVITY and
    # rate_ratio < 0.25 sustained >= 3 steps; also first < 0.5.
    rp = s_pred.mean(dim=2)                            # [B,H]
    rt = s_true.mean(dim=2)
    ratio = (rp + 1e-9) / (rt + 1e-9)
    active = rt.mean(dim=0) > MIN_ACTIVITY             # [H] population gate
    low25 = (ratio < 0.25).float()
    run = torch.zeros(ratio.shape[0])
    sustained = torch.zeros_like(low25)
    for t in range(low25.shape[1]):
        run = torch.where(low25[:, t] > 0, run + 1.0,
                          torch.zeros_like(run))
        sustained[:, t] = run >= 3.0
    h_silent = None
    if bool(sustained.any()) and bool(active.any()):
        idx = sustained.float().argmax(dim=1)
        aff = sustained.any(dim=1)
        h_silent = float(idx[aff].float().mean().item()) + 1.0
    low50 = (ratio < 0.5) & active.unsqueeze(0)
    h_rate50 = float(low50.float().argmax(dim=1).float().mean().item()) + 1.0 \
        if bool(low50.any()) else None
    low25a = (ratio < 0.25) & active.unsqueeze(0)
    h_rate25 = float(low25a.float().argmax(dim=1).float().mean().item()) + 1.0 \
        if bool(low25a.any()) else None
    return {"steps": {str(k): v for k, v in steps_out.items()},
            "H_silent": h_silent,
            "H_rate50": h_rate50, "H_rate25": h_rate25}


def recommend_sigmas(profile, steps=(5, 10, 15)):
    """Spec §11: sigma per reference step from the p75 of |deltaV|,
    clamped to [1e-4, 1e-2]."""
    out = {}
    for name, s in zip(("sigma_small", "sigma_mid", "sigma_large"), steps):
        row = profile["steps"].get(str(s)) or profile["steps"].get(s)
        if row is None:
            continue
        v = row.get("p75") or row.get("p50")
        if v is not None:
            out[name] = min(1e-2, max(1e-4, round(v, 9)))
    return out


def error_crossings(profile, stat="p90"):
    """First step where `stat` (or RMSE) exceeds each error level."""
    out = {}
    for lvl in ERR_LEVELS:
        for st_ in (stat, "rmse"):
            h = None
            for t in sorted(int(k) for k in profile["steps"]):
                v = profile["steps"][str(t)].get(st_)
                if v is not None and v > lvl:
                    h = t
                    break
            out[f"H_err_{lvl:g}_{st_}"] = h
    # near-threshold crossings (the decision-relevant band)
    for band in ("near0.001_p90", "near0.003_p90"):
        for lvl in ERR_LEVELS:
            h = None
            for t in sorted(int(k) for k in profile["steps"]):
                v = profile["steps"][str(t)].get(band)
                if v is not None and v > lvl:
                    h = t
                    break
            out[f"H_err_{lvl:g}_{band}"] = h
    return out


def write_tables(profiles, path):
    lines = ["# State error growth (phase-1, autonomous rollout)", ""]
    for split, prof in profiles.items():
        lines += [f"## split = {split}", "",
                  "| step | median | p75 | p90 | p95 | p99 | RMSE |",
                  "| ---: | -----: | --: | --: | --: | --: | ---: |"]
        for t in sorted(int(k) for k in prof["steps"]):
            r = prof["steps"][str(t)]
            fmt = lambda v: f"{v:.2e}" if v is not None else "—"
            lines.append(f"| {t} | {fmt(r['p50'])} | {fmt(r['p75'])} | "
                         f"{fmt(r['p90'])} | {fmt(r['p95'])} | "
                         f"{fmt(r['p99'])} | {fmt(r['rmse'])} |")
        lines += ["",
                  "| step | near1e-3 p90 | near3e-3 p90 | near1e-2 p90 | "
                  "spike mismatch frac | firing rate ratio |",
                  "| ---: | -----------: | -----------: | -----------: | "
                  "------------------: | ----------------: |"]
        for t in sorted(int(k) for k in prof["steps"]):
            r = prof["steps"][str(t)]
            fmt = lambda v: f"{v:.2e}" if v is not None else "—"
            lines.append(f"| {t} | {fmt(r['near0.001_p90'])} | "
                         f"{fmt(r['near0.003_p90'])} | "
                         f"{fmt(r['near0.01_p90'])} | "
                         f"{r['spike_mismatch_frac']:.4f} | "
                         f"{fmt(r['rate_ratio_mean'])} |")
        lines += ["", f"- H_silent = {prof['H_silent']}, "
                      f"H_rate50 = {prof['H_rate50']}, "
                      f"H_rate25 = {prof['H_rate25']}", ""]
    path.write_text("\n".join(lines))


def make_figures(profiles, out_dir):
    splits = list(profiles)
    # fig01: |dV| quantile growth
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for ax, split in zip(axes, splits):
        prof = profiles[split]
        ts = sorted(int(k) for k in prof["steps"])
        for key, lbl in (("p50", "median"), ("p90", "p90"),
                         ("p95", "p95"), ("p99", "p99")):
            ys = [prof["steps"][str(t)][key] for t in ts]
            ax.plot(ts, ys, marker="o", ms=3, label=lbl)
        ax.set_yscale("log")
        ax.set_xlabel("rollout step")
        ax.set_title(f"|dV| growth — {split}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("|V_pred - V_true|")
    fig.tight_layout()
    fig.savefig(out_dir / "fig01_state_error_growth.png", dpi=150)
    plt.close(fig)

    # fig02: median/p90/p95 vs teacher tolerance bands
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for ax, split in zip(axes, splits):
        prof = profiles[split]
        ts = sorted(int(k) for k in prof["steps"])
        for key, lbl in (("p50", "median"), ("p90", "p90"),
                         ("p95", "p95")):
            ys = [prof["steps"][str(t)][key] for t in ts]
            ax.plot(ts, ys, marker="o", ms=3, label=lbl)
        for lvl in TOL_LINES:
            ax.axhline(lvl, ls="--", lw=1,
                       label=f"tolerance {lvl:g}")
        ax.set_yscale("log")
        ax.set_xlabel("rollout step")
        ax.set_title(f"error vs teacher tolerance — {split}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("|dV|")
    fig.tight_layout()
    fig.savefig(out_dir / "fig02_error_vs_teacher_tolerance.png", dpi=150)
    plt.close(fig)

    # fig08: |dV| histogram at steps 5/10/15 (from stored quantiles we
    # cannot rebuild the histogram, so this is approximated by a quantile
    # ribbon; the full histogram needs raw dV — see fig08 note in JSON)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, split in zip(axes, splits):
        prof = profiles[split]
        ts = [5, 10, 15]
        width = 0.25
        qs = ("p50", "p75", "p90")
        for i, t in enumerate(ts):
            r = prof["steps"][str(t)]
            vals = [r[q] for q in qs]
            ax.bar([q + f" s={t}" for q in qs], vals, width,
                   label=f"step {t}")
        ax.set_yscale("log")
        ax.set_title(f"|dV| quantiles at steps 5/10/15 — {split}")
        ax.grid(alpha=0.3, axis="y")
        ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(out_dir / "fig08_tangent_sigma_distribution.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--n-traj", type=int, default=64)
    parser.add_argument("--H", type=int, default=50)
    parser.add_argument("--ckpt", default=None,
                        help="checkpoint to profile (default: phase-1 gnn)")
    args = parser.parse_args()

    cfg = get_config(args.scale)
    if args.seed is not None:
        cfg.seed = args.seed
    device = get_device(override=args.device)
    conn = get_connectome(cfg, device)
    sim = LIFSimulator(conn, cfg, device)

    ckpt = Path(args.ckpt) if args.ckpt else \
        RESULTS_DIR / "checkpoints" / f"ckpt_gnn_{args.scale}_seed{cfg.seed}.pt"
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = build_model("gnn", cfg, conn, device,
                        dict(blob.get("model_kwargs") or {}))
    model.load_state_dict(blob["state_dict"])
    model.eval()
    print(f"[profile] {ckpt.name} (epoch {blob.get('epoch')})")

    val = load_or_generate("val", sim, cfg)
    th = tune_threshold(model, val, cfg, device, n_windows=512)
    print(f"[threshold] val-tuned {th:.2f} (val only, frozen)")

    profiles = {}
    for split in ("val", "test_seen"):
        data = load_or_generate(split, sim, cfg)
        prof = profile_split(model, data, cfg, device, th,
                             args.n_traj, args.H)
        profiles[split] = prof
        rec = recommend_sigmas(prof)
        cross = error_crossings(prof)
        print(f"[{split}] H_silent={prof['H_silent']} "
              f"H_rate50={prof['H_rate50']} H_rate25={prof['H_rate25']}")
        print(f"[{split}] crossings {cross}")
        print(f"[{split}] recommended sigmas {rec}")
        # compact console table
        print(f"{'step':>5} {'median':>9} {'p75':>9} {'p90':>9} "
              f"{'p95':>9} {'p99':>9} {'RMSE':>9} {'mm frac':>8} "
              f"{'rate_r':>7}")
        for t in STEPS:
            if t > args.H:
                continue
            r = prof["steps"][str(t)]
            rr = r["rate_ratio_mean"]
            rr_s = f"{rr:>7.3f}" if rr is not None else "      —"
            print(f"{t:>5} {r['p50']:>9.2e} {r['p75']:>9.2e} "
                  f"{r['p90']:>9.2e} {r['p95']:>9.2e} {r['p99']:>9.2e} "
                  f"{r['rmse']:>9.2e} {r['spike_mismatch_frac']:>8.4f} "
                  f"{rr_s}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"checkpoint": str(ckpt), "threshold": th,
               "n_traj": args.n_traj, "H": args.H,
               "recommendations": {s: recommend_sigmas(p)
                                   for s, p in profiles.items()},
               "crossings": {s: error_crossings(p)
                             for s, p in profiles.items()},
               "profiles": profiles}
    with open(OUT_DIR / "profile.json", "w") as f:
        json.dump(payload, f, indent=2)
    write_tables(profiles, OUT_DIR / "table_error_growth.md")
    make_figures(profiles, OUT_DIR)
    print(f"[save] {OUT_DIR}")


if __name__ == "__main__":
    main()
