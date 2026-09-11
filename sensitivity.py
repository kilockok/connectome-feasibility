"""Phase-2 Step 1: LIF self-sensitivity diagnostic.

Perturb the TRUE dynamical state of real trajectories with tiny V noise,
continue both branches with the true simulator (same stimulus), and measure
how fast they decorrelate. This bounds how much long-horizon exact spike
matching is achievable AT ALL in this system.

    python sensitivity.py [--scale full] [--n-traj 16]

Writes results/rollout_v2/lif_sensitivity.json and
results/rollout_v2/figures/lif_sensitivity_{spike_f1,v_rmse,population_corr}.png
"""
from __future__ import annotations

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from config import get_config, RESULTS_DIR, add_common_args
from connectome import get_connectome
from dataset import load_or_generate
from device import get_device
from lif import LIFSimulator

EPSILONS = (1e-5, 1e-4, 1e-3, 3e-3, 1e-2)
HORIZONS = (1, 2, 5, 10, 20, 50, 100, 200)
OUT_DIR = RESULTS_DIR / "rollout_v2"
FIG_DIR = OUT_DIR / "figures"


def _f1(ps, ts):
    tp = (ps * ts).sum().item()
    fp = (ps * (1 - ts)).sum().item()
    fn = ((1 - ps) * ts).sum().item()
    p = tp / max(tp + fp, 1.0)
    r = tp / max(tp + fn, 1.0)
    return 2 * p * r / max(p + r, 1e-9)


def _pearson(a: torch.Tensor, b: torch.Tensor, dim) -> torch.Tensor:
    a = a - a.mean(dim=dim, keepdim=True)
    b = b - b.mean(dim=dim, keepdim=True)
    num = (a * b).sum(dim=dim)
    den = a.norm(dim=dim) * b.norm(dim=dim) + 1e-9
    return num / den


def branch_metrics(ref: torch.Tensor, per: torch.Tensor,
                   horizons) -> dict[int, dict]:
    """ref/per [B, H, N, 3]; prefix metrics per horizon, averaged over B."""
    out = {}
    B = ref.shape[0]
    for h in horizons:
        r, p = ref[:, :h], per[:, :h]
        rv, pv = r[..., 0], p[..., 0]
        rs, ps = r[..., 1], p[..., 1]
        v_rmse = ((rv - pv) ** 2).mean().sqrt().item()
        f1 = _f1(ps, rs)
        # per-neuron mean firing rate over window -> corr across neurons
        rr = rs.mean(dim=1)                      # [B, N]
        pr = ps.mean(dim=1)
        rate_corr = _pearson(rr, pr, dim=1).mean().item()
        # active-neuron Jaccard (>=1 spike in window)
        ra = (rs.sum(dim=1) > 0)
        pa = (ps.sum(dim=1) > 0)
        inter = (ra & pa).sum(dim=1).float()
        union = (ra | pa).sum(dim=1).float().clamp(min=1)
        overlap = (inter / union).mean().item()
        # population activity time series corr
        rt = rs.sum(dim=2)                       # [B, h]
        pt = ps.sum(dim=2)
        if h >= 2:
            act_corr = _pearson(rt, pt, dim=1).mean().item()
        else:
            act_corr = float("nan")
        out[int(h)] = {"v_rmse": v_rmse, "spike_f1": f1,
                       "rate_corr": rate_corr, "active_overlap": overlap,
                       "activity_corr": act_corr}
    return out


def make_figures(res: dict, eps_list, horizons):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    hs = list(horizons)

    def series(eps, key):
        return [res[str(eps)][str(h)][key] for h in hs]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for e in eps_list:
        ax.plot(hs, series(e, "spike_f1"), marker="o", ms=3,
                label=f"eps={e}")
    ax.set_xscale("log")
    ax.set_xlabel("horizon (steps)")
    ax.set_ylabel("spike F1 (perturbed vs reference)")
    ax.set_title("LIF self-sensitivity: spike F1 vs horizon")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "lif_sensitivity_spike_f1.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for e in eps_list:
        ax.plot(hs, series(e, "v_rmse"), marker="o", ms=3, label=f"eps={e}")
    ax.set_xscale("log")
    ax.set_xlabel("horizon (steps)")
    ax.set_ylabel("V RMSE")
    ax.set_title("LIF self-sensitivity: V divergence vs horizon")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "lif_sensitivity_v_rmse.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, key, title in (
            (axes[0], "rate_corr", "per-neuron rate correlation"),
            (axes[1], "activity_corr", "population activity correlation"),
            (axes[2], "active_overlap", "active-neuron Jaccard")):
        for e in eps_list:
            ax.plot(hs, series(e, key), marker="o", ms=3, label=f"eps={e}")
        ax.set_xscale("log")
        ax.set_xlabel("horizon")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("LIF self-sensitivity: macro-dynamics vs horizon")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "lif_sensitivity_population_corr.png", dpi=130)
    plt.close(fig)
    print(f"[plots] -> {FIG_DIR}")


def main():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--n-traj", type=int, default=16)
    parser.add_argument("--split", default="val")
    parser.add_argument("--t0", type=int, default=None,
                        help="branch point (default: K)")
    args = parser.parse_args()

    cfg = get_config(args.scale)
    torch.manual_seed(cfg.seed)
    device = get_device(override=args.device)
    conn = get_connectome(cfg, device)
    sim = LIFSimulator(conn, cfg, device)

    data = load_or_generate(args.split, sim, cfg)
    n = min(args.n_traj, data["states"].shape[0])
    t0 = args.t0 if args.t0 is not None else cfg.K
    H = min(max(HORIZONS), cfg.T - t0 - 1)
    horizons = tuple(h for h in HORIZONS if h <= H)
    print(f"[setup] split={args.split} n={n} t0={t0} H={H}")

    states = data["states"][:n]                    # [B,T,N,3]
    stim = data["stimulus"][:n]                    # [B,T,N]
    sil = data["silence"][:n].to(device)

    v0 = states[:, t0, :, 0].to(device)
    s0 = states[:, t0, :, 1].to(device)
    r0 = (states[:, t0, :, 2] * float(cfg.refractory_period)).to(device)
    fut = stim[:, t0 + 1: t0 + 1 + H].to(device)   # stimulus after branch

    with torch.no_grad():
        ref = sim.simulate(fut, silence_mask=sil,
                           state0=(v0, s0, r0))    # [B,H,N,3]

    res = {}
    g = torch.Generator(device="cpu").manual_seed(cfg.seed + 555)
    for eps in EPSILONS:
        dv = (torch.randn(v0.shape, generator=g) * eps).to(device)
        with torch.no_grad():
            per = sim.simulate(fut, silence_mask=sil,
                               state0=(v0 + dv, s0.clone(), r0.clone()))
        m = branch_metrics(ref, per, horizons)
        res[f"{eps:g}"] = {str(h): mm for h, mm in m.items()}
        h_last = horizons[-1]
        print(f"[eps={eps:g}] h=10 f1={m[10]['spike_f1']:.3f} "
              f"h={h_last} f1={m[h_last]['spike_f1']:.3f} "
              f"v_rmse={m[h_last]['v_rmse']:.3f} "
              f"rate_corr={m[h_last]['rate_corr']:.3f} "
              f"act_corr={m[h_last]['activity_corr']:.3f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"scale": args.scale, "split": args.split, "n_traj": n,
               "t0": t0, "horizons": horizons, "epsilons": EPSILONS,
               "metrics": res}
    with open(OUT_DIR / "lif_sensitivity.json", "w") as f:
        json.dump(payload, f, indent=2)
    make_figures(res, [f"{e:g}" for e in EPSILONS], horizons)
    print(f"[save] {OUT_DIR / 'lif_sensitivity.json'}")


if __name__ == "__main__":
    main()
