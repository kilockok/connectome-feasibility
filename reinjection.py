"""Phase-2 Step 2: ground-truth reinjection diagnostic.

Roll the learned model out autonomously, but every K steps re-anchor its
history window to the true states. K=1 is pure teacher forcing, K=inf is
the free rollout. The error-vs-K curve estimates the model's effective
autonomous horizon and separates "local dynamics wrong" from "cannot
recover from own errors".

    python reinjection.py [--scale full] [--models gnn connectome]

Writes results/rollout_v2/reinjection.json and
results/rollout_v2/figures/reinjection_vs_error.png
"""
from __future__ import annotations

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from config import get_config, RESULTS_DIR, CHECKPOINT_DIR, add_common_args
from connectome import get_connectome
from dataset import load_or_generate
from device import get_device
from evaluate import tune_threshold
from lif import LIFSimulator
from models import build_model
from rollout import REFR_HOLD, rollout_metrics

K_LIST = (1, 2, 5, 10, 20, 50, 10_000)      # 10_000 == infinity here
HORIZONS = (1, 2, 5, 10, 20, 25, 50, 100, 200)
OUT_DIR = RESULTS_DIR / "rollout_v2"
FIG_DIR = OUT_DIR / "figures"


@torch.no_grad()
def rollout_reinject(model, states, stim, cfg, k_reinject: int,
                     spike_threshold: float, silence_mask=None,
                     t0: int = 0):
    """Like rollout.rollout but re-anchored to truth every k_reinject steps.

    states [B,T,N,3], stim [B,T,N]; context starts at t0, predictions cover
    [t0+K, t0+K+H). Returns pred [B,H,N,3].
    """
    K = cfg.K
    B, T, N, _ = states.shape
    H = T - t0 - K
    dev = states.device

    def true_window(end):                        # window ending at time end
        w_state = states[:, end - K + 1: end + 1]
        w_stim = stim[:, end - K + 1: end + 1].unsqueeze(-1)
        return torch.cat([w_state, w_stim], dim=-1)

    hist = true_window(t0 + K - 1).clone()       # [B,K,N,4]
    preds = torch.empty(B, H, N, 3, device=dev)
    for s in range(H):
        t_abs = t0 + K + s
        if s > 0 and s % k_reinject == 0:
            hist = true_window(t_abs - 1).clone()
        out = model(hist)
        v = out["v"].clamp(cfg.v_min, cfg.v_th * 3.0)
        sp = (torch.sigmoid(out["s_logits"]) > spike_threshold).to(v.dtype)
        r = out["r"].clamp(0.0, 1.0)
        fired = sp > 0.5
        v = torch.where(fired, torch.full_like(v, cfg.v_reset), v)
        r = torch.where(fired, torch.ones_like(r), r)
        hold = (~fired) & (r > REFR_HOLD)
        v = torch.where(hold, torch.full_like(v, cfg.v_reset), v)
        if silence_mask is not None:
            v = torch.where(silence_mask, torch.full_like(v, cfg.v_rest), v)
            sp = torch.where(silence_mask, torch.zeros_like(sp), sp)
            r = torch.where(silence_mask, torch.zeros_like(r), r)
        preds[:, s, :, 0] = v
        preds[:, s, :, 1] = sp
        preds[:, s, :, 2] = r
        u = stim[:, t_abs]
        feat = torch.stack([v, sp, r, u], dim=-1)
        hist = torch.cat([hist[:, 1:], feat.unsqueeze(1)], dim=1)
    return preds


def main():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--models", nargs="*", default=["gnn", "connectome"])
    parser.add_argument("--split", default="test_seen")
    parser.add_argument("--n-traj", type=int, default=16)
    args = parser.parse_args()

    cfg = get_config(args.scale)
    torch.manual_seed(cfg.seed)
    device = get_device(override=args.device)
    conn = get_connectome(cfg, device)
    sim = LIFSimulator(conn, cfg, device)

    val = load_or_generate("val", sim, cfg)
    data = load_or_generate(args.split, sim, cfg)
    n = min(args.n_traj, data["states"].shape[0])
    states = data["states"][:n].to(device)
    stim = data["stimulus"][:n].to(device)
    sil = data["silence"][:n].to(device)
    K = cfg.K
    true = states[:, K:]
    horizons = [h for h in HORIZONS if h <= true.shape[1]]

    results = {}
    for name in args.models:
        ckpt = CHECKPOINT_DIR / f"ckpt_{name}_{args.scale}_seed{cfg.seed}.pt"
        if not ckpt.exists():
            print(f"[skip] {name}: {ckpt.name} missing")
            continue
        blob = torch.load(ckpt, map_location="cpu", weights_only=False)
        model = build_model(name, cfg, conn, device)
        model.load_state_dict(blob["state_dict"])
        model.eval()
        th = tune_threshold(model, val, cfg, device, n_windows=512)
        print(f"[model] {name}: threshold={th:.2f}")

        per_k = {}
        for k_re in K_LIST:
            pred = rollout_reinject(model, states, stim, cfg, k_re, th,
                                    silence_mask=sil)
            m = rollout_metrics(pred, true, horizons)
            per_k[str(k_re)] = {str(h): mm for h, mm in m.items()}
            h_last = horizons[-1]
            lbl = "inf" if k_re >= 10_000 else str(k_re)
            print(f"  K={lbl:>4s}: h=10 f1={m[10]['spike_f1']:.3f} "
                  f"h=50 f1={m[50]['spike_f1']:.3f} "
                  f"h={h_last} f1={m[h_last]['spike_f1']:.3f} "
                  f"v_rmse={m[h_last]['v_rmse']:.3f}")
        results[name] = {"threshold": th, "per_k": per_k}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "reinjection.json", "w") as f:
        json.dump({"scale": args.scale, "split": args.split, "n_traj": n,
                   "horizons": horizons, "results": results}, f, indent=2)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    ks = list(K_LIST)
    x = list(range(len(ks)))
    xlbl = ["1", "2", "5", "10", "20", "50", "inf"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for name, d in results.items():
        per_k = d["per_k"]
        for ax, h in zip(axes, (10, 50, max(horizons))):
            y = [per_k[str(k)][str(h)]["spike_f1"] for k in ks]
            ax.plot(x, y, marker="o", label=name)
        axes[0].set_ylabel("spike F1")
    for ax, h in zip(axes, (10, 50, max(horizons))):
        ax.set_title(f"F1 vs reinjection period (h={h})")
        ax.set_xticks(x, xlbl)
        ax.set_xlabel("K (steps between truth re-anchoring)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "reinjection_vs_error.png", dpi=130)
    print(f"[save] {OUT_DIR / 'reinjection.json'}")
    print(f"[plots] -> {FIG_DIR / 'reinjection_vs_error.png'}")


if __name__ == "__main__":
    main()
