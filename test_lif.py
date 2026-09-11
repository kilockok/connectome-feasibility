"""Step 1+2 sanity check: LIF simulator and dataset protocol.

Generates 100 trajectories, verifies that dynamics are reasonable
(non-degenerate firing rates, propagation, refractory behaviour), and
saves a diagnostic figure. Also estimates spike class imbalance for
`spike_pos_weight`.

Run:  python test_lif.py [--scale small|full]
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from config import get_config, FIGURES_DIR, add_common_args
from connectome import get_connectome
from dataset import generate_batch
from device import get_device
from lif import LIFSimulator


def main():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--n", type=int, default=100)
    args = parser.parse_args()

    cfg = get_config(args.scale)
    torch.manual_seed(cfg.seed)
    device = get_device(override=args.device)

    conn = get_connectome(cfg, device)
    print(f"[connectome] {conn.summary()}")

    sim = LIFSimulator(conn, cfg, device)

    # ---- generate 100 trajectories (train protocol) -------------------
    seeds = [cfg.traj_seed("train", i) for i in range(args.n)]
    data = generate_batch(seeds, "train", sim, cfg)
    states, stim = data["states"], data["stimulus"]
    S, V, R = states[..., 1], states[..., 0], states[..., 2]

    rate = S.mean().item()                       # fraction of spiking (n,t)
    per_traj = S.mean(dim=(1, 2))                # [B]
    dur_active = (S.sum(dim=1) > 0).float().mean(dim=1)  # frac neurons active
    print(f"[lif] trajectories={args.n} T={cfg.T} N={cfg.n_neurons}")
    print(f"[lif] mean firing rate      : {rate:.4f} spikes/neuron/step")
    print(f"[lif] per-traj rate min/max : {per_traj.min():.4f}/{per_traj.max():.4f}")
    print(f"[lif] frac neurons ever active: mean {dur_active.mean():.3f} "
          f"min {dur_active.min():.3f}")
    print(f"[lif] V range               : [{V.min():.2f}, {V.max():.2f}]")
    print(f"[lif] refractory mean       : {R.mean():.4f} (<=1/tau_ref={1/cfg.refractory_period:.2f})")

    # ---- sanity assertions --------------------------------------------
    assert 1e-4 < rate < 0.4, f"degenerate firing rate {rate}"
    assert per_traj.min() > 0, "some trajectories are completely silent"
    assert V.max() < cfg.v_th * 3.0, "runaway excitation"
    # refractory: a neuron that just spiked must be in refractory next step
    fired = S[:, :-1] > 0.5
    refr_next = R[:, 1:] > 0.5
    frac = refr_next[fired].float().mean().item()
    print(f"[lif] P(refractory | just spiked) = {frac:.3f} (expect ~1)")
    assert frac > 0.99
    # propagation: stimulus should increase firing in non-stimulated neurons
    stimulated = stim.sum(dim=1) > 0                            # [B,N]
    rate_stim = S.masked_select(stimulated.unsqueeze(1).expand_as(S)).mean()
    rate_rest = S.masked_select((~stimulated).unsqueeze(1).expand_as(S)).mean()
    print(f"[lif] rate(stimulated)={rate_stim:.4f}  rate(others)={rate_rest:.4f}")
    assert rate_rest > 1e-4, "no propagation beyond stimulated neurons"

    pos_weight = (1 - rate) / max(rate, 1e-6)
    print(f"[lif] suggested spike_pos_weight ~= {pos_weight:.1f} "
          f"(config default {cfg.spike_pos_weight})")

    # ---- diagnostic figure --------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    b = 0
    axes[0].imshow(S[b].T.cpu(), aspect="auto", cmap="gray_r", interpolation="nearest")
    axes[0].set_ylabel("neuron")
    axes[0].set_title(f"spike raster (traj 0, N={cfg.n_neurons})")
    for nid in range(min(5, cfg.n_neurons)):
        axes[1].plot(V[b, :, nid].cpu() + nid * 1.5, lw=0.7)
    axes[1].set_ylabel("V (offset)")
    axes[1].set_title("membrane potential, 5 neurons")
    axes[2].plot(S[b].mean(dim=1).cpu(), label="population rate")
    axes[2].plot(stim[b].max(dim=1).values.cpu() / cfg.stim_amp_hi,
                 label="stimulus (scaled)")
    axes[2].set_xlabel("t"); axes[2].legend(); axes[2].set_title("population firing rate")
    fig.tight_layout()
    out = FIGURES_DIR / f"sanity_lif_{args.scale}.png"
    fig.savefig(out, dpi=120)
    print(f"[lif] diagnostic figure -> {out}")
    print("[lif] SANITY CHECK PASSED")


if __name__ == "__main__":
    main()
