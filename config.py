"""Central configuration for the connectome dynamics feasibility study.

Everything reproducible flows through this file: seeds, network size,
LIF parameters, dataset protocol, model sizes and training hyperparameters.

Two scales are predefined:
  - small: N=100, T=128  (pipeline smoke test, Step 6)
  - full : N=1000, T=256 (real experiment, Step 7+)
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
CACHE_DIR = RESULTS_DIR / "cache"

for _d in (RESULTS_DIR, FIGURES_DIR, CHECKPOINT_DIR, CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)


@dataclass
class Config:
    # ---------------- reproducibility ----------------
    seed: int = 1234

    # ---------------- network / LIF ------------------
    n_neurons: int = 1000
    T: int = 256                      # timesteps per trajectory
    K: int = 32                       # context window length
    dt: float = 1.0
    tau: float = 5.0
    v_rest: float = 0.0
    v_reset: float = 0.0
    v_th: float = 1.0
    v_min: float = -3.0               # inhibitory reversal floor
    refractory_period: int = 3        # in timesteps

    # ---------------- connectome (placeholder) -------
    # Ring-geometry distance-dependent sparse directed graph with Dale's law.
    # Structured so that the spatial OOD stimulus split is meaningful:
    # activity propagates mostly locally, so the model must use the graph.
    inh_fraction: float = 0.2         # fraction of inhibitory neurons
    conn_target_degree: float = 20.0  # target average in/out degree
    conn_lambda_frac: float = 1.0 / 16.0  # spatial decay, fraction of N
    conn_p_bg: float = 0.004          # background (long-range) prob
    w_exc_lo: float = 1.0
    w_exc_hi: float = 2.5
    w_inh_lo: float = 1.6
    w_inh_hi: float = 4.0
    # tonic bias current per neuron: keeps the network near threshold so
    # stimuli propagate a few hops without runaway excitation
    i_bias_lo: float = 0.15
    i_bias_hi: float = 0.4

    # ---------------- stimulus protocol --------------
    stim_min_neurons: int = 1
    stim_max_neurons: int = 5
    stim_dur_lo: int = 5
    stim_dur_hi: int = 20
    stim_amp_lo: float = 4.0
    stim_amp_hi: float = 7.0
    silence_prob: float = 0.3         # fraction of trajectories with silencing
    silence_max: int = 3

    # ---------------- dataset sizes ------------------
    n_train_traj: int = 2048
    n_val_traj: int = 128
    n_test_traj: int = 128
    n_test_seen_traj: int = 128       # stim in train range, fresh seeds

    # ---------------- model --------------------------
    d_model: int = 128
    d_temporal: int = 64
    nhead: int = 8
    temporal_layers: int = 2
    spatial_layers: int = 3
    dim_ff: int = 256
    dropout: float = 0.0
    gru_hidden: int = 1024
    gnn_layers: int = 3
    conn_alpha: float = 1.0           # scale of log1p(|W|) attention bias

    # ---------------- training -----------------------
    batch_size: int = 16
    epochs: int = 40
    lr: float = 3e-4
    lr_gru: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    patience: int = 8                 # early stopping on val loss
    n_val_windows: int = 1024         # windows sampled for val each epoch
    # fraction of training windows sampled to overlap a stimulus pulse
    # (uniform-only sampling would be dominated by quiescent windows)
    window_activity_bias: float = 0.5

    # ---------------- loss ---------------------------
    lambda_v: float = 1.0
    lambda_s: float = 1.0
    lambda_r: float = 1.0
    spike_pos_weight: float = 30.0    # spike rate ~2-3%, see test_lif

    # ---------------- rollout ------------------------
    rollout_horizons: tuple = (1, 10, 25, 50, 100, 200)
    rollout_hard_reset: bool = True   # spike=1 -> V=V_reset, R=1 (biological)
    n_rollout_traj: int = 32          # trajectories used for rollout eval

    # ---------------- perturbation -------------------
    n_perturb_traj: int = 16
    perturb_rollout: int = 100        # steps of autoregressive rollout
    n_silence_tests: int = 4
    n_lesion_tests: int = 4

    # ---------------- split ranges (fractions of N) --
    def stim_range(self, split: str) -> tuple[int, int]:
        N = self.n_neurons
        if split == "train":
            return (0, int(0.8 * N))
        if split == "val":
            return (int(0.7 * N), int(0.9 * N))
        if split in ("test", "test_ood"):
            return (int(0.8 * N), N)
        if split == "test_seen":
            return (0, int(0.8 * N))
        raise ValueError(f"unknown split {split}")

    # deterministic per-trajectory seeds; identical for every model/run
    def traj_seed(self, split: str, idx: int) -> int:
        offsets = {"train": 1_000_000, "val": 2_000_000,
                   "test": 3_000_000, "test_ood": 3_000_000,
                   "test_seen": 4_000_000}
        return self.seed + offsets[split] + idx

    @property
    def alpha(self) -> float:
        return self.dt / self.tau


def get_config(scale: str = "full") -> Config:
    cfg = Config()
    if scale == "small":
        cfg = replace(
            cfg,
            n_neurons=100, T=128, K=16,
            n_train_traj=512, n_val_traj=64, n_test_traj=64,
            n_test_seen_traj=64,
            batch_size=32, epochs=15, patience=5,
            gru_hidden=512, n_rollout_traj=16,
            rollout_horizons=(1, 10, 25, 50, 80),
            perturb_rollout=50,
        )
    elif scale != "full":
        raise ValueError(f"unknown scale {scale}")
    return cfg


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--scale", default="full", choices=["small", "full"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cpu", "xpu", "cuda"])
    return parser
