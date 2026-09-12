"""Phase-2 (rollout stability) configuration: stage curriculum + experiment matrix.

Deliberately separate from config.py: phase-1 artifacts must stay reproducible
bit-for-bit. Everything phase-2 lives here plus rollout_train.py, dagger.py,
losses.py, surrogate.py, models/mechanistic.py, rollout_eval.py (see PHASE2.md).
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field, replace
from pathlib import Path

from config import Config, CHECKPOINT_DIR, RESULTS_DIR

ROLLOUT2_DIR = RESULTS_DIR / "rollout_v2"
STAGE_CKPT_DIR = CHECKPOINT_DIR / "rollout_v2"
ROLLOUT3_DIR = RESULTS_DIR / "rollout_v3"
STAGE_CKPT_DIR_V3 = CHECKPOINT_DIR / "rollout_v3"
ROLLOUT3R_DIR = RESULTS_DIR / "rollout_v3"
STAGE_CKPT_DIR_V3R = CHECKPOINT_DIR / "rollout_v3r"
ROLLOUT4_DIR = RESULTS_DIR / "rollout_v4"
STAGE_CKPT_DIR_V4 = CHECKPOINT_DIR / "rollout_v4"


def stages_v4(scale: str) -> list[StageConfig]:
    """rollout_v4 2x2 + stability curriculum (spec §6/§14): SINGLE stage
    U=8 for every experiment, teacher 0.90, no noise, DAgger mix 0.20 when
    on, max 12 epochs with patience 4. U is NEVER advanced in v4."""
    bs = 16 if scale == "full" else 32
    return [
        StageConfig("s1_u8", 8, 12, teacher_ratio=0.90, dagger_mix=0.2,
                    lr_factor=1.0, n_traj=512, batch_size=bs,
                    force_ckpt=True),
    ]


# ----------------------------------------------------------------------
@dataclass
class StageConfig:
    name: str
    unroll: int
    epochs: int
    teacher_ratio: float            # per-step prob of feeding the true state
    noise_sigma: float = 0.0        # V-noise std on teacher-forced states
    dagger_mix: float = 0.0         # fraction of on-policy buffer batches
    lr_factor: float = 1.0          # multiplies RolloutConfig.base_lr
    n_traj: int = 512               # train trajectories per epoch
    batch_size: int | None = None   # None -> RolloutConfig.batch_size
    closed_loop: bool = False       # stage 5 marker (teacher_ratio == 0)
    force_ckpt: bool = False        # always gradient-checkpoint this stage


def default_stages(scale: str) -> list[StageConfig]:
    """Progressive horizon curriculum (spec: U=8->64, teacher 0.9->0.0,
    LR halving per stage, sigma ramp 1e-4->3e-3, DAgger mix 0.2->0.7)."""
    bs = 16 if scale == "full" else 32
    late_bs = 8 if scale == "full" else 32
    return [
        StageConfig("s1_u8",   8, 15, teacher_ratio=0.90, noise_sigma=1e-4,
                    dagger_mix=0.2, lr_factor=1.0,   n_traj=512, batch_size=bs),
        StageConfig("s2_u16", 16, 20, teacher_ratio=0.70, noise_sigma=3e-4,
                    dagger_mix=0.3, lr_factor=0.5,   n_traj=512, batch_size=bs,
                    force_ckpt=True),
        StageConfig("s3_u32", 32, 25, teacher_ratio=0.50, noise_sigma=1e-3,
                    dagger_mix=0.5, lr_factor=0.25,  n_traj=384, batch_size=bs),
        StageConfig("s4_u64", 64, 30, teacher_ratio=0.25, noise_sigma=3e-3,
                    dagger_mix=0.7, lr_factor=0.125, n_traj=256,
                    batch_size=late_bs),
        StageConfig("s5_cl",  64, 20, teacher_ratio=0.0,  noise_sigma=3e-3,
                    dagger_mix=0.7, lr_factor=0.0625, n_traj=256,
                    batch_size=late_bs, closed_loop=True),
    ]


def smoke_stages(scale: str) -> list[StageConfig]:
    """Tiny curriculum for pipeline smoke tests (CPU/GPU, minutes)."""
    return [
        StageConfig("s1_u4",  4, 2, teacher_ratio=0.75, noise_sigma=3e-4,
                    dagger_mix=0.3, lr_factor=1.0, n_traj=32),
        StageConfig("s2_u8",  8, 2, teacher_ratio=0.0,  noise_sigma=1e-3,
                    dagger_mix=0.5, lr_factor=0.5, n_traj=32, closed_loop=True),
    ]


def stages_v3(scale: str) -> list[StageConfig]:
    """Phase-3 curriculum (user spec 十/十一): U=4,8,16,32,64 with teacher
    ratio 0.9,0.75,0.5,0.25,0.0. No input-noise axis in phase 3 (the v3
    matrix isolates scheduled sampling / DAgger / mechanistic+threshold
    loss); DAgger mix ramps like v2 and is gated by rc.dagger.
    Late stages always gradient-checkpoint: gnn_temporal costs ~K x a GNN
    step per unroll step."""
    bs = 16 if scale == "full" else 32
    late_bs = 8 if scale == "full" else 16
    ep = (15, 20, 25, 30, 20) if scale == "full" else (8, 10, 12, 12, 10)
    return [
        StageConfig("s1_u4",   4, ep[0], teacher_ratio=0.90, dagger_mix=0.2,
                    lr_factor=1.0,   n_traj=512, batch_size=bs),
        StageConfig("s2_u8",   8, ep[1], teacher_ratio=0.75, dagger_mix=0.3,
                    lr_factor=0.5,   n_traj=512, batch_size=bs),
        StageConfig("s3_u16", 16, ep[2], teacher_ratio=0.50, dagger_mix=0.5,
                    lr_factor=0.25,  n_traj=384, batch_size=bs,
                    force_ckpt=True),
        StageConfig("s4_u32", 32, ep[3], teacher_ratio=0.25, dagger_mix=0.7,
                    lr_factor=0.125, n_traj=256, batch_size=late_bs,
                    force_ckpt=True),
        StageConfig("s5_u64", 64, ep[4], teacher_ratio=0.0,  dagger_mix=0.7,
                    lr_factor=0.0625, n_traj=256, batch_size=late_bs,
                    closed_loop=True, force_ckpt=True),
    ]


def stages_v3r(scale: str, exp: str) -> list[StageConfig]:
    """Per-experiment stage subset for the v3r DAgger+TBPTT matrix (user spec
    §10 / §22): each experiment truncates at its target horizon and never
    advances to U=64 automatically. Epoch caps are tight (feasibility/diagnosis,
    not full training)."""
    bs = 16 if scale == "full" else 32
    ep = (10, 10, 15, 15) if scale == "full" else (4, 4, 5, 5)
    all_stages = [
        StageConfig("s1_u8",   8, ep[0], teacher_ratio=0.90, dagger_mix=0.2,
                    lr_factor=1.0,   n_traj=512, batch_size=bs,
                    force_ckpt=True),   # tangent keeps 2 extra graphs alive
        StageConfig("s2_u16", 16, ep[1], teacher_ratio=0.75, dagger_mix=0.4,
                    lr_factor=0.5,   n_traj=512, batch_size=bs,
                    force_ckpt=True),
        StageConfig("s3_u32", 32, ep[2], teacher_ratio=0.50, dagger_mix=0.5,
                    lr_factor=0.25,  n_traj=384, batch_size=bs,
                    force_ckpt=True),
        StageConfig("s4_u64", 64, ep[3], teacher_ratio=0.25, dagger_mix=0.7,
                    lr_factor=0.125, n_traj=256,
                    batch_size=8 if scale == "full" else 16,
                    force_ckpt=True),
    ]
    n = {"E0": 1, "E1": 1, "E2": 2, "E3": 3}[exp.upper()]
    return all_stages[:n]


# ----------------------------------------------------------------------
@dataclass
class RolloutConfig:
    base: Config
    experiment: str = "G"
    model: str = "gnn"
    scale: str = "full"
    eval_only: bool = False             # experiment A: no training

    # ---- method switches (the A-G ablation axes) ----
    scheduled_sampling: bool = False
    noise: bool = False
    dagger: bool = False
    mechanistic: bool = False
    macro_loss: bool = False            # rate + population + delta terms

    # ---- gradients / loss ----
    surrogate: str = "ste"              # ste | sigmoid | fast_sigmoid
    surrogate_beta: float = 10.0
    focal: bool = False
    focal_gamma: float = 2.0
    gamma: float = 0.98                 # horizon weight w_h = gamma^(h-1)
    lambda_v: float = 1.0
    lambda_s: float = 1.0
    lambda_r: float = 1.0
    lambda_rate: float = 20.0
    lambda_pop: float = 20.0
    lambda_delta: float = 1.0
    pop_groups: int = 10

    # ---- optimisation ----
    base_lr: float = 5e-5               # ~0.15 x phase-1 lr
    batch_size: int = 16
    grad_clip: float = 1.0
    clip_warn_frac: float = 0.2
    patience: int = 5                   # early stop per stage (spec §23: 4-5)
    grad_ckpt: str = "auto"             # auto|on|off (auto: on if unroll >= 32)
    tbptt: int = 0                      # detach history every C steps (0 = off)

    # ---- DAgger ----
    buffer_capacity: int = 200_000
    dagger_collect_every: int = 1
    dagger_collect_traj: int = 64
    dagger_horizon: int = 64

    # ---- validation / protection ----
    n_val_windows: int = 512
    val_rollout_traj: int = 12
    val_rollout_horizon: int = 50
    protect_drop: float = 0.10          # one-step F1 drop guard vs phase-1

    # ---- noise details ----
    spike_flip_p: float = 3e-4
    refrac_jitter_p: float = 3e-4

    # ---- phase-3 additions ----
    version: str = "v2"                 # path namespace: v2 | v3
    threshold_loss: bool = False        # spec 十六 threshold-weighted V loss
    lambda_thresh: float = 1.0
    thresh_alpha: float = 6.0
    thresh_sigma: float = 0.25          # ~quarter of the V operating range
    # tangent / local-stability finite-difference loss (spec 十一)
    tangent: bool = False
    lambda_tangent: float = 0.1
    tangent_sigma: float = 3e-4
    # per-stage DAgger sampling mix override (spec 六 curriculum)
    dagger_mix_override: tuple = ()     # parallel to stages; empty -> use stage
    # rollout_v4 (spec §12-13): multi-scale tangent + perturbed-state loss
    tangent_scales: tuple = ()          # empty -> single tangent_sigma
    tangent_probs: tuple = (0.4, 0.4, 0.2)
    tangent_all_scales: bool = False    # True: weighted SUM of per-scale
                                        # losses EVERY batch (no sampling)
    lambda_perturbed: float = 0.0       # >0 enables L_perturbed_teacher
    score_v4: bool = False              # §17 multi-horizon validation score
    model_kwargs: dict = field(default_factory=dict)  # e.g. k_hist for
    # gnn_temporal; stored in checkpoint blobs so eval rebuilds identically

    stages: list = field(default_factory=list)

    # ------------------------------------------------------------------
    def use_grad_ckpt(self, stage: StageConfig) -> bool:
        if stage.force_ckpt:
            return True
        if self.grad_ckpt == "on":
            return True
        if self.grad_ckpt == "off":
            return False
        return stage.unroll >= 16

    def stage_batch_size(self, stage: StageConfig) -> int:
        return stage.batch_size or self.batch_size


# Experiment matrix (spec section 十二). A is the phase-1 checkpoint itself.
EXPERIMENTS: dict[str, dict] = {
    "A": dict(eval_only=True),
    "B": dict(),                        # progressive unroll only
    "C": dict(scheduled_sampling=True),
    "D": dict(scheduled_sampling=True, noise=True),
    "E": dict(scheduled_sampling=True, dagger=True),
    "F": dict(scheduled_sampling=True, dagger=True, mechanistic=True),
    "G": dict(scheduled_sampling=True, dagger=True, mechanistic=True,
              noise=True, macro_loss=True),
}

# Phase-3 matrix (user spec 二十一, letters D/E/F of that spec). A/B/C are
# not rollout experiments: A = phase-1 GNN, B/C = one-step-trained
# gnn_temporal with k_hist=8 / cfg.K (see train.py --model gnn_temporal).
# Every v3 experiment fine-tunes from the phase-1-style one-step
# gnn_temporal checkpoint (never from another v3 experiment), keeps the v3
# loss weights (lambda_rate/lambda_pop = 0.1 per spec 十五), and always
# trains through scheduled-sampling unrolls with macro loss terms.
_V3_LOSS = dict(macro_loss=True, lambda_rate=0.1, lambda_pop=0.1,
                lambda_delta=1.0)
EXPERIMENTS_V3: dict[str, dict] = {
    # D: GNN+TF K=cfg.K + rollout training with scheduled sampling
    "D": dict(scheduled_sampling=True, **_V3_LOSS),
    # E: D + DAgger
    "E": dict(scheduled_sampling=True, dagger=True, **_V3_LOSS),
    # F: E + mechanistic LIF projection + threshold-weighted V loss
    "F": dict(scheduled_sampling=True, dagger=True, mechanistic=True,
              threshold_loss=True, **_V3_LOSS),
}
MATRICES = {"v2": EXPERIMENTS, "v3": EXPERIMENTS_V3}

# Rollout-v3 DAgger+TBPTT matrix (user "rollout_v3" spec §10). All fine-tune
# from the phase-1 full GNN, use scheduled-sampling unrolls with the v3 loss
# (state+spike+delta+rate + optional tangent), DAgger on-policy replay with a
# rising mix, and TBPTT truncation on the longer horizons. E0 isolates DAgger;
# E1 adds mild input noise; E2/E3 lengthen the horizon with TBPTT.
_V3R_LOSS = dict(macro_loss=True, lambda_rate=0.1, lambda_pop=0.0,
                 lambda_delta=1.0)
EXPERIMENTS_V3R: dict[str, dict] = {
    # E0: U=8, teacher=0.9, DAgger=0.2, no noise, no TBPTT — does DAgger help?
    "E0": dict(scheduled_sampling=True, dagger=True, noise=False,
               tbptt=0, **_V3R_LOSS),
    # E1: E0 + mild state noise
    "E1": dict(scheduled_sampling=True, dagger=True, noise=True,
               tbptt=0, **_V3R_LOSS),
    # E2: U=16, teacher=0.75, DAgger=0.4, TBPTT detach=4
    "E2": dict(scheduled_sampling=True, dagger=True, noise=True,
               tbptt=4, **_V3R_LOSS),
    # E3: U=32, teacher=0.5, DAgger=0.5, TBPTT detach=8
    "E3": dict(scheduled_sampling=True, dagger=True, noise=True,
               tbptt=8, **_V3R_LOSS),
}
MATRICES["v3r"] = EXPERIMENTS_V3R

# Rollout-v4 2x2 DAgger x Tangent ablation (spec §5-9). Everything fixed:
# U=8, teacher 0.9, no noise, STE, E0 macro loss, 12 epochs / patience 4,
# same LR/clip/batch/seed/splits. ONLY DAgger and Tangent toggle.
_V4_BASE = dict(macro_loss=True, lambda_rate=0.1, lambda_pop=0.0,
                lambda_delta=1.0, patience=4, score_v4=True)
EXPERIMENTS_V4: dict[str, dict] = {
    "A0": dict(scheduled_sampling=True, dagger=False, tangent=False,
               **_V4_BASE),                      # neither (unroll-only control)
    "A1": dict(scheduled_sampling=True, dagger=True, tangent=False,
               **_V4_BASE),                      # DAgger only
    "A2": dict(scheduled_sampling=True, dagger=False, tangent=True,
               tangent_sigma=3e-4, lambda_tangent=0.1,
               **_V4_BASE),                      # Tangent only (E0 sigma)
    "A3": dict(scheduled_sampling=True, dagger=True, tangent=True,
               tangent_sigma=3e-4, lambda_tangent=0.1,
               **_V4_BASE),                      # DAgger + Tangent (= E0 logic)
    # STEP 4-6 (spec §14-15): S0 = best 2x2 cell (A2, already trained).
    # S1 = S0 + profiling-recalibrated multi-scale tangent (§11-12);
    # S2 = S1 + direct perturbed-state teacher loss (§13).  No DAgger —
    # the 2x2 showed DAgger+tangent conflict (interaction -0.164).
    "S1": dict(scheduled_sampling=True, dagger=False, tangent=True,
               tangent_sigma=3e-4, lambda_tangent=0.1,
               tangent_scales=(1e-3, 3e-3, 1e-2),
               tangent_probs=(0.4, 0.4, 0.2), **_V4_BASE),
    "S2": dict(scheduled_sampling=True, dagger=False, tangent=True,
               tangent_sigma=3e-4, lambda_tangent=0.1,
               tangent_scales=(1e-3, 3e-3, 1e-2),
               tangent_probs=(0.4, 0.4, 0.2),
               lambda_perturbed=0.2, **_V4_BASE),
    # Final tangent probe: S1 showed multi-scale (>=1e-3 mass) HURTS vs the
    # small sigma; S3 isolates the single mid-scale sigma=1e-3 to locate the
    # sigma-response curve before declaring the tangent branch exhausted.
    "S3": dict(scheduled_sampling=True, dagger=False, tangent=True,
               tangent_sigma=1e-3, lambda_tangent=0.1, **_V4_BASE),
    # Sigma-response edge probe: S3 (1e-3) beat A2 (3e-4) AND the multi-scale
    # mixes with >=3e-3 mass lost to both; sigma=3e-3 single maps the decay
    # edge and closes the tangent-sigma question.
    "S4": dict(scheduled_sampling=True, dagger=False, tangent=True,
               tangent_sigma=3e-3, lambda_tangent=0.1, **_V4_BASE),
    # Sigma-response edge probe: the single-scale curve is MONOTONE RISING
    # (3e-4 -> 0.563, 1e-3 -> 0.572, 3e-3 -> 0.599) while the multi-scale mix
    # with 1e-2 mass collapsed (S1/S1+L_perturbed ~0.52). sigma=1e-2 single
    # isolates whether 1e-2 itself is the poison or mixing scales is.
    "S5": dict(scheduled_sampling=True, dagger=False, tangent=True,
               tangent_sigma=1e-2, lambda_tangent=0.1, **_V4_BASE),
    # The single-scale curve kept RISING through sigma=1e-2 (0.563/0.572/
    # 0.599/0.626): large sigma is not the poison — per-batch mixed-scale
    # sampling is (loss-scale oscillation across batches). sigma=1e-2 already
    # exceeds the measured one-step error (p90 ~7.5e-3): the tangent loss is
    # enlarging the stability basin. S6 maps the next point.
    "S6": dict(scheduled_sampling=True, dagger=False, tangent=True,
               tangent_sigma=3e-2, lambda_tangent=0.1, **_V4_BASE),
    # S7: multi-scale DONE RIGHT (spec §12 remediation). S1 showed per-batch
    # sampled scales fail (loss-scale oscillation); S7 sums the weighted
    # per-scale tangent losses on EVERY batch. Peak single sigma is 1e-2
    # (S5=0.626); S6 turned down (3e-2 -> 0.558).
    "S7": dict(scheduled_sampling=True, dagger=False, tangent=True,
               tangent_sigma=1e-2, lambda_tangent=0.1,
               tangent_scales=(1e-3, 3e-3, 1e-2),
               tangent_probs=(0.4, 0.4, 0.2), tangent_all_scales=True,
               **_V4_BASE),
    # STEP 7-9 (spec §20-26) Temporal Hybrid — entered after the gate passed
    # (collapse 17 <= 20 AND tangent branch exhausted; peak single sigma =
    # 1e-2, S5=0.626). Same recipe as S5 (tangent sigma=1e-2, U=8); ONLY the
    # architecture changes. k_hist truncates inside forward, so the one
    # canonical phase-1 temporal checkpoint (k_hist=cfg.K, untagged) inits
    # every T variant; pos table is cfg.K-sized for all k_hist, making the
    # state dicts compatible.
    # G1 = gnn_wide param-matched to the temporal model (fairness control);
    # G0 = the pure-GNN S5 cell (already trained, no new run).
    "G1": dict(scheduled_sampling=True, dagger=False, tangent=True,
               tangent_sigma=1e-2, lambda_tangent=0.1,
               model_kwargs={"d_model": 128, "gnn_layers": 6},   # 0.899M
               **_V4_BASE),
    "T1": dict(scheduled_sampling=True, dagger=False, tangent=True,
               tangent_sigma=1e-2, lambda_tangent=0.1,
               model_kwargs={"k_hist": 1, "t_layers": 2, "t_heads": 4,
                             "pos_type": "learned", "causal": False,
                             "dropout": 0.05}, **_V4_BASE),   # Markov control
    "T2": dict(scheduled_sampling=True, dagger=False, tangent=True,
               tangent_sigma=1e-2, lambda_tangent=0.1,
               model_kwargs={"k_hist": 16, "t_layers": 2, "t_heads": 4,
                             "pos_type": "learned", "causal": False,
                             "dropout": 0.05}, **_V4_BASE),  # K=16 first
    "T3": dict(scheduled_sampling=True, dagger=False, tangent=True,
               tangent_sigma=1e-2, lambda_tangent=0.1,
               model_kwargs={"k_hist": 32, "t_layers": 2, "t_heads": 4,
                             "pos_type": "learned", "causal": False,
                             "dropout": 0.05}, **_V4_BASE),  # full history
}
MATRICES["v4"] = EXPERIMENTS_V4


def experiment_index(exp: str, matrix: str = "v2") -> int:
    if matrix == "v4":
        # A0..A3 -> 0..3, S0..S4 -> 4..8, T1..T3 -> 21..23, G1 -> 31
        fam = {"A": 0, "S": 4, "T": 20, "G": 30}[exp[0].upper()]
        return fam + int(exp[1])
    letters = {"v2": "ABCDEFG", "v3": "DEF", "v3r": "E0E1E2E3"}[matrix]
    return letters.index(exp.upper())


# ----------------------------------------------------------------------
# Path helpers — phase-2 checkpoints NEVER overwrite phase-1 files, and
# phase-3 (v3) lives in its own namespace (rollout_v3) so neither matrix's
# artifacts can collide (both use experiment letters D/E/F).
def phase1_ckpt_path(cfg: Config, model: str) -> Path:
    return CHECKPOINT_DIR / f"ckpt_{model}_full_seed{cfg.seed}.pt" \
        if cfg.n_neurons == 1000 else \
        CHECKPOINT_DIR / f"ckpt_{model}_small_seed{cfg.seed}.pt"


def _dirs(rc: RolloutConfig) -> tuple[Path, Path]:
    if rc.version == "v2":
        return ROLLOUT2_DIR, STAGE_CKPT_DIR
    if rc.version == "v3":
        return ROLLOUT3_DIR, STAGE_CKPT_DIR_V3
    if rc.version == "v3r":
        return ROLLOUT3R_DIR, STAGE_CKPT_DIR_V3R
    return ROLLOUT4_DIR, STAGE_CKPT_DIR_V4


def stage_ckpt_path(rc: RolloutConfig, stage: StageConfig) -> Path:
    _, stage_dir = _dirs(rc)
    d = stage_dir / rc.experiment
    d.mkdir(parents=True, exist_ok=True)
    return d / (f"ckpt_{rc.model}_{rc.scale}_rollout_{rc.version}_"
                f"{rc.experiment}_{stage.name}_seed{rc.base.seed}.pt")


def final_ckpt_path(rc: RolloutConfig) -> Path:
    return CHECKPOINT_DIR / (f"ckpt_{rc.model}_{rc.scale}_rollout_"
                             f"{rc.version}_{rc.experiment}"
                             f"_seed{rc.base.seed}.pt")


def history_path(rc: RolloutConfig) -> Path:
    out_dir, _ = _dirs(rc)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / (f"history_{rc.model}_{rc.scale}_{rc.version}_"
                      f"{rc.experiment}_seed{rc.base.seed}.csv")


def summary_path(rc: RolloutConfig) -> Path:
    out_dir, _ = _dirs(rc)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / (f"summary_{rc.model}_{rc.scale}_{rc.version}_"
                      f"{rc.experiment}_seed{rc.base.seed}.json")


# ----------------------------------------------------------------------
def add_rollout_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default="gnn",
                        choices=["gnn", "connectome", "gnn_temporal",
                                 "gnn_wide"])
    parser.add_argument("--matrix", default="v2", choices=list(MATRICES),
                        help="experiment matrix: v2 (phase-2 A-G), v3 "
                             "(phase-3 D/E/F), v3r (E0-E3), v4 (A0-A3 2x2)")
    parser.add_argument("--experiment", default="G",
                        help="letter within the selected matrix "
                             "(v2: A-G, v3: D-F)")
    parser.add_argument("--surrogate", default=None,
                        choices=["ste", "sigmoid", "fast_sigmoid"])
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--base-lr", type=float, default=None)
    parser.add_argument("--stages", default=None,
                        help="comma-separated stage names to run "
                             "(default: all)")
    parser.add_argument("--smoke", action="store_true",
                        help="tiny curriculum for pipeline validation")
    parser.add_argument("--buffer-capacity", type=int, default=None)
    parser.add_argument("--tbptt", type=int, default=None)
    parser.add_argument("--grad-ckpt", default=None,
                        choices=["auto", "on", "off"])
    # rollout-v3 (v3r) options
    parser.add_argument("--dagger-mix", default=None,
                        help="comma per-stage DAgger mix, e.g. 0.2,0.3,0.5")
    parser.add_argument("--tangent", action="store_true",
                        help="enable tangent/local-stability loss")
    parser.add_argument("--lambda-tangent", type=float, default=None)
    parser.add_argument("--tangent-sigma", type=float, default=None)
    parser.add_argument("--tangent-scales", default=None,
                        help="comma list of sigmas for multi-scale tangent "
                             "(spec §12), e.g. 1e-3,3e-3,1e-2")
    parser.add_argument("--lambda-perturbed", type=float, default=None,
                        help="weight of L_perturbed_teacher (spec §13)")
    # phase-3 architecture options (used when --model gnn_temporal/gnn_wide)
    parser.add_argument("--k-hist", type=int, default=None,
                        help="gnn_temporal: history steps consumed "
                             "(default: cfg.K)")
    parser.add_argument("--t-layers", type=int, default=None,
                        help="gnn_temporal temporal transformer layers")
    parser.add_argument("--t-heads", type=int, default=None)
    parser.add_argument("--pos", default=None, choices=["learned", "sincos"],
                        help="gnn_temporal temporal positional encoding")
    parser.add_argument("--causal", action="store_true",
                        help="gnn_temporal: causal temporal attention "
                             "(default: full-window)")
    parser.add_argument("--dropout", type=float, default=None)


def build_rollout_config(args, cfg: Config) -> RolloutConfig:
    matrix = getattr(args, "matrix", "v2")
    experiments = MATRICES[matrix]
    exp = args.experiment.upper()
    if exp not in experiments:
        raise SystemExit(f"unknown experiment {exp!r} for matrix {matrix} "
                         f"(choices: {list(experiments)})")
    rc = RolloutConfig(base=cfg, experiment=exp, model=args.model,
                       scale=args.scale, version=matrix,
                       **experiments[exp])
    if args.surrogate is not None:
        rc.surrogate = args.surrogate
    if args.gamma is not None:
        rc.gamma = args.gamma
    if args.base_lr is not None:
        rc.base_lr = args.base_lr
    if args.buffer_capacity is not None:
        rc.buffer_capacity = args.buffer_capacity
    if args.tbptt is not None:
        rc.tbptt = args.tbptt
    if args.grad_ckpt is not None:
        rc.grad_ckpt = args.grad_ckpt
    # per-stage DAgger mix override, e.g. --dagger-mix 0.2,0.3,0.5,0.7
    dmix = getattr(args, "dagger_mix", None)
    if dmix:
        rc.dagger_mix_override = tuple(float(x) for x in dmix.split(","))
    # tangent / local-stability loss
    if getattr(args, "tangent", False):
        rc.tangent = True
    if getattr(args, "lambda_tangent", None) is not None:
        rc.lambda_tangent = args.lambda_tangent
    if getattr(args, "tangent_sigma", None) is not None:
        rc.tangent_sigma = args.tangent_sigma
    if getattr(args, "tangent_scales", None):
        rc.tangent_scales = tuple(float(x) for x in args.tangent_scales.split(","))
    if getattr(args, "lambda_perturbed", None) is not None:
        rc.lambda_perturbed = args.lambda_perturbed
    # v4 T-entries carry model_kwargs in EXPERIMENTS_V4 (k_hist ablation) —
    # do NOT clobber them with CLI defaults; only build from CLI when the
    # experiment left model_kwargs empty.
    if args.model == "gnn_temporal" and not rc.model_kwargs:
        rc.model_kwargs = {
            "k_hist": args.k_hist or cfg.K,
            **({"t_layers": args.t_layers} if args.t_layers else {}),
            **({"t_heads": args.t_heads} if args.t_heads else {}),
            **({"pos_type": args.pos} if args.pos else {}),
            **({"causal": True} if getattr(args, "causal", False) else {}),
            **({"dropout": args.dropout} if args.dropout is not None else {}),
        }
    if rc.scale == "small":
        rc.pop_groups = 5
        rc.dagger_collect_traj = min(rc.dagger_collect_traj, 32)
    if getattr(args, "smoke", False):
        rc.stages = smoke_stages(rc.scale)
    elif matrix == "v3":
        rc.stages = stages_v3(rc.scale)
    elif matrix == "v3r":
        rc.stages = stages_v3r(rc.scale, exp)
    elif matrix == "v4":
        rc.stages = stages_v4(rc.scale)
    else:
        rc.stages = default_stages(rc.scale)
    if args.stages:
        keep = set(args.stages.split(","))
        rc.stages = [s for s in rc.stages if s.name in keep]
        if not rc.stages:
            raise SystemExit(f"no stages matched --stages {args.stages}")
    if matrix == "v2" and (exp in ("B",) or not rc.scheduled_sampling):
        # v2 unroll-only experiments never teacher-force inside the unroll
        rc.stages = [replace(s, teacher_ratio=0.0) for s in rc.stages]
    if not rc.noise:
        rc.stages = [replace(s, noise_sigma=0.0) for s in rc.stages]
    if not rc.dagger:
        rc.stages = [replace(s, dagger_mix=0.0) for s in rc.stages]
    return rc
