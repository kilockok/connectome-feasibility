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


def default_stages(scale: str) -> list[StageConfig]:
    """Progressive horizon curriculum (spec: U=8->64, teacher 0.9->0.0,
    LR halving per stage, sigma ramp 1e-4->3e-3, DAgger mix 0.2->0.7)."""
    bs = 16 if scale == "full" else 32
    late_bs = 8 if scale == "full" else 32
    return [
        StageConfig("s1_u8",   8, 15, teacher_ratio=0.90, noise_sigma=1e-4,
                    dagger_mix=0.2, lr_factor=1.0,   n_traj=512, batch_size=bs),
        StageConfig("s2_u16", 16, 20, teacher_ratio=0.70, noise_sigma=3e-4,
                    dagger_mix=0.3, lr_factor=0.5,   n_traj=512, batch_size=bs),
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
    patience: int = 6                   # early stop per stage
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

    stages: list = field(default_factory=list)

    # ------------------------------------------------------------------
    def use_grad_ckpt(self, stage: StageConfig) -> bool:
        if self.grad_ckpt == "on":
            return True
        if self.grad_ckpt == "off":
            return False
        return stage.unroll >= 32

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


def experiment_index(exp: str) -> int:
    return "ABCDEFG".index(exp.upper())


# ----------------------------------------------------------------------
# Path helpers — phase-2 checkpoints NEVER overwrite phase-1 files.
def phase1_ckpt_path(cfg: Config, model: str) -> Path:
    return CHECKPOINT_DIR / f"ckpt_{model}_full_seed{cfg.seed}.pt" \
        if cfg.n_neurons == 1000 else \
        CHECKPOINT_DIR / f"ckpt_{model}_small_seed{cfg.seed}.pt"


def stage_ckpt_path(rc: RolloutConfig, stage: StageConfig) -> Path:
    d = STAGE_CKPT_DIR / rc.experiment
    d.mkdir(parents=True, exist_ok=True)
    return d / (f"ckpt_{rc.model}_{rc.scale}_rollout_v2_"
                f"{rc.experiment}_{stage.name}_seed{rc.base.seed}.pt")


def final_ckpt_path(rc: RolloutConfig) -> Path:
    return CHECKPOINT_DIR / (f"ckpt_{rc.model}_{rc.scale}_rollout_v2_"
                             f"{rc.experiment}_seed{rc.base.seed}.pt")


def history_path(rc: RolloutConfig) -> Path:
    ROLLOUT2_DIR.mkdir(parents=True, exist_ok=True)
    return ROLLOUT2_DIR / (f"history_{rc.model}_{rc.scale}_"
                           f"{rc.experiment}_seed{rc.base.seed}.csv")


def summary_path(rc: RolloutConfig) -> Path:
    ROLLOUT2_DIR.mkdir(parents=True, exist_ok=True)
    return ROLLOUT2_DIR / (f"summary_{rc.model}_{rc.scale}_"
                           f"{rc.experiment}_seed{rc.base.seed}.json")


# ----------------------------------------------------------------------
def add_rollout_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default="gnn",
                        choices=["gnn", "connectome"])
    parser.add_argument("--experiment", default="G",
                        choices=list(EXPERIMENTS))
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


def build_rollout_config(args, cfg: Config) -> RolloutConfig:
    exp = args.experiment.upper()
    rc = RolloutConfig(base=cfg, experiment=exp, model=args.model,
                       scale=args.scale,
                       **EXPERIMENTS[exp])
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
    if rc.scale == "small":
        rc.pop_groups = 5
        rc.dagger_collect_traj = min(rc.dagger_collect_traj, 32)
    rc.stages = smoke_stages(rc.scale) if getattr(args, "smoke", False) \
        else default_stages(rc.scale)
    if args.stages:
        keep = set(args.stages.split(","))
        rc.stages = [s for s in rc.stages if s.name in keep]
        if not rc.stages:
            raise SystemExit(f"no stages matched --stages {args.stages}")
    if exp in ("B",) or not rc.scheduled_sampling:
        # unroll-only experiments never teacher-force inside the unroll
        rc.stages = [replace(s, teacher_ratio=0.0) for s in rc.stages]
    if not rc.noise:
        rc.stages = [replace(s, noise_sigma=0.0) for s in rc.stages]
    if not rc.dagger:
        rc.stages = [replace(s, dagger_mix=0.0) for s in rc.stages]
    return rc
