"""Surrogate-gradient spike nonlinearity + differentiable state composition.

Phase-2 trains through autonomous unrolls, so the hard spike/reset rules of
rollout.py need differentiable counterparts:

  - hard_spike: forward is the Heaviside on the margin; backward is a
    surrogate gradient ('ste', 'sigmoid' or 'fast_sigmoid').
  - compose_step_learned: rollout.py's composition (clamp, hard reset,
    refractory V-hold) with a surrogate spike.
  - compose_step_mechanistic: the deterministic LIF update given a predicted
    membrane increment dv (used by MechanisticWrapper); the only learned
    quantity is dv, which receives gradients through v_pre (non-fired
    neurons) and through the surrogate spike.

fast_sigmoid normalisation: the classic fast-sigmoid derivative
1 / (1 + beta*|margin|)^2 peaks at 1 at margin=0; we rescale it by beta/4
so its peak equals beta/4 — the peak of the 'sigmoid' surrogate — keeping
the gradient scale comparable across modes. Unlike the sigmoid surrogate
(exponential decay), the polynomial decay keeps a non-negligible gradient
for far-from-threshold neurons, which matters early in fine-tuning when
margins are large.
"""
from __future__ import annotations

import math

import torch

from rollout import REFR_HOLD

SURROGATE_MODES = ("ste", "sigmoid", "fast_sigmoid")


class _HardSpike(torch.autograd.Function):
    """Heaviside forward, surrogate backward; mode/beta are static args.

    Under torch.no_grad (or a margin that does not require grad) no graph is
    built and the call reduces to the plain hard threshold.
    """

    @staticmethod
    def forward(ctx, margin: torch.Tensor, mode: str, beta: float):
        ctx.save_for_backward(margin)
        ctx.mode = mode
        ctx.beta = beta
        return (margin > 0).to(margin.dtype)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (margin,) = ctx.saved_tensors
        beta = ctx.beta
        if ctx.mode == "ste":
            s = torch.sigmoid(margin)
            g = s * (1.0 - s)
        elif ctx.mode == "sigmoid":
            s = torch.sigmoid(beta * margin)
            g = beta * s * (1.0 - s)
        else:  # fast_sigmoid
            g = (beta / 4.0) / (1.0 + beta * margin.abs()).pow(2)
        return grad_out * g, None, None


def hard_spike(margin: torch.Tensor, mode: str = "ste",
               beta: float = 10.0) -> torch.Tensor:
    """Forward: (margin > 0).float(). Backward per mode:
    'ste'          straight-through on sigmoid(margin)
    'sigmoid'      beta * sigmoid'(beta * margin)
    'fast_sigmoid' beta-normalised 1 / (1 + beta*|margin|)^2
    Modes are switched by a plain string; safe under torch.no_grad too
    (returns the hard value, builds no graph)."""
    if mode not in SURROGATE_MODES:
        raise ValueError(f"unknown surrogate mode {mode!r} "
                         f"(expected one of {SURROGATE_MODES})")
    return _HardSpike.apply(margin, mode, beta)


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def compose_step_learned(out: dict, cfg, threshold: float = 0.5,
                         mode: str = "ste", beta: float = 10.0):
    """Differentiable version of rollout.py's hard composition.

    out: model output dict (v, s_logits, r). The spike margin is
    s_logits - logit(threshold), so the forward spike equals
    sigmoid(s_logits) > threshold exactly as in rollout.py.
    Returns (v, sp, r): v clamped to [v_min, 3*v_th]; spike via hard_spike;
    fired -> v=v_reset & r=1; (~fired) & (r > REFR_HOLD) -> v=v_reset
    (the refractory V-hold that train._compose_straight_through lacks);
    r clamped to [0, 1]. Boolean where-masks are detached (as train.py
    does); the returned sp keeps its surrogate gradient, and v/r keep their
    straight-through gradients on the non-fired / non-held branches.
    """
    v = out["v"].clamp(cfg.v_min, cfg.v_th * 3.0)
    sp = hard_spike(out["s_logits"] - _logit(threshold), mode, beta)
    r = out["r"].clamp(0.0, 1.0)
    fired = sp.detach() > 0.5
    v = torch.where(fired, torch.full_like(v, cfg.v_reset), v)
    r = torch.where(fired, torch.ones_like(r), r)
    # as rollout.py: hold is evaluated on the post-reset r; for ~fired
    # neurons it is unchanged, so this is the clamped model r.
    hold = (~fired) & (r.detach() > REFR_HOLD)
    v = torch.where(hold, torch.full_like(v, cfg.v_reset), v)
    return v, sp, r


def compose_step_mechanistic(dv: torch.Tensor, v_t: torch.Tensor,
                             r_t: torch.Tensor, cfg, mode: str = "ste",
                             beta: float = 10.0):
    """Deterministic LIF mechanics given predicted membrane increment dv.

    Mirrors LIFSimulator.simulate's update exactly (r_t is the normalised
    refractory feature, r_t = R_remaining / refractory_period):
        refr = r_t > 1e-6  (simulator: R > 0; R is integer-valued)
        v_pre = clamp(v_t + dv, min=v_min)        (non-refractory branch)
        fire  = hard_spike(v_pre - v_th)          (surrogate grad into dv)
        refr  -> v=v_reset, sp=0
        fire  -> v=v_reset, r_next=1
        else  -> v=v_pre
        r_next = r_t - 1/period clamped >= 0 for every non-fired neuron
    Returns (v_next, sp, r_next, v_pre). All differentiable w.r.t. dv:
    sp keeps the surrogate gradient, v_next keeps the identity gradient on
    the non-fired/non-refractory branch; the fire/refractory masks used in
    the v/r composition are detached.
    """
    refr = r_t > 1e-6
    v_pre = (v_t + dv).clamp(min=cfg.v_min)
    fire = hard_spike(v_pre - cfg.v_th, mode, beta)
    fired = (fire.detach() > 0.5) & ~refr
    r_decay = (r_t - 1.0 / float(cfg.refractory_period)).clamp(min=0.0)
    r_next = torch.where(fired, torch.ones_like(r_t), r_decay)
    v_next = torch.where(fired | refr, torch.full_like(v_pre, cfg.v_reset),
                         v_pre)
    sp = torch.where(refr, torch.zeros_like(fire), fire)
    return v_next, sp, r_next, v_pre
