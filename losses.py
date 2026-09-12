"""Horizon-weighted multi-step losses for phase-2 unrolled training.

multistep_loss supervises an unrolled trajectory: for each step h it compares
the COMPOSED state (v, sp, r from compose_step_learned /
compose_step_mechanistic) and the raw model outputs against the recorded
simulator state, then averages over steps with geometric weights
w_h = gamma^(h-1), normalised by their sum.

Mechanistic-mode spike loss: the main BCE goes to s_logits_aux (the base
model's own spike channel), which keeps learning calibrated spike
probabilities; a 0.5-weighted BCE is added on the mechanistic s_logits (the
(v_pre - v_th) * logit_scale margin) so dv also receives direct threshold
supervision. The mechanistic s_logits path otherwise only carries the
surrogate gradient, whose scale is not BCE-calibrated, which is why it is
not the main term.

Component reporting follows metrics.compute_loss conventions: parts are the
UNWEIGHTED per-component losses (horizon-weighted means), while parts["loss"]
is the lambda-weighted total. Macro terms (rate/pop/delta) contribute only
when rc.macro_loss is True; they are reported as exact zeros otherwise.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def population_groups(n_neurons: int, n_groups: int, device) -> torch.Tensor:
    """[G, N] 0/1 membership, contiguous chunks along the ring (ring-local
    populations; no biological labels available)."""
    idx = torch.arange(n_neurons, device=device)
    gid = idx * n_groups // n_neurons
    groups = torch.zeros(n_groups, n_neurons, device=device)
    groups[gid, idx] = 1.0
    return groups


def focal_bce_with_logits(logits: torch.Tensor, targets: torch.Tensor,
                          pos_weight: torch.Tensor, gamma: float = 2.0):
    """Focal loss with per-class alpha from pos_weight (alpha = pw/(1+pw)).

    FL = -alpha_t * (1 - p_t)^gamma * log(p_t), mean-reduced. Numerically
    stable: the log(p_t) term is the standard BCE-with-logits (computed
    from logits, never from a materialised probability) and the modulating
    factor uses clamped probabilities only inside pow.
    """
    if not torch.is_tensor(pos_weight):
        pos_weight = torch.tensor(float(pos_weight), device=logits.device,
                                  dtype=logits.dtype)
    alpha = pos_weight / (1.0 + pos_weight)
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p = torch.sigmoid(logits)
    p_t = p * targets + (1.0 - p) * (1.0 - targets)
    a_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    mod = (1.0 - p_t).clamp(min=0.0, max=1.0).pow(gamma)
    return (a_t * mod * bce).mean()


def _spike_loss(logits: torch.Tensor, target: torch.Tensor,
                pos_weight: torch.Tensor, rc) -> torch.Tensor:
    if getattr(rc, "focal", False):
        return focal_bce_with_logits(logits, target, pos_weight,
                                     gamma=getattr(rc, "focal_gamma", 2.0))
    return F.binary_cross_entropy_with_logits(logits, target,
                                              pos_weight=pos_weight)


def multistep_loss(step_outs: list[dict],        # len U, raw model outputs
                   step_states: list[tuple],     # len U, composed (v, sp, r)
                   targets: torch.Tensor,        # [B, U+1, N, 3]; index 0 =
                                                 # last TRUE context state,
                                                 # step u supervised by [:, u+1]
                   rc, pos_weight: torch.Tensor,
                   groups: torch.Tensor | None = None):
    """L = sum_h gamma^(h-1) * L_h / sum_h gamma^(h-1), per-step:
    L_V     = mse(composed v, target v)
    L_spike = BCE-with-logits(out s_logits, target s) [or focal if rc.focal];
              mechanistic mode: BCE(s_logits_aux, s) + 0.5 * BCE(s_logits, s)
    L_R     = mse(out r, target r)  (mechanistic: mse(composed r, target r))
    macro terms only if rc.macro_loss:
    L_rate  = mse(mean composed sp, mean target s)      * lambda_rate
              (per-sample rates over neurons, then mean over batch)
    L_pop   = mse(groups @ composed sp / group_size,    * lambda_pop
                  groups @ target s / group_size)
    L_delta = mse(v_h - v_{h-1}, t_h - t_{h-1})         * lambda_delta
              (h-1 = 0 uses targets[:, 0] v and the FED state passed via
               step_states' predecessor — i.e. v_{-1} := targets[:, 0, :, 0])
    threshold-weighted term only if rc.threshold_loss (phase-3, spec 十六):
    L_thresh = mean(w * (composed v - target v)^2)      * lambda_thresh
               with w = 1 + thresh_alpha * exp(-|v_true - v_th| / thresh_sigma)
               — neurons whose TRUE membrane sits near the threshold, where a
               tiny V error flips a spike decision, dominate the term.
    Returns (total, parts dict with every component + 'loss'; the 'thresh'
    key is present only when rc.threshold_loss is on)."""
    U = len(step_outs)
    if not (len(step_states) == U and targets.shape[1] == U + 1):
        raise ValueError(f"len(step_outs)={U}, len(step_states)="
                         f"{len(step_states)}, targets time dim="
                         f"{targets.shape[1]} (need U and U+1)")
    mechanistic = ("dv" in step_outs[0]) or ("s_logits_aux" in step_outs[0])
    macro = bool(getattr(rc, "macro_loss", False)) and groups is not None
    thresh = bool(getattr(rc, "threshold_loss", False))
    v_th = float(rc.base.v_th) if hasattr(rc, "base") else 1.0

    weights = [rc.gamma ** h for h in range(U)]
    w_norm = sum(weights)
    gsize = groups.sum(dim=1).clamp(min=1.0) if macro else None   # [G]

    total = torch.zeros((), device=targets.device, dtype=targets.dtype)
    comp: dict[str, torch.Tensor] = {
        k: torch.zeros((), device=targets.device, dtype=targets.dtype)
        for k in ("v", "spike", "r", "rate", "pop", "delta")}
    if thresh:
        comp["thresh"] = torch.zeros((), device=targets.device,
                                     dtype=targets.dtype)

    v_prev = targets[:, 0, :, 0]           # v_{-1}: last true context V
    for h in range(U):
        out = step_outs[h]
        v, sp, r = step_states[h]
        tgt = targets[:, h + 1]                                # [B, N, 3]
        tv, ts, tr = tgt[..., 0], tgt[..., 1], tgt[..., 2]
        w = weights[h] / w_norm

        lv = F.mse_loss(v, tv)
        if mechanistic:
            ls = (_spike_loss(out["s_logits_aux"], ts, pos_weight, rc)
                  + 0.5 * _spike_loss(out["s_logits"], ts, pos_weight, rc))
            lr_ = F.mse_loss(r, tr)
        else:
            ls = _spike_loss(out["s_logits"], ts, pos_weight, rc)
            lr_ = F.mse_loss(out["r"], tr)

        l_step = rc.lambda_v * lv + rc.lambda_s * ls + rc.lambda_r * lr_
        comp["v"] = comp["v"] + w * lv
        comp["spike"] = comp["spike"] + w * ls
        comp["r"] = comp["r"] + w * lr_
        if thresh:
            wt = 1.0 + float(getattr(rc, "thresh_alpha", 6.0)) * torch.exp(
                -(tv - v_th).abs() / float(getattr(rc, "thresh_sigma", 0.25)))
            l_thresh = (wt * (v - tv) ** 2).mean()
            l_step = l_step + float(getattr(rc, "lambda_thresh", 1.0)) * l_thresh
            comp["thresh"] = comp["thresh"] + w * l_thresh
        if macro:
            l_rate = F.mse_loss(sp.mean(dim=1), ts.mean(dim=1))
            l_pop = F.mse_loss((sp @ groups.T) / gsize,
                               (ts @ groups.T) / gsize)
            l_delta = F.mse_loss(v - v_prev, tv - targets[:, h, :, 0])
            l_step = (l_step + rc.lambda_rate * l_rate
                      + rc.lambda_pop * l_pop + rc.lambda_delta * l_delta)
            comp["rate"] = comp["rate"] + w * l_rate
            comp["pop"] = comp["pop"] + w * l_pop
            comp["delta"] = comp["delta"] + w * l_delta
        total = total + w * l_step
        v_prev = v                          # composed predecessor for delta

    parts = {"loss": float(total.item())}
    parts.update({k: float(c.item()) for k, c in comp.items()})
    return total, parts


# ----------------------------------------------------------------------
# Tangent / local-stability loss (finite-difference Jacobian matching).
def tangent_loss(model, ctx: torch.Tensor, sim, cfg,
                 sigma: float, gen: torch.Generator,
                 use_ckpt: bool = False) -> torch.Tensor:
    """L_tangent = MSE( F_theta(x+dv) - F_theta(x),  F_LIF(x+dv) - F_LIF(x) )
    over the membrane increment dv only (the spike/refractory response to a
    small V nudge is what drives silent-attractor drift). ctx is the model's
    input window [B, K, N, 4]; its last-step V is nudged by sigma*randn to
    form the perturbed window. Teacher increments come from one simulator
    step on (V, S, R) with the next true stimulus (F_LIF, matching the
    convention F_LIF(x_t, U[t+1])). Returns a scalar; gradient flows only
    into the perturbed model branch."""
    B, K, N, _ = ctx.shape
    dev = ctx.device
    v_t = ctx[:, -1, :, 0]
    s_t = ctx[:, -1, :, 1]
    r_t = ctx[:, -1, :, 2]
    u_next = ctx[:, -1, :, 3]                       # stimulus at t+1

    # teacher increments on (V,S,R) — no_grad
    with torch.no_grad():
        dv = (torch.randn(v_t.shape, generator=gen) * sigma).to(dev)
        v_pert = v_t + dv
        r_un = r_t * float(cfg.refractory_period)
        st_c = sim.simulate(u_next[:, None, :], state0=(v_t, s_t, r_un))[:, 0]
        st_p = sim.simulate(u_next[:, None, :], state0=(v_pert, s_t, r_un))[:, 0]
        d_teacher = st_p[..., 0] - st_c[..., 0]     # [B, N] membrane response

    # model increments: rebuild a window whose last-step V is v_t + dv
    ctx_pert = ctx.clone()
    ctx_pert[:, -1, :, 0] = v_t + dv
    if use_ckpt:
        out_c = torch.utils.checkpoint.checkpoint(model, ctx, use_reentrant=False)
        out_p = torch.utils.checkpoint.checkpoint(model, ctx_pert, use_reentrant=False)
    else:
        out_c = model(ctx)
        out_p = model(ctx_pert)
    d_model = out_p["v"] - out_c["v"]               # [B, N]
    return F.mse_loss(d_model, d_teacher)


# ----------------------------------------------------------------------
# rollout_v4 (spec §12-13): multi-scale tangent + perturbed-state teacher
def tangent_loss_full(model, ctx: torch.Tensor, sim, cfg,
                      sigma: float, gen: torch.Generator,
                      pos_weight: torch.Tensor,
                      use_ckpt: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Tangent consistency PLUS the absolute perturbed-state teacher loss.

    L_tangent   = MSE( F_theta(x+dv) - F_theta(x),  F_LIF(x+dv) - F_LIF(x) )
    L_perturbed = state loss of the PERTURBED branch alone:
                  MSE(v_p, v_teacher_p)
                + BCE(s_logits_p, s_teacher_p; pos_weight)
                + MSE(r_p, r_teacher_p)
    Rationale (spec §13): matching only the Delta response does not pin the
    absolute transition; the perturbed branch must itself land on the
    teacher. Returns (L_tangent, L_perturbed); both differentiable w.r.t.
    the model (teacher under no_grad)."""
    B, K, N, _ = ctx.shape
    dev = ctx.device
    v_t = ctx[:, -1, :, 0]
    s_t = ctx[:, -1, :, 1]
    r_t = ctx[:, -1, :, 2]
    u_next = ctx[:, -1, :, 3]

    with torch.no_grad():
        dv = (torch.randn(v_t.shape, generator=gen) * sigma).to(dev)
        v_pert = v_t + dv
        r_un = r_t * float(cfg.refractory_period)
        st_c = sim.simulate(u_next[:, None, :], state0=(v_t, s_t, r_un))[:, 0]
        st_p = sim.simulate(u_next[:, None, :], state0=(v_pert, s_t, r_un))[:, 0]
        d_teacher = st_p[..., 0] - st_c[..., 0]     # [B, N] membrane response

    ctx_pert = ctx.clone()
    ctx_pert[:, -1, :, 0] = v_t + dv
    if use_ckpt:
        out_c = torch.utils.checkpoint.checkpoint(model, ctx, use_reentrant=False)
        out_p = torch.utils.checkpoint.checkpoint(model, ctx_pert, use_reentrant=False)
    else:
        out_c = model(ctx)
        out_p = model(ctx_pert)
    d_model = out_p["v"] - out_c["v"]
    ltan = F.mse_loss(d_model, d_teacher)
    # absolute perturbed-state teacher loss
    lv = F.mse_loss(out_p["v"], st_p[..., 0])
    ls = F.binary_cross_entropy_with_logits(out_p["s_logits"],
                                            st_p[..., 1], pos_weight=pos_weight)
    lr_ = F.mse_loss(out_p["r"], st_p[..., 2])
    lpert = lv + ls + lr_
    return ltan, lpert


def tangent_sigma_sample(rc, gen: torch.Generator) -> float:
    """Multi-scale sigma (spec §12): draw from rc.tangent_scales with
    rc.tangent_probs; fall back to the single rc.tangent_sigma."""
    scales = tuple(getattr(rc, "tangent_scales", ()) or ())
    if not scales:
        return float(rc.tangent_sigma)
    probs = tuple(getattr(rc, "tangent_probs", (0.4, 0.4, 0.2)))
    probs = (probs + (0.0,) * len(scales))[:len(scales)]
    tot = sum(probs) or 1.0
    r = torch.rand((), generator=gen).item() * tot
    acc = 0.0
    for s, p in zip(scales, probs):
        acc += p
        if r <= acc:
            return float(s)
    return float(scales[-1])
