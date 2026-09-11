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
    Returns (total, parts dict with every component + 'loss')."""
    U = len(step_outs)
    if not (len(step_states) == U and targets.shape[1] == U + 1):
        raise ValueError(f"len(step_outs)={U}, len(step_states)="
                         f"{len(step_states)}, targets time dim="
                         f"{targets.shape[1]} (need U and U+1)")
    mechanistic = ("dv" in step_outs[0]) or ("s_logits_aux" in step_outs[0])
    macro = bool(getattr(rc, "macro_loss", False)) and groups is not None

    weights = [rc.gamma ** h for h in range(U)]
    w_norm = sum(weights)
    gsize = groups.sum(dim=1).clamp(min=1.0) if macro else None   # [G]

    total = torch.zeros((), device=targets.device, dtype=targets.dtype)
    comp: dict[str, torch.Tensor] = {
        k: torch.zeros((), device=targets.device, dtype=targets.dtype)
        for k in ("v", "spike", "r", "rate", "pop", "delta")}

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
