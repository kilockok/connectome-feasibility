"""Autoregressive rollout evaluation.

The model is given the true context (X,U)[t0 : t0+K] and the true future
stimulus U[t0+K : t0+K+H] (external drive is a known input), but must feed
back its own predicted states. Metrics are reported at several horizons
from a single max-horizon rollout.

State composition (identical for every model, mirrors the simulator):
  spike = sigmoid(logit) > threshold
  if spike: V = V_reset, R = 1            (hard reset, biological)
  if R > refractory threshold: V = V_reset (refractory hold)
"""
from __future__ import annotations

import torch

REFR_HOLD = 0.15   # normalised refractory above which V is held at reset


@torch.no_grad()
def rollout(model, context: torch.Tensor, future_stim: torch.Tensor,
            cfg, spike_threshold: float = 0.5,
            silence_mask: torch.Tensor | None = None,
            hard_reset: bool | None = None) -> torch.Tensor:
    """context [B,K,N,4], future_stim [B,H,N] -> preds [B,H,N,3]."""
    if hard_reset is None:
        hard_reset = cfg.rollout_hard_reset
    B, K, N, _ = context.shape
    H = future_stim.shape[1]
    hist = context.clone()
    preds = torch.empty(B, H, N, 3, device=context.device)
    for s in range(H):
        out = model(hist)
        v = out["v"].clamp(cfg.v_min, cfg.v_th * 3.0)
        sp = (torch.sigmoid(out["s_logits"]) > spike_threshold).to(v.dtype)
        r = out["r"].clamp(0.0, 1.0)
        if hard_reset:
            fired = sp > 0.5
            v = torch.where(fired, torch.full_like(v, cfg.v_reset), v)
            r = torch.where(fired, torch.ones_like(r), r)
            hold = (~fired) & (r > REFR_HOLD)
            v = torch.where(hold, torch.full_like(v, cfg.v_reset), v)
        if silence_mask is not None:
            sil = silence_mask
            v = torch.where(sil, torch.full_like(v, cfg.v_rest), v)
            sp = torch.where(sil, torch.zeros_like(sp), sp)
            r = torch.where(sil, torch.zeros_like(r), r)
        preds[:, s, :, 0] = v
        preds[:, s, :, 1] = sp
        preds[:, s, :, 2] = r
        feat = torch.stack([v, sp, r, future_stim[:, s]], dim=-1)  # [B,N,4]
        hist = torch.cat([hist[:, 1:], feat.unsqueeze(1)], dim=1)
    return preds


@torch.no_grad()
def naive_rollout(context: torch.Tensor, H: int, cfg) -> torch.Tensor:
    """x[t+1] = x[t] reference: repeat the last context state."""
    last = context[:, -1, :, :3]                      # [B,N,3]
    return last.unsqueeze(1).expand(-1, H, -1, -1).clone()


@torch.no_grad()
def rollout_metrics(pred: torch.Tensor, true: torch.Tensor,
                    horizons) -> dict[int, dict]:
    """pred/true [B, H, N, 3]; metrics per horizon."""
    out = {}
    for h in horizons:
        p, t = pred[:, :h], true[:, :h]
        pv, tv = p[..., 0], t[..., 0]
        ps, ts = p[..., 1], t[..., 1]
        v_rmse = ((pv - tv) ** 2).mean().sqrt().item()

        tp = (ps * ts).sum().item()
        fp = (ps * (1 - ts)).sum().item()
        fn = ((1 - ps) * ts).sum().item()
        prec = tp / max(tp + fp, 1.0)
        rec = tp / max(tp + fn, 1.0)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)

        rate_err = abs(ps.mean().item() - ts.mean().item())
        # population similarity: cosine between per-neuron mean spike rates
        vp = ps.mean(dim=1).flatten(1)                # [B, N]
        vt = ts.mean(dim=1).flatten(1)
        num = (vp * vt).sum(dim=1)
        den = vp.norm(dim=1) * vt.norm(dim=1) + 1e-9
        pop_sim = (num / den).mean().item()
        out[int(h)] = {"v_rmse": v_rmse, "spike_f1": f1,
                       "spike_precision": prec, "spike_recall": rec,
                       "firing_rate_err": rate_err, "pop_similarity": pop_sim,
                       "spike_rate_pred": ps.mean().item(),
                       "spike_rate_true": ts.mean().item()}
    return out


@torch.no_grad()
def run_rollout_eval(model, data: dict, cfg, horizons, spike_threshold=0.5,
                     n_traj=None, t0_offset=0, silence_mask=None,
                     hard_reset=None) -> dict[int, dict]:
    """Average rollout metrics over trajectories of a dataset split.

    Context starts at t0 = t0_offset (default 0), so predictions cover
    [K, K+H). Uses the first `n_traj` trajectories.
    """
    K = cfg.K
    n = n_traj or cfg.n_rollout_traj
    states, stim = data["states"][:n], data["stimulus"][:n]
    H = min(max(horizons), cfg.T - K - t0_offset)
    context_state = states[:, t0_offset:t0_offset + K]              # [B,K,N,3]
    context_stim = stim[:, t0_offset:t0_offset + K].unsqueeze(-1)
    context = torch.cat([context_state, context_stim], dim=-1)      # [B,K,N,4]
    future_stim = stim[:, t0_offset + K:t0_offset + K + H]
    true = states[:, t0_offset + K:t0_offset + K + H]
    if silence_mask is None:
        silence_mask = data.get("silence", None)
        if silence_mask is not None:
            silence_mask = silence_mask[:n]
    pred = rollout(model, context, future_stim, cfg,
                   spike_threshold=spike_threshold,
                   silence_mask=silence_mask, hard_reset=hard_reset)
    horizons = [h for h in horizons if h <= H]
    return rollout_metrics(pred, true, horizons), pred, true
