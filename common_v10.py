"""v10 core: artifact-free candidate machinery.

All predictions use the HARD LIF transition (NULL == base bitwise, see
audit/artifact_audit.md). Estimators are past-only (v9 leakage-fixed
ordering). Candidate set: NULL, GAIN (windowed scalar q filter), ADAPT
(back-out recursion), STP (per-edge (u,x) filter).

Two prediction modes:
  assimilate: teacher-forced one-step predictions on OBSERVED states
              (used for passive fitting and post-intervention scoring);
  rollout   : open-loop hard rollout from a context state under a future
              stimulus segment (used for DESIGN-phase predictive
              disagreement - no future observations available there).
Scoring: information-bearing masked Gaussian NLL (V, per-candidate sigma
from passive residuals) + spike BCE - proper log-likelihood up to const.
"""
import math
import numpy as np
import torch

GAIN_GRID = [(w, l) for w in (8, 16, 32) for l in (0.5, 1.0, 2.0)]
ADAPT_GRID = [(b, t) for b in (0.1, 0.2, 0.35) for t in (8.0, 20.0, 40.0)]
STP_GRID = [(u, trf) for u in (0.08, 0.25, 0.5)
            for trf in ((4.0, 4.0), (8.0, 8.0), (16.0, 24.0))]
CANDIDATES = ('null', 'gain', 'adapt', 'stp')
C_ADAPT = 0.25
TOPK = 2


# ------------------------------------------------------------------ obs terms
@torch.no_grad()
def obs_terms(states, stim, cfg, W, ib):
    v, s, r = states[:, :-1, :, 0], states[:, :-1, :, 1], states[:, :-1, :, 2]
    isyn = torch.einsum('btj,ji->bti', s, W)
    current = isyn + stim + ib
    refr = r > 0
    vn = torch.where(refr, torch.full_like(v, cfg.v_reset),
                     v + cfg.alpha * (-(v - cfg.v_rest) + current)).clamp(min=cfg.v_min)
    fire_base = (~refr) & (vn >= cfg.v_th)
    free = (~refr) & (~fire_base)
    med = isyn.abs().flatten(1).median(1).values[:, None, None]
    info = free & (isyn.abs() > med)
    return dict(v=v, s=s, r=r, u=stim, isyn=isyn, refr=refr, vn=vn,
                fire_base=fire_base, free=free, info=info,
                e=states[:, 1:, :, 0] - vn,
                y_v=states[:, 1:, :, 0], y_s=states[:, 1:, :, 1], y_r=states[:, 1:, :, 2])


def hard_predict(vn, cfg):
    fire = vn >= cfg.v_th
    v_next = torch.where(fire, torch.full_like(vn, cfg.v_reset), vn)
    return v_next, (vn - cfg.v_th) / 0.1, fire


# ------------------------------------------------------------------ filters (past-only)
@torch.no_grad()
def gain_q(terms, cfg, window, lam_scale):
    """q_hat[t] from transitions [t-w, t); [B,T]."""
    e, b, free = terms['e'], cfg.alpha * terms['isyn'], terms['free']
    B, T, N = e.shape
    pos = (b * b)[(b * b) > 0]
    lam = 1e-3 * lam_scale * (pos.mean() if pos.numel() else torch.tensor(1e-6, device=b.device))
    q = torch.zeros(B, T, device=e.device)
    for t in range(1, T):
        s0 = max(0, t - window)
        bb = b[:, s0:t] * free[:, s0:t]
        num = (bb * e[:, s0:t]).sum((1, 2))
        den = (bb * bb).sum((1, 2))
        q[:, t] = num / (den + lam)
    return q


@torch.no_grad()
def adapt_a(terms, beta, tau_a):
    """a_hat[t] assimilated from transitions < t (used for prediction at t)."""
    rho = float(np.exp(-1.0 / tau_a))
    e, free = terms['e'], terms['free']
    B, T, N = e.shape
    a_prev = torch.zeros(B, N, device=e.device)
    a_hat = torch.zeros(B, T, N, device=e.device)
    for t in range(T):
        a_hat[:, t] = a_prev
        a_now = torch.where(free[:, t], -e[:, t] / C_ADAPT, a_prev)
        a_prev = rho * a_now + beta * terms['y_s'][:, t]
    return a_hat


@torch.no_grad()
def stp_g(terms, cfg, conn, U0, taus):
    """Per-edge effective gain g[t] from spikes < t; returns [B,T,N,N] would
    be huge, so return the current-needed products: isyn_pred[t] = s_t @ (W*g_t)
    and the final (u,x) state for continuation."""
    B, T, N = terms['s'].shape
    dev = terms['s'].device
    W = conn.dense_weight(dev)
    mask = (W != 0).float()
    tr, tf = taus
    rho_rec = float(np.exp(-1.0 / tr))
    rho_fac = float(np.exp(-1.0 / tf))
    u = torch.full((B, N, N), U0, device=dev)
    x = torch.ones(B, N, N, device=dev)
    norm = 1.0 / U0
    isyn_pred = torch.zeros(B, T, N, device=dev)
    for t in range(T):
        g = u * x * norm
        isyn_pred[:, t] = torch.einsum('bj,bji->bi', terms['s'][:, t], W * g)
        fire = terms['y_s'][:, t][:, :, None] * mask[None]
        u_new = (u + U0 * (1 - u) * fire).clamp(max=1.0)
        x_new = (x - u_new * x * fire).clamp(0.0, 1.0)
        u = U0 + (u_new - U0) * rho_fac
        x = 1.0 + (x_new - 1.0) * rho_rec
    return isyn_pred, (u, x)


# ------------------------------------------------------------------ assimilate & score
@torch.no_grad()
def candidate_vn(terms, cfg, conn, cand, theta, seg=None):
    """Predicted pre-reset vn for every transition (teacher-forced, past-only)."""
    vn = terms['vn']
    if cand == 'null':
        return vn
    if cand == 'gain':
        q = gain_q(terms, cfg, theta[0], theta[1])
        return vn + cfg.alpha * q[..., None] * terms['isyn']
    if cand == 'adapt':
        a = adapt_a(terms, theta[0], theta[1])
        return vn - C_ADAPT * a * terms['free']
    if cand == 'stp':
        isyn_pred, _ = stp_g(terms, cfg, conn, theta[0], theta[1])
        current = isyn_pred + terms['u'] + (conn.i_bias.to(vn.device)
                                            if conn.i_bias is not None else 0.0)
        out = torch.where(terms['refr'], vn,
                          terms['v'] + cfg.alpha * (-(terms['v'] - cfg.v_rest) + current)
                          ).clamp(min=cfg.v_min)
        return out
    raise ValueError(cand)


@torch.no_grad()
def masked_nll(terms, vn_pred, cfg, t0, t1, sigma2):
    """Info-bearing masked Gaussian NLL (V) + BCE (spikes) per trajectory."""
    vn = vn_pred[:, t0:t1]
    v_next, logit, _ = hard_predict(vn, cfg)
    yv = terms['y_v'][:, t0:t1]
    ys = terms['y_s'][:, t0:t1]
    info = terms['info'][:, t0:t1]
    free = terms['free'][:, t0:t1]
    s2 = max(sigma2, 1e-8)
    nll_v = (0.5 * (v_next - yv) ** 2 / s2) * info
    nll_v = nll_v.sum((1, 2)) / info.sum((1, 2)).clamp(min=1)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logit, ys, reduction='none')
    bce = (bce * free).sum((1, 2)) / free.sum((1, 2)).clamp(min=1)
    return nll_v + bce + 0.5 * math.log(s2)


@torch.no_grad()
def fit_theta_bank(terms, cfg, conn, cand, t0, t1):
    """Grid fit on passive segment; returns top-K (theta, passive_loss, sigma2)."""
    if cand == 'null':
        e = terms['e'][:, t0:t1]
        info = terms['info'][:, t0:t1]
        s2 = float((e[info].square().mean()) if info.any() else 1e-4)
        loss = masked_nll(terms, terms['vn'], cfg, t0, t1, s2)
        return [(None, loss, s2)]
    grids = dict(gain=GAIN_GRID, adapt=ADAPT_GRID, stp=STP_GRID)[cand]
    scores = []
    for theta in grids:
        vn = candidate_vn(terms, cfg, conn, cand, theta)
        # sigma from this candidate's own info-bearing residuals
        v_next, _, _ = hard_predict(vn[:, t0:t1], cfg)
        yv = terms['y_v'][:, t0:t1]
        info = terms['info'][:, t0:t1]
        res = (v_next - yv)[info]
        s2 = float(res.square().mean()) if res.numel() else 1e-4
        loss = masked_nll(terms, vn, cfg, t0, t1, s2)
        scores.append((theta, loss, s2))
    B = scores[0][1].shape[0]
    per_traj = []
    for b in range(B):
        ss = sorted(scores, key=lambda z: float(z[1][b]))
        per_traj.append(ss[:TOPK])
    # return list over grid of (theta, loss[B], s2) but banked: keep full grid
    return scores


# ------------------------------------------------------------------ rollout (design)
@torch.no_grad()
def rollout_candidate(x0, stim_seg, cfg, conn, cand, theta, est, q_const=None):
    """Open-loop hard rollout from state x0 [B,N,3] under stim_seg [B,W,N].
    est carries estimator state at rollout start:
      gain: q_const [B]; adapt: a [B,N]; stp: (u [B,N,N], x [B,N,N]).
    Returns V series [B,W+1,N] and S series [B,W,N] (hard)."""
    B, W_, N = stim_seg.shape
    dev = stim_seg.device
    W = conn.dense_weight(dev)
    ib = conn.i_bias.to(dev) if conn.i_bias is not None else torch.zeros(N, device=dev)
    x = x0.clone()
    if cand == 'adapt':
        a = est.clone()
        rho = float(np.exp(-1.0 / theta[1]))
    elif cand == 'stp':
        u, xx = est
        mask = (W != 0).float()
        tr, tf = theta[1]
        rho_rec = float(np.exp(-1.0 / tr))
        rho_fac = float(np.exp(-1.0 / tf))
        norm = 1.0 / theta[0]
    vs = [x[..., 0]]
    ss = []
    for t in range(W_):
        v, s, r = x.unbind(-1)
        if cand == 'stp':
            g = u * xx * norm
            current = torch.einsum('bj,bji->bi', s, W * g) + stim_seg[:, t] + ib
        else:
            current = s @ W + stim_seg[:, t] + ib
        if cand == 'gain':
            current = current * (1.0 + q_const)[:, None]
        refr = r > 0
        vn = torch.where(refr, torch.full_like(v, cfg.v_reset),
                         v + cfg.alpha * (-(v - cfg.v_rest) + current)).clamp(min=cfg.v_min)
        if cand == 'adapt':
            vn = vn - C_ADAPT * a * (~refr).float()
        fire = (~refr) & (vn >= cfg.v_th)
        vn = torch.where(fire, torch.full_like(vn, cfg.v_reset), vn)
        rn = torch.where(fire, torch.full_like(r, float(cfg.refractory_period)),
                         (r * cfg.refractory_period - 1).clamp(min=0)) / cfg.refractory_period
        x = torch.stack((vn, fire.float(), rn), -1)
        if cand == 'adapt':
            a = rho * a + theta[0] * fire.float()
        if cand == 'stp':
            fire_m = fire.float()[:, :, None] * mask[None]
            u_new = (u + theta[0] * (1 - u) * fire_m).clamp(max=1.0)
            x_new = (xx - u_new * xx * fire_m).clamp(0.0, 1.0)
            u = theta[0] + (u_new - theta[0]) * rho_fac
            xx = 1.0 + (x_new - 1.0) * rho_rec
        vs.append(x[..., 0])
        ss.append(x[..., 1])
    return torch.stack(vs, 1), torch.stack(ss, 1)
