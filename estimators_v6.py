"""Stage 2: observation-legal q_t = gain_t - 1 estimation.

Scalar windowed estimator and recursive smoother; both read ONLY observable
variables of COMPLETED transitions (V, S, U, known graph, base-LIF form).
Their mechanism knowledge (residual form a*q*I_syn, smoothness prior) is
declared; the z-oracle stays privileged and separate.
"""
import csv
import json
from pathlib import Path
import numpy as np
import torch
from run_latent_v3 import datasets_v3
from run_latent_v2 import setup as setup_v2
from lif_latent_v2 import HiddenStateLIFSimulatorV2

ROOT = Path('results/latent_state_v6')
TAU = 128
DS = (0, 1, 2, 4, 8, 16, 32)
N = 32


@torch.no_grad()
def base_residual_terms(sim, data):
    """Per (traj, t): e_t = V[t+1] - F_base(x_t, u_t) (V channel) and b_t = a*I_syn,t."""
    c = sim.cfg
    S = data['states'][:, :-1, :, 1]
    V = data['states'][:, :-1, :, 0]
    R = data['states'][:, :-1, :, 2]
    U = data['stimulus']
    isyn = S @ sim.W
    current = isyn + U + sim.i_bias
    refr = R > 0
    v_new = torch.where(refr, torch.full_like(V, c.v_reset),
                        V + c.alpha * (-(V - c.v_rest) + current)).clamp(min=c.v_min)
    fire = (~refr) & (v_new >= c.v_th)
    v_new = torch.where(fire, torch.full_like(v_new, c.v_reset), v_new)
    e = data['states'][:, 1:, :, 0] - v_new  # [B,T,N]
    b = c.alpha * isyn                          # [B,T,N]
    free = (~refr) & (~fire)                    # legal residual only here
    return e, b, free


@torch.no_grad()
def scalar_estimates(e, b, free, lam_frac=1e-3, window=16, mask_pct=0.20):
    """q_hat[t] from completed transitions s<=t-1; same-step neuron pooling first,
    then the last `window` information-bearing steps. Mask = weak-I_syn steps
    (per-step b power below the mask_pct quantile of train, caller passes mask_th)."""
    B, T, N = e.shape
    bp = (b * b).sum(-1)
    mask_th = bp.flatten().quantile(1 - mask_pct)  # placeholder, caller overrides
    q = torch.full((B, T), float('nan'), device=e.device)
    lam = lam_frac * bp[bp > 0].mean()
    for t in range(1, T):
        s0 = max(0, t - window)
        bb, ee, ff = b[:, s0:t], e[:, s0:t], free[:, s0:t]
        bb = bb * ff
        num = (bb * ee).sum((1, 2))
        den = (bb * bb).sum((1, 2))
        ok = den > mask_th
        q[:, t] = torch.where(ok, num / (den + lam), torch.full_like(num, float('nan')))
    return q, lam


@torch.no_grad()
def smoother_estimates(e, b, free, kappa=0.3, lam_frac=1e-3):
    """Recursive exponential filter: on information-bearing steps update
    q <- q + kappa*(e/(b+lam/b) - q) aggregated over neurons; else carry over."""
    B, T, N = e.shape
    q = torch.zeros(B, T, device=e.device)
    lam = lam_frac * (b * b)[(b * b) > 0].mean()
    for t in range(1, T):
        bb = b[:, t - 1] * free[:, t - 1]
        ee = e[:, t - 1]
        num = (bb * ee).sum(-1)
        den = (bb * bb).sum(-1)
        ok = den > 0
        inst = num / (den + lam)
        q[:, t] = torch.where(ok, q[:, t - 1] + kappa * (inst - q[:, t - 1]), q[:, t - 1])
    return q


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    cfg, lc, conn = setup_v2('hidden')
    sim = HiddenStateLIFSimulatorV2(conn, cfg, torch.device('cuda'), lc)
    data = datasets_v3('hidden')
    rows = []
    for sp in ('train', 'val', 'test_seen', 'test_ood'):
        e, b, free = base_residual_terms(sim, data[sp])
        # frozen from TRAIN: mask threshold and lambda scale
        rows.append((sp, e, b, free))
    bp_train = (rows[0][2] * rows[0][2]).sum(-1)
    mask_th = bp_train[bp_train > 0].quantile(0.20)
    lam_frac = 1e-3
    true_q = {sp: (torch.tanh(data[sp]['z'][..., 0]) * lc.alpha) for sp, *_ in rows}
    for window in (8, 16, 32):
        for sp, e, b, free in rows:
            B, T, N = e.shape
            q = torch.full((B, T), float('nan'), device=e.device)
            lam = lam_frac * (b * b)[(b * b) > 0].mean()
            for t in range(1, T):
                s0 = max(0, t - window)
                bb = b[:, s0:t] * free[:, s0:t]
                num = (bb * e[:, s0:t]).sum((1, 2))
                den = (bb * bb).sum((1, 2))
                ok = den > mask_th
                q[:, t] = torch.where(ok, num / (den + lam), torch.full_like(num, float('nan')))
            tq = true_q[sp][:, 1:]
            err = (q[:, 1:] - tq).abs()
            info = ~torch.isnan(q[:, 1:])
            mae = float(err[info].mean()) if info.any() else None
            rmse = float(err[info].square().mean().sqrt()) if info.any() else None
            cov = float(info.float().mean())
            rows_out = dict(split=sp, estimator=f'scalar_w{window}', q_mae=mae, q_rmse=rmse,
                            coverage=cov, mask_th=float(mask_th))
            print(rows_out, flush=True)
            with (ROOT / 'metrics_estimators.csv').open('a', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(rows_out))
                if f.tell() == 0:
                    w.writeheader()
                w.writerow(rows_out)
    for kappa in (0.1, 0.2, 0.3, 0.5):
        for sp, e, b, free in rows:
            q = smoother_estimates(e, b, free, kappa)
            tq = true_q[sp]
            err = (q - tq).abs()
            mae = float(err.mean())
            rmse = float(err.square().mean().sqrt())
            rows_out = dict(split=sp, estimator=f'smoother_k{kappa}', q_mae=mae, q_rmse=rmse,
                            coverage=1.0, mask_th=float(mask_th))
            print(rows_out, flush=True)
            with (ROOT / 'metrics_estimators.csv').open('a', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(rows_out))
                if f.tell() == 0:
                    w.writeheader()
                w.writerow(rows_out)


if __name__ == '__main__':
    main()
