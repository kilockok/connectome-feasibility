"""v9 Stage 8: candidate fitting with unseen-intervention validation (Gate F).

Observation-legal recursive estimators per candidate family, EQUAL budget
(9 grid points each):
  GAIN : windowed scalar q=gain-1 estimator (window {8,16,32} x lambda {0.5,1,2})
  ADAPT: back-out recursion (beta {0.1,0.2,0.35} x tau_a {8,20,40}), c=0.25
  STP  : per-edge (u,x) filter (U {0.08,0.25,0.5} x (tau_rec,tau_fac)
         {(4,4),(8,8),(16,24)}), norm=1/U
  NULL : base LIF (no grid).
Fit = one-step prediction loss on the PASSIVE segment (t 32..96) of the
delay|32 intervention trajectories; model selection ONLY on the unseen
INTERVENTION segment (t 96..146, prime+probe response). Split B (held-out
params, 64/family) primary; Split C (32/family) secondary. Confusion +
winning margins, per-seed consistency over 5 trajectory subsamples.
"""
import csv
import json
from pathlib import Path
import numpy as np
import torch
from connectome import Connectome
from run_v7 import setup_cfg
from protocol_v9 import ROOT, FAMILIES
from intervention_v9 import IDX, NC
from teachers_v9 import STP_CLUSTERS

TAU = 96
PASSIVE = (32, 96)
ACTIVE = (96, 146)
GAIN_GRID = [(w, l) for w in (8, 16, 32) for l in (0.5, 1.0, 2.0)]
ADAPT_GRID = [(b, t) for b in (0.1, 0.2, 0.35) for t in (8.0, 20.0, 40.0)]
STP_GRID = [(u, trf) for u in (0.08, 0.25, 0.5) for trf in ((4.0, 4.0), (8.0, 8.0), (16.0, 24.0))]
C_ADAPT = 0.25


@torch.no_grad()
def obs_terms(states, stim, cfg, W, ib):
    """Legal per-step quantities from observations only."""
    v, s, r = states[:, :-1, :, 0], states[:, :-1, :, 1], states[:, :-1, :, 2]
    u = stim
    isyn = torch.einsum('btj,ji->bti', s, W)
    current = isyn + u + ib
    refr = r > 0
    vn = torch.where(refr, torch.full_like(v, cfg.v_reset),
                     v + cfg.alpha * (-(v - cfg.v_rest) + current)).clamp(min=cfg.v_min)
    fire_base = (~refr) & (vn >= cfg.v_th)
    free = (~refr) & (~fire_base)
    e = states[:, 1:, :, 0] - vn                     # residual V [B,T,N]
    return dict(v=v, s=s, r=r, u=u, isyn=isyn, refr=refr, vn=vn,
                fire_base=fire_base, free=free, e=e,
                y_v=states[:, 1:, :, 0], y_s=states[:, 1:, :, 1], y_r=states[:, 1:, :, 2])


def hard_predict(vn, cfg):
    """Hard threshold/reset prediction from pre-reset vn."""
    fire = vn >= cfg.v_th
    v_next = torch.where(fire, torch.full_like(vn, cfg.v_reset), vn)
    logit = (vn - cfg.v_th) / 0.1
    return v_next, logit, fire


def seg_loss(terms, vn_pred, t0, t1, cfg):
    """One-step loss on segment [t0,t1) using predicted pre-reset vn."""
    vn = vn_pred[:, t0:t1]
    v_next, logit, _ = hard_predict(vn, cfg)
    yv = terms['y_v'][:, t0:t1]
    ys = terms['y_s'][:, t0:t1]
    refr = terms['refr'][:, t0:t1]
    # refractory neurons are deterministic (v_reset); score only free steps for V
    free = ~refr
    lv = ((v_next - yv) * free).square().sum((1, 2)) / free.sum((1, 2)).clamp(min=1)
    ls = torch.nn.functional.binary_cross_entropy_with_logits(
        logit, ys, reduction='none')
    ls = (ls * free).sum((1, 2)) / free.sum((1, 2)).clamp(min=1)
    return lv + ls


# ---------------------------------------------------------------- estimators
@torch.no_grad()
def gain_vn(terms, cfg, window, lam_scale):
    """Predicted pre-reset vn with windowed scalar q estimator (past only)."""
    e, b, free = terms['e'], cfg.alpha * terms['isyn'], terms['free']
    B, T, N = e.shape
    lam = 1e-3 * lam_scale * (b * b)[(b * b) > 0].mean()
    vn_pred = terms['vn'].clone()
    isyn = terms['isyn']
    for t in range(1, T):
        s0 = max(0, t - window)
        bb = b[:, s0:t] * free[:, s0:t]
        ee = e[:, s0:t]
        num = (bb * ee).sum((1, 2))
        den = (bb * bb).sum((1, 2))
        q = num / (den + lam)
        vn_pred[:, t] = terms['vn'][:, t] + cfg.alpha * q[:, None] * isyn[:, t]
    return vn_pred


@torch.no_grad()
def adapt_vn(terms, cfg, beta, tau_a):
    """Predicted pre-reset vn with back-out adaptation recursion (past only).

    LEGALITY (fixed after a leakage review): the prediction for transition t
    uses ONLY a_prev (assimilated from transitions < t). The completed
    transition t is assimilated AFTER its prediction:
      1. vn_pred[t] = vn[t] - c*a_prev
      2. a_now = where(free[t], -e[t]/c, a_prev)   (assimilate observation)
      3. a_prev = rho*a_now + beta*fire[t]          (state for t+1)
    """
    rho = float(np.exp(-1.0 / tau_a))
    e, free = terms['e'], terms['free']
    B, T, N = e.shape
    a_prev = torch.zeros(B, N, device=e.device)
    vn_pred = terms['vn'].clone()
    for t in range(T):
        vn_pred[:, t] = terms['vn'][:, t] - C_ADAPT * a_prev * free[:, t]
        a_now = torch.where(free[:, t], -e[:, t] / C_ADAPT, a_prev)
        a_prev = rho * a_now + beta * terms['y_s'][:, t]
    return vn_pred


@torch.no_grad()
def stp_vn(terms, cfg, conn, U0, taus):
    """Predicted pre-reset vn with per-edge (u,x) filter (past spikes only)."""
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
    vn_pred = terms['vn'].clone()
    for t in range(T):
        g = u * x * norm
        isyn = torch.einsum('bj,bji->bi', terms['s'][:, t], W * g)
        current = isyn + terms['u'][:, t] + (conn.i_bias.to(dev) if conn.i_bias is not None else 0.0)
        vn = torch.where(terms['refr'][:, t], terms['vn'][:, t],
                         terms['v'][:, t] + cfg.alpha * (-(terms['v'][:, t] - cfg.v_rest) + current)
                         ).clamp(min=cfg.v_min)
        vn_pred[:, t] = vn
        fire = terms['y_s'][:, t][:, :, None] * mask[None]
        u_new = (u + U0 * (1 - u) * fire).clamp(max=1.0)
        x_new = (x - u_new * x * fire).clamp(0.0, 1.0)
        u = U0 + (u_new - U0) * rho_fac
        x = 1.0 + (x_new - 1.0) * rho_rec
    return vn_pred


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    cfg = setup_cfg()
    conn = Connectome.generate(cfg)
    zcache = ROOT / 'data' / 'intervention_branches.pt'
    all_branches = torch.load(zcache, map_location='cpu', weights_only=False)
    from dataset import sample_traj_params, build_stimulus
    from intervention_v9 import build_extra_stim
    from intervention_v9 import out_neighbors, J
    Inb = out_neighbors(conn, J).numpy()
    g_amp = float(np.load(ROOT / 'data' / 'intervention_phi.npz')['g_amp'])

    def stim_for(cond, n):
        seeds = [cfg.traj_seed('test_seen', IDX[cond] + i) for i in range(n)]
        us = []
        for s in seeds:
            g = torch.Generator().manual_seed(int(s))
            us.append(build_stimulus(sample_traj_params(g, cfg, 'test_seen'), cfg))
        es = torch.stack([build_extra_stim(cfg, 'delay', 32, None, Inb, g_amp) for _ in range(n)])
        return (torch.stack(us) + es)

    rows = []
    for cond in ('testB', 'testC'):
        for fam in FAMILIES + ('null',):
            if fam == 'null' and cond != 'testB':
                continue
            key = (fam if fam != 'null' else 'null', cond if fam != 'null' else 'train',
                   'delay|32|None')
            states = all_branches[key].cuda()
            n = len(states)
            stim = stim_for('testB' if fam == 'null' else cond, n).cuda()
            terms = obs_terms(states, stim, cfg, conn.dense_weight('cuda'),
                              conn.i_bias.cuda() if conn.i_bias is not None else torch.zeros(cfg.n_neurons, device='cuda'))
            scores = {}
            # NULL candidate
            lp = seg_loss(terms, terms['vn'], *PASSIVE, cfg)
            la = seg_loss(terms, terms['vn'], *ACTIVE, cfg)
            scores['null'] = [(lp, la, None)]
            for gi, (w, l) in enumerate(GAIN_GRID):
                vn = gain_vn(terms, cfg, w, l)
                scores.setdefault('gain', []).append(
                    (seg_loss(terms, vn, *PASSIVE, cfg), seg_loss(terms, vn, *ACTIVE, cfg), (w, l)))
            for beta, ta in ADAPT_GRID:
                vn = adapt_vn(terms, cfg, beta, ta)
                scores.setdefault('adapt', []).append(
                    (seg_loss(terms, vn, *PASSIVE, cfg), seg_loss(terms, vn, *ACTIVE, cfg), (beta, ta)))
            for U0, taus in STP_GRID:
                vn = stp_vn(terms, cfg, conn, U0, taus)
                scores.setdefault('stp', []).append(
                    (seg_loss(terms, vn, *PASSIVE, cfg), seg_loss(terms, vn, *ACTIVE, cfg), (U0, taus)))
            # per-trajectory: select grid on passive, family on active
            cands = ['null', 'gain', 'adapt', 'stp']
            bestP, bestA = {}, {}
            for c in cands:
                lp = torch.stack([s[0] for s in scores[c]])      # [grid, B]
                la = torch.stack([s[1] for s in scores[c]])
                gi = lp.argmin(0)
                bestP[c] = lp[gi, torch.arange(n)]
                bestA[c] = la[gi, torch.arange(n)]
                rows += [dict(stage='candidate_grid', split=cond, family=fam, candidate=c,
                              grid=str(scores[c][int(gi[i])][2]) if scores[c][0][2] is not None else 'null',
                              traj=i, passive_loss=float(bestP[c][i]), active_loss=float(bestA[c][i]))
                         for i in range(n)]
            A = torch.stack([bestA[c] for c in cands])           # [cand, B]
            P = torch.stack([bestP[c] for c in cands])
            winA = A.argmin(0)
            winP = P.argmin(0)
            Asorted = A.sort(0).values
            margin = Asorted[1] - Asorted[0]
            for i in range(n):
                rows.append(dict(stage='candidate_select', split=cond, family=fam,
                                 candidate=cands[int(winA[i])], traj=i,
                                 passive_loss=float(P[winA[i], i]), active_loss=float(A[winA[i], i]),
                                 margin=float(margin[i]),
                                 winner_passive=cands[int(winP[i])],
                                 correct=int(cands[int(winA[i])] == fam)))
            print('candfit', cond, fam, 'acc', float((winA == cands.index(fam)).float().mean()),
                  flush=True)
            del states, stim, terms
            torch.cuda.empty_cache()
    path = ROOT / 'metrics' / 'candidate_fitting.csv'
    fields = ['stage', 'split', 'family', 'candidate', 'grid', 'traj',
              'passive_loss', 'active_loss', 'margin', 'winner_passive', 'correct']
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print('CANDIDATE FITTING COMPLETE', len(rows), flush=True)


if __name__ == '__main__':
    main()

