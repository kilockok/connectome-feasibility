"""v10 Stage 5: candidate prediction landscape over the intervention library.

Design contexts: 8 trajectories per family from the v9 TRAIN split (train
params). For each context: passive fit -> theta banks (top-2) -> open-loop
candidate rollouts from the TAU state under every library intervention ->
predictive disagreement D(M_i,M_j;d) marginalized over the theta bank.

Objectives (preregistered):
  U_pair(d) = min over candidate pairs of mean z-distance
  U_JS(d)   = mean over pairs of symmetric KL (Gaussian predictive,
              sigma from passive residuals + theta-bank spread)
  U_robust  = Q0.1 over design contexts of U_JS - lambda_cost * cost
No test teachers are used anywhere in this stage (design leakage rule).
"""
import csv
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from connectome import Connectome
from run_v7 import setup_cfg
from protocol_v9 import ROOT as ROOT9, FAMILIES
from common_v10 import (CANDIDATES, obs_terms, candidate_vn, hard_predict, masked_nll,
                        rollout_candidate, GAIN_GRID, ADAPT_GRID, STP_GRID, C_ADAPT)
from library_v10 import build_stim, response_window, cost

ROOT = Path('results/latent_state_v10')
TAU = 96
PASSIVE = (32, 96)
LAMBDA_COST = 0.05
N_CTX = 8
GRIDS = dict(gain=GAIN_GRID, adapt=ADAPT_GRID, stp=STP_GRID, null=[None])


@torch.no_grad()
def context_state(states, stim, cfg, conn, cand, theta):
    """Estimator state at TAU for open-loop continuation."""
    terms = obs_terms(states[:, :TAU + 1], stim[:, :TAU], cfg,
                      conn.dense_weight(states.device),
                      conn.i_bias.to(states.device) if conn.i_bias is not None
                      else torch.zeros(states.shape[2], device=states.device))
    if cand == 'gain':
        from common_v10 import gain_q
        q = gain_q(terms, cfg, theta[0], theta[1])
        return q[:, -1], terms
    if cand == 'adapt':
        from common_v10 import adapt_a
        a = adapt_a(terms, theta[0], theta[1])
        return a[:, -1], terms
    if cand == 'stp':
        from common_v10 import stp_g
        _, ux = stp_g(terms, cfg, conn, theta[0], theta[1])
        return ux, terms
    return None, terms


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    cfg = setup_cfg()
    conn = Connectome.generate(cfg)
    dev = torch.device('cuda')
    blob = torch.load(ROOT9 / 'data' / 'v9_data.pt', map_location='cpu', weights_only=False)
    store = blob['store']
    lib = json.loads((ROOT / 'protocol' / 'intervention_library.json').read_text())
    entries = lib['entries']
    Inb = np.array(lib['J'] and [])  # placeholder
    from intervention_v9 import out_neighbors, J
    Inb_full = out_neighbors(conn, J)
    Wcpu = conn.dense_weight('cpu')
    strength = Wcpu[J].abs().sum(0)[Inb_full]
    Inb = Inb_full[strength.argsort(descending=True)].numpy()
    g_amp = float(lib['g_amp'])

    rows = []
    cache = ROOT / 'data' / 'landscape.npz'
    Us = {}
    for fi, fam in enumerate(FAMILIES):
        d = store[f'{fam}/train']
        # pick N_CTX evenly spaced trajectories
        idxs = torch.linspace(0, len(d['states']) - 1, N_CTX).long()
        states = d['states'][idxs].to(dev)
        stim = d['stimulus'][idxs].to(dev)
        # passive fit: choose theta bank per candidate (top-2 by masked NLL on PASSIVE)
        terms_full = obs_terms(states[:, :TAU + 1], stim[:, :TAU], cfg,
                               conn.dense_weight(dev),
                               conn.i_bias.to(dev) if conn.i_bias is not None
                               else torch.zeros(cfg.n_neurons, device=dev))
        banks = {}
        sigmas = {}
        for cand in CANDIDATES:
            scores = []
            for theta in GRIDS[cand]:
                vn = candidate_vn(terms_full, cfg, conn, cand, theta)
                v_next, _, _ = hard_predict(vn[:, PASSIVE[0]:PASSIVE[1]], cfg)
                yv = terms_full['y_v'][:, PASSIVE[0]:PASSIVE[1]]
                info = terms_full['info'][:, PASSIVE[0]:PASSIVE[1]]
                res = (v_next - yv)
                s2 = (res * info).square().sum((1, 2)) / info.sum((1, 2)).clamp(min=1)
                loss = (0.5 * res ** 2 / s2[:, None, None].clamp(min=1e-8) * info).sum((1, 2)) / info.sum((1, 2)).clamp(min=1)
                scores.append((theta, loss, s2))
            # top-2 per context
            banks[cand] = []
            for b in range(N_CTX):
                ss = sorted(scores, key=lambda z: float(z[1][b]))
                banks[cand].append([(s[0], float(s[2][b])) for s in ss[:2]])
            sigmas[cand] = torch.stack([s[2] for s in scores]).min(0).values
        # estimator states are intervention-independent: compute ONCE
        estates = {}
        for cand in CANDIDATES:
            estates[cand] = []
            for b in range(N_CTX):
                per_k = []
                for k in range(2):
                    th = banks[cand][b][min(k, len(banks[cand][b]) - 1)][0]
                    if cand == 'null':
                        per_k.append((th, None))
                        continue
                    st_b, _ = context_state(states[b:b + 1], stim[b:b + 1], cfg, conn, cand, th)
                    per_k.append((th, st_b))
                estates[cand].append(per_k)
        x0s = states[:, TAU]                            # [N_CTX, N, 3]
        # rollout for every intervention
        U_ctx = {e['id']: dict(pair=[], js=[]) for e in entries}
        for e in entries:
            es = torch.stack([build_stim(e, cfg, Inb, g_amp)] * N_CTX).to(dev)
            stim_seg = stim[:, TAU:] + es[:, TAU:]
            t0, t1 = response_window(e)
            w0, w1 = t0 - TAU, t1 - TAU
            preds = {}
            for cand in CANDIDATES:
                mus = []
                for k in range(2):
                    if cand == 'gain':
                        q = torch.stack([estates[cand][b][k][1] for b in range(N_CTX)]).squeeze(1)
                        th = estates[cand][0][k][0]
                        v, s = rollout_candidate(x0s, stim_seg, cfg, conn, cand, th, None, q_const=q)
                    elif cand == 'adapt':
                        a = torch.stack([estates[cand][b][k][1] for b in range(N_CTX)]).squeeze(1)
                        ths = [estates[cand][b][k][0] for b in range(N_CTX)]
                        if len(set(ths)) == 1:
                            v, s = rollout_candidate(x0s, stim_seg, cfg, conn, cand, ths[0], a)
                        else:
                            vs = [rollout_candidate(x0s[b:b + 1], stim_seg[b:b + 1], cfg, conn,
                                                    cand, ths[b], a[b:b + 1])[0] for b in range(N_CTX)]
                            v = torch.cat(vs)
                    elif cand == 'stp':
                        uu = torch.cat([estates[cand][b][k][1][0] for b in range(N_CTX)])
                        xx = torch.cat([estates[cand][b][k][1][1] for b in range(N_CTX)])
                        ths = [estates[cand][b][k][0] for b in range(N_CTX)]
                        if len(set(ths)) == 1:
                            v, s = rollout_candidate(x0s, stim_seg, cfg, conn, cand, ths[0], (uu, xx))
                        else:
                            vs = [rollout_candidate(x0s[b:b + 1], stim_seg[b:b + 1], cfg, conn,
                                                    cand, ths[b], (uu[b:b + 1], xx[b:b + 1]))[0] for b in range(N_CTX)]
                            v = torch.cat(vs)
                    else:
                        v, s = rollout_candidate(x0s, stim_seg, cfg, conn, cand, None, None)
                    mus.append(v[:, w0:w1 + 1])
                mu = torch.stack(mus).mean(0)              # marginalize theta
                var_theta = torch.stack(mus).var(0)
                preds[cand] = (mu, var_theta)
            # pairwise distances on top-responsive neurons
            mu_base = preds['null'][0]
            resp = sum((preds[c][0] - mu_base).abs().mean(1) for c in CANDIDATES if c != 'null')
            topn = resp.mean(0).topk(min(32, resp.shape[1])).indices
            zs, skls = [], []
            for i, ci in enumerate(CANDIDATES):
                for j, cj in enumerate(CANDIDATES):
                    if j <= i:
                        continue
                    mi, mj = preds[ci][0][:, :, topn], preds[cj][0][:, :, topn]
                    si2 = sigmas[ci][:, None, None] + preds[ci][1][:, :, topn] + 1e-6
                    sj2 = sigmas[cj][:, None, None] + preds[cj][1][:, :, topn] + 1e-6
                    z = ((mi - mj).abs() / (si2 + sj2).sqrt()).mean((1, 2))
                    kl = 0.5 * (si2 / sj2 + (mi - mj) ** 2 / sj2 - 1 + torch.log(sj2 / si2))
                    kl = kl + 0.5 * (sj2 / si2 + (mi - mj) ** 2 / si2 - 1 + torch.log(si2 / sj2))
                    skl = (0.5 * kl).clamp(min=0).mean((1, 2))
                    zs.append(z)
                    skls.append(skl)
            zs = torch.stack(zs)          # [pairs, N_CTX]
            skls = torch.stack(skls)
            U_ctx[e['id']]['pair'] = zs.min(0).values.cpu()
            U_ctx[e['id']]['js'] = skls.mean(0).cpu()
            print('landscape', fam, e['id'], flush=True)
        Us[fam] = U_ctx
        del states, stim
        torch.cuda.empty_cache()
    # aggregate: robust objective
    out = []
    for e in entries:
        c = cost(e, 24)
        upair = torch.cat([Us[f][e['id']]['pair'] for f in FAMILIES])
        ujs = torch.cat([Us[f][e['id']]['js'] for f in FAMILIES])
        row = dict(id=e['id'], family=e['family'], cost=c,
                   u_pair_mean=float(upair.mean()), u_pair_q10=float(upair.quantile(0.1)),
                   u_js_mean=float(ujs.mean()), u_js_q10=float(ujs.quantile(0.1)),
                   u_robust=float(ujs.quantile(0.1)) - LAMBDA_COST * c)
        out.append(row)
    path = ROOT / 'metrics' / 'intervention_scores.csv'
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader(); w.writerows(out)
    np.savez(ROOT / 'data' / 'landscape_raw.npz',
             **{f'{fam}|{eid}|pair': Us[fam][eid]['pair'].numpy() for fam in FAMILIES for eid in [e['id'] for e in entries]},
             **{f'{fam}|{eid}|js': Us[fam][eid]['js'].numpy() for fam in FAMILIES for eid in [e['id'] for e in entries]})
    out_sorted = sorted(out, key=lambda r: -r['u_robust'])
    for r in out_sorted[:15]:
        print(f"{r['id']:22s} fam={r['family']:8s} u_robust={r['u_robust']:.4f} "
              f"u_pair_q10={r['u_pair_q10']:.3f} cost={r['cost']:.2f}", flush=True)
    # Figure 2: landscape
    fig, ax = plt.subplots(figsize=(13, 4.5))
    fams = sorted({r['family'] for r in out})
    colors = dict(zip(fams, plt.cm.tab10.colors))
    xs = np.arange(len(out_sorted))
    ax.bar(xs, [r['u_robust'] for r in out_sorted],
           color=[colors[r['family']] for r in out_sorted])
    ax.set_xticks(xs, [r['id'] for r in out_sorted], rotation=75, ha='right', fontsize=6)
    ax.set_ylabel('U_robust = Q0.1(U_JS) - lambda*cost')
    ax.set_title('v10 Figure 2: candidate predictive divergence over intervention space (sorted)')
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[f]) for f in fams]
    ax.legend(handles, fams, fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT / 'figures' / 'fig2_landscape.png', dpi=140)
    print('LANDSCAPE COMPLETE')


if __name__ == '__main__':
    main()
