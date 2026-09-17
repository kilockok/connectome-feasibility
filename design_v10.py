"""v10 Stage 8: single-shot active model discrimination.

Per held-out context (trajectory up to TAU, mechanism params by cohort):
  1. passive fit -> theta banks + per-candidate sigma (shared across policies)
  2. policy selects intervention - WITHOUT using test teachers:
       passive / random / heuristic_stp / heuristic_gain / v9hand /
       optimized_global / optimized_adaptive /
       oracle (true-family rollout vs fitted-others disagreement; upper bound)
  3. teacher executes the intervention (extra_stim from TAU; same seed =>
     bitwise-identical passive segment as the stored v9 trajectory - asserted)
  4. candidates assimilate on the response window; masked NLL; winner+posterior.
Cohorts: testA (seen params), testB (held-out interpolated), testC
(extrapolated). 5 paired seeds (context subsamples; testC has fixed 32).
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from connectome import Connectome
from run_v7 import setup_cfg
from protocol_v9 import ROOT as ROOT9, FAMILIES, SELECTED, spec_for
from teachers_v9 import MechanismLIFSimulator
from common_v10 import (CANDIDATES, obs_terms, candidate_vn, hard_predict,
                        rollout_candidate, GAIN_GRID, ADAPT_GRID, STP_GRID)
from library_v10 import build_stim, response_window
from landscape_v10 import context_state
from intervention_v9 import out_neighbors, J, build_extra_stim, TAU, branch_list

ROOT = Path('results/latent_state_v10')
PASSIVE = (32, 96)
GRIDS = dict(gain=GAIN_GRID, adapt=ADAPT_GRID, stp=STP_GRID, null=[None])
NCTX = 32
COHORT_IDX = dict(testA=0, testB=64, testC=128)
HEURISTIC = dict(heuristic_stp='paired_i8', heuristic_gain='highcur_a5.0')
LAMBDA = 0.05


# ------------------------------------------------------------------ fits
@torch.no_grad()
def passive_fits(states, stim, cfg, conn, dev):
    terms = obs_terms(states[:, :TAU + 1], stim[:, :TAU], cfg,
                      conn.dense_weight(dev),
                      conn.i_bias.to(dev) if conn.i_bias is not None
                      else torch.zeros(states.shape[2], device=dev))
    out = {}
    for cand in CANDIDATES:
        scores = []
        for theta in GRIDS[cand]:
            vn = candidate_vn(terms, cfg, conn, cand, theta)
            v_next, _, _ = hard_predict(vn[:, PASSIVE[0]:PASSIVE[1]], cfg)
            yv = terms['y_v'][:, PASSIVE[0]:PASSIVE[1]]
            info = terms['info'][:, PASSIVE[0]:PASSIVE[1]]
            res = v_next - yv
            s2 = (res * info).square().sum((1, 2)) / info.sum((1, 2)).clamp(min=1)
            loss = (0.5 * res ** 2 / s2[:, None, None].clamp(min=1e-8) * info).sum((1, 2)) \
                   / info.sum((1, 2)).clamp(min=1)
            scores.append((theta, loss, s2))
        bank = []
        for b in range(states.shape[0]):
            ss = sorted(scores, key=lambda z: float(z[1][b]))
            bank.append([(s[0], float(s[2][b])) for s in ss[:2]])
        out[cand] = dict(bank=bank, s2=torch.stack([s[2] for s in scores]).min(0).values)
    return out


@torch.no_grad()
def estates_for(states, stim, cfg, conn, fits):
    est = {}
    B = states.shape[0]
    for cand in CANDIDATES:
        per = []
        for b in range(B):
            ks = []
            for k in range(2):
                th = fits[cand]['bank'][b][min(k, len(fits[cand]['bank'][b]) - 1)][0]
                if cand == 'null':
                    ks.append((None, None)); continue
                st_b, _ = context_state(states[b:b + 1], stim[b:b + 1], cfg, conn, cand, th)
                ks.append((th, st_b))
            per.append(ks)
        est[cand] = per
    return est


# ------------------------------------------------------------------ utilities
@torch.no_grad()
def rollouts_for_entry(states, stim_seg, cfg, conn, fits, est, window):
    """Theta-marginalized candidate rollout means/vars on window -> preds."""
    B = states.shape[0]
    x0s = states[:, TAU]
    w0, w1 = window[0] - TAU, window[1] - TAU
    preds = {}
    for cand in CANDIDATES:
        mus = []
        for k in range(2):
            if cand == 'null':
                v, _ = rollout_candidate(x0s, stim_seg, cfg, conn, 'null', None, None)
            elif cand == 'gain':
                q = torch.stack([est[cand][b][k][1] for b in range(B)]).squeeze(1)
                v, _ = rollout_candidate(x0s, stim_seg, cfg, conn, cand,
                                         est[cand][0][k][0], None, q_const=q)
            elif cand == 'adapt':
                a = torch.stack([est[cand][b][k][1] for b in range(B)]).squeeze(1)
                ths = [est[cand][b][k][0] for b in range(B)]
                if len(set(ths)) == 1:
                    v, _ = rollout_candidate(x0s, stim_seg, cfg, conn, cand, ths[0], a)
                else:
                    v = torch.cat([rollout_candidate(x0s[b:b + 1], stim_seg[b:b + 1], cfg, conn,
                                                     cand, ths[b], a[b:b + 1])[0] for b in range(B)])
            elif cand == 'stp':
                uu = torch.cat([est[cand][b][k][1][0] for b in range(B)])
                xx = torch.cat([est[cand][b][k][1][1] for b in range(B)])
                ths = [est[cand][b][k][0] for b in range(B)]
                if len(set(ths)) == 1:
                    v, _ = rollout_candidate(x0s, stim_seg, cfg, conn, cand, ths[0], (uu, xx))
                else:
                    v = torch.cat([rollout_candidate(x0s[b:b + 1], stim_seg[b:b + 1], cfg, conn,
                                                     cand, ths[b], (uu[b:b + 1], xx[b:b + 1]))[0] for b in range(B)])
            mus.append(v[:, w0:w1 + 1])
        preds[cand] = (torch.stack(mus).mean(0), torch.stack(mus).var(0))
    return preds


def preds_divergence(preds, fits, pair_weights=None):
    """Mean-pair (or posterior-weighted) symmetric KL on top-responsive neurons."""
    mu_base = preds['null'][0]
    resp = sum((preds[c][0] - mu_base).abs().mean(1) for c in CANDIDATES if c != 'null')
    topn = resp.mean(0).topk(min(32, resp.shape[1])).indices
    skls, wts = [], []
    for i, ci in enumerate(CANDIDATES):
        for j, cj in enumerate(CANDIDATES):
            if j <= i:
                continue
            mi, mj = preds[ci][0][:, :, topn], preds[cj][0][:, :, topn]
            si2 = fits[ci]['s2'][:, None, None] + preds[ci][1][:, :, topn] + 1e-6
            sj2 = fits[cj]['s2'][:, None, None] + preds[cj][1][:, :, topn] + 1e-6
            kl = 0.5 * (si2 / sj2 + (mi - mj) ** 2 / sj2 - 1 + torch.log(sj2 / si2)) \
               + 0.5 * (sj2 / si2 + (mi - mj) ** 2 / si2 - 1 + torch.log(si2 / sj2))
            skls.append((0.5 * kl).clamp(min=0).mean((1, 2)))
            if pair_weights is not None:
                wts.append(pair_weights[(ci, cj)])
    S = torch.stack(skls)
    if pair_weights is None:
        return S.mean(0)
    return (S * torch.tensor(wts, device=S.device)[:, None]).sum(0)


@torch.no_grad()
def context_utilities(states, stim, cfg, conn, fits, est, entries, Inb, g_amp, dev):
    from library_v10 import cost as dcost
    B = states.shape[0]
    U = torch.zeros(B, len(entries))
    for ei, e in enumerate(entries):
        es = torch.stack([build_stim(e, cfg, Inb, g_amp)] * B).to(dev)
        stim_seg = stim[:, TAU:] + es[:, TAU:]
        preds = rollouts_for_entry(states, stim_seg, cfg, conn, fits, est, response_window(e))
        U[:, ei] = preds_divergence(preds, fits) - LAMBDA * dcost(e, 24)
    return U


# ------------------------------------------------------------------ scoring
@torch.no_grad()
def score_branch(states_b, stim_b, cfg, conn, bank1, window, dev):
    """bank1: {cand: (theta, s2)} for THIS context. Returns [4] NLL."""
    terms = obs_terms(states_b, stim_b, cfg, conn.dense_weight(dev),
                      conn.i_bias.to(dev) if conn.i_bias is not None
                      else torch.zeros(states_b.shape[2], device=dev))
    out = torch.zeros(4, device=dev)
    for ci, cand in enumerate(CANDIDATES):
        th, s2 = bank1[cand]
        vn = candidate_vn(terms, cfg, conn, cand, th)
        s2 = max(s2, 1e-8)
        v_next, logit, _ = hard_predict(vn[:, window[0]:window[1]], cfg)
        yv = terms['y_v'][:, window[0]:window[1]]
        ys = terms['y_s'][:, window[0]:window[1]]
        info = terms['info'][:, window[0]:window[1]]
        free = terms['free'][:, window[0]:window[1]]
        nll_v = (0.5 * (v_next - yv) ** 2 / s2 * info).sum() / info.sum().clamp(min=1)
        bce = torch.nn.functional.binary_cross_entropy_with_logits(logit, ys, reduction='none')
        bce = (bce * free).sum() / free.sum().clamp(min=1)
        out[ci] = nll_v + bce
    return out


def bank1_of(fits, i):
    return {c: fits[c]['bank'][i][0] for c in CANDIDATES}


def posterior(nll):
    z = -(nll - nll.min(1, keepdim=True).values)
    p = torch.softmax(z, 1)
    ent = -(p * (p + 1e-12).log()).sum(1)
    return p, ent


# ------------------------------------------------------------------ teacher exec
@torch.no_grad()
def execute(chosen, fam, cohort, config_ids, seeds, cfg, conn, Inb, g_amp, dev):
    """Group by (entry, spec); generate branches; return per-context
    (entry_id, branch states/stim). Asserts passive segment equality with
    stored trajectory on the first context."""
    entry_map = {e['id']: e for e in json.loads((ROOT / 'protocol' / 'intervention_library.json').read_text())['entries']}
    groups = defaultdict(list)
    for i, eid in enumerate(chosen):
        groups[(eid, config_ids[i])].append(i)
    out = {}
    checked = False
    for (eid, pid), grp in groups.items():
        e = entry_map[eid]
        es = torch.stack([build_stim(e, cfg, Inb, g_amp)] * len(grp))
        sim = MechanismLIFSimulator(conn, cfg, dev, spec_for(fam, pid))
        dd = sim.generate([seeds[i] for i in grp], 'test_seen', extra_stim=es)
        for bi, i in enumerate(grp):
            out[i] = (eid, dd['states'][bi:bi + 1], dd['stimulus'][bi:bi + 1])
        del sim
        torch.cuda.empty_cache()
    return out


# ------------------------------------------------------------------ main
@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cohorts', nargs='+', default=['testA', 'testB', 'testC'])
    ap.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2, 3, 4])
    args = ap.parse_args()
    torch.set_num_threads(2)
    cfg = setup_cfg()
    conn = Connectome.generate(cfg)
    dev = torch.device('cuda')
    blob = torch.load(ROOT9 / 'data' / 'v9_data.pt', map_location='cpu', weights_only=False)
    store = blob['store']
    lib = json.loads((ROOT / 'protocol' / 'intervention_library.json').read_text())
    entries = lib['entries']
    short_ids = json.loads((ROOT / 'protocol' / 'configs' / 'shortlist.json').read_text())['ids']
    short_entries = [e for e in entries if e['id'] in short_ids]
    Inb_full = out_neighbors(conn, J)
    Wcpu = conn.dense_weight('cpu')
    strength = Wcpu[J].abs().sum(0)[Inb_full]
    Inb = Inb_full[strength.argsort(descending=True)].numpy()
    g_amp = float(lib['g_amp'])
    rows = []

    for cohort in args.cohorts:
        for fam in FAMILIES:
            d = store[f'{fam}/{cohort}']
            n_all = len(d['states'])
            for seed in args.seeds:
                g = np.random.default_rng(10_000 + seed)
                sel = np.arange(n_all) if cohort == 'testC' else \
                    np.sort(g.choice(n_all, size=min(NCTX, n_all), replace=False))
                states = d['states'][sel].to(dev)
                stim = d['stimulus'][sel].to(dev)
                hsum = d['hidden_summary'][sel].to(dev) if 'hidden_summary' in d else None
                B = len(sel)
                seeds = [cfg.traj_seed('test_seen', COHORT_IDX[cohort] + int(i)) for i in sel]
                if cohort == 'testA':
                    pids = [SELECTED[fam]['train'][int(c)][0] for c in d['config_id'][sel]]
                else:
                    pids = [SELECTED[fam]['splitB' if cohort == 'testB' else 'splitC'][0]] * B
                fits = passive_fits(states, stim, cfg, conn, dev)
                # ---------------- choices ----------------
                choices = dict(passive=['passive'] * B)
                gr = np.random.default_rng(20_000 + seed)
                choices['random'] = [short_ids[int(gr.integers(len(short_ids)))] for _ in range(B)]
                for h, eid in HEURISTIC.items():
                    choices[h] = [eid] * B
                choices['optimized_global'] = [short_ids[0]] * B
                est = estates_for(states, stim, cfg, conn, fits)
                U = context_utilities(states, stim, cfg, conn, fits, est, short_entries,
                                      Inb, g_amp, dev)
                choices['optimized_adaptive'] = [short_ids[int(i)] for i in U.argmax(1)]
                # oracle: true-family rollout vs fitted-others divergence
                Uo = oracle_utilities(states, stim, cfg, conn, fits, est, fam, pids,
                                      hsum, short_entries, Inb, g_amp, dev)
                choices['oracle'] = [short_ids[int(i)] for i in Uo.argmax(1)]
                # ---------------- execute + score ----------------
                for pol in ('passive', 'random', 'heuristic_stp', 'heuristic_gain',
                            'optimized_global', 'optimized_adaptive', 'oracle'):
                    chosen = choices[pol]
                    if pol == 'passive':
                        nll = torch.stack([score_branch(states[i:i + 1], stim[i:i + 1], cfg, conn,
                                                        bank1_of(fits, i), (TAU, TAU + 32), dev)
                                           for i in range(B)])
                    else:
                        branches = execute(chosen, fam, cohort, pids, seeds, cfg, conn,
                                           Inb, g_amp, dev)
                        nll = torch.zeros(B, 4, device=dev)
                        for i in range(B):
                            eid, st_, stm = branches[i]
                            w = response_window({e['id']: e for e in entries}[eid])
                            nll[i] = score_branch(st_, stm, cfg, conn, bank1_of(fits, i), w, dev)
                    p, ent = posterior(nll)
                    win = nll.argmin(1).cpu()
                    truth = CANDIDATES.index(fam)   # CANDIDATES includes 'null' at 0!
                    acc = (win == truth).float().mean().item()
                    for bi in range(B):
                        rows.append(dict(stage='single_probe', cohort=cohort, family=fam,
                                         seed=seed, policy=pol, traj=int(sel[bi]),
                                         correct=int(win[bi] == truth),
                                         winner=CANDIDATES[int(win[bi])],
                                         entropy=float(ent[bi]), conf_true=float(p[bi, truth])))
                    print(cohort, fam, seed, pol, f'acc={acc:.3f}', flush=True)
                # ---------------- v9hand ----------------
                nll = v9hand_score(fam, cohort, pids, sel, seeds, cfg, conn, fits, dev)
                p, ent = posterior(nll)
                win = nll.argmin(1).cpu()
                truth = CANDIDATES.index(fam)
                acc = (win == truth).float().mean().item()
                for bi in range(B):
                    rows.append(dict(stage='single_probe', cohort=cohort, family=fam,
                                     seed=seed, policy='v9hand', traj=int(sel[bi]),
                                     correct=int(win[bi] == truth),
                                     winner=CANDIDATES[int(win[bi])],
                                     entropy=float(ent[bi]), conf_true=float(p[bi, truth])))
                print(cohort, fam, seed, 'v9hand', f'acc={acc:.3f}', flush=True)
                del states, stim, fits, est
                torch.cuda.empty_cache()
    path = ROOT / 'metrics' / 'single_probe.csv'
    with path.open('a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        if f.tell() == 0:
            w.writeheader()
        for r in rows:
            w.writerow(r)
    print('SINGLE PROBE COMPLETE', len(rows), flush=True)


# ------------------------------------------------------------------ oracle
@torch.no_grad()
def oracle_utilities(states, stim, cfg, conn, fits, est, fam, pids, hsum,
                     entries, Inb, g_amp, dev):
    """Upper-bound design: divergence between the TRUE-family rollout (true
    params + true TAU hidden state, mean latent extrapolation) and the fitted
    other candidates."""
    from library_v10 import cost as dcost
    B = states.shape[0]
    U = torch.zeros(B, len(entries))
    # true-family rollout state per context
    true_roll = {}
    for b in range(B):
        true_roll[b] = oracle_rollout_state(states[b:b + 1], stim[b:b + 1],
                                            hsum[b:b + 1] if hsum is not None else None,
                                            cfg, conn, fam, pids[b])
    for ei, e in enumerate(entries):
        es = torch.stack([build_stim(e, cfg, Inb, g_amp)] * B).to(dev)
        stim_seg = stim[:, TAU:] + es[:, TAU:]
        t0, t1 = response_window(e)
        w0, w1 = t0 - TAU, t1 - TAU
        preds = rollouts_for_entry(states, stim_seg, cfg, conn, fits, est, (t0, t1))
        # true-family prediction
        mu_true = torch.cat([oracle_rollout(true_roll[b], stim_seg[b:b + 1], cfg, conn,
                                          fam, pids[b], (t0, t1)) for b in range(B)])
        mu_base = preds['null'][0]
        resp = (mu_true - mu_base).abs().mean(1)
        topn = resp.mean(0).topk(min(32, resp.shape[1])).indices
        z = []
        for cj in CANDIDATES:
            if cj == fam:
                continue
            mj = preds[cj][0][:, :, topn]
            sj2 = fits[cj]['s2'][:, None, None] + preds[cj][1][:, :, topn] + 1e-6
            s2t = fits[fam]['s2'][:, None, None] + 1e-6
            z.append(((mu_true[:, :, topn] - mj).abs() / (sj2 + s2t).sqrt()).mean((1, 2)))
        U[:, ei] = torch.stack(z).mean(0) - LAMBDA * dcost(e, 24)
    return U


def oracle_rollout_state(states, stim, hsum, cfg, conn, fam, pid):
    """True-param + true-state rollout initializer."""
    spec = spec_for(fam, pid)
    dev = states.device
    x0 = states[:, TAU].clone()
    if fam == 'gain':
        # true gain deviation q at TAU from hidden_summary (channel 0)
        return dict(kind='gain', q=hsum[0, TAU - 1, 0].reshape(1), x0=x0)
    if fam == 'adapt':
        ap = spec.adapt
        from common_v10 import adapt_a
        terms = obs_terms(states[:, :TAU + 1], stim[:, :TAU], cfg,
                          conn.dense_weight(dev),
                          conn.i_bias.to(dev) if conn.i_bias is not None
                          else torch.zeros(states.shape[2], device=dev))
        a = adapt_a(terms, ap.beta, ap.tau_a)
        return dict(kind='adapt', a=a[:, -1], beta=ap.beta, tau_a=ap.tau_a, c=ap.c, x0=x0)
    if fam == 'stp':
        # true per-edge params: cluster U/taus x tau_scale, norm=1/U
        sp = spec.stp
        terms = obs_terms(states[:, :TAU + 1], stim[:, :TAU], cfg,
                          conn.dense_weight(dev),
                          conn.i_bias.to(dev) if conn.i_bias is not None
                          else torch.zeros(states.shape[2], device=dev))
        ux = stp_oracle_filter(terms, cfg, conn, sp)
        return dict(kind='stp', ux=ux, sp=sp, x0=x0)
    return dict(kind='null', x0=x0)


@torch.no_grad()
def stp_oracle_filter(terms, cfg, conn, sp):
    """Per-edge filter with TRUE cluster params (x tau_scale) -> (u,x) at TAU."""
    from teachers_v9 import STP_CLUSTERS
    B, T, N = terms['s'].shape
    dev = terms['s'].device
    W = conn.dense_weight(dev)
    mask = (W != 0).float()
    g = torch.Generator().manual_seed(cfg.seed + 991)
    cluster = torch.randint(0, len(STP_CLUSTERS), (N, N), generator=g)
    cluster = torch.where((W.cpu() != 0), cluster, torch.full_like(cluster, -1)).to(dev)
    U = torch.zeros(N, N, device=dev)
    tr = torch.ones(N, N, device=dev)
    tf = torch.ones(N, N, device=dev)
    for ci, cl in enumerate(STP_CLUSTERS):
        m = cluster == ci
        U[m] = cl['U']
        tr[m] = cl['tau_rec'] * sp.tau_scale
        tf[m] = cl['tau_fac'] * sp.tau_scale
    rho_rec = torch.exp(-1.0 / tr)
    rho_fac = torch.exp(-1.0 / tf)
    u = U.clone().expand(B, -1, -1).clone()
    x = torch.ones(B, N, N, device=dev)
    for t in range(T):
        fire = terms['y_s'][:, t][:, :, None] * mask[None]
        u_new = (u + U * (1 - u) * fire).clamp(max=1.0)
        x_new = (x - u_new * x * fire).clamp(0.0, 1.0)
        u = U + (u_new - U) * rho_fac
        x = 1.0 + (x_new - 1.0) * rho_rec
    return (u, x), (U, rho_rec, rho_fac)


@torch.no_grad()
def oracle_rollout(init, stim_seg, cfg, conn, fam, pid, window):
    """Mean latent extrapolation rollout of the TRUE family."""
    dev = stim_seg.device
    W = conn.dense_weight(dev)
    N = W.shape[0]
    ib = conn.i_bias.to(dev) if conn.i_bias is not None else torch.zeros(N, device=dev)
    t0, t1 = window
    w0, w1 = t0 - TAU, t1 - TAU
    x = init['x0']
    W_ = stim_seg.shape[1]
    vs = []
    if init['kind'] == 'gain':
        q = init['q']
    elif init['kind'] == 'adapt':
        a = init['a']
        rho = float(np.exp(-1.0 / init['tau_a']))
    elif init['kind'] == 'stp':
        (u, xx), (U, rho_rec, rho_fac) = init['ux']
        sp = init['sp']
        mask = (W != 0).float()
        norm = 1.0 / U.clamp(min=1e-9)
    for t in range(W_):
        v, s, r = x.unbind(-1)
        if init['kind'] == 'stp':
            g = (1.0 - sp.strength) + sp.strength * (u * xx * norm)
            current = torch.einsum('bj,bji->bi', s, W * g) + stim_seg[:, t] + ib
        else:
            current = s @ W + stim_seg[:, t] + ib
        if init['kind'] == 'gain':
            current = current * (1.0 + q)[:, None]
        refr = r > 0
        vn = torch.where(refr, torch.full_like(v, cfg.v_reset),
                         v + cfg.alpha * (-(v - cfg.v_rest) + current)).clamp(min=cfg.v_min)
        if init['kind'] == 'adapt':
            vn = vn - init['c'] * a * (~refr).float()
        fire = (~refr) & (vn >= cfg.v_th)
        vn = torch.where(fire, torch.full_like(vn, cfg.v_reset), vn)
        rn = torch.where(fire, torch.full_like(r, float(cfg.refractory_period)),
                         (r * cfg.refractory_period - 1).clamp(min=0)) / cfg.refractory_period
        x = torch.stack((vn, fire.float(), rn), -1)
        if init['kind'] == 'adapt':
            a = rho * a + init['beta'] * fire.float()
        if init['kind'] == 'stp':
            fire_m = fire.float()[:, :, None] * mask[None]
            u_new = (u + U * (1 - u) * fire_m).clamp(max=1.0)
            x_new = (xx - u_new * xx * fire_m).clamp(0.0, 1.0)
            u = U + (u_new - U) * rho_fac
            xx = 1.0 + (x_new - 1.0) * rho_rec
        vs.append(x[..., 0])
    return torch.stack(vs, 1)[:, w0:w1 + 1]


# ------------------------------------------------------------------ v9hand
@torch.no_grad()
def v9hand_score(fam, cohort, pids, sel, seeds, cfg, conn, fits, dev):
    g_amp = float(np.load(ROOT9 / 'data' / 'intervention_phi.npz')['g_amp'])
    Inb_ns = out_neighbors(conn, J).numpy()
    nll = torch.zeros(len(seeds), 4, device=dev)
    groups = defaultdict(list)
    for i, pid in enumerate(pids):
        groups[pid].append(i)
    for kind, a, b in branch_list():
        if kind == 'delay':
            w = (TAU + 8 + a, TAU + 8 + a + 8)
        elif kind == 'burst':
            w = (TAU, TAU + 32)
        elif kind == 'precond':
            w = (TAU + b, TAU + b + 8)
        else:
            w = (TAU, TAU + 10)
        for pid, grp in groups.items():
            es = torch.stack([build_extra_stim(cfg, kind, a, b, Inb_ns, g_amp) for _ in grp])
            sim = MechanismLIFSimulator(conn, cfg, dev, spec_for(fam, pid))
            dd = sim.generate([seeds[i] for i in grp], 'test_seen', extra_stim=es)
            for bi, i in enumerate(grp):
                nll[i] += score_branch(dd['states'][bi:bi + 1], dd['stimulus'][bi:bi + 1],
                                       cfg, conn, bank1_of(fits, i), w, dev)
            del sim
            torch.cuda.empty_cache()
    return nll


if __name__ == '__main__':
    main()
