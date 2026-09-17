"""v10 Stage 9: sequential active identification (accuracy vs probe budget).

Per context: posterior over the 4 candidates, updated after each probe.
Probes EXECUTE SEQUENTIALLY IN TIME (cumulative extra_stim; candidates
assimilate continuously). Theta banks are refit on ALL observations so far
before every probe choice. Greedy policy maximizes posterior-weighted
pairwise disagreement via rollouts from the current state; random policy
draws unused entries. Budget curve recorded at 0,1,2,4,8 probes.
Cohort: testB (held-out params), 5 paired seeds, 32 contexts/family.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
from connectome import Connectome
from run_v7 import setup_cfg
from protocol_v9 import ROOT as ROOT9, FAMILIES, SELECTED, spec_for
from teachers_v9 import MechanismLIFSimulator
from common_v10 import (CANDIDATES, obs_terms, candidate_vn, hard_predict,
                        rollout_candidate, GAIN_GRID, ADAPT_GRID, STP_GRID)
from library_v10 import build_stim, response_window, cost as dcost
from intervention_v9 import out_neighbors, J, TAU
from design_v10 import passive_fits, posterior

ROOT = Path('results/latent_state_v10')
GRIDS = dict(gain=GAIN_GRID, adapt=ADAPT_GRID, stp=STP_GRID, null=[None])
NCTX = 32
BUDGETS = (0, 1, 2, 4, 8)
MAXP = 8
SLOT_MARGIN = 4
LAMBDA = 0.05


def stim_anchored(e, cfg, Inb, g_amp, anchor):
    u = build_stim(e, cfg, Inb, g_amp)
    if anchor == TAU:
        return u
    out = torch.zeros_like(u)
    nz = u.any(1).nonzero(as_tuple=True)[0]
    if len(nz):
        lo, hi = int(nz.min()), int(nz.max())
        out[anchor:anchor + hi - lo + 1] = u[lo:hi + 1]
    return out


def window_anchored(e, anchor):
    t0, t1 = response_window(e)
    return (t0 - TAU + anchor, t1 - TAU + anchor)


@torch.no_grad()
def fits_upto(states, stim, upto, cfg, conn, dev):
    """theta banks (top-2) + sigma, fitted on [0, upto)."""
    terms = obs_terms(states[:, :upto + 1], stim[:, :upto], cfg,
                      conn.dense_weight(dev),
                      conn.i_bias.to(dev) if conn.i_bias is not None
                      else torch.zeros(states.shape[2], device=dev))
    fit_win = (32, upto)
    out = {}
    for cand in CANDIDATES:
        scores = []
        for theta in GRIDS[cand]:
            vn = candidate_vn(terms, cfg, conn, cand, theta)
            v_next, _, _ = hard_predict(vn[:, fit_win[0]:fit_win[1]], cfg)
            yv = terms['y_v'][:, fit_win[0]:fit_win[1]]
            info = terms['info'][:, fit_win[0]:fit_win[1]]
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
def estates_upto(states, stim, upto, cfg, conn, fits, dev):
    from landscape_v10 import context_state
    from common_v10 import gain_q, adapt_a, stp_g
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
                terms = obs_terms(states[b:b + 1, :upto + 1], stim[b:b + 1, :upto], cfg,
                                  conn.dense_weight(dev),
                                  conn.i_bias.to(dev) if conn.i_bias is not None
                                  else torch.zeros(states.shape[2], device=dev))
                if cand == 'gain':
                    ks.append((th, gain_q(terms, cfg, th[0], th[1])[:, -1]))
                elif cand == 'adapt':
                    ks.append((th, adapt_a(terms, th[0], th[1])[:, -1]))
                elif cand == 'stp':
                    _, ux = stp_g(terms, cfg, conn, th[0], th[1])
                    ks.append((th, ux))
            per.append(ks)
        est[cand] = per
    return est


@torch.no_grad()
def rollouts_at(states, stim_seg, upto, window, cfg, conn, fits, est):
    B = states.shape[0]
    x0s = states[:, upto]
    w0, w1 = window[0] - upto, window[1] - upto
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


def pair_skl(preds, fits):
    mu_base = preds['null'][0]
    resp = sum((preds[c][0] - mu_base).abs().mean(1) for c in CANDIDATES if c != 'null')
    topn = resp.mean(0).topk(min(32, resp.shape[1])).indices
    skls = {}
    for i, ci in enumerate(CANDIDATES):
        for j, cj in enumerate(CANDIDATES):
            if j <= i:
                continue
            mi, mj = preds[ci][0][:, :, topn], preds[cj][0][:, :, topn]
            si2 = fits[ci]['s2'][:, None, None] + preds[ci][1][:, :, topn] + 1e-6
            sj2 = fits[cj]['s2'][:, None, None] + preds[cj][1][:, :, topn] + 1e-6
            kl = 0.5 * (si2 / sj2 + (mi - mj) ** 2 / sj2 - 1 + torch.log(sj2 / si2)) \
               + 0.5 * (sj2 / si2 + (mi - mj) ** 2 / si2 - 1 + torch.log(si2 / sj2))
            skls[(ci, cj)] = (0.5 * kl).clamp(min=0).mean((1, 2))
    return skls


@torch.no_grad()
def score_window(terms, cfg, conn, cand, theta, s2, window):
    vn = candidate_vn(terms, cfg, conn, cand, theta)
    s2 = max(s2, 1e-8)
    v_next, logit, _ = hard_predict(vn[:, window[0]:window[1]], cfg)
    yv = terms['y_v'][:, window[0]:window[1]]
    ys = terms['y_s'][:, window[0]:window[1]]
    info = terms['info'][:, window[0]:window[1]]
    free = terms['free'][:, window[0]:window[1]]
    nll_v = (0.5 * (v_next - yv) ** 2 / s2 * info).sum((1, 2)) / info.sum((1, 2)).clamp(min=1)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logit, ys, reduction='none')
    bce = (bce * free).sum((1, 2)) / free.sum((1, 2)).clamp(min=1)
    return nll_v + bce


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument('--policies', nargs='+', default=['optimized', 'random'])
    ap.add_argument('--families', nargs='+', default=list(FAMILIES))
    import os
    tag = os.environ.get('V10_TAG', 'main')
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
    entry_map = {e['id']: e for e in entries}
    Inb_full = out_neighbors(conn, J)
    Wcpu = conn.dense_weight('cpu')
    strength = Wcpu[J].abs().sum(0)[Inb_full]
    Inb = Inb_full[strength.argsort(descending=True)].numpy()
    g_amp = float(lib['g_amp'])
    rows = []
    done_units = set()
    sp_path = ROOT / 'metrics' / f'sequential_{tag}.csv'
    if sp_path.exists():
        import collections
        cnt = collections.Counter()
        for r in csv.DictReader(sp_path.open()):
            cnt[(r['family'], r['seed'], r['policy'])] += 1
        for k, c in cnt.items():
            if c >= 160:   # 5 budgets x 32 contexts
                done_units.add(k)
    for fam in args.families:
        d = store[f'{fam}/testB']
        pid = SELECTED[fam]['splitB'][0]
        n_all = len(d['states'])
        for seed in args.seeds:
            g = np.random.default_rng(10_000 + seed)
            sel = np.sort(g.choice(n_all, size=min(NCTX, n_all), replace=False))
            states0 = d['states'][sel].to(dev)
            stim0 = d['stimulus'][sel].to(dev)
            B = len(sel)
            seeds = [cfg.traj_seed('test_seen', 64 + int(i)) for i in sel]
            for pol in args.policies:
                if (fam, str(seed), pol) in done_units:
                    print('skip done', fam, seed, pol, flush=True)
                    continue
                logp = torch.zeros(B, 4, device=dev)
                chosen_hist = [[] for _ in range(B)]
                stim_cum = torch.zeros(B, cfg.T, cfg.n_neurons)
                branch_states, branch_stim = states0.clone(), stim0.clone()
                slot = TAU
                truth = CANDIDATES.index(fam)
                for rnd in range(0, MAXP + 1):
                    # ---------- record at budget ----------
                    if rnd in BUDGETS:
                        p = torch.softmax(-(logp - logp.min(1, keepdim=True).values).clamp(max=50), 1)
                        ent = -(p * (p + 1e-12).log()).sum(1)
                        win = p.argmax(1)
                        for bi in range(B):
                            rows.append(dict(stage='sequential', cohort='testB', family=fam,
                                             seed=seed, policy=pol, traj=int(sel[bi]),
                                             budget=rnd, correct=int(win[bi] == truth),
                                             winner=CANDIDATES[int(win[bi])],
                                             entropy=float(ent[bi]),
                                             conf_true=float(p[bi, truth]),
                                             n_probes=rnd))
                    if rnd == MAXP:
                        break
                    # ---------- infeasible slot: close out remaining budgets ----------
                    min_end = min(window_anchored(e, slot)[1] for e in short_entries)
                    if min_end > cfg.T - 2:
                        p = torch.softmax(-(logp - logp.min(1, keepdim=True).values).clamp(max=50), 1)
                        ent = -(p * (p + 1e-12).log()).sum(1)
                        win = p.argmax(1)
                        for future in range(rnd, MAXP + 1):
                            if future not in BUDGETS:
                                continue
                            for bi in range(B):
                                rows.append(dict(stage='sequential', cohort='testB', family=fam,
                                                 seed=seed, policy=pol, traj=int(sel[bi]),
                                                 budget=future, correct=int(win[bi] == truth),
                                                 winner=CANDIDATES[int(win[bi])],
                                                 entropy=float(ent[bi]),
                                                 conf_true=float(p[bi, truth]),
                                                 n_probes=rnd))
                        break
                    # ---------- fits + estates on all observations so far ----------
                    fits = fits_upto(branch_states, branch_stim, slot, cfg, conn, dev)
                    # ---------- choose probe ----------
                    if pol == 'optimized':
                        est = estates_upto(branch_states, branch_stim, slot, cfg, conn, fits, dev)
                        pm = torch.softmax(-(logp - logp.min(1, keepdim=True).values).clamp(max=50), 1).mean(0)
                        U = torch.zeros(B, len(short_entries))
                        for ei, e in enumerate(short_entries):
                            w = window_anchored(e, slot)
                            if w[1] > cfg.T - 2:
                                U[:, ei] = -1e9
                                continue
                            es = torch.stack([stim_anchored(e, cfg, Inb, g_amp, slot)] * B).to(dev)
                            stim_seg = branch_stim[:, slot:] + es[:, slot:]
                            preds = rollouts_at(branch_states, stim_seg, slot, w, cfg, conn, fits, est)
                            skls = pair_skl(preds, fits)
                            u = torch.zeros(B, device=dev)
                            for (ci, cj), v in skls.items():
                                u = u + float(pm[CANDIDATES.index(ci)] * pm[CANDIDATES.index(cj)]) * v
                            U[:, ei] = u - LAMBDA * dcost(e, 24)
                        _se_ids = [e['id'] for e in short_entries]
                        for bi in range(B):
                            for ui in set(chosen_hist[bi]):
                                U[bi, _se_ids.index(ui)] = -1e9
                        pick = [short_entries[int(i)]['id'] for i in U.argmax(1)]
                    else:
                        gr = np.random.default_rng(30_000 + seed * 100 + rnd)
                        pick = []
                        for bi in range(B):
                            avail = [s for s in short_ids
                                     if s not in set(chosen_hist[bi])
                                     and window_anchored(entry_map[s], slot)[1] <= cfg.T - 2]
                            pick.append(avail[int(gr.integers(len(avail)))])
                    # ---------- execute ----------
                    for bi in range(B):
                        stim_cum[bi] += stim_anchored(entry_map[pick[bi]], cfg, Inb, g_amp, slot)
                        chosen_hist[bi].append(pick[bi])
                    sim = MechanismLIFSimulator(conn, cfg, dev, spec_for(fam, pid))
                    dd = sim.generate(seeds, 'test_seen', extra_stim=stim_cum.to(dev))
                    branch_states, branch_stim = dd['states'], dd['stimulus']
                    del sim
                    torch.cuda.empty_cache()
                    # ---------- score the new windows ----------
                    terms = obs_terms(branch_states, branch_stim, cfg, conn.dense_weight(dev),
                                      conn.i_bias.to(dev) if conn.i_bias is not None
                                      else torch.zeros(cfg.n_neurons, device=dev))
                    for bi in range(B):
                        w = window_anchored(entry_map[pick[bi]], slot)
                        for ci, cand in enumerate(CANDIDATES):
                            th, s2 = fits[cand]['bank'][bi][0]
                            one = dict((k, v[bi:bi + 1]) for k, v in terms.items())
                            logp[bi, ci] += float(score_window(one, cfg, conn, cand, th, s2, w))
                    slot = max(window_anchored(entry_map[p], slot)[1] for p in pick) + SLOT_MARGIN
                print('testB', fam, seed, pol, 'done', flush=True)
                # incremental write per completed policy unit
                hdr = not sp_path.exists()
                with sp_path.open('a', newline='') as f:
                    wcsv = csv.DictWriter(f, fieldnames=list(rows[0]))
                    if hdr:
                        wcsv.writeheader()
                    for r in rows:
                        wcsv.writerow(r)
                rows = []
                del branch_states, branch_stim
                torch.cuda.empty_cache()
    if rows:
        path = ROOT / 'metrics' / f'sequential_{tag}.csv'
        with path.open('a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            if f.tell() == 0:
                w.writeheader()
            for r in rows:
                w.writerow(r)
    print('SEQUENTIAL COMPLETE', len(rows), flush=True)


if __name__ == '__main__':
    main()
