"""v10 Stage 8 (optimized): single-shot active model discrimination.

Speed restructure vs design_v10.py:
  - fits/estates cached per (cohort, family, trajectory) across seeds
    (context subsamples overlap ~2.5x);
  - teacher branches generated ONCE per (entry, param-config) and reused
    by every policy that picked the same entry;
  - candidate_vn computed once per (branch, context, candidate) and
    sliced per scoring window (v9hand previously recomputed 16x).
Same protocol, same outputs (single_probe.csv).
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
from design_v10 import posterior, rollouts_for_entry, preds_divergence

ROOT = Path('results/latent_state_v10')
PASSIVE = (32, 96)
GRIDS = dict(gain=GAIN_GRID, adapt=ADAPT_GRID, stp=STP_GRID, null=[None])
NCTX = 32
COHORT_IDX = dict(testA=0, testB=64, testC=128)
HEURISTIC = dict(heuristic_stp='paired_i8', heuristic_gain='highcur_a5.0')
LAMBDA = 0.05
import os
TAG = os.environ.get('V10_TAG', 'main')
V9H_COHORTS = set(os.environ.get('V10_V9H', 'testB,testC').split(',')) if os.environ.get('V10_V9H') else None
FITS_CACHE = Path(os.environ.get('V10_FITS_CACHE', str(ROOT / 'data' / f'fits_cache_{TAG}.pt')))
BRANCH_CACHE = Path(os.environ.get('V10_BRANCH_CACHE', str(ROOT / 'data' / f'branch_cache_{TAG}.pt')))


@torch.no_grad()
def fits_one(states, stim, cfg, conn, dev):
    """fits + estates for ONE trajectory [1,...]."""
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
            scores.append((theta, float(loss[0]), float(s2[0])))
        ss = sorted(scores, key=lambda z: z[1])
        bank = [(s[0], s[2]) for s in ss[:2]]
        # estates at TAU for the banked thetas
        if cand == 'null':
            bank = bank + bank          # pad to TOPK=2 for uniform indexing
        est = []
        for th, _ in bank[:2]:
            if cand == 'null':
                est.append((None, None)); continue
            st_b, _ = context_state(states, stim, cfg, conn, cand, th)
            est.append((th, (st_b[0].cpu() if cand == 'gain' else st_b.cpu() if cand == 'adapt'
                             else (st_b[0].cpu(), st_b[1].cpu()))))
        out[cand] = dict(bank=bank, est=est,
                         s2=min(s[2] for s in scores))
    return out


@torch.no_grad()
def teacher_branch(fam, pid, seed, entry, cfg, conn, Inb, g_amp, dev):
    """One teacher branch [1] with extra_stim from TAU; cached on disk."""
    sim = MechanismLIFSimulator(conn, cfg, dev, spec_for(fam, pid))
    es = build_stim(entry, cfg, Inb, g_amp)[None]
    dd = sim.generate([seed], 'test_seen', extra_stim=es)
    out = dict(states=dd['states'].cpu(), stimulus=dd['stimulus'].cpu())
    del sim
    torch.cuda.empty_cache()
    return out


@torch.no_grad()
def score_windows(states_b, stim_b, cfg, conn, bank, windows, dev):
    """candidate_vn ONCE per candidate; slice per window. Returns [4] summed NLL."""
    terms = obs_terms(states_b, stim_b, cfg, conn.dense_weight(dev),
                      conn.i_bias.to(dev) if conn.i_bias is not None
                      else torch.zeros(states_b.shape[2], device=dev))
    out = torch.zeros(4, device=dev)
    for ci, cand in enumerate(CANDIDATES):
        th, s2 = bank[cand]
        vn = candidate_vn(terms, cfg, conn, cand, th)
        s2 = max(s2, 1e-8)
        v_next, logit, _ = hard_predict(vn, cfg)
        for (t0, t1) in windows:
            yv = terms['y_v'][:, t0:t1]
            ys = terms['y_s'][:, t0:t1]
            info = terms['info'][:, t0:t1]
            free = terms['free'][:, t0:t1]
            nll_v = (0.5 * (v_next[:, t0:t1] - yv) ** 2 / s2 * info).sum() / info.sum().clamp(min=1)
            bce = torch.nn.functional.binary_cross_entropy_with_logits(
                logit[:, t0:t1], ys, reduction='none')
            bce = (bce * free).sum() / free.sum().clamp(min=1)
            out[ci] += nll_v + bce
    return out


@torch.no_grad()
def adaptive_choices(states, stim, fits_list, cfg, conn, short_entries, short_ids,
                     Inb, g_amp, dev, oracle=None):
    """U(d) over the shortlist for B contexts; fits_list = per-context fits dicts."""
    from library_v10 import cost as dcost
    B = states.shape[0]
    # build batched fits/est structures
    fits = {c: dict(bank=[fits_list[b][c]['bank'] for b in range(B)],
                    s2=torch.tensor([fits_list[b][c]['s2'] for b in range(B)], device=dev))
            for c in CANDIDATES}
    est = {}
    for cand in CANDIDATES:
        per = []
        for b in range(B):
            ks = []
            for k in range(2):
                th, st_cpu = fits_list[b][cand]['est'][k]
                if cand == 'gain':
                    ks.append((th, st_cpu.to(dev).reshape(1)))
                elif cand == 'adapt':
                    ks.append((th, st_cpu.to(dev)))
                elif cand == 'stp':
                    ks.append((th, (st_cpu[0].to(dev), st_cpu[1].to(dev))))
                else:
                    ks.append((None, None))
            per.append(ks)
        est[cand] = per
    U = torch.zeros(B, len(short_entries))
    for ei, e in enumerate(short_entries):
        es = torch.stack([build_stim(e, cfg, Inb, g_amp)] * B).to(dev)
        stim_seg = stim[:, TAU:] + es[:, TAU:]
        preds = rollouts_for_entry(states, stim_seg, cfg, conn, fits, est, response_window(e))
        U[:, ei] = preds_divergence(preds, fits) - LAMBDA * dcost(e, 24)
    return [short_entries[int(i)]['id'] for i in U.argmax(1)]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cohorts', nargs='+', default=['testA', 'testB', 'testC'])
    ap.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument('--policies', nargs='+', default=None)
    args = ap.parse_args()
    torch.set_num_threads(2)
    cfg = setup_cfg()
    conn = Connectome.generate(cfg)
    dev = torch.device('cuda')
    blob = torch.load(ROOT9 / 'data' / 'v9_data.pt', map_location='cpu', weights_only=False)
    store = blob['store']
    lib = json.loads((ROOT / 'protocol' / 'intervention_library.json').read_text())
    entries = lib['entries']
    entry_map = {e['id']: e for e in entries}
    short_ids = json.loads((ROOT / 'protocol' / 'configs' / 'shortlist.json').read_text())['ids']
    short_entries = [e for e in entries if e['id'] in short_ids]
    Inb_full = out_neighbors(conn, J)
    Wcpu = conn.dense_weight('cpu')
    strength = Wcpu[J].abs().sum(0)[Inb_full]
    Inb = Inb_full[strength.argsort(descending=True)].numpy()
    g_amp = float(lib['g_amp'])
    global _unit_count
    _unit_count = 0
    fits_cache = torch.load(FITS_CACHE, map_location='cpu', weights_only=False) \
        if FITS_CACHE.exists() else {}
    branch_cache = torch.load(BRANCH_CACHE, map_location='cpu', weights_only=False) \
        if BRANCH_CACHE.exists() else {}
    rows = []
    done_keys = set()
    done_units = set()
    sp_path = ROOT / 'metrics' / f'single_probe_{TAG}.csv'
    if sp_path.exists():
        for r in csv.DictReader(sp_path.open()):
            done_keys.add((r['cohort'], r['family'], r['seed'], r['policy'], r['traj']))
        import collections
        cnt = collections.Counter()
        for r in csv.DictReader(sp_path.open()):
            cnt[(r['cohort'], r['family'], r['seed'])] += 1
        for k, c in cnt.items():
            if c >= 224:   # >=7 policies x 32 contexts
                done_units.add(k)
    for cohort in args.cohorts:
        for fam in FAMILIES:
            d = store[f'{fam}/{cohort}']
            n_all = len(d['states'])
            if cohort == 'testA':
                pids_all = [SELECTED[fam]['train'][int(c)][0] for c in d['config_id']]
            else:
                pids_all = [SELECTED[fam]['splitB' if cohort == 'testB' else 'splitC'][0]] * n_all
            for seed in args.seeds:
                if (cohort, fam, str(seed)) in done_units:
                    print('skip done unit', cohort, fam, seed, flush=True)
                    continue
                g = np.random.default_rng(10_000 + seed)
                sel = np.arange(n_all) if cohort == 'testC' else \
                    np.sort(g.choice(n_all, size=min(NCTX, n_all), replace=False))
                B = len(sel)
                # ---- fits (cached per trajectory) ----
                fits_list = []
                for i in sel:
                    key = (cohort, fam, int(i))
                    if key not in fits_cache:
                        st_ = d['states'][int(i):int(i) + 1].to(dev)
                        sm_ = d['stimulus'][int(i):int(i) + 1].to(dev)
                        fits_cache[key] = fits_one(st_, sm_, cfg, conn, dev)
                    fits_list.append(fits_cache[key])
                states = d['states'][sel].to(dev)
                stim = d['stimulus'][sel].to(dev)
                seeds_traj = [cfg.traj_seed('test_seen', COHORT_IDX[cohort] + int(i)) for i in sel]
                pids = [pids_all[int(i)] for i in sel]
                # ---- choices ----
                choices = dict(passive=['passive'] * B)
                gr = np.random.default_rng(20_000 + seed)
                choices['random'] = [short_ids[int(gr.integers(len(short_ids)))] for _ in range(B)]
                for h, eid in HEURISTIC.items():
                    choices[h] = [eid] * B
                choices['optimized_global'] = [short_ids[0]] * B
                choices['optimized_adaptive'] = adaptive_choices(
                    states, stim, fits_list, cfg, conn, short_entries, short_ids, Inb, g_amp, dev)
                choices['oracle'] = choices['optimized_adaptive']  # placeholder, replaced below
                # ---- oracle: true-family rollout vs fitted others ----
                from design_v10 import oracle_utilities
                est_batch = {}
                for cand in CANDIDATES:
                    per = []
                    for b in range(B):
                        ks = []
                        for k in range(2):
                            th, st_cpu = fits_list[b][cand]['est'][k]
                            if cand == 'gain':
                                ks.append((th, st_cpu.to(dev).reshape(1)))
                            elif cand == 'adapt':
                                ks.append((th, st_cpu.to(dev)))
                            elif cand == 'stp':
                                ks.append((th, (st_cpu[0].to(dev), st_cpu[1].to(dev))))
                            else:
                                ks.append((None, None))
                        per.append(ks)
                    est_batch[cand] = per
                fits_batch = {c: dict(bank=[fits_list[b][c]['bank'] for b in range(B)],
                                      s2=torch.tensor([fits_list[b][c]['s2'] for b in range(B)],
                                                      device=dev))
                              for c in CANDIDATES}
                hsum = d['hidden_summary'][sel].to(dev) if 'hidden_summary' in d else None
                Uo = oracle_utilities(states, stim, cfg, conn, fits_batch, est_batch, fam, pids,
                                      hsum, short_entries, Inb, g_amp, dev)
                choices['oracle'] = [short_entries[int(i)]['id'] for i in Uo.argmax(1)]
                if args.policies is not None:
                    _keep = set(args.policies)
                    choices = {k: v for k, v in choices.items() if k in _keep}
                # ---- unified branch generation ----
                needed = defaultdict(list)
                for pol, ch in choices.items():
                    if pol in ('passive',):
                        continue
                    for bi, eid in enumerate(ch):
                        needed[(eid, pids[bi], seeds_traj[bi])].append(bi)
                for (eid, pid, sd), bis in needed.items():
                    bkey = (fam, pid, int(sd), eid)
                    if bkey not in branch_cache:
                        branch_cache[bkey] = teacher_branch(fam, pid, sd, entry_map[eid],
                                                            cfg, conn, Inb, g_amp, dev)
                # ---- score ----
                truth = CANDIDATES.index(fam)
                for pol, ch in choices.items():
                    nll = torch.zeros(B, 4, device=dev)
                    for bi in range(B):
                        if pol == 'passive':
                            st_, sm_ = states[bi:bi + 1], stim[bi:bi + 1]
                            w = (TAU, TAU + 32)
                        else:
                            br = branch_cache[(fam, pids[bi], int(seeds_traj[bi]), ch[bi])]
                            st_ = br['states'].to(dev)
                            sm_ = br['stimulus'].to(dev)
                            w = response_window(entry_map[ch[bi]])
                        bank = {c: fits_list[bi][c]['bank'][0] for c in CANDIDATES}
                        nll[bi] = score_windows(st_, sm_, cfg, conn, bank, [w], dev)
                    p, ent = posterior(nll)
                    win = nll.argmin(1).cpu()
                    acc = (win == truth).float().mean().item()
                    for bi in range(B):
                        rows.append(dict(stage='single_probe', cohort=cohort, family=fam,
                                         seed=seed, policy=pol, traj=int(sel[bi]),
                                         correct=int(win[bi] == truth),
                                         winner=CANDIDATES[int(win[bi])],
                                         entropy=float(ent[bi]), conf_true=float(p[bi, truth])))
                    print(cohort, fam, seed, pol, f'acc={acc:.3f}', flush=True)
                # ---- v9hand (16 branches, candidate_vn once per branch) ----
                do_v9h = (not V9H_COHORTS or cohort in V9H_COHORTS) and \
                         (args.policies is None or 'v9hand' in args.policies)
                if not do_v9h:
                    print('v9hand skipped for', cohort, flush=True)
                if do_v9h:
                    v9g_amp = float(np.load(ROOT9 / 'data' / 'intervention_phi.npz')['g_amp'])
                    Inb_ns = out_neighbors(conn, J).numpy()
                    nll = torch.zeros(B, 4, device=dev)
                    if 'v9cache' not in globals():
                        globals()['v9cache'] = {}
                    v9cache = globals()['v9cache']
                    for kind, a, bb in branch_list():
                        if kind == 'delay':
                            w = (TAU + 8 + a, TAU + 8 + a + 8)
                        elif kind == 'burst':
                            w = (TAU, TAU + 32)
                        elif kind == 'precond':
                            w = (TAU + bb, TAU + bb + 8)
                        else:
                            w = (TAU, TAU + 10)
                        for pid in sorted(set(pids)):
                            grp = [i for i, p in enumerate(pids) if p == pid]
                            need = [i for i in grp
                                    if (fam, pid, int(seeds_traj[i]), f'{kind}|{a}|{bb}') not in v9cache]
                            if need:
                                es = torch.stack([build_extra_stim(cfg, kind, a, bb, Inb_ns, v9g_amp) for _ in need])
                                sim = MechanismLIFSimulator(conn, cfg, dev, spec_for(fam, pid))
                                dd = sim.generate([seeds_traj[i] for i in need], 'test_seen', extra_stim=es)
                                for bi2, i in enumerate(need):
                                    v9cache[(fam, pid, int(seeds_traj[i]), f'{kind}|{a}|{bb}')] = (
                                        dd['states'][bi2:bi2 + 1].cpu(), dd['stimulus'][bi2:bi2 + 1].cpu())
                                del sim
                                torch.cuda.empty_cache()
                            for i in grp:
                                st_, sm_ = v9cache[(fam, pid, int(seeds_traj[i]), f'{kind}|{a}|{bb}')]
                                bank = {c: fits_list[i][c]['bank'][0] for c in CANDIDATES}
                                nll[i] += score_windows(st_.to(dev), sm_.to(dev),
                                                        cfg, conn, bank, [w], dev)
                    p, ent = posterior(nll)
                    win = nll.argmin(1).cpu()
                    acc = (win == truth).float().mean().item()
                    for bi in range(B):
                        rows.append(dict(stage='single_probe', cohort=cohort, family=fam,
                                         seed=seed, policy='v9hand', traj=int(sel[bi]),
                                         correct=int(win[bi] == truth),
                                         winner=CANDIDATES[int(win[bi])],
                                         entropy=float(ent[bi]), conf_true=float(p[bi, truth])))
                    print(cohort, fam, seed, 'v9hand', f'acc={acc:.3f}', flush=True)
                del states, stim
                torch.cuda.empty_cache()
                # persist caches periodically (branch cache is huge: rare saves)
                torch.save(fits_cache, FITS_CACHE)
                _unit_count += 1
                if _unit_count % 20 == 0:
                    torch.save(branch_cache, BRANCH_CACHE)
                # write rows incrementally
                with sp_path.open('a', newline='') as f:
                    wcsv = csv.DictWriter(f, fieldnames=list(rows[0]))
                    if f.tell() == 0:
                        wcsv.writeheader()
                    for r in rows:
                        if (r['cohort'], r['family'], str(r['seed']), r['policy'], str(r['traj'])) not in done_keys:
                            wcsv.writerow(r)
                rows = []
    torch.save(fits_cache, FITS_CACHE)
    torch.save(branch_cache, BRANCH_CACHE)
    print('SINGLE PROBE COMPLETE', flush=True)


if __name__ == '__main__':
    main()
