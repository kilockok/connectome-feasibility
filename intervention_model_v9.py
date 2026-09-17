"""v9 Stage 7b: model-predicted intervention fingerprints.

Frozen Phase-A corrector, teacher-forced one-step V predictions along every
probe branch (NO rollout). Same 16-d bounded fingerprint from predicted V.
Tests: (i) alignment: cosine(mean teacher phi, mean model phi) per family;
(ii) transfer: teacher-trained classifier applied to model phi;
(iii) base_pre fingerprint = no-mechanism control.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
from connectome import Connectome
from run_v7 import setup_cfg
from teachers_v9 import MechanismLIFSimulator, MechSpec
from protocol_v9 import ROOT, FAMILIES, SEEDS, SELECTED, spec_for
from models.residual_v8 import build_v8
from models.residual_v7 import base_pre
from intervention_v9 import (gen_branches, fingerprint, vauc, CONDS, IDX, NC,
                             cls_metrics, TAU)

T0, T1 = 76, 146          # prediction range [T0, T1)


@torch.no_grad()
def predict_series(model, states, stim, cfg, W, ib, base_only=False):
    """Teacher-forced V predictions for steps T0..T1-1. Returns [B, T1-T0, N]."""
    B = len(states)
    ends = torch.arange(T0 - 1, T1 - 1, device=states.device)
    idx = ends[:, None] + torch.arange(-31, 1, device=states.device)
    x = torch.cat((states[:, idx], stim[:, idx].unsqueeze(-1)), -1)  # [B,E,K,N,4]
    E = len(ends)
    x = x.reshape(B * E, 32, states.shape[2], 4)
    preds = []
    for i in range(0, len(x), 1024):
        xb = x[i:i + 1024].cuda(non_blocking=True)
        if base_only:
            out = base_pre(xb[:, -1], cfg, W, ib)
            preds.append(out['v_base'].cpu())
        else:
            preds.append(model(xb)['v'].cpu())
    v = torch.cat(preds).reshape(B, E, states.shape[2])
    return v


def phi_from_pred(vpred, branches_true, Inb):
    """Same fingerprint but with predicted V series replacing true V states.

    branches_true used only for nothing else: response windows are absolute
    time, so we rebuild a pseudo 'states' tensor: [B,257,N,3] with predicted
    V in [T0,T1) and true V elsewhere (pre-window baselines also predicted
    when inside the range)."""
    B = vpred.shape[0]
    states = branches_true.clone()
    states[:, T0:T1, :, 0] = vpred
    # rebuild the branch dict shape fingerprint expects: key -> states
    return states


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', nargs='+', type=int, default=list(SEEDS))
    args = ap.parse_args()
    torch.set_num_threads(2)
    cfg = setup_cfg()
    conn = Connectome.generate(cfg)
    W = conn.dense_weight('cuda')
    ib = conn.i_bias.cuda() if conn.i_bias is not None else torch.zeros(cfg.n_neurons, device='cuda')
    from intervention_v9 import out_neighbors, J
    Inb = out_neighbors(conn, J).numpy()
    z = np.load(ROOT / 'data' / 'intervention_phi.npz')
    g_amp = float(z['g_amp'])
    dev = torch.device('cuda')

    # ---- collect branch states once (shared across corrector seeds) ----
    bcache = ROOT / 'data' / 'intervention_branches.pt'
    if bcache.exists():
        all_branches = torch.load(bcache, map_location='cpu', weights_only=False)
    else:
        all_branches = {}
        for fam in FAMILIES + ('null',):
            for cond in ('train', 'testB', 'testC'):
                if fam == 'null' and cond != 'train':
                    continue
                if fam == 'null':
                    spec = MechSpec(name='null')
                    groups = [('null', list(range(NC[cond])))]
                else:
                    if cond == 'train':
                        pids = [n for n, _ in SELECTED[fam]['train']]
                        assign = [pids[i % 3] for i in range(NC[cond])]
                    elif cond == 'testB':
                        assign = [SELECTED[fam]['splitB'][0]] * NC[cond]
                    else:
                        assign = [SELECTED[fam]['splitC'][0]] * NC[cond]
                    groups = [(pid, [i for i, a in enumerate(assign) if a == pid])
                              for pid in sorted(set(assign))]
                for pid in groups:
                    pass
                # generate per config group, then reassemble in order
                fam_cond = {}
                for pid, idxs in groups:
                    seeds = [cfg.traj_seed('test_seen', IDX[cond] + i) for i in idxs]
                    sim = MechanismLIFSimulator(conn, cfg, dev,
                                                spec if fam == 'null' else spec_for(fam, pid))
                    br = gen_branches(sim, cfg, seeds, 'test_seen', Inb, g_amp)
                    stim = None
                    # also need stimulus per branch: regenerate cheaply via sim cache? store it
                    for k, v in br.items():
                        fam_cond.setdefault(k, {})[pid] = (idxs, v)
                    del sim, br
                    torch.cuda.empty_cache()
                # reassemble into CANONICAL row order (i = 0..n-1 as assigned)
                for k, per in fam_cond.items():
                    n_total = sum(len(per[pid][0]) for pid in per)
                    ref = next(iter(per.values()))[1]
                    states = torch.empty((n_total,) + ref.shape[1:], dtype=ref.dtype)
                    for pid in per:
                        idxs, v = per[pid]
                        for row, i in zip(range(len(v)), idxs):
                            states[i] = v[row]
                    all_branches[(fam, cond, k)] = states
                print('branches', fam, cond, flush=True)
        # stimulus is identical per (cond, branch) across families: grab from any
        torch.save(all_branches, bcache)
    print('branch cache ready', flush=True)

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    zt = np.load(ROOT / 'data' / 'intervention_phi.npz')
    phi_t, labels, conds = zt['phi'], zt['labels'], zt['conds']
    tr = conds == 'train'
    sc = StandardScaler().fit(phi_t[tr])
    clf = LogisticRegression(max_iter=3000).fit(sc.transform(phi_t[tr]), labels[tr])

    rows = []
    for cseed in args.seeds:
        summary = json.loads((ROOT / 'metrics' / 'training' / f'ordered_seed{cseed}.json').read_text())
        corr = build_v8('ordered', conn, cfg).cuda()
        corr.load_state_dict(torch.load(summary['checkpoint'], map_location='cuda',
                                        weights_only=False)['state_dict'])
        corr = corr.eval()
        for base_only in (False, True):
            phis, labs, cnds = [], [], []
            for fam in FAMILIES:
                for cond in CONDS:
                    br = {k: all_branches[(fam, cond, k)]
                          for k in list(all_branches) if False} if False else None
                    # collect this (fam,cond) branch dict
                    keys = [k for (f, c, k) in all_branches if f == fam and c == cond]
                    branches = {k: all_branches[(fam, cond, k)] for k in keys}
                    # stimulus: recover from states? need stim tensor; regenerate cheap:
                    # store stimulus inside states? no -> we saved only states.
                    # We need stimulus for windows: re-derive from any family's gen? seeds match,
                    # so stimulus = natural + extra; natural part identical across families.
                    # We stored only states; simplest: regenerate stimulus via the same sim.
                    phi_b = []
                    for k, states in branches.items():
                        kind, a, b = k.split('|')
                        a = {'None': None}.get(a, int(a) if kind == 'delay' else a)
                        b = None if b == 'None' else int(b)
                        # stimulus rebuild
                        from dataset import sample_traj_params, build_stimulus
                        from intervention_v9 import build_extra_stim
                        n = len(states)
                        seeds = [cfg.traj_seed('test_seen', IDX[cond] + i) for i in range(n)]
                        us = []
                        for s in seeds:
                            g = torch.Generator().manual_seed(int(s))
                            p = sample_traj_params(g, cfg, 'test_seen')
                            u = build_stimulus(p, cfg)
                            us.append(u)
                        u = torch.stack(us)
                        from intervention_v9 import build_extra_stim as bes
                        es = torch.stack([bes(cfg, kind, a, b, Inb, g_amp) for _ in range(n)])
                        stim = (u + es)
                        vpred = predict_series(corr if not base_only else None,
                                               states, stim, cfg, W, ib, base_only=base_only)
                        branches[k] = phi_from_pred(vpred, states, Inb)
                    phi = fingerprint(branches, Inb)
                    phis.append(phi)
                    labs += [FAMILIES.index(fam)] * len(phi)
                    cnds += [cond] * len(phi)
                    print('model phi', cseed, 'base' if base_only else 'corr', fam, cond, flush=True)
            phis = torch.cat(phis).numpy()
            labs = np.array(labs)
            cnds = np.array(cnds)
            for cond in CONDS:
                te = cnds == cond
                prob = clf.predict_proba(sc.transform(phis[te]))
                m = cls_metrics(labs[te], prob)
                rows.append(dict(stage='base_fingerprint' if base_only else 'model_fingerprint',
                                 seed=cseed, split=cond, **m))
                # alignment: cosine between mean teacher phi and mean model phi (train cond)
            tec = cnds == 'train'
            cos_rows = []
            for fi, fam in enumerate(FAMILIES):
                mt = phi_t[tr & (labels == fi)].mean(0)
                mm = phis[tec & (labs == fi)].mean(0)
                cos = float(np.dot(mt, mm) / (np.linalg.norm(mt) * np.linalg.norm(mm) + 1e-12))
                cos_rows.append(cos)
            rows.append(dict(stage='base_align' if base_only else 'model_align',
                             seed=cseed, split='train', bal_acc=np.mean(cos_rows),
                             macro_f1=float(np.min(cos_rows)), auroc=float('nan')))
            print('align', cseed, 'base' if base_only else 'corr',
                  [f'{c:.3f}' for c in cos_rows], flush=True)
            del phis
        del corr
        torch.cuda.empty_cache()
    path = ROOT / 'metrics' / 'intervention_identification.csv'
    with path.open('a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        for r in rows:
            w.writerow(r)
    print('MODEL FINGERPRINT COMPLETE', len(rows), flush=True)


if __name__ == '__main__':
    main()
