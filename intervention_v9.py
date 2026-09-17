"""v9 Stage 7: controlled intervention fingerprints (Gate E).

Legal stimulus-only probe protocols on top of the natural protocol
(extra_stim), identical across families (same seeds). Branches per base
seed (16): delay recovery d in {0,2,4,8,16,32}; burst structure
{even,paired,burst,longgap} matched charge; edge-local vs neuron-global
preconditioning {edge,global} x probe at {8,16}; history load {high,low}.
Fingerprint phi = 13-d vector of WITHIN-TRAJECTORY response ratios (no
absolute amplitudes -> no effect-size shortcut).

Teacher fingerprints: classification trained on train-param trajectories,
tested on held-out (splitB) and extrapolated (splitC) parameters.
Model fingerprints: frozen Phase-A corrector, teacher-forced one-step
predictions along each branch (NO rollout); same phi from predicted V;
alignment (cosine to teacher phi) + transfer (teacher-trained classifier
applied to model phi) + base_pre fingerprint as the no-mechanism control.
"""
import argparse
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
from teachers_v9 import MechanismLIFSimulator
from protocol_v9 import ROOT, FAMILIES, SEEDS, SELECTED, spec_for
from models.residual_v8 import build_v8
from models.residual_v7 import base_pre
from latent_data import windows as lw

TAU = 96
J = [10, 11, 12]
A_NEURONS = [20, 21, 22, 23, 24]
BG_NEURONS = [30, 35, 40, 45, 50, 55, 60, 65, 70, 75]
DS = (0, 2, 4, 8, 16, 32)
PATTERNS = ('even', 'paired', 'burst', 'longgap')
CONDS = ('train', 'testB', 'testC')
IDX = dict(train=300, testB=400, testC=500)
NC = dict(train=64, testB=64, testC=32)


def out_neighbors(conn, j):
    W = conn.dense_weight('cpu')
    nb = (W[j] != 0).any(0).nonzero(as_tuple=True)[0]
    return nb


def build_extra_stim(cfg, kind, a, b, I=None, g_amp=0.0):
    """Return a [T,N] extra stimulus for one branch."""
    T, N = cfg.T, cfg.n_neurons
    u = torch.zeros(T, N)
    if kind == 'delay':
        u[TAU:TAU + 4, J] = 6.0
        u[TAU + 8 + a:TAU + 8 + a + 2, J] = 6.0
    elif kind == 'burst':
        if a == 'even':
            u[TAU:TAU + 10, J] = 2.4
        elif a == 'paired':
            u[TAU:TAU + 2, J] = 6.0
            u[TAU + 6:TAU + 8, J] = 6.0
        elif a == 'burst':
            for k in range(4):
                u[TAU + 2 * k:TAU + 2 * k + 2, J] = 3.0
        elif a == 'longgap':
            u[TAU:TAU + 2, J] = 6.0
            u[TAU + 18:TAU + 20, J] = 6.0
    elif kind == 'precond':
        if a == 'edge':
            u[TAU:TAU + 4, J] = 6.0
        else:
            u[TAU:TAU + 4, I] = g_amp
        u[TAU + b:TAU + b + 2, J] = 6.0
    elif kind == 'history':
        if a == 'high':
            u[TAU - 16:TAU - 8, BG_NEURONS] = 5.0
        u[TAU:TAU + 2, A_NEURONS] = 5.0
    return u


def branch_list():
    return ([('delay', d, None) for d in DS] +
            [('burst', p, None) for p in PATTERNS] +
            [('precond', 'edge', 8), ('precond', 'global', 8),
             ('precond', 'edge', 16), ('precond', 'global', 16)] +
            [('history', 'high', None), ('history', 'low', None)])


@torch.no_grad()
def gen_branches(sim, cfg, seeds, split, I, g_amp, batch=16):
    """Generate all branches for all seeds. Returns dict kind|a|b -> states."""
    out = {}
    for kind, a, b in branch_list():
        es = torch.stack([build_extra_stim(cfg, kind, a, b, I, g_amp) for _ in range(batch)])
        chunks = []
        for i in range(0, len(seeds), batch):
            ss = seeds[i:i + batch]
            d = sim.generate(ss, split, extra_stim=es[:len(ss)])
            chunks.append(d['states'].cpu())
        out[f'{kind}|{a}|{b}'] = torch.cat(chunks)
    return out


def vauc(states, neurons, t0, t1):
    """Voltage AUC: mean V over neurons x time (above v_min baseline)."""
    return states[:, t0:t1, :, 0][:, :, neurons].mean((1, 2))


def spcount(states, neurons, t0, t1):
    return states[:, t0:t1, :, 1][:, :, neurons].sum((1, 2))


def _bnorm(w):
    """Block-wise L1 shape normalization (bounded, amplitude-free)."""
    return w / w.abs().sum().clamp(min=1e-9)


def fingerprint(branches, Inb):
    """13-d bounded shape fingerprint; every block normalized per trajectory."""
    resp_d = {}
    for d in DS:
        st = branches[f'delay|{d}|{None}']
        t0 = TAU + 8 + d
        resp_d[d] = vauc(st, Inb, t0 + 2, t0 + 8) - vauc(st, Inb, t0 - 6, t0)
    wd = torch.stack([resp_d[d] for d in DS], -1)              # [B,6]
    fd = list(_bnorm(wd).unbind(-1))                            # 6 dims (curve shape)
    resp_b = {}
    for ptn in PATTERNS:
        st = branches[f'burst|{ptn}|{None}']
        resp_b[ptn] = vauc(st, Inb, TAU + 20, TAU + 32) - vauc(st, Inb, TAU - 12, TAU)
    wb = torch.stack([resp_b[p] for p in PATTERNS], -1)        # [B,4]
    fb = list(_bnorm(wb).unbind(-1))                            # 4 dims
    wp = []
    for kind in ('edge', 'global'):
        for dt in (8, 16):
            st = branches[f'precond|{kind}|{dt}']
            wp.append(vauc(st, Inb, TAU + dt + 2, TAU + dt + 8)
                      - vauc(st, Inb, TAU + dt - 6, TAU + dt))
    wp = torch.stack(wp, -1)                                    # [B,4] e8,g8,e16,g16
    fp = list(_bnorm(wp).unbind(-1))
    wh = []
    for kind in ('high', 'low'):
        st = branches[f'history|{kind}|{None}']
        wh.append(vauc(st, A_NEURONS, TAU + 2, TAU + 10) - vauc(st, A_NEURONS, TAU - 8, TAU))
    wh = torch.stack(wh, -1)                                    # [B,2]
    fh = list(_bnorm(wh).unbind(-1))
    phi = torch.stack(fd + fb + fp + fh, -1)                    # [B,16]
    return torch.nan_to_num(phi, nan=0.0, posinf=0.0, neginf=0.0)


@torch.no_grad()
def calibrate_global_amp(cfg, conn, I, seeds):
    """Global-branch injection amp matching the edge branch's mean V integral
    in I on the NULL teacher (frozen calibration, recorded)."""
    sim = MechanismLIFSimulator(conn, cfg, torch.device('cuda'),
                                __import__('teachers_v9', fromlist=['MechSpec']).MechSpec(name='null'))
    es_edge = torch.stack([build_extra_stim(cfg, 'precond', 'edge', 8) for _ in seeds])
    d = sim.generate(seeds, 'test_seen', extra_stim=es_edge)
    tgt = vauc(d['states'].cuda(), I, TAU, TAU + 8).mean().item()
    lo, hi = 0.0, 20.0
    for _ in range(18):
        mid = (lo + hi) / 2
        es = torch.stack([build_extra_stim(cfg, 'precond', 'global', 8, I, mid) for _ in seeds])
        d = sim.generate(seeds, 'test_seen', extra_stim=es)
        val = vauc(d['states'].cuda(), I, TAU, TAU + 8).mean().item()
        if val < tgt:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2, tgt


def cls_metrics(y, prob):
    from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
    yp = prob.argmax(1)
    out = dict(bal_acc=float(balanced_accuracy_score(y, yp)),
               macro_f1=float(f1_score(y, yp, average='macro')))
    try:
        out['auroc'] = float(roc_auc_score(y, prob, multi_class='ovr', average='macro'))
    except Exception:
        out['auroc'] = float('nan')
    return out


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--part', choices=['teacher', 'model'], default='teacher')
    ap.add_argument('--seeds', nargs='+', type=int, default=list(SEEDS))
    args = ap.parse_args()
    torch.set_num_threads(2)
    cfg = setup_cfg()
    conn = Connectome.generate(cfg)
    Inb = out_neighbors(conn, J).numpy()
    dev = torch.device('cuda')
    cache = ROOT / 'data' / 'intervention_phi.npz'
    if args.part == 'teacher':
        g_amp, tgt = calibrate_global_amp(cfg, conn, Inb, [cfg.traj_seed('test_seen', i) for i in range(300, 308)])
        print(f'global amp calibrated: {g_amp:.3f} (target V AUC {tgt:.4f})', flush=True)
        phis, labels, conds_out = [], [], []
        for fam in FAMILIES:
            for cond in CONDS:
                specs = []
                if cond == 'train':
                    pids = [n for n, _ in SELECTED[fam]['train']]
                    assign = [pids[i % 3] for i in range(NC[cond])]
                elif cond == 'testB':
                    assign = [SELECTED[fam]['splitB'][0]] * NC[cond]
                else:
                    assign = [SELECTED[fam]['splitC'][0]] * NC[cond]
                # group seeds by config for batch generation
                fam_phi = []
                for pid in sorted(set(assign)):
                    idx = [i for i, a in enumerate(assign) if a == pid]
                    seeds = [cfg.traj_seed('test_seen', IDX[cond] + i) for i in idx]
                    sim = MechanismLIFSimulator(conn, cfg, dev, spec_for(fam, pid))
                    branches = gen_branches(sim, cfg, seeds, 'test_seen', Inb, g_amp)
                    fam_phi.append((idx, fingerprint(branches, Inb)))
                    del sim, branches
                    torch.cuda.empty_cache()
                    print('teacher phi', fam, cond, pid, len(seeds), flush=True)
                phi = torch.cat([p for _, p in sorted(fam_phi, key=lambda q: q[0][0])])
                phis.append(phi)
                labels += [FAMILIES.index(fam)] * len(phi)
                conds_out += [cond] * len(phi)
        # null reference fingerprints (not classified)
        sim = MechanismLIFSimulator(conn, cfg, dev,
                                    __import__('teachers_v9', fromlist=['MechSpec']).MechSpec(name='null'))
        seeds = [cfg.traj_seed('test_seen', IDX['train'] + i) for i in range(64)]
        branches = gen_branches(sim, cfg, seeds, 'test_seen', Inb, g_amp)
        null_phi = fingerprint(branches, Inb)
        np.savez(cache, phi=torch.cat(phis).numpy(), labels=np.array(labels),
                 conds=np.array(conds_out), null_phi=null_phi.numpy(), g_amp=g_amp)
        print('PHI SAVED', flush=True)
    # ---------------- classification + figures (teacher) ----------------
    z = np.load(cache)
    phi, labels, conds = z['phi'], z['labels'], z['conds']
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    rows = []
    if args.part == 'teacher':
        tr = conds == 'train'
        sc = StandardScaler().fit(phi[tr])
        clf = LogisticRegression(max_iter=3000).fit(sc.transform(phi[tr]), labels[tr])
        for cond in CONDS:
            te = conds == cond
            prob = clf.predict_proba(sc.transform(phi[te]))
            m = cls_metrics(labels[te], prob)
            rows.append(dict(stage='teacher_fingerprint', split=cond, seed=0, **m))
            print('teacher-fp', cond, m, flush=True)
            np.save(ROOT / 'metrics' / f'intervention_confusion_{cond}.npy',
                    __import__('sklearn.metrics', fromlist=['confusion_matrix']).confusion_matrix(
                        labels[te], prob.argmax(1), labels=[0, 1, 2]))
        # figure: per-family mean fingerprint
        dims = [f'd{d}' for d in DS] + list(PATTERNS) + ['edge8', 'glob8', 'edge16', 'glob16'] + ['hist_hi', 'hist_lo']
        fig, ax = plt.subplots(figsize=(11, 4.5))
        x = np.arange(16)
        for fi, fam in enumerate(FAMILIES):
            mu = phi[tr & (labels == fi)].mean(0)
            sd = phi[tr & (labels == fi)].std(0) / np.sqrt(tr.sum() // 3)
            ax.errorbar(x + fi * 0.25 - 0.25, mu, yerr=sd, marker='o', ms=4, label=fam, capsize=2)
        nmu = z['null_phi'].mean(0)
        ax.plot(x, nmu, 'k--', label='null', alpha=.7)
        ax.axhline(1.0, color='gray', ls=':', lw=.8)
        ax.set_xticks(x, dims, rotation=45, ha='right')
        ax.set_ylabel('response ratio (within-trajectory)')
        ax.set_title('v9 teacher intervention fingerprint (mean +/- s.e.m., train params)')
        ax.legend()
        fig.tight_layout()
        fig.savefig(ROOT / 'figures' / 'intervention_fingerprint.png', dpi=140)
        with (ROOT / 'metrics' / 'intervention_identification.csv').open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
        print('TEACHER FINGERPRINT COMPLETE', flush=True)


if __name__ == '__main__':
    main()
