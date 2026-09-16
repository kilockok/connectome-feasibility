"""Stage 3+4: conditional latent decoding and residual mediation analysis.

Probe chains (all trajectory-disjoint, frozen backbone, capacity-fixed):
  z <- x(current stats) | z <- x+Set | z <- x+Derivative | z <- x+h_Ordered
Targets: z_pos, z_vel, (sin phi, cos phi) with phi from the oscillator's
elliptical coordinates, regime=sign(z_pos) classification, plus CCA and a
randomized-label control. Mediation: direct h->gain-1, latent-mediated
h->z_pos->tanh->gain-1, oracle z_pos->gain-1; descriptive mediation fraction.
"""
import csv
import json
from pathlib import Path
import numpy as np
import torch
from run_latent_v3 import ROOT as V3ROOT, datasets_v3
from run_latent_v4 import ROOT as V4ROOT
from run_latent_v2 import setup as setup_v2
from latent_data import sample_indices, windows
from alias_eval_v5 import load_checked
from latent_probe_v2 import fit_ridge, apply_probe

ROOT = Path('results/latent_state_v5')
SEEDS = (1234, 1235, 1236, 1237, 1238)
OMEGA = 0.06


@torch.no_grad()
def features_for(model, label, data, count, seed=9191):
    """Per-window representations for the probe chains."""
    bi, ti = sample_indices(data, count, seed)
    feats = {k: [] for k in ('current', 'set', 'deriv', 'ordered', 'z')}
    for st in range(0, count, 32):
        b, t = bi[st:st + 32], ti[st:st + 32]
        x, _ = windows(data, b, t, 32)
        z = data['z'][b, t]
        # current: last-token population stats
        cur = torch.cat((x[:, -1].mean(1), x[:, -1, :, 1].mean(1, keepdim=True)), -1)
        feats['current'].append(cur.cpu())
        if label == 'ordered':
            _, h = model.encode(x)
            feats['ordered'].append(h.cpu())
        elif label == 'set':
            _, h = model.encode(x)
            feats['set'].append(h.cpu())
        elif label == 'deriv':
            _, h = model.encode(x)
            feats['deriv'].append(h.cpu())
        feats['z'].append(z.cpu())
    return {k: torch.cat(v) for k, v in feats.items() if v}


def targets(z):
    zp, zv = z[:, 0], z[:, 1]
    phi = torch.atan2(zv / OMEGA, zp)
    return dict(z_pos=zp, z_vel=zv, sin_phi=torch.sin(phi), cos_phi=torch.cos(phi),
                regime=(zp > 0).float(), gain_minus1=torch.tanh(zp))


def r2(pred, true):
    pred, true = pred.double(), true.double()
    den = (true - true.mean()).square().sum()
    return float(1 - (pred - true).square().sum() / den.clamp(min=1e-12))


def clf_metrics(score, y):
    from probes_v3 import classification_metrics
    return classification_metrics(score, y)


def eval_probe_chain(feat_train, feat_val, feat_test, z_train, z_val, z_test, shuffle_labels=False):
    tg_train, tg_val, tg_test = targets(z_train), targets(z_val), targets(z_test)
    out = {}
    if shuffle_labels:
        g = torch.Generator().manual_seed(7)
        perm = torch.randperm(len(z_train), generator=g)
        z_train = z_train[perm]
        tg_train = targets(z_train)
    zcols = torch.stack([tg_train['z_pos'], tg_train['z_vel'], tg_train['sin_phi'], tg_train['cos_phi']], 1)
    model = fit_ridge((feat_train, zcols), (feat_val, torch.stack(
        [tg_val['z_pos'], tg_val['z_vel'], tg_val['sin_phi'], tg_val['cos_phi']], 1)))
    pred = apply_probe(model, feat_test)
    for j, name in enumerate(('z_pos', 'z_vel', 'sin_phi', 'cos_phi')):
        out[f'r2_{name}'] = r2(pred[:, j], tg_test[name])
    # regime via z_pos score
    out['regime_auroc'] = clf_metrics(pred[:, 0], tg_test['regime'])['auroc']
    return out


@torch.no_grad()
def cca(feat_tr, z_tr, feat_te, z_te, q=2, ridge=1e-2):
    """Fit CCA directions on train, evaluate canonical correlations on held-out."""
    f = feat_tr.double() - feat_tr.double().mean(0)
    zz = z_tr.double() - z_tr.double().mean(0)
    cov = torch.cov(torch.cat((f, zz), 1).T)
    d = f.shape[1]
    sxx, syy, sxy = cov[:d, :d], cov[d:, d:], cov[:d, d:]
    sxx = sxx + ridge * sxx.diag().mean() * torch.eye(d, dtype=sxx.dtype)
    syy = syy + ridge * syy.diag().mean() * torch.eye(syy.shape[0], dtype=syy.dtype)
    wx = torch.linalg.solve(sxx, sxy)
    wy = torch.linalg.solve(syy, sxy.T)
    fc = feat_te.double() - feat_te.double().mean(0)
    zc = z_te.double() - z_te.double().mean(0)
    out = []
    for j in range(q):
        a, b = fc @ wx[:, j], zc @ wy[:, j]
        out.append(float(((a - a.mean()) * (b - b.mean())).sum()
                         / (a.norm() * b.norm()).clamp(min=1e-12)))
    return out


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', nargs='+', type=int, default=list(SEEDS))
    args = p.parse_args()
    torch.set_num_threads(2)
    cfg, lc, conn = setup_v2('hidden')
    data = datasets_v3('hidden')
    conn100 = conn
    rows, cca_rows, med_rows = [], [], []
    for seed in args.seeds:
        reps = {}
        for label, key in (('global_k32', 'ordered'), ('set_k32', 'set'), ('deriv', 'deriv')):
            model, th = load_checked(label, seed, conn100, torch.device('cuda'))
            reps[key] = {sp: features_for(model, 'ordered' if key == 'ordered' else key,
                                          data[sp], 2048 if sp == 'train' else 1024)
                         for sp in ('train', 'val', 'test_seen', 'test_ood')}
            del model
            torch.cuda.empty_cache()
        # assemble chains: current | +set | +deriv | +ordered
        for sp in ('train', 'val', 'test_seen', 'test_ood'):
            pass
        chains = {}
        for sp in reps['ordered']:
            cur = reps['ordered'][sp]['current']
            chains[sp] = dict(
                current=cur,
                set=torch.cat((cur, reps['set'][sp]['set']), -1),
                deriv=torch.cat((cur, reps['deriv'][sp]['deriv']), -1),
                ordered=torch.cat((cur, reps['ordered'][sp]['ordered']), -1))
            chains[sp]['z'] = reps['ordered'][sp]['z']
        for chain in ('current', 'set', 'deriv', 'ordered'):
            for split in ('test_seen', 'test_ood'):
                res = eval_probe_chain(chains['train'][chain], chains['val'][chain],
                                       chains[split][chain], chains['train']['z'],
                                       chains['val']['z'], chains[split]['z'])
                res_shuf = eval_probe_chain(chains['train'][chain], chains['val'][chain],
                                            chains[split][chain], chains['train']['z'],
                                            chains['val']['z'], chains[split]['z'],
                                            shuffle_labels=True)
                rows.append(dict(seed=seed, chain=chain, split=split,
                                 **{f'{k}_shuffled' : v for k, v in res_shuf.items()}, **res))
                # mediation: gain-1 direct / mediated / oracle
        for split in ('test_seen', 'test_ood'):
            tg = lambda z: targets(z)['gain_minus1']
            for chain in ('current', 'set', 'deriv', 'ordered'):
                md = fit_ridge((chains['train'][chain], tg(chains['train']['z'])[:, None]),
                               (chains['val'][chain], tg(chains['val']['z'])[:, None]))
                direct = r2(apply_probe(md, chains[split][chain])[:, 0], tg(chains[split]['z']))
                mz = fit_ridge((chains['train'][chain], chains['train']['z'][:, 0:1]),
                               (chains['val'][chain], chains['val']['z'][:, 0:1]))
                zhat = apply_probe(mz, chains[split][chain])[:, 0]
                mediated = r2(torch.tanh(zhat), tg(chains[split]['z']))
                med_rows.append(dict(seed=seed, chain=chain, split=split,
                                     direct_gain_r2=direct, mediated_gain_r2=mediated,
                                     zhat_r2=r2(zhat, chains[split]['z'][:, 0])))
            oz = fit_ridge((chains['train']['z'][:, 0:1], tg(chains['train']['z'])[:, None]),
                           (chains['val']['z'][:, 0:1], tg(chains['val']['z'])[:, None]))
            med_rows.append(dict(seed=seed, chain='oracle_zpos', split=split,
                                 direct_gain_r2=r2(apply_probe(oz, chains[split]['z'][:, 0:1])[:, 0],
                                                   tg(chains[split]['z'])),
                                 mediated_gain_r2=None, zhat_r2=1.0))
        for chain in ('ordered', 'set'):
            cs = cca(chains['train'][chain], chains['train']['z'],
                     chains['test_seen'][chain], chains['test_seen']['z'])
            cca_rows.append(dict(seed=seed, chain=chain, cca1=cs[0], cca2=cs[1] if len(cs) > 1 else None))
        print('probes done seed', seed, flush=True)
    with (ROOT / 'table2_latent_decoding.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with (ROOT / 'table2_cca.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(cca_rows[0])); w.writeheader(); w.writerows(cca_rows)
    with (ROOT / 'table3_residual_recovery.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(med_rows[0])); w.writeheader(); w.writerows(med_rows)
    print('PROBE ROWS', len(rows), 'MED ROWS', len(med_rows), flush=True)


if __name__ == '__main__':
    main()
