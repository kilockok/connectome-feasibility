"""v9 Stage 9: open-set / unknown-mechanism test.

UNKNOWN = additive global colored current (OU), matched effect, NEVER in any
training set. NULL (base LIF) is also non-training-class. Tests every
identification channel for forced-classification behavior:
  passive probes (frozenZ linear, raw_ordered, shortcut) -> max-softmax /
  entropy / energy + AUROC known-vs-unknown;
  teacher-fingerprint classifier -> same on OU fingerprints;
  candidate fitting -> winner distribution + margin on OU (margin collapse?).
"""
import csv
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from connectome import Connectome
from run_v7 import setup_cfg
from teachers_v9 import MechanismLIFSimulator, MechSpec
from protocol_v9 import ROOT, FAMILIES, SEEDS, OU_TEST
from identify_v9 import (stratified_sets, load_store, sample_windows, shortcut_features,
                         train_raw, load_raw)
from latent_data import windows as lw
from models.residual_v8 import build_v8
from intervention_v9 import gen_branches, fingerprint, out_neighbors, J, IDX, cls_metrics

OU_IDX0 = 600
N_OU = 64


@torch.no_grad()
def gen_ou(cfg, conn):
    cache = ROOT / 'data' / 'ou_test.pt'
    if cache.exists():
        return torch.load(cache, map_location='cpu', weights_only=False)
    dev = torch.device('cuda')
    out = {}
    for pid, p in OU_TEST:
        sim = MechanismLIFSimulator(conn, cfg, dev, MechSpec(name=f'ou_{pid}', ou=p))
        seeds = [cfg.traj_seed('test_seen', OU_IDX0 + (0 if pid == 'u1' else 32) + i)
                 for i in range(32)]
        d = sim.generate(seeds, 'test_seen')
        out[pid] = {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in d.items()}
        print('ou gen', pid, flush=True)
    torch.save(out, cache)
    return out


@torch.no_grad()
def gen_ou_branches(cfg, conn, Inb, g_amp):
    cache = ROOT / 'data' / 'ou_branches.pt'
    if cache.exists():
        return torch.load(cache, map_location='cpu', weights_only=False)
    dev = torch.device('cuda')
    out = {}
    for pid, p in OU_TEST:
        sim = MechanismLIFSimulator(conn, cfg, dev, MechSpec(name=f'ou_{pid}', ou=p))
        seeds = [cfg.traj_seed('test_seen', OU_IDX0 + (0 if pid == 'u1' else 32) + i)
                 for i in range(32)]
        out[pid] = gen_branches(sim, cfg, seeds, 'test_seen', Inb, g_amp)
        print('ou branches', pid, flush=True)
        del sim
        torch.cuda.empty_cache()
    torch.save(out, cache)
    return out


def open_set_scores(prob):
    eps = 1e-12
    mx = prob.max(1)
    ent = -(prob * np.log(prob + eps)).sum(1)
    energy = -np.log(np.exp(prob - prob.max(1, keepdims=True)).sum(1) + eps)  # unnormalized logits unavailable; use softmax energy proxy
    return mx, ent, energy


def auroc_binary(pos, neg):
    from sklearn.metrics import roc_auc_score
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    s = np.concatenate([pos, neg])
    return float(roc_auc_score(y, s))


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    cfg = setup_cfg()
    conn = Connectome.generate(cfg)
    W = conn.dense_weight('cuda')
    ib = conn.i_bias.cuda() if conn.i_bias is not None else torch.zeros(cfg.n_neurons, device='cuda')
    Inb = out_neighbors(conn, J).numpy()
    store = load_store()
    strata = stratified_sets()
    ou = gen_ou(cfg, conn)
    rows = []

    # ---------- passive channels ----------
    # known = testB windows (stratified); unknown = OU windows; null = null windows
    def ou_windows(per=1024, seed=7777):
        g = torch.Generator().manual_seed(seed)
        xs = []
        for pid in ('u1', 'u2'):
            d = ou[pid]
            b = torch.randint(len(d['states']), (per // 2,), generator=g)
            t = torch.randint(31, d['stimulus'].shape[1], (per // 2,), generator=g)
            x, _ = lw(d, b, t, 32)
            xs.append(x)
        return torch.cat(xs).cuda()

    def null_windows(per=1024, seed=8888):
        d = store['null/test']
        g = torch.Generator().manual_seed(seed)
        b = torch.randint(len(d['states']), (per,), generator=g)
        t = torch.randint(31, d['stimulus'].shape[1], (per,), generator=g)
        x, _ = lw(d, b, t, 32)
        return x.cuda()

    x_ou = ou_windows()
    x_null = null_windows()

    # frozenZ linear probe (seed 1234)
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    cseed = SEEDS[0]
    summary = json.loads((ROOT / 'metrics' / 'training' / f'ordered_seed{cseed}.json').read_text())
    corr = build_v8('ordered', conn, cfg).cuda()
    corr.load_state_dict(torch.load(summary['checkpoint'], map_location='cuda',
                                    weights_only=False)['state_dict'])
    corr = corr.eval()

    def encode(x):
        return torch.cat([corr.g.encode(x[i:i + 128])[0].mean(1).cpu()
                          for i in range(0, len(x), 128)]).numpy()
    x, y, _ = sample_windows(store, strata, 'train', 4096, 6000 + cseed)
    Z = encode(x)
    sc = StandardScaler().fit(Z)
    lin = LogisticRegression(max_iter=3000).fit(sc.transform(Z), y.cpu().numpy())
    xe, ye, _ = sample_windows(store, strata, 'testB', 1024, 7000 + cseed)
    p_known = lin.predict_proba(sc.transform(encode(xe)))
    p_ou = lin.predict_proba(sc.transform(encode(x_ou)))
    p_null = lin.predict_proba(sc.transform(encode(x_null)))
    for name, pk, pu in (('frozenZ_linear', p_known, p_ou),):
        mk, ek, _ = open_set_scores(pk)
        mu, eu, _ = open_set_scores(pu)
        rows.append(dict(channel=name, known='testB', unknown='ou',
                         conf_known=float(mk.mean()), conf_unknown=float(mu.mean()),
                         ent_known=float(ek.mean()), ent_unknown=float(eu.mean()),
                         auroc_reject=auroc_binary(ek, eu)))
        print(name, 'conf known/ou', mk.mean(), mu.mean(), 'ent auroc', auroc_binary(ek, eu), flush=True)
    mk, ek, _ = open_set_scores(p_known)
    mn, en, _ = open_set_scores(p_null)
    rows.append(dict(channel='frozenZ_linear', known='testB', unknown='null',
                     conf_known=float(mk.mean()), conf_unknown=float(mn.mean()),
                     ent_known=float(ek.mean()), ent_unknown=float(en.mean()),
                     auroc_reject=auroc_binary(ek, en)))
    print('null: conf', mn.mean(), 'ent auroc', auroc_binary(ek, en), flush=True)

    # raw_ordered classifier
    rsummary = train_raw('ordered', SEEDS[0], conn, cfg, store, strata)
    raw = load_raw(rsummary, conn)
    with torch.no_grad():
        pk = torch.cat([nn.functional.softmax(raw(xe[i:i + 256]), 1) for i in range(0, len(xe), 256)]).cpu().numpy()
        pu = torch.cat([nn.functional.softmax(raw(x_ou[i:i + 256]), 1) for i in range(0, len(x_ou), 256)]).cpu().numpy()
        pn = torch.cat([nn.functional.softmax(raw(x_null[i:i + 256]), 1) for i in range(0, len(x_null), 256)]).cpu().numpy()
    for uname, p_u, in (('ou', pu), ('null', pn)):
        mk, ek, _ = open_set_scores(pk)
        mu, eu, _ = open_set_scores(p_u)
        rows.append(dict(channel='raw_ordered', known='testB', unknown=uname,
                         conf_known=float(mk.mean()), conf_unknown=float(mu.mean()),
                         ent_known=float(ek.mean()), ent_unknown=float(eu.mean()),
                         auroc_reject=auroc_binary(ek, eu)))
        print('raw_ordered', uname, mk.mean(), mu.mean(), auroc_binary(ek, eu), flush=True)

    # shortcut channel
    F = shortcut_features(x, cfg, W, ib).cpu().numpy()
    sc2 = StandardScaler().fit(F)
    clf2 = LogisticRegression(max_iter=3000).fit(sc2.transform(F), y.cpu().numpy())
    pk = clf2.predict_proba(sc2.transform(shortcut_features(xe, cfg, W, ib).cpu().numpy()))
    pu = clf2.predict_proba(sc2.transform(shortcut_features(x_ou, cfg, W, ib).cpu().numpy()))
    pn = clf2.predict_proba(sc2.transform(shortcut_features(x_null, cfg, W, ib).cpu().numpy()))
    for uname, p_u in (('ou', pu), ('null', pn)):
        mk, ek, _ = open_set_scores(pk)
        mu, eu, _ = open_set_scores(p_u)
        rows.append(dict(channel='shortcut', known='testB', unknown=uname,
                         conf_known=float(mk.mean()), conf_unknown=float(mu.mean()),
                         ent_known=float(ek.mean()), ent_unknown=float(eu.mean()),
                         auroc_reject=auroc_binary(ek, eu)))
        print('shortcut', uname, mk.mean(), mu.mean(), auroc_binary(ek, eu), flush=True)

    # ---------- fingerprint channel ----------
    g_amp = float(np.load(ROOT / 'data' / 'intervention_phi.npz')['g_amp'])
    ou_br = gen_ou_branches(cfg, conn, Inb, g_amp)
    phi_ou = torch.cat([fingerprint(ou_br['u1'], Inb), fingerprint(ou_br['u2'], Inb)]).numpy()
    z = np.load(ROOT / 'data' / 'intervention_phi.npz')
    phi_t, labels, conds = z['phi'], z['labels'], z['conds']
    tr = conds == 'train'
    scf = StandardScaler().fit(phi_t[tr])
    clff = LogisticRegression(max_iter=3000).fit(scf.transform(phi_t[tr]), labels[tr])
    te = conds == 'testB'
    pk = clff.predict_proba(scf.transform(phi_t[te]))
    pu = clff.predict_proba(scf.transform(phi_ou))
    pn = clff.predict_proba(scf.transform(z['null_phi']))
    for uname, p_u in (('ou', pu), ('null', pn)):
        mk, ek, _ = open_set_scores(pk)
        mu, eu, _ = open_set_scores(p_u)
        rows.append(dict(channel='teacher_fingerprint', known='testB', unknown=uname,
                         conf_known=float(mk.mean()), conf_unknown=float(mu.mean()),
                         ent_known=float(ek.mean()), ent_unknown=float(eu.mean()),
                         auroc_reject=auroc_binary(ek, eu)))
        print('teacher_fingerprint', uname, mk.mean(), mu.mean(), auroc_binary(ek, eu), flush=True)
    # forced-classification distribution on OU
    pred_ou = clff.predict(scf.transform(phi_ou))
    for fi, fam in enumerate(FAMILIES):
        rows.append(dict(channel='fingerprint_forced_ou', known=fam, unknown='ou',
                         conf_known=float((pred_ou == fi).mean()), conf_unknown=float('nan'),
                         ent_known=float('nan'), ent_unknown=float('nan'),
                         auroc_reject=float('nan')))
    print('forced OU distribution', {FAMILIES[i]: float((pred_ou == i).mean()) for i in range(3)}, flush=True)

    path = ROOT / 'metrics' / 'unknown_detection.csv'
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print('UNKNOWN DETECTION COMPLETE', len(rows), flush=True)


if __name__ == '__main__':
    main()
