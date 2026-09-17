"""v9 analysis: representation geometry, calibration, parameter generalization.

Geometry: does the frozen corrector representation cluster by MECHANISM
FAMILY or by residual amplitude / firing regime? LDA ratio, silhouette,
family-vs-effect linear probes, PCA (figure only).
Calibration: reliability/ECE/Brier of the frozenZ-linear probe on testA/B;
confidence on NULL/OU/testB.
Parameter generalization: identification accuracy on seen (testA) vs
interpolated-held-out (testB) vs extrapolated (testC) parameters.
"""
import csv
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from connectome import Connectome
from run_v7 import setup_cfg
from protocol_v9 import ROOT, FAMILIES, SEEDS
from identify_v9 import (stratified_sets, load_store, sample_windows, shortcut_features)
from models.residual_v8 import build_v8


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    cfg = setup_cfg()
    conn = Connectome.generate(cfg)
    W = conn.dense_weight('cuda')
    ib = conn.i_bias.cuda() if conn.i_bias is not None else torch.zeros(cfg.n_neurons, device='cuda')
    store = load_store()
    strata = stratified_sets()
    cseed = SEEDS[0]
    summary = json.loads((ROOT / 'metrics' / 'training' / f'ordered_seed{cseed}.json').read_text())
    corr = build_v8('ordered', conn, cfg).cuda()
    corr.load_state_dict(torch.load(summary['checkpoint'], map_location='cuda',
                                    weights_only=False)['state_dict'])
    corr = corr.eval()

    def encode(x):
        return torch.cat([corr.g.encode(x[i:i + 128])[0].mean(1).cpu()
                          for i in range(0, len(x), 128)]).numpy()

    # ---------------- geometry ----------------
    x, y, _ = sample_windows(store, strata, 'train', 3072, 6000 + cseed)
    Z = encode(x)
    eff = shortcut_features(x, cfg, W, ib)[:, 6].cpu().numpy()   # in-window residual RMS
    ynp = y.cpu().numpy()
    Zc = Z - Z.mean(0)
    # LDA ratio: between-family scatter / within-family scatter
    Sw = np.zeros(Z.shape[1])
    mu = {}
    for fi in range(3):
        Sw += Zc[ynp == fi].var(0) * (ynp == fi).sum()
        mu[fi] = Zc[ynp == fi].mean(0)
    Sw /= len(Z)
    mu_all = np.stack([mu[i] for i in range(3)])
    Sb = mu_all.var(0)
    lda_ratio = float((Sb / (Sw + 1e-12)).mean())
    # effect-size clustering comparison: tercile labels
    terc = np.quantile(eff, [1/3, 2/3])
    elab = np.digitize(eff, terc)
    Sw_e = np.zeros(Z.shape[1])
    mu_e = {}
    for fi in range(3):
        Sw_e += Zc[elab == fi].var(0) * (elab == fi).sum()
        mu_e[fi] = Zc[elab == fi].mean(0)
    Sw_e /= len(Z)
    Sb_e = np.stack([mu_e[i] for i in range(3)]).var(0)
    lda_ratio_eff = float((Sb_e / (Sw_e + 1e-12)).mean())
    # silhouette
    from sklearn.metrics import silhouette_score
    sil_fam = float(silhouette_score(Zc, ynp, sample_size=3000, random_state=0))
    sil_eff = float(silhouette_score(Zc, elab, sample_size=3000, random_state=0))
    # linear probes: family vs effect from z
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score
    scz = StandardScaler().fit(Z)
    fam_acc = float(cross_val_score(LogisticRegression(max_iter=2000), scz.transform(Z), ynp, cv=3).mean())
    eff_r2 = float(cross_val_score(Ridge(), scz.transform(Z), eff, cv=3, scoring='r2').mean())
    print(f'geometry: lda_fam {lda_ratio:.5f} lda_eff {lda_ratio_eff:.5f} '
          f'sil_fam {sil_fam:.4f} sil_eff {sil_eff:.4f} fam_acc {fam_acc:.3f} eff_r2 {eff_r2:.3f}', flush=True)
    geom = dict(lda_family=lda_ratio, lda_effect=lda_ratio_eff, silhouette_family=sil_fam,
                silhouette_effect=sil_eff, family_probe_acc=fam_acc, effect_probe_r2=eff_r2)
    (ROOT / 'metrics' / 'representation_geometry.json').write_text(json.dumps(geom, indent=1))

    # PCA figure (illustration only)
    from sklearn.decomposition import PCA
    pc = PCA(2).fit_transform(scz.transform(Z))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for fi, fam in enumerate(FAMILIES):
        axes[0].scatter(pc[ynp == fi, 0], pc[ynp == fi, 1], s=2, alpha=.3, label=fam)
    axes[0].set_title('frozen Z PCA colored by family')
    axes[0].legend(markerscale=4)
    sc_ax = axes[1].scatter(pc[:, 0], pc[:, 1], s=2, c=eff, cmap='viridis', alpha=.4)
    axes[1].set_title('frozen Z PCA colored by effect size')
    fig.colorbar(sc_ax, ax=axes[1], label='in-window residual RMS')
    fig.tight_layout()
    fig.savefig(ROOT / 'figures' / 'representation_geometry.png', dpi=140)

    # ---------------- calibration ----------------
    xe, ye, _ = sample_windows(store, strata, 'testB', 1024, 7000 + cseed)
    Ze = encode(xe)
    from sklearn.linear_model import LogisticRegression as LR
    lin = LR(max_iter=3000).fit(scz.transform(Z), ynp)
    rows = []
    for split in ('testA', 'testB', 'testC'):
        xs, ys, _ = sample_windows(store, strata, split, 1024, 7000 + cseed)
        ps = lin.predict_proba(scz.transform(encode(xs)))
        ynp2 = ys.cpu().numpy()
        # ECE + Brier (one-vs-rest macro)
        conf = ps.max(1)
        corr_pred = (ps.argmax(1) == ynp2).astype(float)
        bins = np.linspace(0, 1, 11)
        ece = 0.0
        reli = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (conf >= lo) & (conf < hi)
            if m.sum() > 0:
                ece += float(m.sum() / len(conf) * abs(conf[m].mean() - corr_pred[m].mean()))
                reli.append((float(conf[m].mean()), float(corr_pred[m].mean()), int(m.sum())))
        brier = float(np.mean(np.sum((ps - np.eye(3)[ynp2]) ** 2, 1)))
        rows.append(dict(split=split, ece=ece, brier=brier, conf_mean=float(conf.mean()),
                         acc=float(corr_pred.mean())))
        print('calibration', split, rows[-1], flush=True)
    with (ROOT / 'metrics' / 'calibration.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    # reliability figure
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot([0, 1], [0, 1], 'k--', alpha=.5)
    for c, m, n in reli:
        ax.plot(c, m, 'o', ms=4 + n / 200)
    ax.set_xlabel('predicted confidence'); ax.set_ylabel('empirical accuracy')
    ax.set_title(f'frozenZ-linear reliability (testB, ECE={rows[1]["ece"]:.3f})')
    fig.tight_layout()
    fig.savefig(ROOT / 'figures' / 'calibration.png', dpi=140)

    # ---------------- parameter generalization ----------------
    pr = list(csv.DictReader((ROOT / 'metrics' / 'passive_identification.csv').open()))
    fig, ax = plt.subplots(figsize=(6.5, 4))
    models = ['shortcut', 'raw_k2', 'raw_ordered', 'frozenZ_linear', 'oracle']
    xm = np.arange(len(models))
    for j, sp in enumerate(('testA', 'testB', 'testC')):
        acc = []
        for m in models:
            v = [float(r['bal_acc']) for r in pr if r['model'] == m and r['split'] == sp and r['note'] == '']
            acc.append(np.mean(v) if v else np.nan)
        ax.plot(xm + j * 0.25 - 0.25, acc, 'o-', label={'testA': 'seen params', 'testB': 'held-out params', 'testC': 'extrapolated'}[sp])
    ax.axhline(1 / 3, color='gray', ls=':')
    ax.set_xticks(xm, models, rotation=20)
    ax.set_ylabel('balanced accuracy')
    ax.set_title('passive identification vs parameter distance')
    ax.legend()
    fig.tight_layout()
    fig.savefig(ROOT / 'figures' / 'parameter_generalization.png', dpi=140)
    print('ANALYSIS COMPLETE', flush=True)


if __name__ == '__main__':
    main()
