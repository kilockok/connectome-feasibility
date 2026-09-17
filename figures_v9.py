"""v9 final figures + lda_effect fix + confusion matrices."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('results/latent_state_v9')
FAMS = ('gain', 'adapt', 'stp')

# ---- lda_effect recompute (clean) ----
geom = json.loads((ROOT / 'metrics' / 'representation_geometry.json').read_text())
import torch
from connectome import Connectome
from run_v7 import setup_cfg
from protocol_v9 import SEEDS
from identify_v9 import stratified_sets, load_store, sample_windows, shortcut_features
from models.residual_v8 import build_v8
torch.set_num_threads(2)
cfg = setup_cfg()
conn = Connectome.generate(cfg)
W = conn.dense_weight('cuda')
ib = conn.i_bias.cuda() if conn.i_bias is not None else torch.zeros(cfg.n_neurons, device='cuda')
store = load_store()
strata = stratified_sets()
summary = json.loads((ROOT / 'metrics' / 'training' / f'ordered_seed{SEEDS[0]}.json').read_text())
corr = build_v8('ordered', conn, cfg).cuda()
corr.load_state_dict(torch.load(summary['checkpoint'], map_location='cuda', weights_only=False)['state_dict'])
corr = corr.eval()
with torch.no_grad():
    x, y, _ = sample_windows(store, strata, 'train', 3072, 6000 + SEEDS[0])
    Z = torch.cat([corr.g.encode(x[i:i + 128])[0].mean(1).cpu() for i in range(0, len(x), 128)]).numpy()
    eff = shortcut_features(x, cfg, W, ib)[:, 6].cpu().numpy()
ok = np.isfinite(eff)
Z, eff, ynp = Z[ok], eff[ok], y.cpu().numpy()[ok]
Zc = Z - Z.mean(0)
terc = np.quantile(eff, [1 / 3, 2 / 3])
elab = np.digitize(eff, terc)
Sw_e = np.zeros(Z.shape[1])
mu_e = []
for fi in range(3):
    sel = elab == fi
    Sw_e += Zc[sel].var(0) * sel.sum()
    mu_e.append(Zc[sel].mean(0))
Sw_e /= len(Z)
Sb_e = np.stack(mu_e).var(0)
geom['lda_effect'] = float((Sb_e / (Sw_e + 1e-12)).mean())
(ROOT / 'metrics' / 'representation_geometry.json').write_text(json.dumps(geom, indent=1))
print('lda_effect', geom['lda_effect'], 'lda_family', geom['lda_family'])
del corr, x, Z
torch.cuda.empty_cache()

# ---- passive confusion figure ----
pr = list(csv.DictReader((ROOT / 'metrics' / 'passive_identification.csv').open()))
cm = np.load(ROOT / 'metrics' / 'passive_confusion.npz')
fig, axes = plt.subplots(2, 4, figsize=(15, 7))
show = [('shortcut', 'testB'), ('eventrich', 'testB'), ('raw_k2', 'testB'), ('raw_ordered', 'testB'),
        ('frozenZ_linear', 'testB'), ('oracle', 'testB'), ('raw_ordered', 'testA'), ('raw_ordered', 'testC')]
for ax, (m, sp) in zip(axes.flat, show):
    keys = [k for k in cm.files if k.startswith(f'{m}|{sp}|') and k.endswith('|')]
    C = np.mean([cm[k] for k in keys], 0)
    Cn = C / C.sum(1, keepdims=True)
    im = ax.imshow(Cn, vmin=0, vmax=1, cmap='Blues')
    ax.set_title(f'{m} ({sp})', fontsize=9)
    ax.set_xticks(range(3), FAMS, rotation=45, fontsize=8)
    ax.set_yticks(range(3), FAMS, fontsize=8)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f'{Cn[i, j]:.2f}', ha='center', va='center', fontsize=8)
fig.suptitle('v9 passive identification confusion (row-normalized, mean over seeds)')
fig.tight_layout()
fig.savefig(ROOT / 'figures' / 'passive_confusion.png', dpi=130)

# ---- intervention confusion figure ----
fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
for ax, sp in zip(axes, ('train', 'testB', 'testC')):
    C = np.load(ROOT / 'metrics' / f'intervention_confusion_{sp}.npy')
    Cn = C / C.sum(1, keepdims=True)
    ax.imshow(Cn, vmin=0, vmax=1, cmap='Greens')
    ax.set_title(f'teacher fingerprint ({sp})', fontsize=9)
    ax.set_xticks(range(3), FAMS, rotation=45, fontsize=8)
    ax.set_yticks(range(3), FAMS, fontsize=8)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f'{Cn[i, j]:.2f}', ha='center', va='center', fontsize=9)
fig.suptitle('v9 intervention fingerprint confusion (row-normalized)')
fig.tight_layout()
fig.savefig(ROOT / 'figures' / 'intervention_confusion.png', dpi=130)

# ---- candidate fit figure ----
cf = [r for r in csv.DictReader((ROOT / 'metrics' / 'candidate_fitting.csv').open())
      if r['stage'] == 'candidate_select']
cands = ('null', 'gain', 'adapt', 'stp')
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax, sp in zip(axes, ('testB', 'testC')):
    fams_sp = sorted({r['family'] for r in cf if r['split'] == sp})
    C = np.zeros((4, 4))
    for r in cf:
        if r['split'] == sp:
            C[fams_sp.index(r['family']) if r['family'] in fams_sp else 0][cands.index(r['candidate'])] += 1
    labels_y = fams_sp
    Cn = C / C.sum(1, keepdims=True)
    ax.imshow(Cn, vmin=0, vmax=1, cmap='Oranges')
    ax.set_title(f'candidate fitting ({sp})', fontsize=9)
    ax.set_xticks(range(4), cands, rotation=45, fontsize=8)
    ax.set_yticks(range(len(labels_y)), labels_y, fontsize=8)
    for i in range(len(labels_y)):
        for j in range(4):
            ax.text(j, i, f'{Cn[i, j]:.2f}', ha='center', va='center', fontsize=8)
fig.suptitle('candidate fitting: family selected on UNSEEN intervention segment (row-normalized)')
fig.tight_layout()
fig.savefig(ROOT / 'figures' / 'candidate_fit.png', dpi=130)

# ---- cross-mechanism story figure ----
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
# 1: passive identification
models = ['effectmag', 'shortcut', 'eventrich', 'raw_k2', 'frozenZ_linear', 'raw_ordered', 'oracle']
acc = {sp: [] for sp in ('testA', 'testB', 'testC')}
for m in models:
    for sp in acc:
        v = [float(r['bal_acc']) for r in pr if r['model'] == m and r['split'] == sp and r['note'] == '']
        acc[sp].append(np.mean(v) if v else np.nan)
xm = np.arange(len(models))
for j, sp in enumerate(('testA', 'testB', 'testC')):
    axes[0].plot(xm, acc[sp], 'o-', label=sp, ms=4)
axes[0].axhline(1 / 3, color='gray', ls=':')
axes[0].set_xticks(xm, models, rotation=35, ha='right', fontsize=8)
axes[0].set_ylabel('balanced accuracy')
axes[0].set_title('passive identification')
axes[0].legend(fontsize=8)
# 2: intervention vs passive
ii = list(csv.DictReader((ROOT / 'metrics' / 'intervention_identification.csv').open()))
stages = ['passive best\n(raw_ordered)', 'shortcut', 'teacher\nfingerprint', 'model\nfingerprint', 'base\nfingerprint']
def geti(stage, sp):
    v = [float(r['bal_acc']) for r in ii if r['stage'] == stage and r['split'] == sp]
    return np.mean(v) if v else np.nan
vals_b = [np.mean([float(r['bal_acc']) for r in pr if r['model'] == 'raw_ordered' and r['split'] == 'testB' and r['note'] == '']),
          np.mean([float(r['bal_acc']) for r in pr if r['model'] == 'shortcut' and r['split'] == 'testB' and r['note'] == '']),
          geti('teacher_fingerprint', 'testB'), geti('model_fingerprint', 'testB'), geti('base_fingerprint', 'testB')]
vals_c = [np.mean([float(r['bal_acc']) for r in pr if r['model'] == 'raw_ordered' and r['split'] == 'testC' and r['note'] == '']),
          np.mean([float(r['bal_acc']) for r in pr if r['model'] == 'shortcut' and r['split'] == 'testC' and r['note'] == '']),
          geti('teacher_fingerprint', 'testC'), geti('model_fingerprint', 'testC'), geti('base_fingerprint', 'testC')]
xs = np.arange(len(stages))
axes[1].bar(xs - 0.2, vals_b, 0.4, label='held-out params')
axes[1].bar(xs + 0.2, vals_c, 0.4, label='extrapolated')
axes[1].axhline(1 / 3, color='gray', ls=':')
axes[1].set_xticks(xs, stages, fontsize=8)
axes[1].set_title('intervention fingerprint vs passive')
axes[1].legend(fontsize=8)
# 3: candidate fitting accuracy per family
acc_fam = {}
for r in cf:
    if r['split'] == 'testB':
        acc_fam.setdefault(r['family'], []).append(float(r['correct']))
fam_names = list(acc_fam)
axes[2].bar(fam_names, [np.mean(v) for v in acc_fam.values()], color='tab:orange')
axes[2].axhline(0.25, color='gray', ls=':')
axes[2].set_ylabel('accuracy (winner = true family)')
axes[2].set_title('candidate fitting, unseen-intervention selection (testB)')
fig.suptitle('v9 cross-mechanism summary')
fig.tight_layout()
fig.savefig(ROOT / 'figures' / 'cross_mechanism_story.png', dpi=130)
print('FIGURES COMPLETE')
