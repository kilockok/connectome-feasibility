"""v9 Stage 5: passive mechanism identification (Gates C/D).

3-way window-level classification (gain/adapt/stp) on effect-stratified
subsets. Models: M0 ShortcutStats, M1 RawK2, M2 RawEventRich, M3 RawOrdered,
M4 Oracle (fixed-width hidden summaries), FrozenZ (frozen Phase-A corrector
representation) with linear/MLP probes. Controls: label shuffle, time
shuffle, effect-magnitude-only. Splits A (seen params), B (held-out params,
primary), C (extrapolation). No mechanism label ever enters the corrector.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from connectome import Connectome
from latent_data import windows, history_control
from run_v7 import setup_cfg
from models.residual_v8 import build_v8, event_features
from models.latent_temporal_v2 import SpatialEncoderV2, GlobalTemporalPredictorV2
from protocol_v9 import ROOT, FAMILIES, SEEDS

STATS = ('residual_v_rms', 'spike_disagree', 'f1_drop', 'rate_change', 'rate_mech')
SPLITS = ('testA', 'testB', 'testC')
K = 32
N_TRAIN = 4096
N_EVAL = 1024


# ---------------------------------------------------------------- strata
def stratified_sets(seed=0, nb=4):
    """Trajectory indices kept per (family, split) by resRMS x disagreement
    grid min-count matching (same protocol as the Gate-A audit)."""
    import csv as _csv
    rows = list(_csv.DictReader((ROOT / 'metrics' / 'effect_matching.csv').open()))
    for r in rows:
        for k in STATS:
            r[k] = float(r[k])
        r['traj'] = int(r['traj'])
    out = {}
    g = np.random.default_rng(seed)
    for split in ('train', 'val') + tuple(SPLITS):
        data = {f: [r for r in rows if r['family'] == f and r['split'] == split] for f in FAMILIES}
        if not any(data.values()):
            # val has no effect stats (early-stopping only): keep all trajectories
            for f in FAMILIES:
                out[(f, split)] = list(range(63))
            continue
        allr = [r for f in FAMILIES for r in data[f]]
        rb = np.quantile([r['residual_v_rms'] for r in allr], np.linspace(0, 1, nb + 1))
        db = np.quantile([r['spike_disagree'] for r in allr], np.linspace(0, 1, nb + 1))
        for i in range(nb):
            for j in range(nb):
                cell = {f: [r for r in data[f]
                            if rb[i] <= r['residual_v_rms'] <= rb[i + 1] + 1e-9
                            and db[j] <= r['spike_disagree'] <= db[j + 1] + 1e-9]
                        for f in FAMILIES}
                n = min(len(c) for c in cell.values())
                if n == 0:
                    continue
                for f in FAMILIES:
                    idx = g.choice(len(cell[f]), size=n, replace=False)
                    out.setdefault((f, split), []).extend(int(cell[f][k]['traj']) for k in idx)
    return out


# ---------------------------------------------------------------- windows
def load_store():
    blob = torch.load(ROOT / 'data' / 'v9_data.pt', map_location='cpu', weights_only=False)
    return blob['store']


def sample_windows(store, strata, split, per_family, seed):
    """Balanced windows from stratified trajectories. Returns x [B,K,N,4],
    y [B], plus hidden summaries [B,K,2]."""
    g = torch.Generator().manual_seed(seed)
    xs, ys, hs = [], [], []
    for fi, fam in enumerate(FAMILIES):
        d = store[f'{fam}/{split}']
        keep = torch.tensor(strata[(fam, split)])
        b = keep[torch.randint(len(keep), (per_family,), generator=g)]
        t = torch.randint(31, d['stimulus'].shape[1], (per_family,), generator=g)
        x, _ = windows(d, b, t, K)
        xs.append(x)
        ys.append(torch.full((per_family,), fi, dtype=torch.long))
        ends = t[:, None] + torch.arange(1 - K, 1)
        hs.append(d['hidden_summary'][b[:, None], ends])
    return torch.cat(xs).cuda(), torch.cat(ys).cuda(), torch.cat(hs).cuda()


# ---------------------------------------------------------------- features
def shortcut_features(x, cfg, W, i_bias):
    """Window-legal global stats only [B,9]."""
    v, s, r, u = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    isyn = s[:, -1] @ W
    res = []
    for t in range(1, x.shape[1]):
        vn = v[:, t - 1] + cfg.alpha * (-(v[:, t - 1] - cfg.v_rest)
                                        + (s[:, t - 1] @ W) + u[:, t - 1] + i_bias)
        vn = vn.clamp(min=cfg.v_min)
        res.append((v[:, t] - vn))
    res = torch.stack(res, 1)
    refr = r > 0
    free = ~refr[:, 1:]
    res_free = (res * free).square().sum((1, 2)) / free.sum((1, 2)).clamp(min=1)
    fire_base = ((~refr[:, :-1]) & (vn_dummy(x, cfg, W, i_bias) >= cfg.v_th)).float()
    flip = ((s[:, 1:] - fire_base).abs() * free).sum((1, 2)) / free.sum((1, 2)).clamp(min=1)
    feats = torch.stack((
        s.mean((1, 2)), v.mean((1, 2)), v.std((1, 2)), r.mean((1, 2)),
        u.mean((1, 2)), isyn.abs().mean(1), res_free.sqrt(), flip,
        s[:, -8:].mean((1, 2))), -1)
    return feats


def vn_dummy(x, cfg, W, i_bias):
    v, s, r, u = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    vn = v[:, :-1] + cfg.alpha * (-(v[:, :-1] - cfg.v_rest)
                                  + torch.einsum('bkj,ji->bki', s[:, :-1], W)
                                  + u[:, :-1] + i_bias)
    return vn.clamp(min=cfg.v_min)


def eventrich_features(x):
    ef = event_features(x, 'rich')          # [B,N,9]
    return torch.cat((ef.mean(1), ef.amax(1)), -1)      # [B,18]


def oracle_features(h):
    """Fixed-width [B,10] hidden-drive summaries (same width for all families)."""
    outs = []
    for ch in range(2):
        z = h[..., ch]
        m = z.mean(1)
        sd = z.std(1)
        mn = z.min(1).values
        mx = z.max(1).values
        zc = z - m[:, None]
        ac = ((zc[:, 1:] * zc[:, :-1]).sum(1) / (zc * zc).sum(1).clamp(min=1e-9))
        outs += [m, sd, mn, mx, ac]
    return torch.stack(outs, -1)


def effectmag_features(x, cfg, W, i_bias):
    f = shortcut_features(x, cfg, W, i_bias)
    return f[:, [6]]


# ---------------------------------------------------------------- raw classifiers
class RawK2(nn.Module):
    def __init__(self, conn, d=64, layers=2, nclass=3):
        super().__init__()
        self.spatial = SpatialEncoderV2(conn, d, layers)
        self.head = nn.Linear(2 * d, nclass)

    def forward(self, x):
        h = torch.cat((self.spatial(x[:, -1]), self.spatial(x[:, -2])), -1)
        return self.head(h.mean(1))


class RawOrdered(nn.Module):
    def __init__(self, conn, d=64, layers=2, nclass=3):
        super().__init__()
        self.g = GlobalTemporalPredictorV2(conn, k=K, d=d, layers=layers)
        self.head = nn.Linear(2 * d, nclass)

    def forward(self, x):
        fused, _ = self.g.encode(x)
        return self.head(fused.mean(1))


def train_raw(kind, seed, conn, cfg, store, strata, steps=1500, batch=64, shuffle_labels=False):
    tag = f'{kind}_seed{seed}' + ('_labelshuffle' if shuffle_labels else '')
    summary_path = ROOT / 'metrics' / 'identify' / f'{tag}.json'
    if summary_path.exists():
        return json.loads(summary_path.read_text())
    torch.manual_seed(seed)
    model = (RawK2(conn) if kind == 'k2' else RawOrdered(conn)).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    x, y, _ = sample_windows(store, strata, 'train', N_TRAIN, 5000 + seed)
    xv, yv, _ = sample_windows(store, strata, 'val', 1024, 9001)
    if shuffle_labels:
        y = y[torch.randperm(len(y))]
    best, best_state = -1.0, None
    for step in range(1, steps + 1):
        idx = torch.randint(0, len(x), (batch,), device='cuda')
        logits = model(x[idx])
        loss = nn.functional.cross_entropy(logits, y[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 250 == 0:
            model.eval()
            with torch.no_grad():
                acc = sum((model(xv[i:i + 256]).argmax(1) == yv[i:i + 256]).float().mean()
                          for i in range(0, len(xv), 256)) / (len(xv) // 256 + 1)
            if float(acc) > best:
                best = float(acc)
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            model.train()
            print(f'{tag} step={step} val_acc={float(acc):.4f}', flush=True)
    ckpt = ROOT / 'checkpoints' / 'identify' / f'{tag}.pt'
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state_dict=best_state, kind=kind, seed=seed,
                    shuffle_labels=shuffle_labels, val_acc=best), ckpt)
    summary = dict(kind=kind, seed=seed, shuffle_labels=shuffle_labels,
                   val_acc=best, checkpoint=str(ckpt))
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))
    del model, opt
    torch.cuda.empty_cache()
    return summary


def load_raw(summary, conn):
    model = (RawK2(conn) if summary['kind'] == 'k2' else RawOrdered(conn)).cuda()
    model.load_state_dict(torch.load(summary['checkpoint'], map_location='cuda',
                                     weights_only=False)['state_dict'])
    return model.eval()


# ---------------------------------------------------------------- metrics
def cls_metrics(y_true, prob):
    from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score, confusion_matrix
    yp = prob.argmax(1)
    out = dict(bal_acc=float(balanced_accuracy_score(y_true, yp)),
               macro_f1=float(f1_score(y_true, yp, average='macro')))
    try:
        out['auroc'] = float(roc_auc_score(y_true, prob, multi_class='ovr', average='macro'))
    except Exception:
        out['auroc'] = float('nan')
    out['confusion'] = confusion_matrix(y_true, yp, labels=[0, 1, 2]).tolist()
    return out


# ---------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', nargs='+', type=int, default=list(SEEDS))
    args = p.parse_args()
    torch.set_num_threads(2)
    cfg = setup_cfg()
    conn = Connectome.generate(cfg)
    W = conn.dense_weight('cuda')
    ib = conn.i_bias.cuda() if conn.i_bias is not None else torch.zeros(cfg.n_neurons, device='cuda')
    store = load_store()
    strata = stratified_sets()
    results = []

    def record(model, split, seed, prob, y, note=''):
        m = cls_metrics(y.cpu().numpy(), prob)
        row = dict(model=model, split=split, seed=seed, note=note,
                   bal_acc=m['bal_acc'], macro_f1=m['macro_f1'], auroc=m['auroc'])
        results.append((row, m['confusion']))
        print(model, split, seed, note, f"bal={m['bal_acc']:.3f} f1={m['macro_f1']:.3f}", flush=True)

    # ---- feature probes (M0/M2/M4/effectmag): 5 window-sampling seeds ----
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    feat_fns = dict(shortcut=lambda x, h: shortcut_features(x, cfg, W, ib),
                    eventrich=lambda x, h: eventrich_features(x),
                    oracle=lambda x, h: oracle_features(h),
                    effectmag=lambda x, h: effectmag_features(x, cfg, W, ib))
    for name, fn in feat_fns.items():
        for wseed in SEEDS:
            x, y, h = sample_windows(store, strata, 'train', N_TRAIN, 6000 + wseed)
            F = fn(x, h).cpu().numpy()
            sc = StandardScaler().fit(F)
            clf = LogisticRegression(max_iter=3000).fit(sc.transform(F), y.cpu().numpy())
            for split in SPLITS:
                xe, ye, he = sample_windows(store, strata, split, N_EVAL, 7000 + wseed)
                Fe = fn(xe, he).cpu().numpy()
                prob = clf.predict_proba(sc.transform(Fe))
                record(name, split, wseed, prob, ye)
            del x, F
            torch.cuda.empty_cache()

    # ---- raw trained classifiers (M1 k2, M3 ordered) ----
    for kind in ('k2', 'ordered'):
        for seed in args.seeds:
            summary = train_raw(kind, seed, conn, cfg, store, strata)
            model = load_raw(summary, conn)
            with torch.no_grad():
                for split in SPLITS:
                    xe, ye, _ = sample_windows(store, strata, split, N_EVAL, 7000 + seed)
                    prob = torch.cat([nn.functional.softmax(model(xe[i:i + 256]), 1)
                                      for i in range(0, len(xe), 256)]).cpu().numpy()
                    record(f'raw_{kind}', split, seed, prob, ye)
                    # time-shuffle control
                    xs = history_control(xe, 'shuffle', torch.Generator().manual_seed(4242))
                    prob = torch.cat([nn.functional.softmax(model(xs[i:i + 256]), 1)
                                      for i in range(0, len(xs), 256)]).cpu().numpy()
                    record(f'raw_{kind}', split, seed, prob, ye, note='time_shuffle')
            del model
            torch.cuda.empty_cache()

    # ---- label-shuffle control on the ordered classifier ----
    for seed in (SEEDS[0], SEEDS[1]):
        summary = train_raw('ordered', seed, conn, cfg, store, strata, shuffle_labels=True)
        model = load_raw(summary, conn)
        with torch.no_grad():
            xe, ye, _ = sample_windows(store, strata, 'testB', N_EVAL, 7000 + seed)
            prob = torch.cat([nn.functional.softmax(model(xe[i:i + 256]), 1)
                              for i in range(0, len(xe), 256)]).cpu().numpy()
            record('raw_ordered', 'testB', seed, prob, ye, note='label_shuffle')
        del model
        torch.cuda.empty_cache()

    # ---- FrozenZ probes ----
    from models.residual_v8 import ResidualModelV8  # noqa
    for cseed in args.seeds:
        summary = json.loads((ROOT / 'metrics' / 'training' / f'ordered_seed{cseed}.json').read_text())
        corr = build_v8('ordered', conn, cfg).cuda()
        corr.load_state_dict(torch.load(summary['checkpoint'], map_location='cuda',
                                        weights_only=False)['state_dict'])
        corr = corr.eval()
        def encode(x):
            with torch.no_grad():
                return torch.cat([corr.g.encode(x[i:i + 128])[0].mean(1).cpu()
                                  for i in range(0, len(x), 128)]).numpy()
        x, y, _ = sample_windows(store, strata, 'train', N_TRAIN, 6000 + cseed)
        Z = encode(x)
        ynp = y.cpu().numpy()
        sc = StandardScaler().fit(Z)
        lin = LogisticRegression(max_iter=3000).fit(sc.transform(Z), ynp)
        mlp = nn.Sequential(nn.Linear(Z.shape[1], 64), nn.GELU(), nn.Linear(64, 3)).cuda()
        opt = torch.optim.AdamW(mlp.parameters(), lr=1e-3, weight_decay=1e-4)
        Zt = torch.tensor(sc.transform(Z), dtype=torch.float32, device='cuda')
        yt = torch.tensor(ynp, device='cuda')
        for step in range(800):
            idx = torch.randint(0, len(Zt), (256,), device='cuda')
            loss = nn.functional.cross_entropy(mlp(Zt[idx]), yt[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        for split in SPLITS:
            xe, ye, _ = sample_windows(store, strata, split, N_EVAL, 7000 + cseed)
            Ze = encode(xe)
            prob = lin.predict_proba(sc.transform(Ze))
            record('frozenZ_linear', split, cseed, prob, ye)
            with torch.no_grad():
                prob = nn.functional.softmax(
                    mlp(torch.tensor(sc.transform(Ze), dtype=torch.float32, device='cuda')), 1).cpu().numpy()
            record('frozenZ_mlp', split, cseed, prob, ye)
        # label-shuffle control on the linear probe
        ysh = np.random.default_rng(7).permutation(ynp)
        lin_sh = LogisticRegression(max_iter=3000).fit(sc.transform(Z), ysh)
        xe, ye, _ = sample_windows(store, strata, 'testB', N_EVAL, 7000 + cseed)
        Ze = encode(xe)
        record('frozenZ_linear', 'testB', cseed, lin_sh.predict_proba(sc.transform(Ze)), ye,
               note='label_shuffle')
        del corr, mlp
        torch.cuda.empty_cache()

    # ---- write ----
    mdir = ROOT / 'metrics'
    with (mdir / 'passive_identification.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(results[0][0]))
        w.writeheader()
        for row, _ in results:
            w.writerow(row)
    conf = {f"{r['model']}|{r['split']}|{r['seed']}|{r['note']}": c for r, c in results}
    np.savez(mdir / 'passive_confusion.npz', **conf)
    print('PASSIVE IDENTIFICATION COMPLETE', len(results), flush=True)


if __name__ == '__main__':
    main()

