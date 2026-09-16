"""Stage 6: representation probes on the global temporal token (z_ctx).

Regression: ridge + small MLP -> (z_pos, z_vel). Classification:
logistic -> sign(z_vel) with AUROC / accuracy / balanced accuracy.
Geometry: PCA of z_ctx, correlation of first components with z_pos/z_vel.
Backbone stays frozen; probes fit on train, ridge/early-stop on val,
evaluated on held-out windows of test_seen/test_ood.
"""
import json
import numpy as np
import torch
from run_latent_v3 import ROOT, datasets_v3
from run_latent_v2 import setup as setup_v2
from train_latent_v2 import load_model_v2
from latent_data import sample_indices, windows, history_control
from latent_probe_v2 import fit_ridge, apply_probe, probe_metrics, MLPProbe, fit_mlp, apply_mlp


@torch.no_grad()
def global_token_features(model, data, count=1024, seed=9191, control='ordered'):
    bi, ti = sample_indices(data, count, seed)
    sg = torch.Generator().manual_seed(seed + 19)
    feats, labels = [], []
    for start in range(0, count, 32):
        b, t = bi[start:start + 32], ti[start:start + 32]
        x, _ = windows(data, b, t, model.k)
        x = history_control(x, control, sg)
        _, zctx = model.encode(x)
        feats.append(zctx.cpu())
        labels.append(data['z'][b.to(x.device), t.to(x.device)].cpu())
    return torch.cat(feats), torch.cat(labels)


def classification_metrics(score, sign_label):
    s = score.double()
    y = sign_label.double()
    # AUROC via rank statistic
    order = torch.argsort(s)
    ranks = torch.empty_like(order, dtype=torch.float64)
    ranks[order] = torch.arange(1, len(s) + 1, dtype=torch.float64)
    pos = y > 0
    npos, nneg = pos.sum(), (~pos).sum()
    auroc = float((ranks[pos].sum() - npos * (npos + 1) / 2) / (npos * nneg)) if npos and nneg else None
    pred = (s > 0).double()
    acc = float((pred == y).double().mean())
    tpr = float((pred[pos] == 1).double().mean()) if pos.any() else None
    tnr = float((pred[~pos] == 0).double().mean()) if (~pos).any() else None
    bal = (tpr + tnr) / 2 if tpr is not None and tnr is not None else None
    return dict(auroc=auroc, accuracy=acc, balanced_accuracy=bal)


def fit_logistic(f, y, fv, yv, epochs=500, lr=1e-2):
    mean, std = f.mean(0), f.std(0).clamp(min=1e-5)
    w = torch.zeros(f.shape[1], dtype=torch.float64, requires_grad=True)
    b = torch.zeros((), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr)
    x = ((f - mean) / std).double()
    xv = ((fv - mean) / std).double()
    yd, yvd = y.double(), yv.double()
    best, wait, best_state = float('inf'), 0, None
    for ep in range(epochs):
        opt.zero_grad()
        logits = x @ w + b
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, yd)
        loss.backward()
        opt.step()
        with torch.no_grad():
            vl = torch.nn.functional.binary_cross_entropy_with_logits(xv @ w + b, yvd)
        if vl < best - 1e-8:
            best, wait = float(vl), 0
            best_state = (w.detach().clone(), b.detach().clone())
        else:
            wait += 1
            if wait >= 50:
                break
    w0, b0 = best_state
    return dict(mean=mean, std=std, w=w0, b=b0)


def apply_logistic(model, f):
    return ((f.double() - model['mean']) / model['std']) @ model['w'] + model['b']


def pca_geometry(f, z):
    f = f.double()
    fc = f - f.mean(0)
    u, s, v = torch.pca_lowrank(fc, q=4)
    comp = fc @ v
    out = {}
    for j in range(4):
        for tj, name in enumerate(('z_pos', 'z_vel')):
            a, b = comp[:, j], z[:, tj].double()
            cd = a.norm() * b.norm()
            out[f'pc{j + 1}_{name}'] = float(((a - a.mean()) * (b - b.mean())).sum()
                                             / ((a - a.mean()).norm() * (b - b.mean()).norm()).clamp(min=1e-12))
    return out


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', nargs='+', type=int, default=[1234, 1235, 1236, 1237, 1238])
    args = p.parse_args()
    torch.set_num_threads(2)
    cfg, lc, conn = setup_v2('hidden')
    data = datasets_v3('hidden')
    out = ROOT / 'probes'
    out.mkdir(parents=True, exist_ok=True)
    rows, geo_rows = [], []
    for seed in args.seeds:
        for label in ('global_k32', 'gshuffle'):
            summary = json.loads((ROOT / 'replication' / 'hidden' / 'training'
                                  / f'{label}_seed{seed}.json').read_text())
            model, blob = load_model_v2(summary, conn, torch.device('cuda'))
            control = summary['control']
            features = {sp: global_token_features(model, data[sp], 2048 if sp == 'train' else 1024,
                                                  control=control)
                        for sp in ('train', 'val', 'test_seen', 'test_ood')}
            ridge = fit_ridge(features['train'], features['val'])
            mlp, stats = fit_mlp((features['train'][0].float(), features['train'][1].float()),
                                 (features['val'][0].float(), features['val'][1].float()))
            sign_train = (features['train'][1][:, 1] > 0).float()
            sign_val = (features['val'][1][:, 1] > 0).float()
            logit = fit_logistic(features['train'][0], sign_train, features['val'][0], sign_val)
            for sp, (f, z) in features.items():
                if sp == 'train':
                    continue
                rm = probe_metrics(apply_probe(ridge, f), z)
                mm = probe_metrics(apply_mlp(mlp, stats, f.float()), z)
                cm = classification_metrics(apply_logistic(logit, f), (z[:, 1] > 0).float())
                rows.append(dict(model=label, seed=seed, split=sp,
                                 ridge_zpos_r2=rm['z_pos']['r2'], ridge_zvel_r2=rm['z_vel']['r2'],
                                 ridge_zpos_mae=rm['z_pos']['mae'], ridge_zvel_mae=rm['z_vel']['mae'],
                                 mlp_zpos_r2=mm['z_pos']['r2'], mlp_zvel_r2=mm['z_vel']['r2'],
                                 sign_auroc=cm['auroc'], sign_acc=cm['accuracy'],
                                 sign_balacc=cm['balanced_accuracy']))
            geo = pca_geometry(features['test_seen'][0], features['test_seen'][1])
            geo_rows.append(dict(model=label, seed=seed, **geo))
            print('probe done', label, seed, flush=True)
            del model
            torch.cuda.empty_cache()
    from calibrate_latent import write_csv
    write_csv(out / 'representation_probes.csv', rows)
    write_csv(out / 'representation_geometry.csv', geo_rows)
    print('PROBE ROWS', len(rows), flush=True)


if __name__ == '__main__':
    main()
