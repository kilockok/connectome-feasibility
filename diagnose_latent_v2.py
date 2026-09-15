"""Stage 9: observability diagnostics BEFORE main training.

Estimator ladder for (z_pos, z_vel) at the window endpoint:
  E0 current_only        flattened x[t] -> z
  E1 unordered_stats     pooled window statistics -> z (ridge)
  E2 ordered_flatlinear  flattened x[t-K:t] -> z (dual-form ridge)
  E3 unordered_mlp       pooled window statistics -> z (small MLP)
  E4 ordered_seq         per-step population features -> GRU -> z
  E5 shuffled_seq        same as E4, history shuffled (current token fixed)

Gates into main training (fixed before results):
  oracle ceiling passed (Stage 8) AND E4 > E0 on z_pos R2 AND
  E4 > max(E1, E3) on z_pos R2 AND E4 > E5 on z_pos R2.
"""
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from run_latent_v2 import ROOT, setup, datasets
from latent_data import sample_indices, windows, history_control

K = 32
OUT = ROOT / 'observability'


def ridge_fit(fx, z, fv, zv, lambdas=(.1, 1., 10., 100., 1000.)):
    """Closed-form ridge, lambda selected per target on val. Dual form for wide f."""
    fx, fv = fx.double(), fv.double()
    mean, std = fx.mean(0), fx.std(0).clamp(min=1e-6)
    x = (fx - mean) / std
    v = (fv - mean) / std
    zm = z.double().mean(0)
    gram = x @ x.T
    best = {}
    for lam in lambdas:
        w = x.T @ torch.linalg.solve(gram + lam * torch.eye(len(gram)).double(), (z.double() - zm))
        err = ((v @ w + zm) - zv.double()).square().mean(0)
        for j in range(z.shape[1]):
            if j not in best or err[j] < best[j][0]:
                best[j] = (err[j], lam, w[:, j])
    weights = torch.stack([best[j][2] for j in range(z.shape[1])], 1)
    return dict(mean=mean, std=std, zm=zm, w=weights,
                lambdas=[best[j][1] for j in range(z.shape[1])])


def ridge_apply(model, f):
    return ((f.double() - model['mean']) / model['std']) @ model['w'] + model['zm']


def metrics(pred, true):
    pred, true = pred.double(), true.double()
    out = {}
    for j, name in enumerate(('z_pos', 'z_vel')):
        p, t = pred[:, j], true[:, j]
        den = (t - t.mean()).square().sum()
        pc, tc = p - p.mean(), t - t.mean()
        out[name] = dict(r2=float(1 - (p - t).square().sum() / den) if den > 1e-12 else None,
                         mae=float((p - t).abs().mean()),
                         corr=float((pc * tc).sum() / (pc.norm() * tc.norm()).clamp(min=1e-12)))
    return out


def get_windows(data, split, count, seed):
    bi, ti = sample_indices(data[split], count, seed)
    x, _ = windows(data[split], bi, ti, K)
    z = data[split]['z'][bi, ti]
    return x, z, (bi, ti)


def stat_features(x):
    """Order-free pooled statistics of a window [B,K,N,4]."""
    mean = x.mean((1, 2)); std = x.std((1, 2), unbiased=False)
    mn = x.amin((1, 2)); mx = x.amax((1, 2))
    last = x[:, -1].mean(1)
    pop = x.mean(2)                                  # [B,K,4] per-step population mean
    pstd = pop.std(1, unbiased=False)
    trend = pop[:, -1] - pop.mean(1)
    rate = x[..., 1].mean((1, 2))[:, None]
    return torch.cat((mean, std, mn, mx, last, pstd, trend, rate), -1)


def step_features(x):
    """Per-step population features [B,K,F] for sequence models."""
    pop_mean = x.mean(2)                             # [B,K,4]
    pop_std = x.std(2, unbiased=False)               # [B,K,4]
    return torch.cat((pop_mean, pop_std), -1)        # [B,K,8]


class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 128), nn.GELU(), nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 2))

    def forward(self, f):
        return self.net(f)


class GRUEstimator(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.gru = nn.GRU(f, 64, batch_first=True)
        self.head = nn.Linear(64, 2)

    def forward(self, s):
        h, _ = self.gru(s)
        return self.head(h[:, -1])


def train_torch(model, fx, z, fv, zv, epochs=300, lr=1e-3, seq=False):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best, wait = (float('inf'), None), 0
    for ep in range(epochs):
        model.train(); opt.zero_grad(set_to_none=True)
        loss = (model(fx) - z).square().mean()
        loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = (model(fv) - zv).square().mean()
        if vl < best[0] - 1e-6:
            best, wait = (float(vl), {k: v.clone() for k, v in model.state_dict().items()}), 0
        else:
            wait += 1
            if wait >= 30:
                break
    model.load_state_dict(best[1])
    return model.eval()


def main():
    torch.set_num_threads(2)
    OUT.mkdir(parents=True, exist_ok=True)
    cfg, lc, conn = setup('hidden')
    data = datasets('hidden', cfg, lc, conn)
    dev = torch.device('cuda')
    splits = {sp: get_windows(data, sp, 2048 if sp == 'train' else 1024, 4242)
              for sp in ('train', 'val', 'test_seen', 'test_ood')}
    rows = {}

    def ridge_row(name, feat_fn):
        f = {sp: feat_fn(v[0]).cpu() for sp, v in splits.items()}
        m = ridge_fit(f['train'], splits['train'][1].cpu(), f['val'], splits['val'][1].cpu())
        rows[name] = {sp: metrics(ridge_apply(m, f[sp]), z[1].cpu()) for sp, z in splits.items() if sp != 'train'}

    # E0 current-only: flattened current token
    ridge_row('E0_current_only', lambda x: x[:, -1].flatten(1))
    # E1 unordered statistics
    ridge_row('E1_unordered_stats', stat_features)
    # E2 ordered flattened linear (dual ridge handles K*N*4 wide features)
    ridge_row('E2_ordered_flatlinear', lambda x: x.flatten(1))

    sg = torch.Generator().manual_seed(77)
    fstat = {sp: stat_features(v[0]) for sp, v in splits.items()}
    m3 = train_torch(MLP(fstat['train'].shape[1]).to(dev), fstat['train'], splits['train'][1],
                     fstat['val'], splits['val'][1])
    rows['E3_unordered_mlp'] = {sp: metrics(m3(fstat[sp]).cpu(), z[1].cpu())
                                for sp, z in splits.items() if sp != 'train'}

    fseq = {sp: step_features(v[0]) for sp, v in splits.items()}
    m4 = train_torch(GRUEstimator(fseq['train'].shape[2]).to(dev), fseq['train'], splits['train'][1],
                     fseq['val'], splits['val'][1])
    rows['E4_ordered_seq'] = {sp: metrics(m4(fseq[sp]).cpu(), z[1].cpu())
                              for sp, z in splits.items() if sp != 'train'}

    fshuf = {}
    for sp, v in splits.items():
        xs = history_control(v[0].clone(), 'shuffle', sg)
        fshuf[sp] = step_features(xs)
    m5 = train_torch(GRUEstimator(fshuf['train'].shape[2]).to(dev), fshuf['train'], splits['train'][1],
                     fshuf['val'], splits['val'][1])
    rows['E5_shuffled_seq'] = {sp: metrics(m5(fshuf[sp]).cpu(), z[1].cpu())
                               for sp, z in splits.items() if sp != 'train'}

    flat = []
    for name, per in rows.items():
        for sp, m in per.items():
            for target, mm in m.items():
                flat.append(dict(estimator=name, split=sp, target=target, **mm))
        print(name, 'seen z_pos R2=%.3f z_vel R2=%.3f' % (per['test_seen']['z_pos']['r2'], per['test_seen']['z_vel']['r2']), flush=True)
    import csv
    with (OUT / 'observability_analysis.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(flat[0])); w.writeheader(); w.writerows(flat)
    g = lambda name: rows[name]['test_seen']['z_pos']['r2']
    gates = dict(ordered_vs_current=g('E4_ordered_seq') > g('E0_current_only'),
                 ordered_vs_unordered=g('E4_ordered_seq') > max(g('E1_unordered_stats'), g('E3_unordered_mlp')),
                 ordered_vs_shuffled=g('E4_ordered_seq') > g('E5_shuffled_seq'),
                 z_pos_r2={k: g(k) for k in rows},
                 z_vel_r2_seen={k: rows[k]['test_seen']['z_vel']['r2'] for k in rows})
    gates['passed'] = all(gates[k] for k in ('ordered_vs_current', 'ordered_vs_unordered', 'ordered_vs_shuffled'))
    (OUT / 'gates.json').write_text(json.dumps(gates, indent=2))
    print(json.dumps(gates, indent=2))


if __name__ == '__main__':
    main()
