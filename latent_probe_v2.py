"""Frozen-backbone probes (ridge + MLP) for (z_pos, z_vel) and v2 interventions."""
import torch
from eval_latent import predict_windows, summarize, first_sustained
from latent_data import windows, history_control
from lif_latent_v2 import HiddenStateLIFSimulatorV2


def probe_metrics(pred, true):
    pred, true = pred.double(), true.double()
    out = {}
    for j, name in enumerate(('z_pos', 'z_vel')):
        p, t = pred[:, j], true[:, j]
        den = (t - t.mean()).square().sum()
        pc, tc = p - p.mean(), t - t.mean()
        out[name] = dict(r2=float(1 - (p - t).square().sum() / den) if den > 1e-12 else None,
                         correlation=float((pc * tc).sum() / (pc.norm() * tc.norm()).clamp(min=1e-12)),
                         mae=float((p - t).abs().mean()))
    return out


def fit_ridge(train, val):
    f, z = train
    vf, vz = val
    f, vf, z, vz = f.double(), vf.double(), z.double(), vz.double()
    mean = f.mean(0); std = f.std(0).clamp(min=1e-5); zm = z.mean(0)
    x = (f - mean) / std; v = (vf - mean) / std
    gram = x.T @ x
    best = {}
    for ridge in (.01, .1, 1., 10., 100.):
        w = torch.linalg.solve(gram + ridge * torch.eye(gram.shape[0], dtype=torch.float64), x.T @ (z - zm))
        err = ((v @ w + zm) - vz).square().mean(0)
        for j in range(z.shape[1]):
            if j not in best or err[j] < best[j][0]:
                best[j] = (err[j], ridge, w[:, j])
    return dict(mean=mean, std=std, z_mean=zm,
                weight=torch.stack([best[j][2] for j in range(z.shape[1])], 1),
                ridge=[best[j][1] for j in range(z.shape[1])])


def apply_probe(probe, f):
    return ((f.double().cpu() - probe['mean']) / probe['std']) @ probe['weight'] + probe['z_mean']


class MLPProbe(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(d, 128), torch.nn.GELU(),
                                       torch.nn.Linear(128, 64), torch.nn.GELU(), torch.nn.Linear(64, 2))

    def forward(self, f):
        return self.net(f)


def fit_mlp(train, val, epochs=300):
    f, z = train
    vf, vz = val
    mean, std = f.mean(0), f.std(0).clamp(min=1e-5)
    zm = z.mean(0)
    model = MLPProbe(f.shape[1]).to(f.device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    best, wait = float('inf'), 0
    best_state = None
    for ep in range(epochs):
        model.train(); opt.zero_grad(set_to_none=True)
        loss = (model((f - mean) / std) - (z - zm)).square().mean()
        loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = (model((vf - mean) / std) - (vz - zm)).square().mean()
        if vl < best - 1e-6:
            best, wait = float(vl), 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= 30:
                break
    model.load_state_dict(best_state)
    return model.eval(), dict(mean=mean, std=std, z_mean=zm)


def apply_mlp(model, stats, f):
    mean, std, zm = stats['mean'], stats['std'], stats['z_mean']
    with torch.no_grad():
        return model((f - mean) / std) + zm


@torch.no_grad()
def fused_features(model, x):
    """Uniform probe input for every v2 architecture: mean/std over neurons of the
    fused decoder input (spatial code + broadcast context where present)."""
    fused, _ = model.encode(x)
    return torch.cat((fused.mean(1), fused.std(1, unbiased=False)), -1)


@torch.no_grad()
def predict_with_features(model, x):
    fused, _ = model.encode(x)
    y = model.decoder(fused)
    return dict(v=y[..., 0], s_logits=y[..., 1], r=y[..., 2]), fused


@torch.no_grad()
def extract_features(model, data, control='ordered', count=1024, seed=9191):
    from latent_data import sample_indices
    bi, ti = sample_indices(data, count, seed)
    sg = torch.Generator().manual_seed(seed + 19)
    feats, labels = [], []
    for start in range(0, count, 32):
        b, t = bi[start:start + 32], ti[start:start + 32]
        x, _ = windows(data, b, t, model.k)
        x = history_control(x, control, sg)
        feats.append(fused_features(model, x).cpu())
        labels.append(data['z'][b.to(x.device), t.to(x.device)].cpu())
    return torch.cat(feats), torch.cat(labels)


def run_probe_v2(model, data, control='ordered'):
    """Ridge + MLP probes on frozen features; targets (z_pos, z_vel)."""
    features = {sp: extract_features(model, data[sp], control, 2048 if sp == 'train' else 1024)
                for sp in ('train', 'val', 'test_seen', 'test_ood')}
    ridge = fit_ridge(features['train'], features['val'])
    mlp, stats = fit_mlp((features['train'][0].float(), features['train'][1].float()),
                         (features['val'][0].float(), features['val'][1].float()))
    results = {}
    for sp, (f, z) in features.items():
        if sp == 'train':
            continue
        results[sp] = dict(ridge=probe_metrics(apply_probe(ridge, f), z),
                           mlp=probe_metrics(apply_mlp(mlp, stats, f.float()), z))
    assert all(p.grad is None for p in model.parameters())
    return (ridge, mlp, stats), results, features


INTERVENTIONS = ((128, 'vel_flip'), (128, 'phase_jump', 1.5), (128, 'regime', None))


@torch.no_grad()
def intervention_v2(model, probe_pack, conn, cfg, lc, threshold, control='ordered', n=32):
    """Velocity sign flip, phase jump, regime switch at t=128; relative recovery.

    regime switch reflects (z_pos, z_vel) -> (-z_pos, -z_vel) at the jump step.
    Recovery: error <= E_base + 0.2*(E_peak - E_base) for 3 consecutive delays,
    where E_base = pre-jump MAE (delays -16..-1), E_peak = max MAE in delays 0..8.
    """
    ridge, _, _ = probe_pack
    sim = HiddenStateLIFSimulatorV2(conn, cfg, next(model.parameters()).device, lc)
    rows = []
    for spec in INTERVENTIONS:
        at, kind = spec[0], spec[1]
        if kind == 'regime':
            probe = None  # per-trajectory reflection; resolved below
        d0 = sim.generate(list(range(80_000_000, 80_000_000 + n)), 'test_seen')
        if kind == 'regime':
            zr = d0['z'][:, at].clone()
            interventions = [(at, 'regime', float(-zr[i, 0]), float(-zr[i, 1])) for i in range(n)]
            # generate per-trajectory regimes in chunks of identical intervention
            datas = []
            for i in range(n):
                datas.append(sim.generate([80_000_000 + i], 'test_seen', interventions[i]))
            d = {k: torch.cat([di[k] for di in datas]) for k in datas[0]}
        else:
            d = sim.generate(list(range(80_000_000, 80_000_000 + n)), 'test_seen', spec)
        sg = torch.Generator().manual_seed(32)
        curve = []
        for delay in range(-16, 65):
            end = at + delay
            x, y = windows(d, torch.arange(n), torch.full((n,), end), model.k)
            x = history_control(x, control, sg)
            ztrue = d['z'][:, end]
            out, fused = predict_with_features(model, x)
            f = torch.cat((fused.mean(1), fused.std(1, unbiased=False)), -1).cpu()
            pred = apply_probe(ridge, f)
            mm = summarize({k: v.cpu() for k, v in out.items()}, y.cpu(), threshold)
            curve.append(dict(kind=kind, delay=delay,
                              probe_mae=float((pred - ztrue.cpu()).abs().mean()),
                              probe_mae_pos=float((pred[:, 0] - ztrue.cpu()[:, 0]).abs().mean()),
                              probe_mae_vel=float((pred[:, 1] - ztrue.cpu()[:, 1]).abs().mean()),
                              true_z_pos=float(ztrue[:, 0].mean()), **mm))
        e_base = sum(r['probe_mae'] for r in curve if -16 <= r['delay'] < 0) / 16
        e_peak = max(r['probe_mae'] for r in curve if 0 <= r['delay'] <= 8)
        thresh = e_base + .2 * (e_peak - e_base)
        onset = first_sustained([r['probe_mae'] <= thresh for r in curve if r['delay'] >= 0])
        latency = onset - 1 if onset is not None else None
        for r in curve:
            r['e_base'], r['e_peak'], r['recovery_threshold'] = e_base, e_peak, thresh
            r['recovery_latency'] = latency
            r['latency_definition'] = ('first of 3 consecutive delays with probe MAE <= '
                                       'E_base+0.2*(E_peak-E_base); null=censored >64')
            rows.append(r)
    return rows
