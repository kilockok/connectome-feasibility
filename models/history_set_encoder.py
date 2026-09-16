"""v4 history encoders: strictly permutation-invariant SetHistory (DeepSets),
handcrafted StatsHistory, and the explicit DerivativeBaseline.

SetHistory sees NO temporal order by construction: per-step GNN encoding ->
population pooling -> per-step phi -> permutation-invariant aggregation
(mean/std/max over time) -> rho. No position embedding, no attention, no
recurrence, no temporal convolution anywhere in the history path.
"""
import torch
from torch import nn
from models.latent_temporal_v2 import SpatialEncoderV2, LocalTemporalPredictorV2, GlobalTemporalPredictorV2
from models.latent_temporal_v2 import build_v2


class SetHistoryPredictor(nn.Module):
    """DeepSets over history steps; strictly permutation-invariant over time."""

    def __init__(self, conn, k=32, d=64, layers=2, phi_h=192, rho_h=192):
        super().__init__()
        self.k, self.oracle = k, False
        self.spatial = SpatialEncoderV2(conn, d, layers)
        self.phi = nn.Sequential(nn.Linear(d, phi_h), nn.GELU(), nn.Linear(phi_h, d))
        self.rho = nn.Sequential(nn.Linear(3 * d, rho_h), nn.GELU(), nn.Linear(rho_h, d))
        self.decoder = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 3))

    def encode(self, x, z=None, return_sequence=False):
        if x.shape[-1] != 4:
            raise ValueError('Public input is exactly V,S,R,U')
        if z is not None:
            raise ValueError('Hidden z may not enter a non-oracle model')
        x = x[:, -self.k:]
        spatial = self.spatial(x)                  # [B,K,N,D]
        g = spatial.mean(dim=2)                    # [B,K,D] per-step population token
        q = self.phi(g)                            # [B,K,D]
        agg = torch.cat((q.mean(1), q.std(1, unbiased=False), q.amax(1)), -1)
        h_set = self.rho(agg)                      # [B,D]
        b, k, n, d = spatial.shape
        fused = torch.cat((spatial[:, -1], h_set[:, None, :].expand(b, n, d)), -1)
        return (fused, q) if return_sequence else (fused, h_set)

    def forward(self, x, z=None, return_features=False):
        fused, features = self.encode(x, z)
        y = self.decoder(fused)
        out = dict(v=y[..., 0], s_logits=y[..., 1], r=y[..., 2])
        return (out, features) if return_features else out


def stats_features(x):
    """Handcrafted order-free window statistics [B, 25]."""
    pop_mean = x.mean(2)                           # [B,K,4] per-step population means
    rate = x[..., 1].mean(2, keepdim=True)         # [B,K,1] per-step spike rate
    per_step = torch.cat((pop_mean, rate), -1)     # [B,K,5]
    w_mean = per_step.mean(1)
    w_std = per_step.std(1, unbiased=False)
    w_min = per_step.amin(1)
    w_max = per_step.amax(1)
    current = per_step[:, -1]
    return torch.cat((w_mean, w_std, w_min, w_max, current), -1)


class StatsHistoryPredictor(nn.Module):
    """Handcrafted population moments over the window + small MLP."""

    def __init__(self, conn, k=32, d=64, layers=2, mlp_h=64):
        super().__init__()
        self.k, self.oracle = k, False
        self.spatial = SpatialEncoderV2(conn, d, layers)
        self.mlp = nn.Sequential(nn.Linear(25, mlp_h), nn.GELU(),
                                 nn.Linear(mlp_h, mlp_h), nn.GELU(), nn.Linear(mlp_h, d))
        self.decoder = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 3))

    def encode(self, x, z=None, return_sequence=False):
        if z is not None:
            raise ValueError('Hidden z may not enter a non-oracle model')
        x = x[:, -self.k:]
        spatial = self.spatial(x)
        h_stats = self.mlp(stats_features(x))
        b, k, n, d = spatial.shape
        fused = torch.cat((spatial[:, -1], h_stats[:, None, :].expand(b, n, d)), -1)
        return (fused, h_stats) if return_sequence else (fused, h_stats)

    def forward(self, x, z=None, return_features=False):
        fused, features = self.encode(x, z)
        y = self.decoder(fused)
        out = dict(v=y[..., 0], s_logits=y[..., 1], r=y[..., 2])
        return (out, features) if return_features else out


def derivative_features(x):
    """Explicit short-timescale dynamical trend features [B, 10]."""
    pop = x.mean(2)                                # [B,K,4]
    rate = x[..., 1].mean(2)                       # [B,K]
    v = pop[..., 0]; r = pop[..., 2]; u = pop[..., 3]
    cur = torch.cat((pop[:, -1], rate[:, -1, None]), -1)              # [B,5]
    dv1 = v[:, -1] - v[:, -2]                                            # [B]
    dv4 = (v[:, -1] - v[:, -5].clamp(min=-1e9)) / 3 if x.shape[1] >= 5 else dv1
    dr1 = rate[:, -1] - rate[:, -2]
    dr4 = rate[:, -1] - rate[:, -4: -1].mean(-1) if x.shape[1] >= 5 else dr1
    t = torch.arange(4, device=x.device, dtype=x.dtype)
    tm = t - t.mean()
    slope = ((v[:, -4:] - v[:, -4:].mean(1, keepdim=True)) * tm).sum(-1) / (tm * tm).sum() \
        if x.shape[1] >= 4 else dv4
    return torch.cat((cur, dv1[:, None], dv4[:, None], dr1[:, None], dr4[:, None], slope[:, None]), -1)


class DerivativeBaselinePredictor(nn.Module):
    """Explicit local temporal derivative baseline."""

    def __init__(self, conn, k=32, d=64, layers=2, mlp_h=64):
        super().__init__()
        self.k, self.oracle = k, False
        self.spatial = SpatialEncoderV2(conn, d, layers)
        self.mlp = nn.Sequential(nn.Linear(10, mlp_h), nn.GELU(), nn.Linear(mlp_h, d))
        self.decoder = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 3))

    def encode(self, x, z=None, return_sequence=False):
        if z is not None:
            raise ValueError('Hidden z may not enter a non-oracle model')
        x = x[:, -self.k:]
        spatial = self.spatial(x)
        h = self.mlp(derivative_features(x))
        b, k, n, d = spatial.shape
        fused = torch.cat((spatial[:, -1], h[:, None, :].expand(b, n, d)), -1)
        return (fused, h) if return_sequence else (fused, h)

    def forward(self, x, z=None, return_features=False):
        fused, features = self.encode(x, z)
        y = self.decoder(fused)
        out = dict(v=y[..., 0], s_logits=y[..., 1], r=y[..., 2])
        return (out, features) if return_features else out


V4_LABELS = ('set_k32', 'stats_k32', 'deriv')


def build_v4(label, conn):
    """v4 model table; v2 labels are delegated unchanged."""
    if label == 'set_k32':
        ref = GlobalTemporalPredictorV2(conn)
        target = sum(p.numel() for p in ref.parameters())
        best = None
        for phi_h in (128, 160, 192, 224, 256):
            for rho_h in (128, 160, 192, 224, 256):
                m = SetHistoryPredictor(conn, phi_h=phi_h, rho_h=rho_h)
                n = sum(p.numel() for p in m.parameters())
                rel = abs(n - target) / target
                if best is None or rel < best[0]:
                    best = (rel, phi_h, rho_h, n)
        if best[0] > 0.10:
            raise RuntimeError(f'Cannot parameter-match SetHistory within 10%: {best}')
        return SetHistoryPredictor(conn, phi_h=best[1], rho_h=best[2]), \
            dict(cls='set', k=32, phi_h=best[1], rho_h=best[2], params=best[3], match_target=target)
    if label == 'stats_k32':
        return StatsHistoryPredictor(conn), dict(cls='stats', k=32)
    if label == 'deriv':
        return DerivativeBaselinePredictor(conn), dict(cls='deriv', k=32)
    model, spec = build_v2(label, conn)
    return model, dict(spec, cls=spec.get('cls', 'v2'))


def load_model_v4(summary, conn, device):
    blob = torch.load(summary['checkpoint'], map_location=device, weights_only=False)
    spec = dict(blob['spec'])
    cls = spec.pop('cls', 'v2')
    if cls == 'set':
        spec.pop('params', None); spec.pop('match_target', None)
        model = SetHistoryPredictor(conn, **spec)
    elif cls == 'stats':
        model = StatsHistoryPredictor(conn, **spec)
    elif cls == 'deriv':
        model = DerivativeBaselinePredictor(conn, **spec)
    elif cls == 'global':
        model = GlobalTemporalPredictorV2(conn, **spec)
    else:
        model = LocalTemporalPredictorV2(conn, **spec)
    model = model.to(device)
    model.load_state_dict(blob['state_dict'])
    return model.eval(), blob
