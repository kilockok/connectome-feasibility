"""latent_state_v2 models: spatial baseline, local temporal, global temporal.

The hidden variable is GLOBAL (one gain for the whole graph), so v2 adds a
Global Population Temporal Path: per-timestep GNN encoding -> population
mean pooling -> causal temporal Transformer over the pooled sequence ->
broadcast back to every neuron for decoding. Local temporal models (v1 path,
per-neuron [B*N,K,D] attention) are kept as controls.

All causal blocks run with causal=True. The labelled oracle is the only
model that may receive z, now two-dimensional (z_pos, z_vel).
"""
import torch
from torch import nn
class SpatialEncoderV2(nn.Module):
    """Same architecture as latent_state_v1.SpatialEncoder, configurable input width."""

    def __init__(self, conn, d=64, layers=2, in_features=4):
        super().__init__()
        self.inp = nn.Linear(in_features, d)
        self.identity = nn.Parameter(torch.randn(1, conn.n_neurons, d) * .02)
        self.msg = nn.ModuleList([nn.Linear(d, d) for _ in range(layers)])
        self.update = nn.ModuleList([nn.Sequential(nn.Linear(2 * d, 2 * d), nn.GELU(),
                                                   nn.Linear(2 * d, d)) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
        self.types = nn.Parameter(torch.zeros(layers, 2, d))
        w = conn.dense_weight('cpu')
        deg = (w != 0).sum(0).clamp(min=1).sqrt()
        self.register_buffer('adj', (w / deg[None]).T.contiguous())
        self.register_buffer('neuron_type', conn.neuron_type.cpu())

    def forward(self, x):
        h = self.inp(x) + self.identity
        for i, (msg, update, norm) in enumerate(zip(self.msg, self.update, self.norms)):
            m = msg(h) + self.types[i, self.neuron_type]
            agg = torch.matmul(self.adj, m)
            h = norm(h + update(torch.cat((h, agg), -1)))
        return h


from models.gnn_temporal import TemporalBlock, sinusoidal_table


class LocalTemporalPredictorV2(nn.Module):
    """v1 local temporal path; oracle z is 2-D."""

    def __init__(self, conn, k=32, d=64, layers=2, temporal=True, oracle=False, dropout=0.0):
        super().__init__()
        self.k, self.oracle, self.temporal = k, oracle, temporal
        if oracle and k != 1:
            raise ValueError('The labelled oracle uses current z only')
        self.spatial = SpatialEncoderV2(conn, d, layers, 6 if oracle else 4)
        if temporal:
            self.register_buffer('pos', sinusoidal_table(32, d))
            self.blocks = nn.ModuleList([TemporalBlock(d, 4, 4 * d, dropout) for _ in range(2)])
            self.norm = nn.LayerNorm(d)
        self.decoder = nn.Sequential(nn.Linear(2 * d if temporal else d, d), nn.GELU(), nn.Linear(d, 3))

    def encode(self, x, z=None, return_sequence=False):
        if x.shape[-1] != 4:
            raise ValueError('Public input is exactly V,S,R,U')
        x = x[:, -self.k:]
        if self.oracle:
            if z is None:
                raise ValueError('Oracle requires explicitly supplied z')
            if z.shape[-1] != 2:
                raise ValueError('v2 oracle z is (z_pos, z_vel)')
            zz = z[:, None, None, :].expand(*x.shape[:-1], 2)
            x = torch.cat((x, zz), -1)
        elif z is not None:
            raise ValueError('Hidden z may not enter a non-oracle model')
        spatial = self.spatial(x)
        if not self.temporal:
            return spatial[:, -1], spatial[:, -1]
        b, k, n, d = spatial.shape
        seq = spatial.permute(0, 2, 1, 3).reshape(b * n, k, d) + self.pos[:, -k:]
        for block in self.blocks:
            seq, _ = block(seq, causal=True)
        seq = self.norm(seq).reshape(b, n, k, d).permute(0, 2, 1, 3)
        fused = torch.cat((spatial[:, -1], seq[:, -1]), -1)
        return (fused, seq) if return_sequence else (fused, seq[:, -1])

    def forward(self, x, z=None, return_features=False):
        fused, features = self.encode(x, z)
        y = self.decoder(fused)
        out = dict(v=y[..., 0], s_logits=y[..., 1], r=y[..., 2])
        return (out, features) if return_features else out


class GlobalTemporalPredictorV2(nn.Module):
    """Per-step spatial encoding -> population pooling -> causal global temporal
    Transformer -> broadcast context -> per-neuron decoder."""

    def __init__(self, conn, k=32, d=64, layers=2, heads=4, dropout=0.0, pooling='mean'):
        super().__init__()
        if pooling != 'mean':
            raise ValueError('v2 first study fixes mean pooling')
        self.k = k
        self.oracle = False
        self.spatial = SpatialEncoderV2(conn, d, layers)
        self.register_buffer('pos', sinusoidal_table(32, d))
        self.blocks = nn.ModuleList([TemporalBlock(d, heads, 4 * d, dropout) for _ in range(2)])
        self.norm = nn.LayerNorm(d)
        self.decoder = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 3))

    def encode(self, x, z=None, return_sequence=False):
        if x.shape[-1] != 4:
            raise ValueError('Public input is exactly V,S,R,U')
        if z is not None:
            raise ValueError('Hidden z may not enter a non-oracle model')
        x = x[:, -self.k:]
        spatial = self.spatial(x)                       # [B,K,N,D]
        b, k, n, d = spatial.shape
        g = spatial.mean(dim=2) + self.pos[:, -k:]      # [B,K,D] population token
        seq = g
        for block in self.blocks:
            seq, _ = block(seq, causal=True)
        seq = self.norm(seq)
        z_ctx = seq[:, -1]                              # [B,D] global context
        fused = torch.cat((spatial[:, -1], z_ctx[:, None, :].expand(b, n, d)), -1)
        return (fused, seq) if return_sequence else (fused, z_ctx)

    def forward(self, x, z=None, return_features=False):
        fused, features = self.encode(x, z)
        y = self.decoder(fused)
        out = dict(v=y[..., 0], s_logits=y[..., 1], r=y[..., 2])
        return (out, features) if return_features else out


def specification_v2(label, conn):
    """M0..M8 model table for the v2 controlled comparison."""
    if label == 'gnn_k1':                       # M0 spatial-only baseline
        return dict(cls='local', k=1, temporal=False)
    if label == 'local_k16':                    # M1
        return dict(cls='local', k=16)
    if label == 'local_k32':                    # M2
        return dict(cls='local', k=32)
    if label == 'global_k16':                   # M3
        return dict(cls='global', k=16)
    if label == 'global_k32':                   # M4
        return dict(cls='global', k=32)
    if label == 'wide':                         # M5 parameter-matched to M4
        ref = GlobalTemporalPredictorV2(conn)
        target = sum(p.numel() for p in ref.parameters())
        best = None
        for d in (64, 80, 96, 112, 128, 160):
            for layers in (2, 3, 4, 5):
                m = LocalTemporalPredictorV2(conn, k=1, d=d, layers=layers, temporal=False)
                n = sum(p.numel() for p in m.parameters())
                candidate = (abs(n - target), d, layers)
                if best is None or candidate < best:
                    best = candidate
        return dict(cls='local', k=1, d=best[1], layers=best[2], temporal=False)
    if label in ('gshuffle', 'glast'):          # M6 / M7 (training-side history controls)
        return dict(cls='global', k=32)
    if label == 'oracle':                       # M8 labelled true-z upper bound
        return dict(cls='local', k=1, temporal=False, oracle=True)
    raise ValueError(label)


def build_v2(label, conn):
    spec = specification_v2(label, conn)
    cls = spec.pop('cls')
    model = (GlobalTemporalPredictorV2 if cls == 'global' else LocalTemporalPredictorV2)(conn, **spec)
    return model, dict(spec, cls=cls)
