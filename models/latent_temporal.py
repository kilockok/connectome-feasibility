"""Spatial-first GNN and causal temporal inference, isolated from legacy models."""
import torch
from torch import nn
from models.gnn_temporal import TemporalBlock, sinusoidal_table


class SpatialEncoder(nn.Module):
    def __init__(self, conn, d=64, layers=2, oracle=False):
        super().__init__()
        self.inp = nn.Linear(5 if oracle else 4, d)
        self.identity = nn.Parameter(torch.randn(1, conn.n_neurons, d)*.02)
        self.msg = nn.ModuleList([nn.Linear(d, d) for _ in range(layers)])
        self.update = nn.ModuleList([nn.Sequential(nn.Linear(2*d, 2*d), nn.GELU(),
                                                   nn.Linear(2*d, d)) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
        self.types = nn.Parameter(torch.zeros(layers, 2, d))
        w = conn.dense_weight('cpu')
        deg = (w != 0).sum(0).clamp(min=1).sqrt()
        self.register_buffer('adj', (w/deg[None]).T.contiguous())
        self.register_buffer('neuron_type', conn.neuron_type.cpu())

    def forward(self, x):
        # Dense adjacency is appropriate to N<=1000 and avoids [B,K,E,D] materialization.
        h = self.inp(x) + self.identity
        for i, (msg, update, norm) in enumerate(zip(self.msg, self.update, self.norms)):
            m = msg(h)+self.types[i, self.neuron_type]
            agg = torch.matmul(self.adj, m)
            h = norm(h+update(torch.cat((h, agg), -1)))
        return h


class LatentPredictor(nn.Module):
    def __init__(self, conn, k=32, d=64, layers=2, temporal=True, oracle=False, dropout=.05):
        super().__init__()
        self.k, self.oracle, self.temporal = k, oracle, temporal
        if oracle and k != 1:
            raise ValueError('The labelled oracle uses current z only')
        self.spatial = SpatialEncoder(conn, d, layers, oracle)
        if temporal:
            self.register_buffer('pos', sinusoidal_table(32, d))
            self.blocks = nn.ModuleList([TemporalBlock(d, 4, 4*d, dropout) for _ in range(2)])
            self.norm = nn.LayerNorm(d)
        self.decoder = nn.Sequential(nn.Linear(2*d if temporal else d, d), nn.GELU(), nn.Linear(d, 3))

    def encode(self, x, z=None, return_sequence=False):
        if x.shape[-1] != 4:
            raise ValueError('Public input is exactly V,S,R,U')
        x = x[:, -self.k:]
        if self.oracle:
            if z is None:
                raise ValueError('Oracle requires explicitly supplied z')
            zz = z[:, None, None, None].expand(*x.shape[:-1], 1)
            x = torch.cat((x, zz), -1)
        elif z is not None:
            raise ValueError('Hidden z may not enter a non-oracle model')
        spatial = self.spatial(x)
        if not self.temporal:
            return spatial[:, -1], spatial[:, -1]
        b, k, n, d = spatial.shape
        seq = spatial.permute(0, 2, 1, 3).reshape(b*n, k, d)+self.pos[:, -k:]
        for block in self.blocks:
            seq, _ = block(seq, causal=True)
        seq = self.norm(seq).reshape(b, n, k, d).permute(0, 2, 1, 3)
        context = seq[:, -1]
        fused = torch.cat((spatial[:, -1], context), -1)
        return (fused, seq) if return_sequence else (fused, context)

    def forward(self, x, z=None, return_features=False):
        fused, features = self.encode(x, z)
        y = self.decoder(fused)
        out = dict(v=y[..., 0], s_logits=y[..., 1], r=y[..., 2])
        return (out, features) if return_features else out


def specification(label, conn):
    if label == 'gnn_k1':
        return dict(k=1, temporal=False)
    if label == 'oracle':
        return dict(k=1, temporal=False, oracle=True)
    if label == 'wide':
        ref = LatentPredictor(conn)
        target = sum(p.numel() for p in ref.parameters())
        best = None
        for d in (64, 80, 96, 112, 128):
            for layers in (2, 3, 4):
                m = LatentPredictor(conn, k=1, d=d, layers=layers, temporal=False)
                n = sum(p.numel() for p in m.parameters())
                candidate = (abs(n-target), d, layers)
                if best is None or candidate < best:
                    best = candidate
        return dict(k=1, d=best[1], layers=best[2], temporal=False)
    if label.startswith('hybrid_k'):
        return dict(k=int(label.removeprefix('hybrid_k')))
    if label in ('shuffle', 'last'):
        return dict(k=32)
    raise ValueError(label)
