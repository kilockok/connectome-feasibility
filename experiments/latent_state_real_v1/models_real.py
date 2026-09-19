"""Models for latent_state_real_v1 (region-level real data, fs=1.2 Hz).

All models: input window Y[t-L:t] (z-scored region fluorescence, [B,L,R]),
output Y_hat at horizon h ([B,R]). No mechanism labels; observation-space
prediction only.

  M1 temporal-only : causal transformer over frame tokens; per-region readout
  M2 connectome GNN: 2-layer message passing on A (log1p tbar, row-normed,
                     directed); frame-wise; no temporal latent
  M3 hybrid        : y_hat = base_graph(y_t, A) + delta(e_t, z_t),
                     z_t = causal transformer over GNN-token history
  variants: m3_nolatent (z=0), m1_param-matched H (M3 with A=I),
            graph shuffles handled by swapping A at build time.
"""
import torch
from torch import nn
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))  # feasibility root
from models.gnn_temporal import TemporalBlock, sinusoidal_table


def norm_adj(A_raw):
    """log1p(|T-bar|), row-normalized (row = dst sums to 1 over src)."""
    A_raw = torch.as_tensor(A_raw, dtype=torch.float32)
    A = torch.log1p(A_raw.abs())
    A = A / A.sum(0, keepdim=True).clamp(min=1e-9)
    return A                                             # A[src, dst]


class GraphEncoder(nn.Module):
    """2-layer message passing on the current frame y_t [B,R] -> [B,R,d]."""

    def __init__(self, A_raw, R, d=64, identity=True):
        super().__init__()
        self.register_buffer('A', norm_adj(A_raw))
        self.inp = nn.Linear(1, d)
        self.node = nn.Parameter(torch.randn(1, R, d) * .02) if identity else None
        self.msg = nn.ModuleList([nn.Linear(d, d) for _ in range(2)])
        self.upd = nn.ModuleList([nn.Sequential(nn.Linear(2 * d, 2 * d), nn.GELU(),
                                                nn.Linear(2 * d, d)) for _ in range(2)])
        self.norm = nn.ModuleList([nn.LayerNorm(d) for _ in range(2)])

    def forward(self, y):                                # y [B,R] -> h [B,R,d]
        h = self.inp(y[..., None])
        if self.node is not None:
            h = h + self.node
        for msg, upd, nm in zip(self.msg, self.upd, self.norm):
            m = msg(h)
            agg = torch.matmul(self.A.transpose(0, 1), m)      # sum over src -> dst
            h = nm(h + upd(torch.cat((h, agg), -1)))
        return h


class M2Graph(nn.Module):
    """M2: structure-constrained dynamics without temporal latent."""

    def __init__(self, A_raw, R, d=64):
        super().__init__()
        self.enc = GraphEncoder(A_raw, R, d)
        self.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        # learned linear structural base (B2-class, learned jointly)
        self.a = nn.Parameter(torch.tensor(1.0))
        self.b = nn.Parameter(torch.tensor(0.0))
        self.c = nn.Parameter(torch.zeros(R))
        self.register_buffer('An', norm_adj(A_raw))

    def base(self, y):
        return self.a * y + self.b * (y @ self.An) + self.c

    def forward(self, x):
        y = x[:, -1]
        return self.base(y) + self.head(self.enc(y))[..., 0]


class M1Temporal(nn.Module):
    """M1: causal transformer over frame tokens; no graph."""

    def __init__(self, R, L=32, d=64, heads=4):
        super().__init__()
        self.L = L
        self.inp = nn.Linear(R, d)
        self.register_buffer('pos', sinusoidal_table(L, d))
        self.blocks = nn.ModuleList([TemporalBlock(d, heads, 4 * d, 0.0) for _ in range(2)])
        self.norm = nn.LayerNorm(d)
        self.reg = nn.Parameter(torch.randn(1, R, d) * .02)
        self.head = nn.Sequential(nn.Linear(d + 1 + d, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, x):                                # [B,L,R] -> [B,R]
        B, L, R = x.shape
        tok = self.inp(x) + self.pos[:, -L:]
        for blk in self.blocks:
            tok, _ = blk(tok, causal=True)
        z = self.norm(tok)[:, -1]                        # [B,d]
        y = x[:, -1]                                     # [B,R]
        feat = torch.cat((z[:, None, :].expand(B, R, -1),
                          y[..., None],
                          self.reg.expand(B, R, -1)), -1)
        return self.head(feat)[..., 0]

    def latent(self, x):
        tok = self.inp(x) + self.pos[:, -x.shape[1]:]
        for blk in self.blocks:
            tok, _ = blk(tok, causal=True)
        return self.norm(tok)


class M3Hybrid(nn.Module):
    """M3: y_hat = base_graph(y_t, A) + delta(e_t, z_t)."""

    def __init__(self, A_raw, R, L=32, d=64, dz=64, heads=4, latent=True):
        super().__init__()
        self.latent = latent
        self.enc = GraphEncoder(A_raw, R, d)
        self.L = L
        self.token = nn.Linear(2 * d + 3, dz)
        self.register_buffer('pos', sinusoidal_table(L, dz))
        self.blocks = nn.ModuleList([TemporalBlock(dz, heads, 4 * dz, 0.0) for _ in range(2)])
        self.norm = nn.LayerNorm(dz)
        self.base_lin = nn.Linear(2 * R, R)              # learned [y_t, A y_t] base
        self.delta = nn.Sequential(nn.Linear(d + dz, d), nn.GELU(), nn.Linear(d, 1))
        self.register_buffer('An', norm_adj(A_raw))

    def base(self, y):
        return self.base_lin(torch.cat((y, y @ self.An), -1))

    def encode_window(self, x):
        B, L, R = x.shape
        e = torch.stack([self.enc(x[:, t]) for t in range(L)], 1)   # [B,L,R,d]
        stats = torch.stack((x.mean(2), x.std(2), (x[..., -1] if False else x.abs().mean(2))), -1)
        tok = self.token(torch.cat((e.mean(2), e.amax(2), stats), -1)) + self.pos[:, -L:]
        for blk in self.blocks:
            tok, _ = blk(tok, causal=True)
        return self.norm(tok), e

    def forward(self, x, z_override=None):
        z_seq, e = self.encode_window(x)
        z = z_seq[:, -1] if self.latent else torch.zeros_like(z_seq[:, -1])
        if z_override is not None:
            z = z_override.reshape(x.shape[0], -1)
        fused = torch.cat((e[:, -1], z[:, None, :].expand(-1, x.shape[2], -1)), -1)
        return self.base(x[:, -1]) + self.delta(fused)[..., 0]

    def z_last(self, x):
        """Latent at the final window position [B,dz] (for z-shuffle)."""
        z_seq, _ = self.encode_window(x)
        return z_seq[:, -1]

    def forward_parts(self, x):
        z_seq, e = self.encode_window(x)
        z = z_seq[:, -1] if self.latent else torch.zeros_like(z_seq[:, -1])
        fused = torch.cat((e[:, -1], z[:, None, :].expand(-1, x.shape[2], -1)), -1)
        return dict(base=self.base(x[:, -1]), delta=self.delta(fused)[..., 0], z=z_seq)


class B1NoHistory(nn.Module):
    """B1: continuous rate dynamics without history - per-region shared MLP
    on the current frame only (plus region embedding)."""

    def __init__(self, R, d=64):
        super().__init__()
        self.reg = nn.Parameter(torch.randn(1, R, d) * .02)
        self.net = nn.Sequential(nn.Linear(1 + d, d), nn.GELU(), nn.Linear(d, d),
                                 nn.GELU(), nn.Linear(d, 1))

    def forward(self, x):
        y = x[:, -1]
        B, R = y.shape
        feat = torch.cat((y[..., None], self.reg.expand(B, R, -1)), -1)
        return self.net(feat)[..., 0]


def build_real(kind, A_raw, R, latent=True, d=64):
    if kind == 'b1':
        return B1NoHistory(R, d=d)
    if kind == 'm1':
        return M1Temporal(R, d=d)
    if kind == 'm2':
        return M2Graph(A_raw, R, d=d)
    if kind == 'm3':
        return M3Hybrid(A_raw, R, d=d, latent=True)
    if kind == 'm3_nolatent':
        return M3Hybrid(A_raw, R, d=d, latent=False)
    raise ValueError(kind)
