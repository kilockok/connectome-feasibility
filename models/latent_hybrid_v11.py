"""v11 hybrid neural simulator: hard-LIF prior + connectome GNN + explicit latent z_t.

    e_t   = SpatialEncoderV2(x_t; connectome)          # per-step GNN, [B,K,N,d]
    tok_t = proj([mean_N e_t, max_N e_t, phys(x_t)])   # per-step population token
    z_t   = CausalTransformer(tok_{1..t})              # [B,K,dz], exposed per step
    y     = base_pre(x_t) + head([e_t, z_t])           # additive residual on the
                                                        # differentiable base (soft
                                                        # base in training only)

Kinds:
  full   : GNN + latent transformer (B5)
  k0     : no latent history (z = 0; decoder on e_t only)  (B3)
  nognn  : per-neuron MLP instead of GNN messages          (B4)
k: context length (0 forces k0 semantics). The latent is exposed per step
(`latent` returns z for every window position) so Stage 2 interventions can
substitute/shift/freeze it.

No mechanism labels anywhere. The model never sees hidden teacher state.
"""
import torch
from torch import nn

from models.latent_temporal_v2 import SpatialEncoderV2
from models.gnn_temporal import TemporalBlock, sinusoidal_table
from models.residual_v7 import base_pre


class PerNeuronMLP(nn.Module):
    """B4 ablation: same per-step interface as SpatialEncoderV2, no messages."""

    def __init__(self, n_neurons, d=64, in_features=4):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_features, d), nn.GELU(),
                                 nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.identity = nn.Parameter(torch.randn(1, n_neurons, d) * .02)

    def forward(self, x):
        return self.net(x) + self.identity


def phys_features(x):
    """Legal per-step population summaries from x [B,K,N,4]: rate, V mean,
    V std, stimulus mean."""
    s, v, u = x[..., 1], x[..., 0], x[..., 3]
    return torch.stack((s.mean(2), v.mean(2), v.std(2), u.mean(2)), -1)


class LatentHybridV11(nn.Module):
    def __init__(self, conn, cfg, kind='full', k=32, d=64, dz=64, layers=2,
                 heads=4, dropout=0.0):
        super().__init__()
        if kind not in ('full', 'k0', 'nognn'):
            raise ValueError(kind)
        self.kind = kind
        self.cfg = cfg
        self.k = k
        self.dz = dz
        if kind == 'nognn':
            self.spatial = PerNeuronMLP(conn.n_neurons, d)
        else:
            self.spatial = SpatialEncoderV2(conn, d, layers)
        self.token = nn.Linear(2 * d + 4, dz)
        self.register_buffer('pos', sinusoidal_table(32, dz))
        self.blocks = nn.ModuleList([TemporalBlock(dz, heads, 4 * dz, dropout)
                                     for _ in range(2)])
        self.norm = nn.LayerNorm(dz)
        self.head = nn.Sequential(nn.Linear(d + dz, d), nn.GELU(), nn.Linear(d, 3))
        W = conn.dense_weight('cpu')
        self.register_buffer('W', W)
        self.register_buffer('i_bias', conn.i_bias.cpu()
                             if conn.i_bias is not None else torch.zeros(conn.n_neurons))

    def latent(self, x):
        """z sequence for every position of the window x [B,K,N,4] -> [B,K,dz]."""
        e = self.spatial(x)                                 # [B,K,N,d]
        tok = self.token(torch.cat((e.mean(2), e.amax(2), phys_features(x)), -1))
        seq = tok + self.pos[:, -x.shape[1]:]
        for block in self.blocks:
            seq, _ = block(seq, causal=True)
        return self.norm(seq), e

    def forward(self, x, z_override=None):
        """x [B,K,N,4] -> next-state prediction dict. z_override [B,dz]
        replaces z at the last position (Stage 2 latent interventions)."""
        x = x[:, -self.k:] if self.k >= 1 else x[:, -1:]
        z_seq, e = self.latent(x)
        if self.kind == 'k0' or self.k == 0:
            z = torch.zeros(x.shape[0], self.dz, device=x.device)
        else:
            z = z_seq[:, -1]
        if z_override is not None:
            z = z_override
        fused = torch.cat((e[:, -1], z[:, None, :].expand(-1, x.shape[2], -1)), -1)
        corr = self.head(fused)
        base = base_pre(x[:, -1], self.cfg, self.W, self.i_bias)
        return dict(v=base['v_base'] + corr[..., 0],
                    s_logits=base['logit_base'] + corr[..., 1],
                    r=base['r_base'] + corr[..., 2],
                    corr=corr, z=z_seq)


def build_v11(kind, conn, cfg, k=32):
    return LatentHybridV11(conn, cfg, kind=kind, k=k)
