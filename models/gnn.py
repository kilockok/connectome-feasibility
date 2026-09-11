"""GNN baseline: temporal encoder + edge-weighted message passing.

Per-neuron temporal embedding (shared with the transformers), then M rounds
of message passing along connectome edges:

    m_i = sum_j w_ji * (W_msg h_j) / sqrt(deg_i)
    h_i = h_i + MLP([h_i, m_i])

Implemented with index_add on edge lists (no torch_geometric dependency).
Signed weights are kept as-is; each round has its own transform so
excitatory/inhibitory messages are learnable per type via an edge-type
embedding added to the message.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from models.transformer import TemporalEncoder


class MessageRound(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.msg = nn.Linear(d, d)
        self.type_emb = nn.Parameter(torch.zeros(2, d))
        self.update = nn.Sequential(nn.Linear(2 * d, 2 * d), nn.GELU(),
                                    nn.Linear(2 * d, d))

    def forward(self, h, edge_index, edge_weight, presyn_type, deg_sqrt):
        # h [B, N, d]
        B, N, d = h.shape
        src, dst = edge_index[0], edge_index[1]
        m = self.msg(h) + self.type_emb[presyn_type]      # [B, N, d]
        contrib = m[:, src] * edge_weight[None, :, None]  # [B, E, d]
        agg = torch.zeros_like(h)
        idx = dst[None, :, None].expand(B, -1, d)
        agg.scatter_add_(1, idx, contrib)
        agg = agg / deg_sqrt[None, :, None]
        return h + self.update(torch.cat([h, agg], dim=-1))


class GNNBaseline(nn.Module):
    def __init__(self, cfg, connectome):
        super().__init__()
        self.cfg = cfg
        self.temporal = TemporalEncoder(cfg)
        self.proj = nn.Linear(cfg.d_temporal, cfg.d_model)
        self.neuron_emb = nn.Parameter(
            torch.randn(1, cfg.n_neurons, cfg.d_model) * 0.02)
        self.rounds = nn.ModuleList(
            [MessageRound(cfg.d_model) for _ in range(cfg.gnn_layers)])
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, 3)

        ei = connectome.edge_index.cpu()
        ew = connectome.edge_weight.cpu()
        N = cfg.n_neurons
        deg = torch.zeros(N).scatter_add_(
            0, ei[1], torch.ones_like(ew)).clamp(min=1.0)
        self.register_buffer("edge_index", ei)
        self.register_buffer("edge_weight", ew)
        self.register_buffer("presyn_type", connectome.neuron_type.cpu())
        self.register_buffer("deg_sqrt", deg.sqrt())

    def forward(self, x: torch.Tensor, attn_bias=None) -> dict:
        h = self.proj(self.temporal(x)) + self.neuron_emb
        for rnd in self.rounds:
            h = rnd(h, self.edge_index, self.edge_weight,
                    self.presyn_type, self.deg_sqrt)
        y = self.head(self.norm(h))
        return {"v": y[..., 0], "s_logits": y[..., 1], "r": y[..., 2]}
