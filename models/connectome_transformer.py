"""Connectome-constrained Transformer.

Same architecture as the vanilla transformer, but spatial attention is
masked/biased by the connectome:

    score(i, j) = Q_i . K_j / sqrt(d)
                  + alpha_h * log(1 + |W_ji|)      (only if edge j -> i)
                  + edge_type_emb[type_j]          (excitatory/inhibitory)
                  - LARGE                           (if no edge j -> i)

Self-attention (i == j) is always allowed with zero bias so no row of the
attention matrix is fully masked.

Note: the additive bias is a dense [N, N] tensor, which is fine for
N <= ~1000 with fused SDPA kernels (memory stays ~O(N^2 * H) in the
kernel's working set only). For N >> 2000 this must become a sparse /
neighborhood attention (see README).

The connectome is exposed so perturbation experiments can remove edges
(via `attn_bias_override`) or silence neurons exactly as in the simulator.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from models.transformer import VanillaTransformer

NEG = -1e4


class ConnectomeTransformer(VanillaTransformer):
    def __init__(self, cfg, connectome):
        super().__init__(cfg)
        N = cfg.n_neurons
        ei = connectome.edge_index.cpu()
        ew = connectome.edge_weight.cpu()
        ntype = connectome.neuron_type.cpu()

        edge = torch.zeros(N, N, dtype=torch.bool)
        edge[ei[1], ei[0]] = True                       # edge[dst, src]
        logw = torch.zeros(N, N)
        logw[ei[1], ei[0]] = torch.log1p(ew.abs())

        mask = torch.full((N, N), NEG)
        mask[edge] = 0.0
        mask.fill_diagonal_(0.0)

        self.register_buffer("conn_mask", mask)         # [N, N]
        self.register_buffer("conn_logw", logw)         # [N, N]
        self.register_buffer("conn_edge", edge)
        # presynaptic type per key column j
        self.register_buffer("presyn_type", ntype)      # [N]

        self.alpha = nn.Parameter(torch.full((cfg.nhead,), cfg.conn_alpha))
        self.type_emb = nn.Parameter(torch.zeros(2))

    def attn_bias(self) -> torch.Tensor:
        # [1, H, N, N] = mask + alpha_h * logw + type_emb[type_j] * edge
        bias = self.conn_mask.unsqueeze(0).unsqueeze(0)
        bias = bias + self.alpha[None, :, None, None] * self.conn_logw
        bias = bias + (self.type_emb[self.presyn_type][None, :]
                       * self.conn_edge).unsqueeze(0).unsqueeze(0)
        return bias

    @torch.no_grad()
    def rebuild_bias_from_connectome(self, conn) -> None:
        """Replace the attention bias buffers from a (lesioned) connectome.

        Used by the perturbation experiments: removing edge A->B in the
        model is exactly removing A->B from the attention mask."""
        N = self.cfg.n_neurons
        ei = conn.edge_index.cpu()
        ew = conn.edge_weight.cpu()
        edge = torch.zeros(N, N, dtype=torch.bool)
        edge[ei[1], ei[0]] = True
        logw = torch.zeros(N, N)
        logw[ei[1], ei[0]] = torch.log1p(ew.abs())
        mask = torch.full((N, N), NEG)
        mask[edge] = 0.0
        mask.fill_diagonal_(0.0)
        dev = self.conn_mask.device
        self.conn_mask = mask.to(dev)
        self.conn_logw = logw.to(dev)
        self.conn_edge = edge.to(dev)
