"""GRU baseline: flatten the neuron dimension, no spatial structure.

Input  [B, K, N, 4] -> reshape [B, K, N*4] -> GRU -> last hidden -> MLP
Output dict(v=[B,N], s_logits=[B,N], r=[B,N])
"""
from __future__ import annotations

import torch
import torch.nn as nn


class GRUBaseline(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        N, F = cfg.n_neurons, 4
        self.gru = nn.GRU(N * F, cfg.gru_hidden, num_layers=1, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(cfg.gru_hidden, cfg.gru_hidden), nn.GELU(),
            nn.Linear(cfg.gru_hidden, N * 3),
        )
        self.N = N

    def forward(self, x: torch.Tensor, attn_bias=None) -> dict:
        B, K, N, F = x.shape
        h = x.reshape(B, K, N * F)
        out, _ = self.gru(h)
        y = self.head(out[:, -1]).reshape(B, N, 3)
        return {"v": y[..., 0], "s_logits": y[..., 1], "r": y[..., 2]}
