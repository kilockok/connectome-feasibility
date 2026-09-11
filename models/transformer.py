"""Vanilla Transformer baseline (no connectome) + shared building blocks.

Architecture (spatial transformer over neuron tokens):
  per neuron: K-step history (V,S,R,U) -> causal temporal encoder -> h_n
  h_n + neuron identity embedding -> L x transformer block over N tokens
  -> per-neuron head -> (V, spike logit, refractory)

Full N x N attention is allowed. Temporal encoding uses causal attention
over the K=32-step history. All attention goes through
F.scaled_dot_product_attention (memory-efficient on XPU).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

N_FEATURES = 4   # V, S, R, U


class MHA(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % nhead == 0
        self.h = nhead
        self.dh = d_model // nhead
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.drop = dropout

    def forward(self, x: torch.Tensor, attn_bias: torch.Tensor | None = None,
                causal: bool = False) -> torch.Tensor:
        B, L, D = x.shape
        qkv = (self.qkv(x).reshape(B, L, 3, self.h, self.dh)
               .permute(2, 0, 3, 1, 4))                  # [3, B, H, L, dh]
        out = F.scaled_dot_product_attention(
            qkv[0], qkv[1], qkv[2], attn_mask=attn_bias,
            dropout_p=self.drop if self.training else 0.0,
            is_causal=causal)
        return self.proj(out.permute(0, 2, 1, 3).reshape(B, L, D))


class Block(nn.Module):
    """Pre-norm transformer block."""
    def __init__(self, d_model: int, nhead: int, dim_ff: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MHA(d_model, nhead, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, dim_ff), nn.GELU(),
                                 nn.Linear(dim_ff, d_model))

    def forward(self, x, attn_bias=None, causal=False):
        x = x + self.attn(self.ln1(x), attn_bias=attn_bias, causal=causal)
        x = x + self.ffn(self.ln2(x))
        return x


class TemporalEncoder(nn.Module):
    """Per-neuron causal encoder over the K-step history."""
    def __init__(self, cfg):
        super().__init__()
        d = cfg.d_temporal
        self.K = cfg.K
        self.inp = nn.Linear(N_FEATURES, d)
        self.pos = nn.Parameter(torch.randn(1, cfg.K, d) * 0.02)
        self.blocks = nn.ModuleList(
            [Block(d, 4, d * 2, cfg.dropout) for _ in range(cfg.temporal_layers)])
        self.norm = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x [B, K, N, F] -> [B, N, d]
        B, K, N, Ft = x.shape
        h = x.permute(0, 2, 1, 3).reshape(B * N, K, Ft)
        h = self.inp(h) + self.pos
        for blk in self.blocks:
            h = blk(h, causal=True)
        return self.norm(h[:, -1]).reshape(B, N, -1)


class VanillaTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.temporal = TemporalEncoder(cfg)
        self.proj = nn.Linear(cfg.d_temporal, cfg.d_model)
        self.neuron_emb = nn.Parameter(
            torch.randn(1, cfg.n_neurons, cfg.d_model) * 0.02)
        self.blocks = nn.ModuleList(
            [Block(cfg.d_model, cfg.nhead, cfg.dim_ff, cfg.dropout)
             for _ in range(cfg.spatial_layers)])
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, 3)

    def attn_bias(self) -> torch.Tensor | None:
        return None

    def forward(self, x: torch.Tensor, attn_bias_override=None) -> dict:
        h = self.temporal(x)
        h = self.proj(h) + self.neuron_emb
        bias = attn_bias_override if attn_bias_override is not None \
            else self.attn_bias()
        for blk in self.blocks:
            h = blk(h, attn_bias=bias)
        y = self.head(self.norm(h))
        return {"v": y[..., 0], "s_logits": y[..., 1], "r": y[..., 2]}
