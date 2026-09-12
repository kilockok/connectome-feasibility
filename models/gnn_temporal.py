"""GNN + Temporal Transformer (phase-3): per-timestep connectome message
passing, then per-neuron temporal self-attention over the history window.

Architecture (spec: spatial GNN first, temporal attention second):

    x [B, K, N, 4]                    (V, S, R_norm, U per timestep)
      -> slice last k_hist timesteps
      -> per timestep: feature embed + neuron embedding, then M rounds of
         edge-weighted message passing on the connectome (models.gnn
         MessageRound, unchanged)            -> h_t [B, N, D] per timestep
      -> stack to [B, N, k, D], reshape to [B*N, k, D]
      -> temporal positional encoding (learned or sinusoidal; encodes
         timestep ORDER only, never neuron identity)
      -> L pre-norm transformer blocks along the TIME axis (each neuron
         attends over its own history independently — no cross-neuron
         attention; space is the GNN's job)
      -> readout at the last position, LayerNorm, linear head
      -> {"v", "s_logits", "r"}  (same output contract as every model, so
         MechanisticWrapper / rollout / DAgger all work unchanged)

Memory note: the per-timestep GNN sees B*k graphs per forward, so the
[B, E, D] message tensor in MessageRound scales with k. encode_spatial
processes the flattened (B*k) graphs in chunks (default 32) — exact, since
graphs never interact inside the GNN.

`k_hist` truncates the window INSIDE forward, so one dataset / rollout /
DAgger pipeline (window length cfg.K) serves the whole history ablation:
k_hist=1 is the last-state-only (Markov) control, k_hist=cfg.K the full
model. Positional encodings are sized for cfg.K and sliced to k_hist.

return_attn=True (diagnostics only) makes the temporal blocks compute
attention weights explicitly (instead of fused SDPA) and returns the LAST
block's head-averaged attention from the predicting (last) position:
attn [B, N, k] — "how much does neuron i look at each past timestep".
Attention weights are an interpretability cue, not causal evidence.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.gnn import MessageRound
from models.transformer import N_FEATURES


def sinusoidal_table(max_len: int, d: int) -> torch.Tensor:
    """Standard sin/cos table [1, max_len, d]."""
    pos = torch.arange(max_len, dtype=torch.float32)[:, None]
    div = torch.exp(torch.arange(0, d, 2, dtype=torch.float32)
                    * (-math.log(10000.0) / d))
    pe = torch.zeros(max_len, d)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
    return pe.unsqueeze(0)


class TemporalBlock(nn.Module):
    """Pre-norm transformer block over the time axis (length k) of a
    [B*, k, D] sequence. With need_weights=True the attention matrix is
    computed explicitly and returned (diagnostics; slower, more memory)."""

    def __init__(self, d_model: int, nhead: int, dim_ff: int,
                 dropout: float = 0.0):
        super().__init__()
        assert d_model % nhead == 0
        self.h = nhead
        self.dh = d_model // nhead
        self.ln1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, dim_ff), nn.GELU(),
                                 nn.Linear(dim_ff, d_model))
        self.drop = dropout

    def forward(self, x: torch.Tensor, causal: bool = False,
                need_weights: bool = False):
        B, L, D = x.shape
        h = self.ln1(x)
        qkv = (self.qkv(h).reshape(B, L, 3, self.h, self.dh)
               .permute(2, 0, 3, 1, 4))                    # [3,B,H,L,dh]
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = None
        if need_weights:
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh)
            if causal:
                mask = torch.triu(torch.ones(L, L, dtype=torch.bool,
                                             device=x.device), diagonal=1)
                att = att.masked_fill(mask, float("-inf"))
            att = att.softmax(dim=-1)
            o = att @ v
        else:
            o = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.drop if self.training else 0.0,
                is_causal=causal)
        x = x + self.proj(o.permute(0, 2, 1, 3).reshape(B, L, D))
        x = x + self.ffn(self.ln2(x))
        return x, att


class GNNTemporalTransformer(nn.Module):
    """Connectome GNN per timestep + per-neuron temporal Transformer.

    k_hist (default cfg.K): how many of the last timesteps the model
    consumes; the input window (cfg.K) is truncated internally, so data
    pipelines never change across the history ablation.
    """

    def __init__(self, cfg, connectome, k_hist: int | None = None,
                 d_model: int | None = None, t_layers: int = 2,
                 t_heads: int = 4, t_ff: int | None = None,
                 dropout: float | None = None, pos_type: str = "learned",
                 causal: bool = False, gnn_chunk: int = 32):
        super().__init__()
        self.cfg = cfg
        d_model = d_model or cfg.d_model
        t_ff = t_ff or 4 * d_model
        dropout = cfg.dropout if dropout is None else dropout
        self.k_hist = int(k_hist) if k_hist else cfg.K
        if not 1 <= self.k_hist <= cfg.K:
            raise ValueError(f"k_hist must be in [1, cfg.K={cfg.K}], "
                             f"got {self.k_hist}")
        if pos_type not in ("learned", "sincos"):
            raise ValueError(f"pos_type must be learned|sincos, got {pos_type}")
        self.pos_type = pos_type
        self.causal = causal
        self.gnn_chunk = max(1, int(gnn_chunk))
        self.d_model = d_model

        # ---- spatial (per-timestep) encoder ------------------------------
        self.inp = nn.Linear(N_FEATURES, d_model)
        self.neuron_emb = nn.Parameter(
            torch.randn(1, cfg.n_neurons, d_model) * 0.02)
        self.rounds = nn.ModuleList(
            [MessageRound(d_model) for _ in range(cfg.gnn_layers)])
        self.gnn_norm = nn.LayerNorm(d_model)

        # ---- temporal encoder --------------------------------------------
        if pos_type == "learned":
            self.pos = nn.Parameter(torch.randn(1, cfg.K, d_model) * 0.02)
        else:
            self.register_buffer("pos", sinusoidal_table(cfg.K, d_model))
        self.t_blocks = nn.ModuleList(
            [TemporalBlock(d_model, t_heads, t_ff, dropout)
             for _ in range(t_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 3)

        # ---- connectome buffers (identical to GNNBaseline) ---------------
        ei = connectome.edge_index.cpu()
        ew = connectome.edge_weight.cpu()
        N = cfg.n_neurons
        deg = torch.zeros(N).scatter_add_(
            0, ei[1], torch.ones_like(ew)).clamp(min=1.0)
        self.register_buffer("edge_index", ei)
        self.register_buffer("edge_weight", ew)
        self.register_buffer("presyn_type", connectome.neuron_type.cpu())
        self.register_buffer("deg_sqrt", deg.sqrt())

    # ------------------------------------------------------------------
    def encode_spatial(self, x: torch.Tensor) -> torch.Tensor:
        """x [B, k, N, F] -> [B, k, N, D]; the B*k graphs are processed in
        chunks so the [graphs, E, D] message tensor stays bounded."""
        B, k, N, Ft = x.shape
        flat = x.reshape(B * k, N, Ft)
        outs = []
        for part in flat.split(self.gnn_chunk):
            h = self.inp(part) + self.neuron_emb
            for rnd in self.rounds:
                h = rnd(h, self.edge_index, self.edge_weight,
                        self.presyn_type, self.deg_sqrt)
            outs.append(self.gnn_norm(h))
        return torch.cat(outs, dim=0).reshape(B, k, N, -1)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, return_attn: bool = False):
        """x [B, K, N, 4] -> {"v","s_logits","r"}; with return_attn also the
        last block's head-averaged attention from the last position
        [B, N, k_hist]."""
        x = x[:, -self.k_hist:]
        h = self.encode_spatial(x)                        # [B, k, N, D]
        B, k, N, D = h.shape
        h = h.permute(0, 2, 1, 3).reshape(B * N, k, D)
        h = h + self.pos[:, -k:]
        att = None
        for i, blk in enumerate(self.t_blocks):
            need = return_attn and (i == len(self.t_blocks) - 1)
            h, a = blk(h, causal=self.causal, need_weights=need)
            if need:
                att = a
        z = self.norm(h[:, -1]).reshape(B, N, D)
        y = self.head(z)
        out = {"v": y[..., 0], "s_logits": y[..., 1], "r": y[..., 2]}
        if return_attn:
            # [B*N, heads, k, k] -> predicting row, head mean -> [B, N, k]
            w = att.mean(dim=1)[:, -1, :].reshape(B, N, k)
            return out, w
        return out


def gnn_temporal_kwargs(cfg, k_hist: int | None = None, **overrides) -> dict:
    """Canonical model_kwargs dict for checkpoints / rebuilds."""
    kw = {"k_hist": k_hist or cfg.K, "t_layers": 2, "t_heads": 4,
          "pos_type": "learned", "causal": False}
    kw.update({k: v for k, v in overrides.items() if v is not None})
    return kw
