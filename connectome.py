"""Connectome representation + placeholder generator.

Canonical representation (as in FlyWire exports):
    edge_index  : LongTensor [2, E]  (row 0 = src j, row 1 = dst i, edge j -> i)
    edge_weight : FloatTensor [E]    (signed: >0 excitatory, <0 inhibitory)

The placeholder is a ring-geometry distance-dependent sparse directed graph
with Dale's law (a neuron is either excitatory or inhibitory everywhere).
The local structure makes the spatial OOD stimulus split meaningful.
Replace `Connectome.generate(...)` with a FlyWire subgraph loader later;
everything downstream only relies on edge_index / edge_weight.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class Connectome:
    n_neurons: int
    edge_index: torch.Tensor      # [2, E] (src, dst)
    edge_weight: torch.Tensor     # [E]
    neuron_type: torch.Tensor     # [N] 0 = excitatory, 1 = inhibitory
    i_bias: torch.Tensor | None = None  # [N] tonic bias current

    # ------------------------------------------------------------------
    @classmethod
    def generate(cls, cfg, seed: int | None = None) -> "Connectome":
        """Distance-dependent random directed graph on a ring."""
        N = cfg.n_neurons
        g = torch.Generator().manual_seed(cfg.seed + 77 if seed is None else seed)

        # neuron types (Dale's law): first inh_fraction are inhibitory
        perm = torch.randperm(N, generator=g)
        neuron_type = torch.zeros(N, dtype=torch.long)
        neuron_type[perm[: int(cfg.inh_fraction * N)]] = 1

        # ring distance matrix, vectorised in chunks to bound memory.
        # peak prob p0 chosen so E[degree] ~= conn_target_degree:
        # E[deg] ~= 2*p0*lambda + N*p_bg  (ring has two sides)
        lam = max(cfg.conn_lambda_frac * N, 1.0)
        p0 = min((cfg.conn_target_degree - N * cfg.conn_p_bg) / (2.0 * lam), 0.9)
        p0 = max(p0, 0.0)
        src_list, dst_list, w_list = [], [], []
        for i0 in range(0, N, 4096):
            dst = torch.arange(i0, min(i0 + 4096, N))
            d = (dst[:, None] - torch.arange(N)[None, :]).abs().float()
            d = torch.minimum(d, N - d)                      # ring distance
            p = p0 * torch.exp(-d / lam) + cfg.conn_p_bg
            p = torch.clamp(p, max=0.9)
            rnd = torch.rand(p.shape, generator=g)
            mask = (rnd < p) & (torch.arange(N)[None, :] != dst[:, None])
            row, col = mask.nonzero(as_tuple=True)          # row: local dst, col: src
            if row.numel() == 0:
                continue
            src, dst_g = col, row + i0
            w = torch.empty(src.numel()).uniform_(0.0, 1.0, generator=g)
            is_inh = neuron_type[src] == 1
            w_exc = cfg.w_exc_lo + (cfg.w_exc_hi - cfg.w_exc_lo) * w
            w_inh = -(cfg.w_inh_lo + (cfg.w_inh_hi - cfg.w_inh_lo) * w)
            w = torch.where(is_inh, w_inh, w_exc)
            src_list.append(src); dst_list.append(dst_g); w_list.append(w)

        edge_index = torch.stack([torch.cat(src_list), torch.cat(dst_list)])
        edge_weight = torch.cat(w_list)
        i_bias = (cfg.i_bias_lo + (cfg.i_bias_hi - cfg.i_bias_lo)
                  * torch.rand(N, generator=g))
        return cls(N, edge_index, edge_weight, neuron_type, i_bias)

    # ------------------------------------------------------------------
    @property
    def n_edges(self) -> int:
        return self.edge_index.shape[1]

    def dense_weight(self, device=None) -> torch.Tensor:
        """Dense W [N, N] with W[j, i] = weight of edge j -> i.

        Only used for simulation and dense attention bias at N <= ~2000.
        For larger N this must be replaced by sparse ops (see README).
        """
        W = torch.zeros(self.n_neurons, self.n_neurons, device=device)
        W[self.edge_index[0].to(W.device), self.edge_index[1].to(W.device)] = \
            self.edge_weight.to(W.device, W.dtype)
        return W

    def to(self, device) -> "Connectome":
        return Connectome(self.n_neurons, self.edge_index.to(device),
                          self.edge_weight.to(device), self.neuron_type.to(device),
                          self.i_bias.to(device) if self.i_bias is not None else None)

    def without_edges(self, drop_mask: torch.Tensor) -> "Connectome":
        """Return a copy with edges where drop_mask is True removed (lesion)."""
        keep = ~drop_mask
        return Connectome(self.n_neurons, self.edge_index[:, keep],
                          self.edge_weight[keep], self.neuron_type.clone(),
                          self.i_bias.clone() if self.i_bias is not None else None)

    # ------------------------------------------------------------------
    def save(self, path: str | Path):
        torch.save({"n_neurons": self.n_neurons, "edge_index": self.edge_index,
                    "edge_weight": self.edge_weight, "neuron_type": self.neuron_type,
                    "i_bias": self.i_bias}, path)

    @classmethod
    def load(cls, path: str | Path) -> "Connectome":
        d = torch.load(path, map_location="cpu", weights_only=True)
        return cls(d["n_neurons"], d["edge_index"], d["edge_weight"],
                   d["neuron_type"], d.get("i_bias"))

    def summary(self) -> str:
        n_inh = int((self.neuron_type == 1).sum())
        return (f"N={self.n_neurons}, E={self.n_edges}, "
                f"avg_degree={self.n_edges / self.n_neurons:.1f}, "
                f"exc={self.n_neurons - n_inh}, inh={n_inh}")


def get_connectome(cfg, device=None) -> Connectome:
    """Load the cached connectome or generate it (fixed seed => same graph
    for every model and every dataset, which the comparison relies on)."""
    path = CACHE_PATH(cfg)
    if path.exists():
        conn = Connectome.load(path)
    else:
        conn = Connectome.generate(cfg)
        conn.save(path)
    return conn.to(device) if device is not None else conn


def CACHE_PATH(cfg):
    from config import CACHE_DIR
    return CACHE_DIR / f"connectome_N{cfg.n_neurons}_seed{cfg.seed}.pt"
