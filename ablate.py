"""Graph ablation controls (sanity checks 1-3).

All ablations modify ONLY the graph seen by the model (attention bias);
the LIF simulator always runs the true connectome, so the task itself is
unchanged and any performance delta is attributable to the model's use of
the graph <-> dynamics correspondence.

  shuffle-edges   : degree-preserving random rewire (topology destroyed)
  shuffle-weights : topology kept, weights permuted within exc/inh pools
  identity        : no edges at all (self-attention only; leakage probe)
"""
from __future__ import annotations

import torch

from connectome import Connectome


def rewire_connectome(conn: Connectome, seed: int,
                      n_swaps: int | None = None) -> Connectome:
    """Check 1: degree-preserving random rewire.

    Repeatedly swaps the targets of two edges: (a->b, c->d) => (a->d, c->b).
    Preserves exactly: every neuron's in- and out-degree, the edge count,
    and the weight pool (weights travel with their source edge, so Dale's
    law still holds). Destroys all topology (ring geometry / distance
    dependence). Swaps that would create self-loops or parallel edges are
    rejected.
    """
    g = torch.Generator().manual_seed(seed)
    src = conn.edge_index[0].tolist()
    dst = conn.edge_index[1].tolist()
    E = len(src)
    edges = set(zip(src, dst))
    target = n_swaps or 10 * E
    done, attempts = 0, 0
    while done < target and attempts < 20 * target:
        attempts += 1
        i, j = torch.randint(0, E, (2,), generator=g).tolist()
        if i == j:
            continue
        a, b, c, d = src[i], dst[i], src[j], dst[j]
        if a == c or b == d or a == d or c == b:
            continue                          # no-op / self-loop
        if (a, d) in edges or (c, b) in edges:
            continue                          # parallel edge
        edges.discard((a, b)); edges.discard((c, d))
        edges.add((a, d)); edges.add((c, b))
        dst[i], dst[j] = d, b
        done += 1
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    return Connectome(conn.n_neurons, edge_index, conn.edge_weight.clone(),
                      conn.neuron_type.clone(),
                      conn.i_bias.clone() if conn.i_bias is not None else None)


def shuffle_weights(conn: Connectome, seed: int) -> Connectome:
    """Check 2: keep topology, shuffle weights within exc / inh pools.

    Sign structure (Dale's law) is preserved per edge because pools are
    defined by the presynaptic neuron's type; only the weight<->topology
    correlation (e.g. distance dependence) is destroyed.
    """
    g = torch.Generator().manual_seed(seed)
    w = conn.edge_weight
    src_type = conn.neuron_type[conn.edge_index[0]]
    new_w = w.clone()
    for t in (0, 1):
        idx = (src_type == t).nonzero(as_tuple=True)[0]
        perm = torch.randperm(idx.numel(), generator=g)
        new_w[idx] = w[idx[perm]]
    return Connectome(conn.n_neurons, conn.edge_index.clone(), new_w,
                      conn.neuron_type.clone(),
                      conn.i_bias.clone() if conn.i_bias is not None else None)


def identity_connectome(conn: Connectome) -> Connectome:
    """Check 3: null graph — every neuron attends only to itself."""
    empty_idx = torch.zeros(2, 0, dtype=torch.long)
    empty_w = torch.zeros(0)
    return Connectome(conn.n_neurons, empty_idx, empty_w,
                      conn.neuron_type.clone(),
                      conn.i_bias.clone() if conn.i_bias is not None else None)


def ablate_connectome(conn: Connectome, mode: str, seed: int) -> Connectome:
    if mode == "shuffle-edges":
        return rewire_connectome(conn, seed)
    if mode == "shuffle-weights":
        return shuffle_weights(conn, seed)
    if mode == "identity":
        return identity_connectome(conn)
    raise ValueError(f"unknown ablation {mode}")


ABLATIONS = ("shuffle-edges", "shuffle-weights", "identity")
