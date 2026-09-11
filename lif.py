"""Discrete-time batched LIF simulator.

    tau * dV/dt = -(V - V_rest) + I_syn + I_ext
    V[t+1] = V[t] + dt/tau * (-(V[t]-V_rest) + I_syn[t] + I_ext[t])
    I_syn[i,t] = sum_j W[j,i] * spike[j,t]
    V >= V_th  ->  spike=1, V=V_reset, enter refractory period

State features stored per neuron per step: [V, spike, refractory].
The refractory feature is the remaining refractory time normalised to [0,1].

Supports: excitatory/inhibitory edges, external stimulation, neuron
silencing, edge lesion (via a lesioned connectome), fixed initial state,
batched trajectory generation.

Note: synaptic integration uses a dense [N, N] weight matrix, which is
fine for N <= ~2000. For larger N replace `S @ W` with a sparse matmul.
"""
from __future__ import annotations

import torch

from connectome import Connectome
from config import Config


class LIFSimulator:
    def __init__(self, connectome: Connectome, cfg: Config, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.N = connectome.n_neurons
        self.W = connectome.dense_weight(device)          # W[src, dst]
        if connectome.i_bias is not None:
            self.i_bias = connectome.i_bias.to(device)
        else:
            self.i_bias = torch.zeros(self.N, device=device)
        self.connectome = connectome.to(device)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def simulate(
        self,
        stimulus: torch.Tensor,               # [B, T, N]
        v0: torch.Tensor | None = None,       # [B, N]
        silence_mask: torch.Tensor | None = None,  # [B, N] bool, True = silenced
        state0: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Run a batched simulation. Returns states [B, T, N, 3] (V, S, R).

        state0 optionally gives the full initial dynamical state
        (V, S, R_remaining) for branching continuations; R_remaining is the
        unnormalised refractory counter.
        """
        cfg = self.cfg
        B, T, N = stimulus.shape
        a = cfg.alpha
        dev = self.device

        if state0 is not None:
            V, S, R = (t.to(dev).clone() for t in state0)
        else:
            V = torch.zeros(B, N, device=dev) if v0 is None else v0.to(dev).clone()
            S = torch.zeros(B, N, device=dev)
            R = torch.zeros(B, N, device=dev)

        states = torch.empty(B, T, N, 3, device=dev)
        for t in range(T):
            I = S @ self.W + stimulus[:, t] + self.i_bias
            refr = R > 0
            V_new = torch.where(refr, torch.full_like(V, cfg.v_reset),
                                V + a * (-(V - cfg.v_rest) + I))
            V_new = torch.clamp(V_new, min=cfg.v_min)
            fire = (~refr) & (V_new >= cfg.v_th)
            S = fire.to(V.dtype)
            V = torch.where(fire, torch.full_like(V, cfg.v_reset), V_new)
            R = torch.where(fire, torch.full_like(R, float(cfg.refractory_period)),
                            torch.clamp(R - 1.0, min=0.0))
            if silence_mask is not None:
                sil = silence_mask.to(dev)
                S = torch.where(sil, torch.zeros_like(S), S)
                V = torch.where(sil, torch.full_like(V, cfg.v_rest), V)
                R = torch.where(sil, torch.zeros_like(R), R)
            states[:, t, :, 0] = V
            states[:, t, :, 1] = S
            states[:, t, :, 2] = R / float(cfg.refractory_period)
        return states

    # ------------------------------------------------------------------
    def with_lesion(self, drop_edge_mask: torch.Tensor) -> "LIFSimulator":
        """New simulator with selected edges removed (edge lesion)."""
        lesioned = self.connectome.without_edges(drop_edge_mask)
        return LIFSimulator(lesioned, self.cfg, self.device)
