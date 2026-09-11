"""Synthetic perturbation dataset.

Each trajectory is fully determined by an integer seed, so the dataset is
reproducible without storing gigabytes on disk. Training generates batches
on the fly from a fixed pool of trajectory seeds; evaluation splits are
generated once and cached to disk so that every model sees identical data.

OOD protocol (spatial stimulus split):
  train      : stimulated neurons drawn from [0, 0.8N)
  val        : [0.7N, 0.9N)
  test / OOD : [0.8N, N)   <- neurons never directly stimulated in training
  test_seen  : [0, 0.8N) with fresh seeds (in-distribution control)

A trajectory: random V0, 1-5 stimulated neurons, each with a random onset
and a 5-20 step pulse of random amplitude; a fraction also silences 1-3
random neurons (the silenced ids are recorded in the metadata).
"""
from __future__ import annotations

import torch

from config import Config, CACHE_DIR
from connectome import Connectome
from lif import LIFSimulator


def sample_traj_params(g: torch.Generator, cfg: Config, split: str) -> dict:
    """Random perturbation parameters for one trajectory."""
    N, T = cfg.n_neurons, cfg.T
    lo, hi = cfg.stim_range(split)

    n_stim = int(torch.randint(cfg.stim_min_neurons, cfg.stim_max_neurons + 1, (1,), generator=g))
    perm = torch.randperm(hi - lo, generator=g)[:n_stim] + lo
    onset = torch.randint(0, max(T - cfg.stim_dur_hi, 1), (n_stim,), generator=g)
    dur = torch.randint(cfg.stim_dur_lo, cfg.stim_dur_hi + 1, (n_stim,), generator=g)
    amp = cfg.stim_amp_lo + (cfg.stim_amp_hi - cfg.stim_amp_lo) * torch.rand(n_stim, generator=g)

    silenced = torch.zeros(N, dtype=torch.bool)
    if torch.rand((), generator=g).item() < cfg.silence_prob:
        n_sil = int(torch.randint(1, cfg.silence_max + 1, (1,), generator=g))
        sil_ids = torch.randperm(N, generator=g)[:n_sil]
        silenced[sil_ids] = True

    return {"stim_ids": perm, "onset": onset, "dur": dur, "amp": amp,
            "silenced": silenced}


def build_stimulus(params: dict, cfg: Config) -> torch.Tensor:
    """[T, N] external drive from perturbation parameters."""
    stim = torch.zeros(cfg.T, cfg.n_neurons)
    for nid, on, du, am in zip(params["stim_ids"], params["onset"],
                               params["dur"], params["amp"]):
        stim[int(on): int(on) + int(du), int(nid)] = float(am)
    return stim


@torch.no_grad()
def generate_batch(seeds: list[int], split: str, sim: LIFSimulator,
                   cfg: Config) -> dict:
    """Generate one batch of trajectories from integer seeds."""
    N = cfg.n_neurons
    stim, v0, sil, metas = [], [], [], []
    for s in seeds:
        g = torch.Generator().manual_seed(int(s))
        p = sample_traj_params(g, cfg, split)
        stim.append(build_stimulus(p, cfg))
        v0.append(torch.rand(N, generator=g) * cfg.v_th)
        sil.append(p["silenced"])
        metas.append({"stim_ids": p["stim_ids"], "seed": int(s),
                      "onset": p["onset"], "dur": p["dur"]})
    stimulus = torch.stack(stim).to(sim.device)
    v0 = torch.stack(v0).to(sim.device)
    silence = torch.stack(sil).to(sim.device)
    states = sim.simulate(stimulus, v0=v0, silence_mask=silence)
    return {"states": states, "stimulus": stimulus,
            "silence": silence, "v0": v0, "meta": metas}


def traj_count(cfg: Config, split: str) -> int:
    return {"train": cfg.n_train_traj, "val": cfg.n_val_traj,
            "test": cfg.n_test_traj, "test_ood": cfg.n_test_traj,
            "test_seen": cfg.n_test_seen_traj}[split]


def load_or_generate(split: str, sim: LIFSimulator, cfg: Config,
                     batch_size: int = 64) -> dict:
    """Deterministic cached dataset for evaluation splits."""
    n = traj_count(cfg, split)
    path = (CACHE_DIR / f"data_{split}_N{cfg.n_neurons}_T{cfg.T}"
                        f"_n{n}_seed{cfg.seed}.pt")
    if path.exists():
        d = torch.load(path, map_location="cpu", weights_only=False)
        return {k: (v.to(sim.device) if torch.is_tensor(v) else v)
                for k, v in d.items()}
    parts = []
    for i0 in range(0, n, batch_size):
        seeds = [cfg.traj_seed(split, i) for i in range(i0, min(i0 + batch_size, n))]
        parts.append(generate_batch(seeds, split, sim, cfg))
    out = {
        "states": torch.cat([p["states"] for p in parts]).cpu(),
        "stimulus": torch.cat([p["stimulus"] for p in parts]).cpu(),
        "silence": torch.cat([p["silence"] for p in parts]).cpu(),
        "meta": [m for p in parts for m in p["meta"]],
    }
    torch.save(out, path)
    return {k: (v.to(sim.device) if torch.is_tensor(v) else v)
            for k, v in out.items()}


def sample_t0(metas: list[dict], K: int, T: int, activity_frac: float,
              generator: torch.Generator, device) -> torch.Tensor:
    """Sample window starts; `activity_frac` of windows are placed so the
    window [t0, t0+K] overlaps a stimulus pulse (the rest are uniform)."""
    B = len(metas)
    t_max = T - K - 1
    t0 = torch.randint(0, t_max + 1, (B,), generator=generator)
    for b, m in enumerate(metas):
        if torch.rand((), generator=generator).item() >= activity_frac:
            continue
        on = int(m["onset"][0]); du = int(m["dur"][0])
        lo = max(on - K + 1, 0); hi = min(on + du - 1, t_max)
        if hi >= lo:
            t0[b] = torch.randint(lo, hi + 1, (1,), generator=generator)
    return t0.to(device)


def make_windows(states: torch.Tensor, stimulus: torch.Tensor, K: int,
                 t0: torch.Tensor | None = None,
                 generator: torch.Generator | None = None):
    """Slice (context, target) windows.

    Input  : X[t0:t0+K] and U[t0:t0+K]  (4 features per step: V,S,R,U)
    Target : X[t0+K]
    t0 can be given (fixed windows) or sampled uniformly.
    """
    B, T, N, F = states.shape
    t_max = T - K - 1                                    # need target at t0+K
    if t0 is None:
        t0 = torch.randint(0, t_max + 1, (B,), generator=generator)
    t0 = t0.to(states.device)
    idx = t0[:, None] + torch.arange(K, device=states.device)[None, :]  # [B,K]
    bidx = torch.arange(B, device=states.device)[:, None]
    x_state = states[bidx, idx]                          # [B,K,N,F]
    x_stim = stimulus[bidx, idx].unsqueeze(-1)           # [B,K,N,1]
    x = torch.cat([x_state, x_stim], dim=-1)             # [B,K,N,F+1]
    y = states[bidx, (t0 + K)[:, None]][:, 0]            # [B,N,F]
    return x, y, t0
