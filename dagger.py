"""Phase-2 DAgger: priority replay buffer + on-policy data collection.

The model is rolled out autonomously (closed-loop, hard state composition,
no gradients) on fresh TRAIN-split trajectories. At every rollout step the
history window the model just consumed is stored as context, paired with a
one-step teacher label from the true LIF simulator:

    target = F_LIF(x_hat_t, U[t+1])

i.e. one simulator step from the model's OWN composed state at the absolute
time t of the prediction, driven by the next true stimulus U[t+1] (the
stimulus of the transition t -> t+1, which the model never sees — see the
time convention in PHASE2.md). Pairs are prioritised by one-step error so a
fixed-capacity buffer keeps the on-policy states the model handles worst.

Only train-split stimuli are used, so the OOD protocol (test_ood neurons
never directly stimulated in training) survives phase 2: "OOD states" in
the buffer are activity that propagated into the OOD region on train
trajectories, never direct stimulation there.
"""
from __future__ import annotations

import math

import torch

from config import Config
from dataset import generate_batch, traj_count
from lif import LIFSimulator
from rollout import REFR_HOLD

__all__ = ["ReplayBuffer", "collect_on_policy"]


# ----------------------------------------------------------------------
class ReplayBuffer:
    """Fixed-capacity CPU buffer of (context, target, priority) triples.

    contexts   [M, K, N, 4] float32 — history windows (V, S, R_norm, U)
    targets    [M, N, 3]    float32 — one-step teacher labels (V, S, R_norm)
    priorities [M]          float32 — higher = more valuable
    depths     [M]          float32 — optional normalised rollout depth
               (step_index / horizon at collection time), stored only when
               passed to add(); used by stats() for logging.

    Storage is preallocated lazily on the first add() (K, N are not known at
    construction) and never grows beyond `capacity`. add() is O(batch) while
    there is room; on overflow the buffer keeps the highest-priority entries
    via a stable descending sort (O(capacity log capacity), ties keep older
    entries — deterministic given the data).
    """

    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        self.capacity = int(capacity)
        self._contexts: torch.Tensor | None = None    # [capacity, K, N, 4]
        self._targets: torch.Tensor | None = None     # [capacity, N, 3]
        self._priorities: torch.Tensor | None = None  # [capacity]
        self._depths: torch.Tensor | None = None      # [capacity]
        self._size = 0

    # ------------------------------------------------------------------
    def _allocate(self, contexts: torch.Tensor, targets: torch.Tensor) -> None:
        K, N = contexts.shape[1], contexts.shape[2]
        self._contexts = torch.empty(self.capacity, K, N, 4,
                                     dtype=contexts.dtype)
        self._targets = torch.empty(self.capacity, N, 3, dtype=targets.dtype)
        self._priorities = torch.empty(self.capacity, dtype=torch.float32)
        self._depths = torch.full((self.capacity,), float("nan"),
                                  dtype=torch.float32)

    def _check_shapes(self, contexts: torch.Tensor, targets: torch.Tensor,
                      priorities: torch.Tensor) -> None:
        if contexts.dim() != 4 or contexts.shape[-1] != 4:
            raise ValueError(f"contexts must be [B, K, N, 4], got "
                             f"{tuple(contexts.shape)}")
        if targets.dim() != 3 or targets.shape[-1] != 3:
            raise ValueError(f"targets must be [B, N, 3], got "
                             f"{tuple(targets.shape)}")
        B = contexts.shape[0]
        if targets.shape[0] != B or priorities.shape != (B,):
            raise ValueError(f"batch sizes disagree: contexts {B}, targets "
                             f"{targets.shape[0]}, priorities "
                             f"{tuple(priorities.shape)}")
        if self._contexts is not None:
            _, K, N, _ = self._contexts.shape
            if contexts.shape[1] != K or contexts.shape[2] != N:
                raise ValueError(f"shape mismatch: buffer holds K={K}, N={N}, "
                                 f"got K={contexts.shape[1]}, "
                                 f"N={contexts.shape[2]}")

    # ------------------------------------------------------------------
    def add(self, contexts: torch.Tensor, targets: torch.Tensor,
            priorities: torch.Tensor,
            depths: torch.Tensor | None = None) -> None:
        """Add a batch of CPU pairs. On overflow keep highest-priority entries.

        contexts [B, K, N, 4], targets [B, N, 3], priorities [B] (CPU).
        `depths` is an optional [B] logging annotation (normalised rollout
        depth), not part of the pinned contract.
        """
        contexts = contexts.detach().cpu()
        targets = targets.detach().cpu()
        priorities = priorities.detach().cpu().to(torch.float32).reshape(-1)
        if depths is not None:
            depths = depths.detach().cpu().to(torch.float32).reshape(-1)
        self._check_shapes(contexts, targets, priorities)
        if depths is not None and depths.shape != priorities.shape:
            raise ValueError("depths must match priorities shape")
        if self._contexts is None:
            self._allocate(contexts, targets)
        if depths is None:
            depths = torch.full_like(priorities, float("nan"))

        n_new = contexts.shape[0]
        if self._size + n_new <= self.capacity:
            sl = slice(self._size, self._size + n_new)
            self._contexts[sl] = contexts
            self._targets[sl] = targets
            self._priorities[sl] = priorities
            self._depths[sl] = depths
            self._size += n_new
            return

        # overflow: keep the top-`capacity` entries by priority (stable
        # descending sort: ties keep older entries first).
        all_c = torch.cat([self._contexts[: self._size], contexts])
        all_t = torch.cat([self._targets[: self._size], targets])
        all_p = torch.cat([self._priorities[: self._size], priorities])
        all_d = torch.cat([self._depths[: self._size], depths])
        order = torch.argsort(all_p, descending=True, stable=True)
        keep = order[: self.capacity]
        self._contexts = all_c[keep].clone()
        self._targets = all_t[keep].clone()
        self._priorities = all_p[keep].clone()
        self._depths = all_d[keep].clone()
        self._size = self.capacity

    # ------------------------------------------------------------------
    def sample(self, n: int, generator: torch.Generator):
        """Uniform sample of n pairs; returns (contexts, targets) CPU tensors.

        Without replacement when len(buffer) >= n, with replacement
        otherwise. All randomness flows through `generator` (CPU).
        """
        if n <= 0:
            raise ValueError(f"n must be positive, got {n}")
        if self._size == 0:
            raise ValueError("cannot sample from an empty buffer")
        if self._size >= n:
            idx = torch.randperm(self._size, generator=generator)[:n]
        else:
            idx = torch.randint(0, self._size, (n,), generator=generator)
        return self._contexts[idx].clone(), self._targets[idx].clone()

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return self._size

    def stats(self) -> dict:
        """Logging summary: mean_priority always; frac of entries deeper than
        half the max stored depth only when depths were supplied to add()."""
        if self._size == 0:
            return {"size": 0, "capacity": self.capacity, "mean_priority": 0.0}
        out = {
            "size": self._size,
            "capacity": self.capacity,
            "mean_priority": float(
                self._priorities[: self._size].mean().item()),
        }
        d = self._depths[: self._size]
        valid = ~torch.isnan(d)
        if valid.any():
            dv = d[valid]
            out["frac_depth>0.5*max"] = float(
                (dv > 0.5 * dv.max()).to(torch.float32).mean().item())
        return out


# ----------------------------------------------------------------------
@torch.no_grad()
def collect_on_policy(model: torch.nn.Module, sim: LIFSimulator, cfg: Config,
                      device, n_traj: int, horizon: int,
                      spike_threshold: float, mechanistic: bool,
                      seed: int, n_batches: int = 4):
    """Closed-loop rollout on fresh TRAIN-split trajectories with LIF teacher.

    Trajectory indices are drawn with a torch.Generator seeded by `seed`
    (randperm over the train pool, first `n_traj`), then generated fresh via
    dataset.generate_batch in `n_batches` chunks of ~n_traj/n_batches to
    bound memory. Starting from the true context window ending at
    t0 = cfg.K - 1 (times 0..K-1), the model is rolled out autonomously for
    `horizon` steps (predictions cover absolute times K .. K+horizon-1),
    exactly like rollout.rollout / reinjection.rollout_reinject: each
    predicted state at absolute time t_abs is appended to the window paired
    with the stimulus at the SAME absolute time, (x_hat[t_abs], U[t_abs]),
    and each trajectory's silence mask is applied to the composed state.

    Composition is model-agnostic HARD composition (under no_grad):
        v  = out["v"].clamp(v_min, 3*v_th)
        sp = sigmoid(out["s_logits"]) > spike_threshold
        r  = out["r"].clamp(0, 1)
        fired            -> v = v_reset, r = 1
        ~fired & r > REFR_HOLD -> v = v_reset
        silenced         -> v = v_rest, sp = 0, r = 0
    This is identical for learned and mechanistic models: mechanistic
    wrappers already apply their deterministic LIF rule inside forward, so
    the same hard composition applies on top. The `mechanistic` flag is
    accepted for API stability / logging and does NOT change the rollout.

    At every step with t_abs + 1 < T (a next stimulus exists) a training
    pair is stored:
        context = the history window the model just consumed (BEFORE this
                  step's prediction), cloned to CPU;
        target  = F_LIF(composed state at t_abs, U[t_abs+1]) — one
                  sim.simulate step with state0=(v_hat, sp_hat,
                  r_hat * refractory_period) (R is UNNORMALISED for state0)
                  and the trajectory's silence mask; states[:, 0].

    Priority per pair (all neuron-means per sample):
        mean|out["v"] - target v|                 (raw head vs teacher)
        + 5   * spike mismatch fraction           (composed sp vs target s)
        + 0.5 * near-threshold fraction           (|v_pre - v_th| < 0.1,
                v_pre = out["v_pre"] when present (mechanistic) else
                out["v"])
        + 0.5 * (step_index / horizon)

    Returns (contexts [M, K, N, 4], targets [M, N, 3], priorities [M]) on
    CPU, ordered chunk-major then step-major (pair p of a chunk = step
    p // B_chunk, trajectory p % B_chunk). M = n_traj * (# steps with
    t_abs+1 < T) <= n_traj * horizon — the natural cap; steps whose
    prediction lands on the last timestep have no next stimulus and are
    skipped. `horizon` is effectively capped at T - K (no stimulus exists
    beyond T-1 to append). The model's training/eval mode is preserved.

    Determinism: trajectory choice is governed by `seed` alone (t0 fixed);
    generate_batch and the LIF sim are deterministic given seeds/states, so
    two calls with the same arguments return identical tensors (bitwise on
    CPU; accelerator kernels must likewise be run deterministically).
    """
    dev = torch.device(device)
    K, T = cfg.K, cfg.T
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    H = min(horizon, T - K)
    if H < 1:
        raise ValueError(f"no predictable steps: horizon={horizon}, "
                         f"T={T}, K={K}")
    n_batches = max(1, int(n_batches))

    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    pool = traj_count(cfg, "train")
    order = torch.randperm(pool, generator=gen)[:n_traj].tolist()  # clips to pool

    was_training = model.training
    model.eval()
    contexts_all: list[torch.Tensor] = []
    targets_all: list[torch.Tensor] = []
    prios_all: list[torch.Tensor] = []
    try:
        chunk = max(1, math.ceil(len(order) / n_batches))
        for c0 in range(0, len(order), chunk):
            idxs = order[c0: c0 + chunk]
            seeds = [cfg.traj_seed("train", i) for i in idxs]
            data = generate_batch(seeds, "train", sim, cfg)
            states = data["states"].to(dev)          # [B, T, N, 3]
            stim = data["stimulus"].to(dev)          # [B, T, N]
            sil = data["silence"].to(dev)            # [B, N] bool

            # true context window: (X, U)[0 .. K-1], ends at t0 = K - 1
            hist = torch.cat([states[:, :K], stim[:, :K].unsqueeze(-1)],
                             dim=-1)                 # [B, K, N, 4]
            for s in range(H):
                t_abs = K + s                        # absolute time predicted
                out = model(hist)

                # ---- hard composition (rollout.py semantics) ----------
                v = out["v"].clamp(cfg.v_min, cfg.v_th * 3.0)
                sp = (torch.sigmoid(out["s_logits"]) > spike_threshold
                      ).to(v.dtype)
                r = out["r"].clamp(0.0, 1.0)
                fired = sp > 0.5
                v = torch.where(fired, torch.full_like(v, cfg.v_reset), v)
                r = torch.where(fired, torch.ones_like(r), r)
                hold = (~fired) & (r > REFR_HOLD)
                v = torch.where(hold, torch.full_like(v, cfg.v_reset), v)
                v = torch.where(sil, torch.full_like(v, cfg.v_rest), v)
                sp = torch.where(sil, torch.zeros_like(sp), sp)
                r = torch.where(sil, torch.zeros_like(r), r)

                # ---- DAgger pair: context + LIF teacher label ---------
                if t_abs + 1 < T:
                    u_next = stim[:, t_abs + 1].to(sim.device)
                    teacher = sim.simulate(
                        u_next[:, None, :],
                        state0=(v, sp, r * float(cfg.refractory_period)),
                        silence_mask=sil,
                    )[:, 0].to(dev)                  # [B, N, 3] at t_abs+1

                    v_pre = out["v_pre"] if "v_pre" in out else out["v"]
                    v_err = (out["v"] - teacher[..., 0]).abs().mean(dim=1)
                    s_mis = (sp != teacher[..., 1]).to(torch.float32
                                                       ).mean(dim=1)
                    near = ((v_pre - cfg.v_th).abs() < 0.1
                            ).to(torch.float32).mean(dim=1)
                    prio = (v_err + 5.0 * s_mis + 0.5 * near
                            + 0.5 * (s / float(horizon)))

                    contexts_all.append(hist.clone().cpu())
                    targets_all.append(teacher.cpu())
                    prios_all.append(prio.cpu())

                # ---- append (x_hat[t_abs], U[t_abs]) to the window ----
                feat = torch.stack([v, sp, r, stim[:, t_abs]], dim=-1)
                hist = torch.cat([hist[:, 1:], feat.unsqueeze(1)], dim=1)
    finally:
        model.train(was_training)

    if contexts_all:
        contexts = torch.cat(contexts_all)
        targets = torch.cat(targets_all)
        prios = torch.cat(prios_all)
    else:  # horizon reached end-of-trajectory immediately: no pairs
        contexts = torch.empty(0, K, cfg.n_neurons, 4)
        targets = torch.empty(0, cfg.n_neurons, 3)
        prios = torch.empty(0)
    mean_p = float(prios.mean().item()) if prios.numel() else float("nan")
    t_rate = float(targets[..., 1].mean().item()) if targets.numel() \
        else float("nan")
    print(f"[dagger] collected {contexts.shape[0]} pairs "
          f"(traj={len(order)}, horizon={horizon}, seed={seed}, "
          f"mechanistic={mechanistic}) | mean priority {mean_p:.4f} | "
          f"mean teacher spike rate {t_rate:.4f}")
    return contexts, targets, prios
