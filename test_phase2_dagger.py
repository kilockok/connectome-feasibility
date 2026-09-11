"""Phase-2 unit tests: DAgger replay buffer + on-policy collection.

Run:  python test_phase2_dagger.py

Small scale, CPU (bitwise-deterministic, ~1 min). Covers:
  1. ReplayBuffer overflow keeps the highest-priority entries; sampling
     shapes/dtypes; sampling with len < n; generator reproducibility.
  2. collect_on_policy end-to-end from the phase-1 GNN checkpoint: shapes,
     finiteness, expected pair count, context U-channel aligned to the true
     stimulus at the right absolute times, full-seed reproducibility.
  3. Teacher labels are valid LIF states (binary spikes, refractory in
     [0, 1] with fired neurons at full refractory, bounded V). Note:
     R_norm is NOT restricted to the 1/refractory_period grid, because the
     teacher starts one step from the model's CONTINUOUS composed r
     (contract: state0 R = r_hat * refractory_period); grid-aligned R only
     arises from true-trajectory starts.
"""
from __future__ import annotations

import math

import torch

from config import get_config, CHECKPOINT_DIR
from connectome import get_connectome
from dagger import ReplayBuffer, collect_on_policy
from dataset import generate_batch, traj_count
from lif import LIFSimulator
from models import build_model

CKPT = CHECKPOINT_DIR / "ckpt_gnn_small_seed1234.pt"


# ----------------------------------------------------------------------
def test_replay_buffer() -> None:
    print("[test] ReplayBuffer")
    K, N, cap = 2, 3, 6
    priorities = torch.tensor([0.1, 0.9, 0.5, 0.3, 0.8, 0.2,
                               0.7, 0.4, 0.6, 0.0])
    # entry i is tagged by filling every element with float(i)
    ctx = torch.stack([torch.full((K, N, 4), float(i)) for i in range(10)])
    tgt = torch.stack([torch.full((N, 3), float(i)) for i in range(10)])
    survivors = torch.tensor([1.0, 2.0, 4.0, 6.0, 7.0, 8.0])  # top-6 priority

    # (a) two adds, second one triggers the eviction path
    buf = ReplayBuffer(capacity=cap)
    buf.add(ctx[:4], tgt[:4], priorities[:4])
    assert len(buf) == 4
    buf.add(ctx[4:], tgt[4:], priorities[4:])
    assert len(buf) == cap
    c, t = buf.sample(cap, torch.Generator().manual_seed(0))  # all entries
    got = c[:, 0, 0, 0].sort().values
    assert torch.equal(got, survivors), f"wrong survivors {got}"
    assert torch.equal(t[:, 0, 0].sort().values, survivors)
    st = buf.stats()
    assert abs(st["mean_priority"] - float(priorities[survivors.long()]
                                             .mean())) < 1e-6
    assert st["size"] == cap and st["capacity"] == cap

    # (b) single add larger than capacity -> same survivors
    buf2 = ReplayBuffer(capacity=cap)
    buf2.add(ctx, tgt, priorities)
    c2, _ = buf2.sample(cap, torch.Generator().manual_seed(1))
    assert torch.equal(c2[:, 0, 0, 0].sort().values, survivors)

    # (c) sample shapes / dtypes / value membership
    c, t = buf.sample(4, torch.Generator().manual_seed(0))
    assert c.shape == (4, K, N, 4) and t.shape == (4, N, 3)
    assert c.dtype == torch.float32 and t.dtype == torch.float32
    vals = c[:, 0, 0, 0]
    assert all(float(v) in survivors.tolist() for v in vals)

    # (d) len < n -> sample with replacement, no crash
    c, t = buf.sample(10, torch.Generator().manual_seed(0))
    assert c.shape == (10, K, N, 4) and t.shape == (10, N, 3)
    vals = c[:, 0, 0, 0]
    assert all(float(v) in survivors.tolist() for v in vals)

    # (e) sampling is governed by the generator (reproducible)
    ca, _ = buf.sample(5, torch.Generator().manual_seed(7))
    cb, _ = buf.sample(5, torch.Generator().manual_seed(7))
    assert torch.equal(ca, cb)

    # (f) guards
    try:
        ReplayBuffer(0)
        raise AssertionError("capacity=0 accepted")
    except ValueError:
        pass
    try:
        ReplayBuffer(4).sample(1, torch.Generator())
        raise AssertionError("empty-buffer sample accepted")
    except ValueError:
        pass
    try:
        buf.add(ctx[:2, :, :2], tgt[:2, :, :2], priorities[:2])  # wrong N
        raise AssertionError("shape-mismatched add accepted")
    except ValueError:
        pass
    print("[test] ReplayBuffer OK")


# ----------------------------------------------------------------------
def test_collect_on_policy():
    print("[test] collect_on_policy (small scale, CPU)")
    cfg = get_config("small")
    device = torch.device("cpu")          # CPU: bitwise reproducibility
    conn = get_connectome(cfg, device)
    sim = LIFSimulator(conn, cfg, device)
    model = build_model("gnn", cfg, conn, device)
    blob = torch.load(CKPT, map_location="cpu", weights_only=False)
    model.load_state_dict(blob["state_dict"])
    model.eval()

    n_traj, horizon, n_batches, seed, thr = 4, 12, 2, 0, 0.5
    args = (model, sim, cfg, device, n_traj, horizon, thr, False, seed)
    contexts, targets, prios = collect_on_policy(*args, n_batches=n_batches)

    # full-seed reproducibility
    c2, t2, p2 = collect_on_policy(*args, n_batches=n_batches)
    assert torch.equal(contexts, c2) and torch.equal(targets, t2)
    assert torch.equal(prios, p2), "same seed must give identical tensors"

    # shapes / finiteness; expected pair count from the skip rule
    expected = n_traj * sum(
        1 for s in range(min(horizon, cfg.T - cfg.K))
        if cfg.K + s + 1 < cfg.T)
    print(f"[test] pairs: got {contexts.shape[0]}, expected {expected}")
    assert contexts.shape == (expected, cfg.K, cfg.n_neurons, 4)
    assert targets.shape == (expected, cfg.n_neurons, 3)
    assert prios.shape == (expected,)
    for name, ten in (("contexts", contexts), ("targets", targets),
                      ("priorities", prios)):
        assert torch.isfinite(ten).all(), f"{name} not finite"
    assert (prios >= 0).all(), "priority terms are all non-negative"

    # context U-channel alignment: regenerate chunk 0 with the same protocol
    gen = torch.Generator().manual_seed(seed)
    order = torch.randperm(traj_count(cfg, "train"),
                           generator=gen)[:n_traj].tolist()
    chunk = math.ceil(n_traj / n_batches)
    seeds = [cfg.traj_seed("train", i) for i in order[:chunk]]
    data = generate_batch(seeds, "train", sim, cfg)
    stim0, states0 = data["stimulus"], data["states"]   # [B_chunk, T, N]
    B_chunk = stim0.shape[0]

    # pair 0 = chunk 0, traj 0, step 0: window covers absolute times 0..K-1
    # (t0 = K-1); U channel must equal the true stimulus at those times, and
    # the state channels are the TRUE states (step 0 context is all-true).
    t0 = cfg.K - 1
    assert torch.equal(contexts[0, :, :, 3], stim0[0, : t0 + 1])
    assert torch.equal(contexts[0, :, :, 3][-1], stim0[0, t0])
    assert torch.equal(contexts[0, :, :, :3], states0[0, : t0 + 1])
    # pair B_chunk = step 1, traj 0: the appended predicted state at t_abs=K
    # is paired with the true stimulus at the same absolute time U[K].
    assert torch.equal(contexts[B_chunk, -1, :, 3], stim0[0, t0 + 1])
    print("[test] collect_on_policy OK")
    return cfg, contexts, targets, prios


# ----------------------------------------------------------------------
def test_teacher_labels(cfg, targets: torch.Tensor) -> None:
    print("[test] teacher-label validity")
    V, S, R = targets[..., 0], targets[..., 1], targets[..., 2]
    binary = (S == 0) | (S == 1)
    frac_binary = binary.to(torch.float32).mean().item()
    print(f"[test] teacher spike rate {S.mean().item():.4f}, "
          f"binary fraction {frac_binary:.4f}")
    assert frac_binary >= 0.9, "teacher spikes must be exactly 0/1"

    rp = R * cfg.refractory_period
    assert R.min() >= 0.0 and R.max() <= 1.0
    # Hard invariants of one LIF step from any (possibly continuous) state:
    # a fired neuron resets V and enters the full refractory period.
    fired = S > 0.5
    if fired.any():
        assert (V[fired] - cfg.v_reset).abs().max() < 1e-6
        assert (R[fired] - 1.0).abs().max() < 1e-6
    # R_norm is not expected on the 1/period grid: the teacher starts from
    # the model's continuous composed r (state0 R = r_hat * period), so
    # non-fired neurons keep R = clamp(r_hat * period - 1, 0) / period.
    on_grid = ((rp - rp.round()).abs() < 1e-4).to(torch.float32).mean()
    print(f"[test] teacher R_norm on-grid fraction "
          f"{on_grid.item():.3f} (continuous r_hat => off-grid allowed)")

    # after one LIF step: fired -> v_reset; else v_min <= V < v_th
    assert V.min() >= cfg.v_min - 1e-4
    assert V.max() <= cfg.v_th + 1e-4
    print(f"[test] teacher V range [{V.min():.3f}, {V.max():.3f}], "
          f"R range [{R.min():.3f}, {R.max():.3f}]")
    print("[test] teacher-label validity OK")


# ----------------------------------------------------------------------
def main() -> None:
    torch.manual_seed(0)
    test_replay_buffer()
    cfg, contexts, targets, prios = test_collect_on_policy()
    test_teacher_labels(cfg, targets)
    print("[test_phase2_dagger] ALL TESTS PASSED")


if __name__ == "__main__":
    main()
