"""Unit tests for phase-3 modules (gnn_temporal model / threshold loss /
v3 rollout config / unroll batch). Run directly:  python test_phase3_temporal.py
CPU-only, small scale, plain asserts — no pytest dependency.
"""
from __future__ import annotations

import argparse

import torch

from config import get_config
from connectome import get_connectome
from dataset import generate_batch
from lif import LIFSimulator
from losses import multistep_loss, population_groups
from models import build_model, count_params, match_gnn_wide
from models.gnn_temporal import (GNNTemporalTransformer, gnn_temporal_kwargs,
                                 sinusoidal_table)
from models.mechanistic import MechanisticWrapper, maybe_wrap
from rollout_config import (EXPERIMENTS, EXPERIMENTS_V3, RolloutConfig,
                            StageConfig, build_rollout_config,
                            experiment_index, final_ckpt_path,
                            stages_v3, stage_ckpt_path)
from surrogate import compose_step_mechanistic

DEVICE = torch.device("cpu")
CFG = get_config("small")


def _conn():
    return get_connectome(CFG, DEVICE)


def _args(**kw):
    """Minimal argparse namespace accepted by build_rollout_config."""
    d = dict(model="gnn_temporal", matrix="v3", experiment="E",
             scale="small", surrogate=None, gamma=None, base_lr=None,
             stages=None, smoke=False, buffer_capacity=None, tbptt=None,
             grad_ckpt=None, k_hist=None, t_layers=None, t_heads=None,
             pos=None, causal=False, dropout=None)
    d.update(kw)
    return argparse.Namespace(**d)


# ----------------------------------------------------------------------
def test_model_shapes() -> None:
    cfg, conn = CFG, _conn()
    B, K, N = 2, cfg.K, cfg.n_neurons
    model = GNNTemporalTransformer(cfg, conn)
    x = torch.randn(B, K, N, 4)
    out = model(x)
    assert set(out) == {"v", "s_logits", "r"}
    for k in out:
        assert out[k].shape == (B, N), f"{k} shape {tuple(out[k].shape)}"
    # backward populates grads through both encoders
    loss = out["v"].sum() + out["s_logits"].sum()
    loss.backward()
    assert model.head.weight.grad is not None
    assert model.inp.weight.grad is not None, "spatial input must get grads"
    assert model.rounds[0].msg.weight.grad is not None, "GNN must get grads"
    assert model.t_blocks[0].qkv.weight.grad is not None, \
        "temporal attention must get grads"
    if model.pos_type == "learned":
        assert model.pos.grad is not None, "learned pos emb must get grads"


def test_k_hist_truncation() -> None:
    """k_hist truncation must equal manually slicing the window: a k_hist=k
    model on the full cfg.K window == the same model on the last k steps."""
    cfg, conn = CFG, _conn()
    B, N = 2, cfg.n_neurons
    torch.manual_seed(0)
    m4 = GNNTemporalTransformer(cfg, conn, k_hist=4)
    m4.eval()
    x = torch.randn(B, cfg.K, N, 4)
    with torch.no_grad():
        a = m4(x)
        b = m4(x[:, -4:])
        c = m4(x[:, -8:])          # input longer than k_hist: still last 4
    for k in a:
        assert torch.equal(a[k], b[k]), f"{k}: truncation != manual slice"
        assert torch.equal(a[k], c[k]), f"{k}: longer input must truncate"
    # k_hist=1 (last-state-only / Markov control)
    m1 = GNNTemporalTransformer(cfg, conn, k_hist=1)
    with torch.no_grad():
        o1 = m1(x)
    assert o1["v"].shape == (B, N)
    # invalid k_hist raises
    try:
        GNNTemporalTransformer(cfg, conn, k_hist=cfg.K + 1)
        raise AssertionError("k_hist > cfg.K must raise")
    except ValueError:
        pass


def test_positional_encodings() -> None:
    cfg, conn = CFG, _conn()
    for pos in ("learned", "sincos"):
        m = GNNTemporalTransformer(cfg, conn, pos_type=pos)
        x = torch.randn(1, cfg.K, cfg.n_neurons, 4)
        out = m(x)
        assert out["v"].shape == (1, cfg.n_neurons)
    table = sinusoidal_table(16, 8)
    assert table.shape == (1, 16, 8)
    # causal variant runs and differs from non-causal on shuffled history
    mc = GNNTemporalTransformer(cfg, conn, causal=True)
    x = torch.randn(1, cfg.K, cfg.n_neurons, 4)
    mc.eval()
    with torch.no_grad():
        o = mc(x)
    assert o["v"].shape == (1, cfg.n_neurons)


def test_attention_return() -> None:
    cfg, conn = CFG, _conn()
    B, N = 1, cfg.n_neurons
    m = GNNTemporalTransformer(cfg, conn, k_hist=8)
    m.eval()
    x = torch.randn(B, cfg.K, N, 4)
    with torch.no_grad():
        out, w = m(x, return_attn=True)
    assert w.shape == (B, N, 8), f"attn shape {tuple(w.shape)}"
    assert torch.allclose(w.sum(dim=-1), torch.ones(B, N), atol=1e-4), \
        "attention rows must sum to 1"
    # temporal-order sensitivity: permuting past positions must change the
    # output of a model with learned positions (guard against a degenerate
    # time-invariant implementation)
    g = torch.Generator().manual_seed(0)
    xs = x.clone()
    perm = torch.randperm(cfg.K - 1, generator=g)
    xs[:, :-1] = xs[:, perm]
    xs = xs[:, torch.cat([perm, torch.tensor([cfg.K - 1])])]
    with torch.no_grad():
        out_s = m(xs)
    assert not torch.allclose(out["v"], out_s["v"], atol=1e-5), \
        "model must be sensitive to temporal order"


def test_mechanistic_compat() -> None:
    """MechanisticWrapper must wrap gnn_temporal unchanged (experiment F)."""
    cfg, conn = CFG, _conn()
    base = GNNTemporalTransformer(cfg, conn, k_hist=4)
    rc = RolloutConfig(base=CFG, mechanistic=True)
    w = maybe_wrap(base, cfg, rc)
    assert isinstance(w, MechanisticWrapper)
    x = torch.randn(2, cfg.K, cfg.n_neurons, 4)
    x[:, -1, :, 2] = 0.0
    with torch.no_grad():
        out = w(x)
    for k in ("v", "s_logits", "r", "dv", "v_pre", "s_logits_aux", "r_aux"):
        assert k in out, f"missing key {k}"
        assert out[k].shape == (2, cfg.n_neurons)
    # and through the differentiable path
    w.train()
    dv = w(x)["dv"]
    v, sp, r, v_pre = compose_step_mechanistic(
        dv, x[:, -1, :, 0], x[:, -1, :, 2], cfg)
    (v.sum() + sp.sum()).backward()
    assert w.base.head.weight.grad is not None


def test_threshold_loss() -> None:
    cfg = CFG
    B, N, U = 2, cfg.n_neurons, 3
    g = torch.Generator().manual_seed(3)
    rc_off = RolloutConfig(base=CFG)
    rc_on = RolloutConfig(base=CFG, threshold_loss=True)
    outs = [{"v": torch.randn(B, N, generator=g, requires_grad=True),
             "s_logits": torch.randn(B, N, generator=g, requires_grad=True),
             "r": torch.rand(B, N, generator=g, requires_grad=True)}
            for _ in range(U)]
    sts = [(o["v"].clamp(-3, 3), torch.sigmoid(o["s_logits"]).round(),
            o["r"].clamp(0, 1)) for o in outs]
    targets = torch.cat([
        torch.randn(B, U + 1, N, 1, generator=g),
        (torch.rand(B, U + 1, N, 1, generator=g) > 0.97).float(),
        torch.rand(B, U + 1, N, 1, generator=g)], dim=-1)
    pos_weight = torch.tensor(cfg.spike_pos_weight)
    groups = population_groups(N, 5, DEVICE)

    _, parts_off = multistep_loss(outs, sts, targets, rc_off, pos_weight,
                                  groups)
    assert "thresh" not in parts_off, "off: no thresh key (v2 compat)"
    total_on, parts_on = multistep_loss(outs, sts, targets, rc_on,
                                        pos_weight, groups)
    assert "thresh" in parts_on and parts_on["thresh"] > 0
    assert torch.isfinite(total_on)
    total_on.backward()
    for o in outs:
        assert o["v"].grad is not None and torch.isfinite(o["v"].grad).all()

    # matches its definition: mean(w * err^2), w = 1 + a*exp(-|tv-vth|/sig)
    rc = rc_on
    weights = [rc.gamma ** h for h in range(U)]
    wn = sum(weights)
    manual = 0.0
    for h in range(U):
        tv = targets[:, h + 1, :, 0]
        wt = 1.0 + rc.thresh_alpha * torch.exp(
            -(tv - cfg.v_th).abs() / rc.thresh_sigma)
        manual += weights[h] / wn * (wt * (sts[h][0] - tv) ** 2).mean().item()
    assert abs(parts_on["thresh"] - manual) < 1e-6, "thresh vs definition"


def test_v3_config() -> None:
    # letters + seeding
    assert experiment_index("D", "v3") == 0
    assert experiment_index("F", "v3") == 2
    assert experiment_index("G", "v2") == 6
    # v3 experiments build and never collide with v2 paths
    for exp, expect in (("D", dict(scheduled_sampling=True, dagger=False,
                                   mechanistic=False, threshold_loss=False)),
                        ("E", dict(scheduled_sampling=True, dagger=True,
                                   mechanistic=False, threshold_loss=False)),
                        ("F", dict(scheduled_sampling=True, dagger=True,
                                   mechanistic=True, threshold_loss=True))):
        rc = build_rollout_config(_args(experiment=exp), CFG)
        for k, v in expect.items():
            assert getattr(rc, k) == v, f"{exp}: {k}={getattr(rc, k)} != {v}"
        assert rc.version == "v3" and rc.macro_loss
        assert rc.lambda_rate == 0.1 and rc.lambda_pop == 0.1
        assert rc.model_kwargs["k_hist"] == CFG.K
        # scheduled sampling keeps stage teacher ratios; dagger gate works
        trs = [s.teacher_ratio for s in rc.stages]
        assert trs[0] > 0 and trs[-1] == 0.0, f"{exp}: teacher {trs}"
        dms = [s.dagger_mix for s in rc.stages]
        if rc.dagger:
            assert dms[0] > 0, f"{exp}: dagger mix must be on"
        else:
            assert all(d == 0.0 for d in dms), f"{exp}: dagger mix must be 0"
    rc2 = build_rollout_config(_args(matrix="v2", model="gnn",
                                     experiment="G"), CFG)
    rc3 = build_rollout_config(_args(matrix="v3", model="gnn_temporal",
                                     experiment="E"), CFG)
    st2 = stage_ckpt_path(rc2, StageConfig("sX", 4, 1, 0.5))
    st3 = stage_ckpt_path(rc3, StageConfig("sX", 4, 1, 0.5))
    assert "rollout_v2" in str(st2) and "rollout_v3" in str(st3)
    assert final_ckpt_path(rc2) != final_ckpt_path(rc3)
    # stages follow the spec: U = 4,8,16,32,64 with teacher 0.9..0.0
    us = [s.unroll for s in stages_v3("small")]
    tr = [s.teacher_ratio for s in stages_v3("small")]
    assert us == [4, 8, 16, 32, 64]
    assert tr == [0.90, 0.75, 0.50, 0.25, 0.0]


def test_param_match() -> None:
    cfg, conn = CFG, _conn()
    ref = build_model("gnn_temporal", cfg, conn, None,
                      gnn_temporal_kwargs(cfg, k_hist=8))
    target = count_params(ref)
    kw = match_gnn_wide(cfg, conn, target)
    wide = build_model("gnn_wide", cfg, conn, None, kw)
    n = count_params(wide)
    assert abs(n - target) / target < 0.25, \
        f"param match off by {(n - target) / target:+.1%}"
    del ref, wide


def test_unroll_batch_smoke() -> None:
    """One tiny unrolled training step end-to-end (the rollout_train core)."""
    from rollout_train import unroll_batch
    cfg, conn = CFG, _conn()
    sim = LIFSimulator(conn, cfg, DEVICE)
    seeds = [cfg.traj_seed("train", i) for i in range(4)]
    data = generate_batch(seeds, "train", sim, cfg)
    states, stim = data["states"], data["stimulus"]
    pos_weight = torch.tensor(cfg.spike_pos_weight)
    groups = population_groups(cfg.n_neurons, 5, DEVICE)

    for mech in (False, True):
        rc = build_rollout_config(_args(experiment="F" if mech else "D"),
                                  CFG)
        model = build_model("gnn_temporal", cfg, conn, DEVICE,
                            rc.model_kwargs)
        model = maybe_wrap(model, cfg, rc)
        model.train()
        stage = StageConfig("t", 3, 1, teacher_ratio=0.5)
        gen = torch.Generator().manual_seed(0)
        t0 = torch.zeros(4, dtype=torch.long)
        loss, parts, tf = unroll_batch(model, states, stim, t0, cfg, rc,
                                       stage, gen, DEVICE, pos_weight,
                                       groups)
        assert torch.isfinite(loss) and 0.0 <= tf <= 1.0
        loss.backward()
        g = model.base.head.weight.grad if mech else model.head.weight.grad
        assert g is not None and torch.isfinite(g).all()
        del model


def main() -> None:
    tests = [
        test_model_shapes,
        test_k_hist_truncation,
        test_positional_encodings,
        test_attention_return,
        test_mechanistic_compat,
        test_threshold_loss,
        test_v3_config,
        test_param_match,
        test_unroll_batch_smoke,
    ]
    for fn in tests:
        fn()
        print(f"[PASS] {fn.__name__}")
    print(f"all {len(tests)} tests passed")


if __name__ == "__main__":
    main()
