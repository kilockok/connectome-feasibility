"""Unit tests for the phase-2 mechanism modules (surrogate / mechanistic /
losses). Run directly:  python test_phase2_mechanisms.py
CPU-only, small scale, plain asserts — no pytest dependency.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import get_config
from connectome import get_connectome
from dataset import generate_batch
from lif import LIFSimulator
from losses import focal_bce_with_logits, multistep_loss, population_groups
from models.gnn import GNNBaseline
from models.mechanistic import MechanisticWrapper, maybe_wrap, REFR_LOGIT
from rollout import REFR_HOLD
from rollout_config import RolloutConfig
from surrogate import (SURROGATE_MODES, compose_step_learned,
                       compose_step_mechanistic, hard_spike)

DEVICE = torch.device("cpu")
CFG = get_config("small")


# ----------------------------------------------------------------------
def test_hard_spike() -> None:
    # forward is the strict Heaviside
    margin = torch.tensor([-2.0, -0.1, 0.0, 0.1, 2.0])
    y = hard_spike(margin)
    assert torch.equal(y, torch.tensor([0., 0., 0., 1., 1.])), \
        f"forward must be (margin > 0), got {y}"

    # safe under no_grad: hard value, no graph
    with torch.no_grad():
        m_ng = margin.clone().requires_grad_(True)
        y_ng = hard_spike(m_ng, mode="sigmoid")
        assert torch.equal(y_ng, y), "no_grad forward must equal hard value"
        assert not y_ng.requires_grad, "no_grad call must not build a graph"

    # unknown mode raises
    try:
        hard_spike(margin, mode="bogus")
        raise AssertionError("unknown mode must raise ValueError")
    except ValueError:
        pass

    # per-mode gradients: finite, non-negative, expected scale at margin=0
    g = torch.Generator().manual_seed(0)
    m = torch.cat([torch.randn(64, generator=g), torch.tensor([0.0])])
    for mode in SURROGATE_MODES:
        mm = m.clone().requires_grad_(True)
        out = hard_spike(mm, mode=mode, beta=10.0)
        assert torch.equal(out.detach(), (m > 0).float()), \
            f"{mode}: forward changed by mode"
        out.sum().backward()
        grad = mm.grad
        assert grad is not None and torch.isfinite(grad).all(), \
            f"{mode}: non-finite gradient"
        assert (grad >= 0).all(), f"{mode}: surrogate grad must be >= 0"
        g0 = grad[-1].item()
        if mode == "ste":
            assert abs(g0 - 0.25) < 1e-6, \
                f"ste grad at margin=0 must be sigmoid'(0)=0.25, got {g0}"
            assert (grad <= 0.25 + 1e-6).all(), "ste grad exceeds sigmoid' max"
        else:
            assert abs(g0 - 10.0 / 4.0) < 1e-5, \
                f"{mode} grad at margin=0 must be beta/4=2.5, got {g0}"
        # surrogate grads decay away from the threshold
        assert grad[0].item() < g0, f"{mode}: grad should peak at margin=0"

    # fast_sigmoid keeps a usable gradient where sigmoid has vanished
    m_far = torch.tensor([2.0], requires_grad=True)
    hard_spike(m_far, mode="sigmoid", beta=10.0).backward()
    g_sig = m_far.grad.item()
    m_far2 = torch.tensor([2.0], requires_grad=True)
    hard_spike(m_far2, mode="fast_sigmoid", beta=10.0).backward()
    g_fast = m_far2.grad.item()
    assert g_fast > 100.0 * g_sig, \
        f"fast_sigmoid should decay polynomially (got {g_fast} vs {g_sig})"
    assert abs(g_fast - (10.0 / 4.0) / (1.0 + 10.0 * 2.0) ** 2) < 1e-9


# ----------------------------------------------------------------------
def _rollout_reference(out: dict, threshold: float):
    """Verbatim reimplementation of rollout.py's hard composition."""
    cfg = CFG
    v = out["v"].clamp(cfg.v_min, cfg.v_th * 3.0)
    sp = (torch.sigmoid(out["s_logits"]) > threshold).to(v.dtype)
    r = out["r"].clamp(0.0, 1.0)
    fired = sp > 0.5
    v = torch.where(fired, torch.full_like(v, cfg.v_reset), v)
    r = torch.where(fired, torch.ones_like(r), r)
    hold = (~fired) & (r > REFR_HOLD)
    v = torch.where(hold, torch.full_like(v, cfg.v_reset), v)
    return v, sp, r


def test_compose_step_learned() -> None:
    cfg = CFG
    B, N = 4, cfg.n_neurons
    g = torch.Generator().manual_seed(1)
    out = {
        "v": torch.rand(B, N, generator=g) * 7.0 - 4.0,     # [-4, 3]
        "s_logits": torch.randn(B, N, generator=g) * 4.0,
        "r": torch.rand(B, N, generator=g) * 2.0 - 0.5,     # [-0.5, 1.5]
    }
    for threshold in (0.5, 0.3):
        v_ref, sp_ref, r_ref = _rollout_reference(out, threshold)
        v, sp, r = compose_step_learned(out, cfg, threshold=threshold)
        assert torch.equal(v, v_ref), f"v mismatch vs rollout.py (t={threshold})"
        assert torch.equal(sp, sp_ref), f"sp mismatch vs rollout.py (t={threshold})"
        assert torch.equal(r, r_ref), f"r mismatch vs rollout.py (t={threshold})"
        # exercise the refractory V-hold explicitly: high r, no spike
        assert ((sp_ref < 0.5) & (r_ref > REFR_HOLD)).any(), \
            "test data should contain refractory-hold neurons"
        held = (sp_ref < 0.5) & (r_ref > REFR_HOLD)
        assert (v[held] == cfg.v_reset).all(), "refractory V-hold not applied"

    # gradients: sp keeps the surrogate gradient on s_logits - logit(t)
    o = {k: val.clone() for k, val in out.items()}
    for k in o:
        o[k] = o[k].requires_grad_(True)
    v, sp, r = compose_step_learned(o, CFG, threshold=0.5, mode="ste")
    (sp.sum() + v.sum() + r.sum()).backward()
    m = o["s_logits"].detach()                      # logit(0.5) == 0
    eg = torch.sigmoid(m) * (1.0 - torch.sigmoid(m))
    assert torch.allclose(o["s_logits"].grad, eg, atol=1e-6), \
        "spike gradient must be the surrogate gradient on the margin"
    fired = (m > 0).float()
    # v grad: 0 where fired, held, or clipped by the [v_min, 3*v_th] clamp
    held = (fired < 0.5) & (o["r"].detach().clamp(0, 1) > REFR_HOLD)
    v_in = ((o["v"].detach() >= cfg.v_min)
            & (o["v"].detach() <= cfg.v_th * 3.0)).float()
    expect_v = ((fired < 0.5) & ~held).float() * v_in
    assert torch.equal(o["v"].grad, expect_v), "v gradient mask mismatch"
    expect_r = (fired < 0.5).float()                # fired -> r overwritten by 1
    # where r was clamped (out of [0,1]) the clamp kills the grad as well
    in_range = ((o["r"].detach() >= 0.0) & (o["r"].detach() <= 1.0)).float()
    assert torch.equal(o["r"].grad, expect_r * in_range), "r gradient mismatch"


# ----------------------------------------------------------------------
def test_mechanistic_exactness() -> None:
    """Critical gate: MechanisticWrapper + true dv == one simulator step."""
    cfg = CFG
    conn = get_connectome(cfg, DEVICE)
    sim = LIFSimulator(conn, cfg, DEVICE)
    seeds = [cfg.traj_seed("train", i) for i in range(4)]
    data = generate_batch(seeds, "train", sim, cfg)
    states, stim, silence = data["states"], data["stimulus"], data["silence"]
    B, T, N, _ = states.shape
    t = 37
    assert cfg.K <= t < T - 1, "need a full context window ending at t"

    v_t = states[:, t, :, 0]
    s_t = states[:, t, :, 1]
    r_t = states[:, t, :, 2]                       # normalised
    r_unnorm = r_t * float(cfg.refractory_period)
    u_next = stim[:, t + 1]

    # ground truth: branch one simulator step from the true state
    nxt = sim.simulate(u_next[:, None], state0=(v_t, s_t, r_unnorm),
                       silence_mask=silence)[:, 0]
    assert torch.equal(nxt, states[:, t + 1]), \
        "branch simulation must reproduce the recorded next state"

    # reconstruct the true dv by replicating the simulator's update math
    i_syn = s_t @ sim.W + u_next + sim.i_bias
    v_upd = v_t + cfg.alpha * (-(v_t - cfg.v_rest) + i_syn)
    v_pre_true = v_upd.clamp(min=cfg.v_min)
    dv_true = v_pre_true - v_t

    class DummyBase(nn.Module):
        """Base model whose channel-0 output is v_t + dv_true."""

        def __init__(self, dv: torch.Tensor):
            super().__init__()
            self.dv = dv

        def forward(self, x: torch.Tensor) -> dict:
            vt = x[:, -1, :, 0]
            return {"v": vt + self.dv,
                    "s_logits": torch.zeros_like(vt),
                    "r": x[:, -1, :, 2]}

    x = torch.cat([states[:, t - cfg.K + 1: t + 1],
                   stim[:, t - cfg.K + 1: t + 1].unsqueeze(-1)], dim=-1)
    assert x.shape == (B, cfg.K, N, 4)
    wrapped = MechanisticWrapper(DummyBase(dv_true), cfg, dv_adapter=False)
    with torch.no_grad():
        out = wrapped(x)

    keep = ~silence                                # wrapper knows no silencing
    # strict '>' vs simulator '>=' is ambiguous only within ~1 ulp of v_th
    edge = (v_pre_true - cfg.v_th).abs() < 1e-6
    n_edge = int((edge & keep).sum())
    assert n_edge <= max(1, int(0.01 * keep.sum())), \
        f"too many near-threshold neurons ({n_edge})"
    mask = keep & ~edge

    assert torch.allclose(out["v_pre"][keep], v_pre_true[keep], atol=1e-5), \
        "v_pre mismatch vs reconstructed simulator value"
    assert torch.allclose(out["v"][mask], nxt[..., 0][mask], atol=1e-5), \
        "v mismatch vs simulator one-step branch"
    sp = (out["s_logits"] > 0).float()             # sigmoid(.) > 0.5 <=> > 0
    assert torch.equal(sp[mask], nxt[..., 1][mask]), \
        "spike mismatch vs simulator one-step branch"
    assert torch.allclose(out["r"][mask], nxt[..., 2][mask], atol=1e-5), \
        "r mismatch vs simulator one-step branch"

    refr = (r_t > 1e-6) & keep
    if refr.any():
        assert (out["s_logits"][refr] <= REFR_LOGIT).all(), \
            "refractory neurons must get very negative s_logits"
        assert (sp[refr] == 0).all(), "refractory neurons must not spike"
        assert (out["v"][refr] == cfg.v_reset).all(), \
            "refractory neurons must be held at v_reset"
    # every spike case must appear in the test data for the gate to be real
    fired = nxt[..., 1] > 0.5
    assert (fired & mask).any(), "test data should contain firing neurons"
    assert (refr & ~edge).any(), "test data should contain refractory neurons"


# ----------------------------------------------------------------------
def _make_rc(**kw) -> RolloutConfig:
    rc = RolloutConfig(base=CFG)
    for k, v in kw.items():
        setattr(rc, k, v)
    return rc


def _manual_loss(step_outs, step_states, targets, rc, pos_weight, groups):
    """Independent reimplementation of the multistep_loss contract."""
    U = len(step_outs)
    mech = ("dv" in step_outs[0]) or ("s_logits_aux" in step_outs[0])
    ws = [rc.gamma ** h for h in range(U)]
    wn = sum(ws)
    gsize = groups.sum(dim=1).clamp(min=1.0) if (rc.macro_loss and groups is not None) else None
    comp = {k: torch.zeros(()) for k in ("v", "spike", "r", "rate", "pop", "delta")}
    total = torch.zeros(())
    v_prev = targets[:, 0, :, 0]
    for h in range(U):
        out = step_outs[h]
        v, sp, r = step_states[h]
        tv, ts, tr = (targets[:, h + 1, :, i] for i in range(3))
        w = ws[h] / wn
        lv = F.mse_loss(v, tv)
        if mech:
            ls = (F.binary_cross_entropy_with_logits(
                      out["s_logits_aux"], ts, pos_weight=pos_weight)
                  + 0.5 * F.binary_cross_entropy_with_logits(
                      out["s_logits"], ts, pos_weight=pos_weight))
            lr_ = F.mse_loss(r, tr)
        else:
            ls = F.binary_cross_entropy_with_logits(
                out["s_logits"], ts, pos_weight=pos_weight)
            lr_ = F.mse_loss(out["r"], tr)
        l = rc.lambda_v * lv + rc.lambda_s * ls + rc.lambda_r * lr_
        comp["v"] += w * lv
        comp["spike"] += w * ls
        comp["r"] += w * lr_
        if rc.macro_loss and groups is not None:
            l_rate = F.mse_loss(sp.mean(dim=1), ts.mean(dim=1))
            l_pop = F.mse_loss((sp @ groups.T) / gsize, (ts @ groups.T) / gsize)
            l_delta = F.mse_loss(v - v_prev, tv - targets[:, h, :, 0])
            l = (l + rc.lambda_rate * l_rate + rc.lambda_pop * l_pop
                 + rc.lambda_delta * l_delta)
            comp["rate"] += w * l_rate
            comp["pop"] += w * l_pop
            comp["delta"] += w * l_delta
        total = total + w * l
        v_prev = v
    return total, comp


def _rand_outs(B, N, g, U=3):
    """Random raw model outputs (leaf tensors) for the learned mode."""
    return [{"v": torch.randn(B, N, generator=g, requires_grad=True),
             "s_logits": torch.randn(B, N, generator=g, requires_grad=True),
             "r": torch.rand(B, N, generator=g, requires_grad=True)}
            for _ in range(U)]


def _rand_targets(B, N, g, U=3):
    """[B, U+1, N, 3]: index 0 = last true context state, then U targets."""
    return torch.cat([
        torch.randn(B, U + 1, N, 1, generator=g),
        (torch.rand(B, U + 1, N, 1, generator=g) > 0.97).float(),
        torch.rand(B, U + 1, N, 1, generator=g)], dim=-1)


def _all_leaves(outs):
    return [t for o in outs for t in o.values()
            if t.requires_grad and t.is_leaf]


def test_multistep_loss() -> None:
    cfg = CFG
    B, N = 2, cfg.n_neurons
    g = torch.Generator().manual_seed(2)
    pos_weight = torch.tensor(cfg.spike_pos_weight)
    groups = population_groups(N, 5, DEVICE)
    assert groups.shape == (5, N) and torch.equal(groups.sum(dim=1),
                                                  torch.full((5,), N / 5))

    # --- learned mode, macro off: finite, backward populates grads, zeros
    rc = _make_rc()                                 # gamma=0.98, macro off
    outs = _rand_outs(B, N, g)
    # composed states derived from the outputs, as the trainer builds them
    sts = [compose_step_learned(o, cfg) for o in outs]
    targets = _rand_targets(B, N, g)
    total, parts = multistep_loss(outs, sts, targets, rc, pos_weight, None)
    assert total.ndim == 0 and torch.isfinite(total), "total must be finite"
    total.backward()
    for leaf in _all_leaves(outs):
        assert leaf.grad is not None and torch.isfinite(leaf.grad).all(), \
            "backward must populate finite grads on every input"
    assert set(parts) == {"loss", "v", "spike", "r", "rate", "pop", "delta"}
    assert abs(parts["loss"] - total.item()) < 1e-6
    assert parts["rate"] == 0.0 and parts["pop"] == 0.0 \
        and parts["delta"] == 0.0, "macro terms must be exactly 0 when off"
    # matches the independent manual reference
    man_total, man_comp = _manual_loss(outs, sts, targets, rc, pos_weight, None)
    assert torch.allclose(total, man_total, atol=1e-6), "total vs manual"
    comp_keys = ("v", "spike", "r", "rate", "pop", "delta")
    for k in comp_keys:
        assert abs(parts[k] - man_comp[k].item()) < 1e-6, f"parts[{k}] vs manual"

    # --- gamma = 1.0 -> uniform average of per-step losses
    rc_u = _make_rc(gamma=1.0)
    total_u, _ = multistep_loss(outs, sts, targets, rc_u, pos_weight, None)
    per_step = []
    for h in range(3):
        v, sp, r = sts[h]
        tv, ts, tr = (targets[:, h + 1, :, i] for i in range(3))
        per_step.append(
            rc.lambda_v * F.mse_loss(v, tv)
            + rc.lambda_s * F.binary_cross_entropy_with_logits(
                outs[h]["s_logits"], ts, pos_weight=pos_weight)
            + rc.lambda_r * F.mse_loss(outs[h]["r"], tr))
    manual_u = torch.stack(per_step).mean()
    assert torch.allclose(total_u, manual_u, atol=1e-6), \
        "gamma=1.0 must equal the uniform average of per-step losses"

    # --- gamma < 1 downweights later steps
    rc_h = _make_rc(gamma=0.5)
    total_h, _ = multistep_loss(outs, sts, targets, rc_h, pos_weight, None)
    ws = [0.5 ** h for h in range(3)]
    manual_h = sum(w * l for w, l in zip(ws, per_step)) / sum(ws)
    assert torch.allclose(total_h, manual_h, atol=1e-6), \
        "horizon weighting mismatch"

    # --- macro terms on: nonzero and match manual
    rc_m = _make_rc(macro_loss=True)
    total_m, parts_m = multistep_loss(outs, sts, targets, rc_m, pos_weight,
                                      groups)
    assert torch.isfinite(total_m)
    assert parts_m["rate"] > 0 and parts_m["pop"] > 0 \
        and parts_m["delta"] > 0, "macro terms must be active"
    man_total_m, man_comp_m = _manual_loss(outs, sts, targets, rc_m,
                                           pos_weight, groups)
    assert torch.allclose(total_m, man_total_m, atol=1e-6), "macro vs manual"
    for k in comp_keys:
        assert abs(parts_m[k] - man_comp_m[k].item()) < 1e-6, \
            f"macro parts[{k}] vs manual"

    # --- focal path runs, is finite, backpropagates (fresh graph: the
    #     first backward already freed the composition graph over outs)
    rc_f = _make_rc(focal=True)
    sts_f = [compose_step_learned(o, cfg) for o in outs]
    total_f, _ = multistep_loss(outs, sts_f, targets, rc_f, pos_weight, None)
    assert torch.isfinite(total_f), "focal total must be finite"
    total_f.backward()

    # --- focal matches its definition on a raw tensor pair
    z = torch.randn(50, generator=g)
    yb = (torch.rand(50, generator=g) > 0.5).float()
    fl = focal_bce_with_logits(z, yb, pos_weight, gamma=2.0)
    p = torch.sigmoid(z)
    pt = p * yb + (1 - p) * (1 - yb)
    alpha = pos_weight / (1 + pos_weight)
    at = alpha * yb + (1 - alpha) * (1 - yb)
    fl_ref = (at * (1 - pt) ** 2
              * F.binary_cross_entropy_with_logits(z, yb, reduction="none")
              ).mean()
    assert torch.allclose(fl, fl_ref, atol=1e-6), "focal vs definition"

    # --- mechanistic mode: grads reach dv through the composed chain,
    #     spike term = BCE(s_logits_aux) + 0.5 * BCE(s_logits)
    v_t = torch.rand(B, N, generator=g) * 0.5
    r_t = torch.zeros(B, N)
    outs_m, sts_m = [], []
    for _ in range(3):
        dv = (0.5 * torch.randn(B, N, generator=g)).requires_grad_(True)
        s_aux = torch.randn(B, N, generator=g, requires_grad=True)
        v, sp, r, v_pre = compose_step_mechanistic(dv, v_t, r_t, cfg)
        s_logits = (v_pre - cfg.v_th) * 8.0
        outs_m.append({"v": v, "s_logits": s_logits, "r": r, "dv": dv,
                       "v_pre": v_pre, "s_logits_aux": s_aux,
                       "r_aux": torch.rand(B, N, generator=g)})
        sts_m.append((v, sp, r))
        v_t, r_t = v, r                              # closed-loop chain
    targets_m = _rand_targets(B, N, g)
    rc_mm = _make_rc(macro_loss=True)
    total_mm, parts_mm = multistep_loss(outs_m, sts_m, targets_m, rc_mm,
                                        pos_weight, groups)
    assert torch.isfinite(total_mm)
    total_mm.backward()
    for o in outs_m:
        assert o["dv"].grad is not None \
            and torch.isfinite(o["dv"].grad).all(), "dv must receive grads"
        assert o["s_logits_aux"].grad is not None \
            and torch.isfinite(o["s_logits_aux"].grad).all()
    assert any(o["dv"].grad.abs().sum() > 0 for o in outs_m), \
        "at least one step's dv must receive nonzero gradient"
    man_total_mm, man_comp_mm = _manual_loss(outs_m, sts_m, targets_m, rc_mm,
                                             pos_weight, groups)
    assert torch.allclose(total_mm, man_total_mm, atol=1e-6), \
        "mechanistic total vs manual"
    for k in comp_keys:
        assert abs(parts_mm[k] - man_comp_mm[k].item()) < 1e-6, \
            f"mechanistic parts[{k}] vs manual"

    # --- length validation
    try:
        multistep_loss(outs[:2], sts, targets, rc, pos_weight, None)
        raise AssertionError("mismatched lengths must raise ValueError")
    except ValueError:
        pass


# ----------------------------------------------------------------------
def test_maybe_wrap() -> None:
    cfg = CFG
    conn = get_connectome(cfg, DEVICE)
    base = GNNBaseline(cfg, conn)

    rc_off = _make_rc(mechanistic=False)
    m = maybe_wrap(base, cfg, rc_off)
    assert m is base, "mechanistic=False must return the model itself"
    assert getattr(m, "is_mechanistic") is False

    rc_on = _make_rc(mechanistic=True)
    w = maybe_wrap(base, cfg, rc_on)
    assert isinstance(w, MechanisticWrapper), "mechanistic=True must wrap"
    assert getattr(w, "is_mechanistic") is True
    sd = w.state_dict()
    assert any(k.startswith("base.") for k in sd), \
        "base weights must be prefixed 'base.'"
    assert "dv_a" in sd and "dv_b" in sd, "adapter params missing"
    assert torch.allclose(sd["dv_a"], torch.ones(cfg.n_neurons)) \
        and torch.allclose(sd["dv_b"], torch.zeros(cfg.n_neurons)), \
        "adapter must start at a=1, b=0 (warm start)"

    B, N = 2, cfg.n_neurons
    x = torch.randn(B, cfg.K, N, 4)
    x[:, -1, :, 2] = 0.0
    x[0, -1, :10, 2] = 1.0                          # refractory neurons
    with torch.no_grad():
        out = w(x)
    for k in ("v", "s_logits", "r", "dv", "v_pre", "s_logits_aux", "r_aux"):
        assert k in out, f"missing output key {k}"
        assert out[k].shape == (B, N), f"{k} shape {tuple(out[k].shape)}"
    assert (out["s_logits"][0, :10] <= REFR_LOGIT).all(), \
        "refractory s_logits must be forced very negative"
    nr = x[:, -1, :, 2] <= 1e-6
    agree = (out["s_logits"] > 0) == (out["v_pre"] > cfg.v_th)
    assert agree[nr].all(), "sigmoid(s_logits)>0.5 must equal v_pre>v_th"

    # warm start: dv == base v-channel minus v_t at init
    with torch.no_grad():
        base_out = base(x)
    assert torch.allclose(out["dv"], base_out["v"] - x[:, -1, :, 0],
                          atol=1e-6), "warm-start dv mismatch"

    # gradients flow to the adapter and the backbone
    w.train()
    out2 = w(x)
    loss = out2["v"].sum() + out2["s_logits"].sum() + out2["r"].sum()
    loss.backward()
    assert w.dv_a.grad is not None and torch.isfinite(w.dv_a.grad).all()
    assert w.dv_a.grad.abs().sum() > 0, "adapter a must receive gradient"
    assert w.base.head.weight.grad is not None \
        and torch.isfinite(w.base.head.weight.grad).all(), \
        "backbone must receive gradients through the wrapper"

    # usable under torch.utils.checkpoint (no side effects)
    from torch.utils.checkpoint import checkpoint
    w.zero_grad(set_to_none=True)
    xg = x.clone().requires_grad_(True)
    out3 = checkpoint(w, xg, use_reentrant=False)
    (out3["v"].sum() + out3["s_logits"].sum()).backward()
    assert xg.grad is not None and torch.isfinite(xg.grad).all()


# ----------------------------------------------------------------------
def main() -> None:
    tests = [
        test_hard_spike,
        test_compose_step_learned,
        test_mechanistic_exactness,
        test_multistep_loss,
        test_maybe_wrap,
    ]
    for fn in tests:
        fn()
        print(f"[PASS] {fn.__name__}")
    print(f"all {len(tests)} tests passed")


if __name__ == "__main__":
    main()
