"""Phase-2/3 rollout trainer (see PHASE2.md interface, section rollout_train.py).

    python rollout_train.py --model gnn --experiment G --scale full
    python rollout_train.py --model gnn_temporal --matrix v3 --experiment E \
        --scale small [--smoke]

Flow (pinned by PHASE2.md):
  1. build model (+ MechanisticWrapper when rc.mechanistic); load the
     phase-1(-style) one-step checkpoint of the same model+scale — hard
     error if missing; never train from scratch.
  2. Reference scores of the initial model: calibrated val one-step F1
     (tune_threshold) and the val rollout score.
  3. DAgger warm-up collection with the initial model.
  4. Progressive stages (each initialised from the previous stage's best,
     fresh AdamW per stage): unrolled batches with per-sample scheduled
     sampling + (v2) input noise, mixed with on-policy buffer batches at
     stage.dagger_mix; multistep_loss over the unroll; per-epoch quick val
     (one-step F1@0.5 + val rollout score) with early stopping and stage-best
     checkpoints; DAgger buffer refresh; stage-end one-step protection.
  5. Copy the last stage's best to the experiment's final checkpoint; write
     the history CSV + summary JSON.

Deterministic seeding: one torch.Generator seeded from
cfg.seed + 7919 * experiment_index(experiment, matrix), re-seeded per epoch
(+epoch). All sampling (trajectory order, t0, Bernoulli feed choice, noise,
buffer draws) flows through it.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import time
from dataclasses import asdict
from pathlib import Path

import torch

from config import get_config, add_common_args
from connectome import get_connectome
from dagger import ReplayBuffer, collect_on_policy
from dataset import (generate_batch, load_or_generate, make_windows,
                     traj_count)
from device import get_device
from evaluate import tune_threshold, onestep_eval
from lif import LIFSimulator
from losses import (multistep_loss, population_groups, tangent_loss,
                    tangent_loss_full, tangent_sigma_sample)
from metrics import compute_loss, compute_metrics
from models import build_model
from models.mechanistic import maybe_wrap
from rollout import rollout, rollout_metrics
from rollout_config import (EXPERIMENTS, EXPERIMENTS_V3, RolloutConfig,
                            StageConfig, add_rollout_args,
                            build_rollout_config, experiment_index,
                            final_ckpt_path, history_path, phase1_ckpt_path,
                            stage_ckpt_path, summary_path)
from surrogate import compose_step_learned, compose_step_mechanistic

VAL_SEED = 4242          # fixed windows for the per-epoch quick val
COLLECT_SEED_OFF = 555   # DAgger warm-up offset


# ----------------------------------------------------------------------
# quick validation helpers
@torch.no_grad()
def quick_onestep_f1(model, val_data, cfg, device, n_windows: int,
                     seed: int = VAL_SEED, threshold: float = 0.5) -> float:
    """Spike F1 at a fixed threshold on fixed val windows."""
    model.eval()
    g = torch.Generator(device="cpu").manual_seed(seed)
    states, stim = val_data["states"], val_data["stimulus"]
    per = max(1, n_windows // states.shape[0])
    tp = fp = fn = 0.0
    for i0 in range(0, states.shape[0], 32):
        sl = slice(i0, min(i0 + 32, states.shape[0]))
        for _ in range(per):
            x, y, _ = make_windows(states[sl], stim[sl], cfg.K, generator=g)
            out = model(x.to(device))
            s_pred = (torch.sigmoid(out["s_logits"]) > threshold).float()
            s_true = y.to(device)[..., 1]
            tp += float((s_pred * s_true).sum())
            fp += float((s_pred * (1 - s_true)).sum())
            fn += float(((1 - s_pred) * s_true).sum())
    prec = tp / max(tp + fp, 1.0)
    rec = tp / max(tp + fn, 1.0)
    return 2 * prec * rec / max(prec + rec, 1e-9)


@torch.no_grad()
def quick_rollout_score(model, val_data, cfg, device, n_traj: int, H: int,
                        threshold: float = 0.5) -> tuple[float, dict]:
    """Multi-horizon closed-loop val score (rollout_v3 spec §14):

        score = 0.2*F1@10 + 0.3*F1@20 + 0.5*F1@50 + 0.2*PopCorr@50
                - 0.2*norm_V_RMSE@50 - 0.2*silent_collapse_penalty

    where norm_V_RMSE = v_rmse / v_th (bounded at 3) and the silent-collapse
    penalty is 1 when any trajectory's rate_ratio = (pred_rate+eps)/
    (true_rate+eps) stays < 0.2 for >= 5 consecutive steps. Horizons are
    clipped to the available rollout length. Returns (score, flat dict)."""
    model.eval()
    H = min(H, cfg.T - cfg.K)
    n = min(n_traj, val_data["states"].shape[0])
    states, stim = val_data["states"][:n], val_data["stimulus"][:n]
    sil = val_data.get("silence")
    sil = sil[:n] if sil is not None else None
    context = torch.cat([states[:, :cfg.K],
                         stim[:, :cfg.K].unsqueeze(-1)], dim=-1).to(device)
    fut = stim[:, cfg.K:cfg.K + H].to(device)
    pred = rollout(model, context, fut, cfg, spike_threshold=threshold,
                   silence_mask=sil.to(device) if sil is not None else None)
    true = states[:, cfg.K:cfg.K + H].to(device)
    hs = sorted({h for h in (10, 20, 50) if h <= H} or {H})

    def m_at(h):
        return rollout_metrics(pred, true, [min(h, H)])[min(h, H)]

    # per-step population firing-rate ratio [B, H]
    ps = pred[..., 1].mean(dim=2)
    ts = true[..., 1].mean(dim=2)
    ratio = (ps + 1e-9) / (ts + 1e-9)
    low = ratio < 0.2                                    # [B, H]
    # sustained collapse: >=5 consecutive low steps
    sustained = torch.zeros_like(low)
    run = torch.zeros(low.shape[0], device=low.device)
    for t in range(low.shape[1]):
        run = torch.where(low[:, t], run + 1.0,
                          torch.zeros_like(run))
        sustained[:, t] = run >= 5.0
    collapse_step = None
    if bool(sustained.any()):
        steps = sustained.float().argmax(dim=1).float()  # first per traj
        aff = sustained.any(dim=1)
        collapse_step = float(steps[aff].mean().item())
    silent_pen = 1.0 if collapse_step is not None else 0.0

    f1 = {h: m_at(h)["spike_f1"] for h in (10, 20, 50) if h <= H}
    h50 = 50 if 50 in f1 else hs[-1]
    m50 = m_at(h50)
    norm_v = min(m50["v_rmse"] / max(float(cfg.v_th), 1e-9), 3.0)
    score = (0.2 * f1.get(10, 0.0) + 0.3 * f1.get(20, 0.0)
             + 0.5 * f1.get(50, f1[hs[-1]]) + 0.2 * m50["pop_similarity"]
             - 0.2 * norm_v - 0.2 * silent_pen)
    flat = {f"roll_f1@{h}": v for h, v in f1.items()}
    flat.update({f"roll_pop@{h50}": m50["pop_similarity"],
                 f"roll_rate_err@{h50}": m50["firing_rate_err"],
                 f"roll_v_rmse@{h50}": m50["v_rmse"],
                 "roll_norm_v": norm_v,
                 "silent_pen": silent_pen,
                 "collapse_step": collapse_step})
    return score, flat


# ----------------------------------------------------------------------
# rollout_v4 validation score (spec §10/§17)
MIN_ACTIVITY_V4 = 1e-4       # spec §10 activity gate


@torch.no_grad()
def quick_rollout_score_v4(model, val_data, cfg, device, n_traj: int,
                           H: int, threshold: float = 0.5
                           ) -> tuple[float, dict]:
    """Rollout-v4 closed-loop val score (spec §17):

        score = 0.15*F1@10 + 0.25*F1@20 + 0.30*F1@50
              + 0.15*rate_score@20 + 0.15*rate_score@50
              - 0.10*norm_V_RMSE@20 - 0.10*silent_penalty

    rate_score@h = mean over ACTIVE trajectories (true rate > 1e-4 at step h)
    of exp(-|log(rate_ratio + eps)|) — 1 when the predicted population rate
    matches the teacher, decaying geometrically; neutral 1.0 when no
    trajectory is active at that step (nothing to track). norm_V_RMSE =
    v_rmse@20 / v_th bounded at 3. silent_penalty = 1 when any trajectory's
    rate_ratio stays < 0.25 for >= 3 consecutive steps while active
    (spec §10)."""
    model.eval()
    H = min(H, cfg.T - cfg.K)
    n = min(n_traj, val_data["states"].shape[0])
    states, stim = val_data["states"][:n], val_data["stimulus"][:n]
    sil = val_data.get("silence")
    sil = sil[:n] if sil is not None else None
    context = torch.cat([states[:, :cfg.K],
                         stim[:, :cfg.K].unsqueeze(-1)], dim=-1).to(device)
    fut = stim[:, cfg.K:cfg.K + H].to(device)
    pred = rollout(model, context, fut, cfg, spike_threshold=threshold,
                   silence_mask=sil.to(device) if sil is not None else None)
    true = states[:, cfg.K:cfg.K + H].to(device)

    def m_at(h):
        return rollout_metrics(pred, true, [min(h, H)])[min(h, H)]

    # per-step per-trajectory firing-rate ratio and active gate [B, H]
    ps = pred[..., 1].mean(dim=2)
    ts = true[..., 1].mean(dim=2)
    ratio = (ps + 1e-9) / (ts + 1e-9)
    active = ts > MIN_ACTIVITY_V4

    def rate_score_at(h):
        idx = min(h, H) - 1
        a = active[:, idx]
        if not bool(a.any()):
            return 1.0                       # quiescent true state: neutral
        r = ratio[:, idx][a]
        return float(torch.exp(-(r + 1e-9).log().abs()).mean().item())

    # silent collapse (spec §10): ratio < 0.25 sustained >= 3 steps, gated
    low = (ratio < 0.25) & active
    sustained = torch.zeros_like(low)
    run = torch.zeros(low.shape[0], device=low.device)
    for t in range(low.shape[1]):
        run = torch.where(low[:, t], run + 1.0, torch.zeros_like(run))
        sustained[:, t] = run >= 3.0
    collapse_step = None
    if bool(sustained.any()):
        steps = sustained.float().argmax(dim=1).float()
        aff = sustained.any(dim=1)
        collapse_step = float(steps[aff].mean().item()) + 1.0
    silent_pen = 1.0 if collapse_step is not None else 0.0

    f1 = {h: m_at(h)["spike_f1"] for h in (10, 20, 50) if h <= H}
    h20 = 20 if 20 <= H else max(f1)
    m20 = m_at(h20)
    norm_v = min(m20["v_rmse"] / max(float(cfg.v_th), 1e-9), 3.0)
    rs20 = rate_score_at(20)
    rs50 = rate_score_at(50) if 50 <= H else rs20
    score = (0.15 * f1.get(10, 0.0) + 0.25 * f1.get(20, 0.0)
             + 0.30 * f1.get(50, f1[max(f1)]) + 0.15 * rs20 + 0.15 * rs50
             - 0.10 * norm_v - 0.10 * silent_pen)
    flat = {f"roll_f1@{h}": v for h, v in f1.items()}
    flat.update({f"roll_rate_score@{h20}": rs20,
                 f"roll_v_rmse@{h20}": m20["v_rmse"],
                 "roll_norm_v": norm_v,
                 "silent_pen": silent_pen,
                 "collapse_step": collapse_step})
    return score, flat


def rollout_score_fn(rc):
    """Per-experiment validation scorer (v3 §14 vs v4 §17)."""
    return quick_rollout_score_v4 if getattr(rc, "score_v4", False) \
        else quick_rollout_score


# ----------------------------------------------------------------------
# experiment-level best checkpoints (rollout_v3 spec §15)
def best_ckpt_path(rc: RolloutConfig, kind: str) -> Path:
    """best_one_step / best_rollout / best_combined live beside the stage
    checkpoints so earlier stages are never destroyed by later ones."""
    from rollout_config import _dirs
    _, stage_dir = _dirs(rc)
    d = stage_dir / rc.experiment
    d.mkdir(parents=True, exist_ok=True)
    return d / (f"ckpt_{rc.model}_{rc.scale}_rollout_{rc.version}_"
                f"{rc.experiment}_best_{kind}_seed{rc.base.seed}.pt")


def _save_blob(rc: RolloutConfig, cfg, model, path: Path, epoch: int,
               stage_name: str, extra: dict) -> None:
    torch.save({"model": rc.model, "scale": rc.scale, "seed": cfg.seed,
                "epoch": epoch, "stage": stage_name,
                "experiment": rc.experiment, "version": rc.version,
                "mechanistic": rc.mechanistic,
                "model_kwargs": rc.model_kwargs, "K": cfg.K, **extra,
                "state_dict": model.state_dict()}, path)


def _pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    if not xs:
        return 0.0
    i = min(len(xs) - 1, max(0, int(round(q * (len(xs) - 1)))))
    return xs[i]


# ----------------------------------------------------------------------
# unrolled training batch
def apply_teacher_noise(state: torch.Tensor, cfg, rc: RolloutConfig,
                        stage: StageConfig, gen: torch.Generator) -> torch.Tensor:
    """Noise on teacher-forced states (v2 noise axis; v3 keeps it off):
    V += sigma*randn; per-neuron spike flip with rc.spike_flip_p (flip-on
    also sets v=v_reset, r=1); refractory jitter +-1/period with
    rc.refrac_jitter_p. state [B, N, 3]; returns a new tensor."""
    out = state.clone()
    if stage.noise_sigma > 0:
        noise = torch.randn(out[..., 0].shape, generator=gen).to(out.device)
        out[..., 0] = out[..., 0] + stage.noise_sigma * noise
    if rc.noise and rc.spike_flip_p > 0:
        flip = (torch.rand(out[..., 1].shape, generator=gen)
                < rc.spike_flip_p).to(out.device)
        flip_on = flip & (out[..., 1] < 0.5)
        flip_off = flip & (out[..., 1] > 0.5)
        out[..., 1] = torch.where(flip_on, torch.ones_like(out[..., 1]),
                                  torch.where(flip_off, torch.zeros_like(out[..., 1]),
                                              out[..., 1]))
        out[..., 0] = torch.where(flip_on,
                                  torch.full_like(out[..., 0], cfg.v_reset),
                                  out[..., 0])
        out[..., 2] = torch.where(flip_on, torch.ones_like(out[..., 2]),
                                  out[..., 2])
    if rc.noise and rc.refrac_jitter_p > 0:
        jm = (torch.rand(out[..., 2].shape, generator=gen)
              < rc.refrac_jitter_p).to(out.device)
        sign = torch.where(
            torch.rand(out[..., 2].shape, generator=gen) < 0.5, -1.0, 1.0
        ).to(out.device)
        delta = sign / float(cfg.refractory_period)
        out[..., 2] = torch.where(jm, out[..., 2] + delta, out[..., 2])
        out[..., 2] = out[..., 2].clamp(0.0, 1.0)
    return out


def unroll_batch(model, states, stimulus, t0, cfg, rc: RolloutConfig,
                 stage: StageConfig, gen: torch.Generator, device,
                 pos_weight, groups):
    """One scheduled-sampling unroll batch. Returns (loss, parts,
    actual_teacher_fraction).

    hist starts as the true window (X,U)[t0 : t0+K]; step u predicts the
    state at absolute time t0+K+u, composes it (surrogate-differentiable),
    then feeds either the true (noised) state or the composed state
    per-sample (Bernoulli(stage.teacher_ratio)); the appended feature pairs
    the fed state at absolute time t0+K+u with the stimulus at the SAME
    absolute time (PHASE2.md time convention).
    targets[:, 0] = last true context state; step u supervised by
    targets[:, u+1].
    """
    B = states.shape[0]
    K, U = cfg.K, stage.unroll
    x, _, _ = make_windows(states, stimulus, K, t0=t0)
    hist = x
    ar = torch.arange(U + 1, device=states.device)
    bidx = torch.arange(B, device=states.device)[:, None]
    t_idx = t0.to(states.device)[:, None] + (K - 1 + ar)[None, :]  # [B, U+1]
    targets = states[bidx, t_idx]                     # [B, U+1, N, 3]

    step_outs: list[dict] = []
    step_states: list[tuple] = []
    fed_true = 0
    use_ckpt = rc.use_grad_ckpt(stage) and model.training
    for u in range(U):
        if use_ckpt:
            out = torch.utils.checkpoint.checkpoint(model, hist,
                                                    use_reentrant=False)
        else:
            out = model(hist)
        if rc.mechanistic:
            v, sp, r, _ = compose_step_mechanistic(
                out["dv"], hist[:, -1, :, 0], hist[:, -1, :, 2], cfg,
                mode=rc.surrogate, beta=rc.surrogate_beta)
        else:
            v, sp, r = compose_step_learned(out, cfg, mode=rc.surrogate,
                                            beta=rc.surrogate_beta)
        step_outs.append(out)
        step_states.append((v, sp, r))

        feed_true = (torch.rand(B, generator=gen) < stage.teacher_ratio
                     ).to(device)
        fed_true += int(feed_true.sum())
        true_state = apply_teacher_noise(targets[:, u + 1], cfg, rc, stage,
                                         gen)
        sel = feed_true[:, None].to(v.dtype)
        v_f = sel * true_state[..., 0] + (1 - sel) * v
        sp_f = sel * true_state[..., 1] + (1 - sel) * sp
        r_f = sel * true_state[..., 2] + (1 - sel) * r
        u_stim = stimulus[bidx[:, 0], t0.to(states.device) + K + u]
        feat = torch.stack([v_f, sp_f, r_f, u_stim], dim=-1)
        hist = torch.cat([hist[:, 1:], feat.unsqueeze(1)], dim=1)
        if rc.tbptt > 0 and (u + 1) % rc.tbptt == 0:
            hist = hist.detach()
    loss, parts = multistep_loss(step_outs, step_states, targets, rc,
                                 pos_weight, groups)
    return loss, parts, fed_true / float(B * U)


# ----------------------------------------------------------------------
# training driver
def train(rc: RolloutConfig, args) -> None:
    cfg = rc.base
    torch.manual_seed(cfg.seed)
    device = get_device(override=args.device)
    conn = get_connectome(cfg, device)
    sim = LIFSimulator(conn, cfg, device)
    pos_weight = torch.tensor(cfg.spike_pos_weight, device=device)
    groups = population_groups(cfg.n_neurons, rc.pop_groups, device)

    # ---- 1. model + phase-1 init (hard error if missing) ---------------
    model = build_model(rc.model, cfg, conn, device, rc.model_kwargs)
    model = maybe_wrap(model, cfg, rc)
    p1 = phase1_ckpt_path(cfg, rc.model)
    if not p1.exists():
        raise SystemExit(f"[fatal] phase-1 checkpoint missing: {p1}\n"
                         f"train the one-step model first "
                         f"(train.py --model {rc.model} --scale {rc.scale})")
    blob = torch.load(p1, map_location="cpu", weights_only=False)
    if blob.get("model_kwargs") and not rc.model_kwargs:
        rc.model_kwargs = dict(blob["model_kwargs"])
    base_sd = blob["state_dict"]
    if rc.mechanistic:
        # the wrapper adds dv_a/dv_b (init a=1, b=0); base keys land under
        # the `base.` submodule
        base_sd = {f"base.{k}": v for k, v in base_sd.items()}
        missing = model.load_state_dict(base_sd, strict=False)
        assert not [k for k in missing.missing_keys
                    if k not in ("dv_a", "dv_b")], missing
    else:
        model.load_state_dict(base_sd)
    print(f"[init] {rc.model} {rc.model_kwargs or ''} mechanistic="
          f"{rc.mechanistic} <- {p1.name} (epoch {blob.get('epoch')}, "
          f"val_loss {blob.get('val_loss')})")

    if rc.eval_only:
        print("[eval-only] experiment A: no training (phase-1 reference)")
        return

    print("[data] loading fixed validation set ...")
    val_data = load_or_generate("val", sim, cfg)

    # ---- 2. reference scores --------------------------------------------
    th_ref = tune_threshold(model, val_data, cfg, device,
                            n_windows=rc.n_val_windows)
    r_ref, _ = onestep_eval(model, val_data, cfg, device, th_ref,
                            n_windows=rc.n_val_windows)
    f1_ref = r_ref["spike_f1"]
    score_fn = rollout_score_fn(rc)
    ref_score, ref_roll = score_fn(
        model, val_data, cfg, device, rc.val_rollout_traj,
        rc.val_rollout_horizon)
    print(f"[reference] calibrated one-step val F1={f1_ref:.3f} "
          f"(th={th_ref:.2f}); rollout score={ref_score:.3f} {ref_roll}")

    # ---- 3. DAgger warm-up ----------------------------------------------
    buffer = ReplayBuffer(rc.buffer_capacity) if rc.dagger else None
    exp_idx = experiment_index(rc.experiment, rc.version)
    base_seed = cfg.seed + 7919 * exp_idx
    gen = torch.Generator(device="cpu")
    if rc.dagger:
        c, t, p = collect_on_policy(
            model, sim, cfg, device, rc.dagger_collect_traj,
            rc.dagger_horizon, spike_threshold=0.5,
            mechanistic=rc.mechanistic, seed=base_seed + COLLECT_SEED_OFF)
        buffer.add(c, t, p)
        print(f"[dagger] warm-up buffer: {buffer.stats()}")

    # ---- 4. stage curriculum ---------------------------------------------
    hist_rows: list[dict] = []
    stage_summaries: list[dict] = []
    protection_events: list[dict] = []
    prev_stage_best = ref_score
    last_stage_ckpt: Path | None = None
    epoch_global = 0
    n_train = traj_count(cfg, "train")

    # ---- experiment-level best checkpoints (spec §15) -------------------
    best_ckpts = {"one_step": {"val": -1.0, "f1": -1.0, "path": None},
                  "rollout": {"val": -1e9, "f1": -1.0, "path": None},
                  "combined": {"val": -1e9, "f1": -1.0, "path": None}}
    warn_streak = 0                     # grad-explosion streak (spec §20)
    dmix_cap: float | None = None       # spec §6: no mix raise if F1 < 0.98

    for si, stage in enumerate(rc.stages):
        if si > 0:
            # spec §15: init from best_combined (F1 >= 0.98), else the
            # best_rollout among F1 >= 0.98, else best_combined anyway.
            if best_ckpts["combined"]["f1"] >= 0.98:
                src = best_ckpts["combined"]
            elif best_ckpts["rollout"]["f1"] >= 0.98:
                src = best_ckpts["rollout"]
            else:
                src = best_ckpts["combined"]
            if src["path"] is not None:
                sb = torch.load(src["path"], map_location="cpu",
                                weights_only=False)
                model.load_state_dict(sb["state_dict"])
                print(f"[stage {stage.name}] initialised from "
                      f"{src['path'].name} (f1={src['f1']:.3f})")
        opt = torch.optim.AdamW(model.parameters(),
                                lr=rc.base_lr * stage.lr_factor,
                                weight_decay=cfg.weight_decay)
        bs = rc.stage_batch_size(stage)
        gen_chunk = max(8 * bs, 64)
        best_score, best_epoch, bad = -1e9, 0, 0
        low_f1_streak = 0
        warn_streak = 0                 # per-stage grad-explosion streak
        # spec §23: one-step floor is a FULL-scale criterion; the tiny smoke
        # model never reaches it, so it is disabled at small scale.
        f1_floor = 0.95 if cfg.n_neurons >= 1000 else 0.0
        for epoch in range(1, stage.epochs + 1):
            epoch_global += 1
            t_start = time.time()
            gen.manual_seed(base_seed + epoch_global)
            params0 = [p.detach().clone() for p in model.parameters()]
            order = torch.randperm(n_train, generator=gen)[:stage.n_traj]
            order = order.tolist()
            ep_parts: dict[str, float] = {}
            ep_gnorms: list[float] = []
            ep_clip = 0
            ep_tf, ep_tf_n = 0.0, 0
            nb = 0
            model.train()
            for c0 in range(0, len(order), gen_chunk):
                seeds = [cfg.traj_seed("train", i)
                         for i in order[c0:c0 + gen_chunk]]
                data = generate_batch(seeds, "train", sim, cfg)
                states_b, stim_b = data["states"], data["stimulus"]
                for b0 in range(0, len(seeds), bs):
                    sl = slice(b0, min(b0 + bs, len(seeds)))
                    # Initialise every per-batch name to None so the
                    # post-batch cleanup never references an unbound name on
                    # the DAgger branch (which only sets loss/parts).
                    hist = targets = step_outs = step_states = None
                    dmix = (rc.dagger_mix_override[min(si, len(rc.dagger_mix_override) - 1)]
                            if rc.dagger_mix_override else stage.dagger_mix)
                    if dmix_cap is not None:
                        dmix = min(dmix, dmix_cap)
                    use_buffer = (rc.dagger and len(buffer) > 0
                                  and torch.rand((), generator=gen).item()
                                  < dmix)
                    if use_buffer:
                        ctx, tgt = buffer.sample(sl.stop - sl.start, gen)
                        out = model(ctx.to(device))
                        loss, parts = compute_loss(out, tgt.to(device), cfg,
                                                   pos_weight)
                        parts = {"loss": parts["loss"]}
                    else:
                        t_max = cfg.T - cfg.K - stage.unroll - 1
                        t0 = torch.randint(0, t_max + 1, (sl.stop - sl.start,),
                                           generator=gen).to(device)
                        loss, parts, tf = unroll_batch(
                            model, states_b[sl], stim_b[sl], t0, cfg, rc,
                            stage, gen, device, pos_weight, groups)
                        ep_tf += tf
                        ep_tf_n += 1
                        if rc.tangent:
                            x0, _, _ = make_windows(states_b[sl], stim_b[sl],
                                                    cfg.K, t0=t0)
                            # multi-scale sigma (spec §12): falls back to the
                            # single rc.tangent_sigma when no scales given.
                            # tangent_all_scales (v4 S7): weighted SUM of
                            # per-scale losses every batch — the per-batch
                            # SAMPLING variant (S1) oscillates the loss scale
                            # across batches and degraded training.
                            # the tangent branch keeps TWO extra forward
                            # graphs alive; always checkpoint it (it roughly
                            # doubles per-step memory at U<=8 otherwise)
                            if getattr(rc, "tangent_all_scales", False) \
                                    and getattr(rc, "tangent_scales", ()):
                                ltan_sum = None
                                for sig_i, p_i in zip(rc.tangent_scales,
                                                      rc.tangent_probs):
                                    l_i = tangent_loss(
                                        model, x0, sim, cfg, float(sig_i),
                                        gen, use_ckpt=True)
                                    term = float(p_i) * l_i
                                    ltan_sum = term if ltan_sum is None \
                                        else ltan_sum + term
                                loss = loss + rc.lambda_tangent * ltan_sum
                                parts["tangent"] = float(ltan_sum.item())
                            elif rc.lambda_perturbed > 0:
                                # spec §13: absolute perturbed-state teacher
                                sig = tangent_sigma_sample(rc, gen)
                                ltan, lpert = tangent_loss_full(
                                    model, x0, sim, cfg, sig, gen,
                                    pos_weight, use_ckpt=True)
                                loss = (loss + rc.lambda_tangent * ltan
                                        + rc.lambda_perturbed * lpert)
                                parts["tangent"] = float(ltan.item())
                                parts["perturbed"] = float(lpert.item())
                            else:
                                sig = tangent_sigma_sample(rc, gen)
                                ltan = tangent_loss(model, x0, sim, cfg,
                                                    sig, gen, use_ckpt=True)
                                loss = loss + rc.lambda_tangent * ltan
                                parts["tangent"] = float(ltan.item())
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    gn = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                        rc.grad_clip)
                    ep_gnorms.append(float(gn))
                    ep_clip += int(gn > rc.grad_clip)
                    opt.step()
                    for k, v in parts.items():
                        ep_parts[k] = ep_parts.get(k, 0.0) + v
                    nb += 1
            # ---- gradient diagnostics (spec §20) --------------------------
            gmax = max(ep_gnorms) if ep_gnorms else 0.0
            clipped_frac = ep_clip / len(ep_gnorms) if ep_gnorms else 0.0
            with torch.no_grad():
                upd_norm = float(torch.sqrt(sum(
                    ((p - p0) ** 2).sum()
                    for p, p0 in zip(model.parameters(), params0))).item())
                param_norm = float(torch.sqrt(sum(
                    (p ** 2).sum() for p in model.parameters())).item())
            grad_bad = gmax > 1e3 or clipped_frac > 0.8
            warn_streak = warn_streak + 1 if grad_bad else 0
            if grad_bad:
                # spec §20: report only — NEVER raise the clip to "fix" it.
                print(f"  [warn-grad] {stage.name} ep{epoch}: max={gmax:.1f} "
                      f"clipped={clipped_frac:.0%} streak={warn_streak}")
            if warn_streak >= 3:
                print(f"[auto-stop] {stage.name}: gradient explosion for "
                      f"{warn_streak} consecutive epochs; stopping stage")
                break
            if clipped_frac > rc.clip_warn_frac:
                print(f"  [warn] {stage.name} ep{epoch}: gradients clipped "
                      f"in {ep_clip}/{len(ep_gnorms)} batches "
                      f"(> {rc.clip_warn_frac:.0%})")

            # ---- per-epoch quick val -------------------------------------
            f1_05 = quick_onestep_f1(model, val_data, cfg, device,
                                     rc.n_val_windows)
            score, roll = score_fn(
                model, val_data, cfg, device, rc.val_rollout_traj,
                rc.val_rollout_horizon)
            # spec §23: one-step ability broken -> auto-stop
            if f1_05 < f1_floor:
                low_f1_streak += 1
                if low_f1_streak >= 2:
                    print(f"[auto-stop] {stage.name}: val F1@0.5 < "
                          f"{f1_floor} for {low_f1_streak} consecutive epochs")
                    break
            else:
                low_f1_streak = 0
            row = {"stage": stage.name, "epoch": epoch,
                   "epoch_global": epoch_global, "unroll": stage.unroll,
                   "lr": rc.base_lr * stage.lr_factor,
                   "teacher_target": stage.teacher_ratio,
                   "teacher_actual": (ep_tf / ep_tf_n) if ep_tf_n else 0.0,
                   "dagger_mix": dmix if rc.dagger else 0.0,
                   "buffer_size": len(buffer) if buffer is not None else 0,
                   "grad_norm_mean": (sum(ep_gnorms) / len(ep_gnorms)
                                      if ep_gnorms else 0.0),
                   "grad_norm_median": _pct(ep_gnorms, 0.5),
                   "grad_norm_p95": _pct(ep_gnorms, 0.95),
                   "grad_norm_max": gmax,
                   "clipped_frac": clipped_frac,
                   "update_norm": upd_norm, "param_norm": param_norm,
                   "val_f1@0.5": f1_05, "val_score": score,
                   **{f"val_{k}": v for k, v in roll.items()},
                   **{f"loss_{k}": v / max(nb, 1)
                      for k, v in ep_parts.items()}}
            hist_rows.append(row)
            print(f"[{stage.name} ep {epoch:3d}] loss={row['loss_loss']:.4f} "
                  f"tf={row['teacher_actual']:.2f} "
                  f"gnorm={row['grad_norm_mean']:.3f}/{gmax:.1f} "
                  f"val_f1@0.5={f1_05:.3f} score={score:.3f} "
                  f"collapse={roll.get('collapse_step')} "
                  f"{time.time() - t_start:.0f}s")
            if epoch == 1:
                # spec §13: component magnitudes before any weight tuning
                comp = " ".join(f"{k}={v / max(nb, 1):.4f}"
                                for k, v in sorted(ep_parts.items()))
                print(f"  [components] {stage.name}: {comp}")
            if score > best_score + 1e-4:
                best_score, best_epoch, bad = score, epoch, 0
                _save_blob(rc, cfg, model, stage_ckpt_path(rc, stage),
                           epoch, stage.name,
                           {"val_f1@0.5": f1_05, "val_score": score, **roll})
                # spec §15: experiment-level best_one_step / best_rollout /
                # best_combined (combined ranks score + f1_0.5)
                if f1_05 > best_ckpts["one_step"]["f1"]:
                    best_ckpts["one_step"].update(f1=f1_05, val=score)
                    p = best_ckpt_path(rc, "one_step")
                    _save_blob(rc, cfg, model, p, epoch, stage.name,
                               {"val_f1@0.5": f1_05, "val_score": score,
                                **roll})
                    best_ckpts["one_step"]["path"] = p
                if score > best_ckpts["rollout"]["val"]:
                    best_ckpts["rollout"].update(f1=f1_05, val=score)
                    p = best_ckpt_path(rc, "rollout")
                    _save_blob(rc, cfg, model, p, epoch, stage.name,
                               {"val_f1@0.5": f1_05, "val_score": score,
                                **roll})
                    best_ckpts["rollout"]["path"] = p
                comb = score + f1_05
                if comb > best_ckpts["combined"]["val"]:
                    best_ckpts["combined"].update(val=comb, f1=f1_05)
                    p = best_ckpt_path(rc, "combined")
                    _save_blob(rc, cfg, model, p, epoch, stage.name,
                               {"val_f1@0.5": f1_05, "val_score": score,
                                **roll})
                    best_ckpts["combined"]["path"] = p
            else:
                bad += 1
                if bad >= rc.patience:
                    print(f"[early stop] {stage.name}: no val improvement "
                          f"for {rc.patience} epochs (spec §23)")
                    break
            if rc.dagger and epoch % rc.dagger_collect_every == 0:
                c, t, p = collect_on_policy(
                    model, sim, cfg, device, rc.dagger_collect_traj,
                    rc.dagger_horizon, spike_threshold=0.5,
                    mechanistic=rc.mechanistic,
                    seed=base_seed + COLLECT_SEED_OFF + epoch_global)
                buffer.add(c, t, p)
                print(f"[dagger] refresh: {buffer.stats()}")

        # ---- stage end: calibrated one-step + protection -----------------
        ck = stage_ckpt_path(rc, stage)
        if ck.exists():
            sb = torch.load(ck, map_location="cpu", weights_only=False)
            model.load_state_dict(sb["state_dict"])
        th_cal = tune_threshold(model, val_data, cfg, device,
                                n_windows=rc.n_val_windows)
        r_cal, _ = onestep_eval(model, val_data, cfg, device, th_cal,
                                n_windows=rc.n_val_windows)
        f1_cal = r_cal["spike_f1"]
        print(f"[stage {stage.name} end] best score={best_score:.3f} "
              f"(ep {best_epoch}); calibrated one-step F1={f1_cal:.3f} "
              f"(th={th_cal:.2f}) vs ref {f1_ref:.3f}")
        stage_summaries.append({"stage": stage.name, "unroll": stage.unroll,
                                "best_score": best_score,
                                "best_epoch": best_epoch,
                                "calibrated_f1": f1_cal,
                                "threshold": th_cal})
        protected = (f1_cal < f1_ref - rc.protect_drop
                     and best_score <= prev_stage_best)
        if protected:
            protection_events.append(
                {"after_stage": stage.name, "f1_cal": f1_cal,
                 "f1_ref": f1_ref, "best_score": best_score,
                 "prev_stage_best": prev_stage_best})
            print(f"[protection] one-step F1 dropped "
                  f"{f1_ref - f1_cal:.3f} >= {rc.protect_drop} with no "
                  f"rollout improvement; stopping experiment, keeping the "
                  f"previous stage best")
            break
        if rc.dagger and f1_cal < 0.98 and dmix_cap is None:
            dmix_cap = stage.dagger_mix
            print(f"[dagger-cap] calibrated F1 {f1_cal:.3f} < 0.98; DAgger mix "
                  f"capped at {dmix_cap} for later stages (spec §6)")
        prev_stage_best = max(prev_stage_best, best_score)
        last_stage_ckpt = ck if ck.exists() else last_stage_ckpt

    # ---- 5. final artefacts ----------------------------------------------
    # spec §15: the final is best_combined (never destroys earlier stages;
    # their best ckpts stay on disk as final candidates too).
    final_src = (best_ckpts["combined"]["path"]
                 or best_ckpts["rollout"]["path"] or last_stage_ckpt)
    if final_src is not None and Path(final_src).exists():
        shutil.copyfile(final_src, final_ckpt_path(rc))
        print(f"[final] {final_ckpt_path(rc)} <- {Path(final_src).name}")
    hp = history_path(rc)
    if hist_rows:
        with open(hp, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(hist_rows[0].keys()))
            writer.writeheader()
            writer.writerows(hist_rows)
    rc_view = {k: v for k, v in asdict(rc).items() if k != "base"}
    rc_view["base"] = {"seed": cfg.seed, "n_neurons": cfg.n_neurons,
                       "T": cfg.T, "K": cfg.K}
    sp = summary_path(rc)
    with open(sp, "w") as f:
        json.dump({"config": rc_view, "f1_ref": f1_ref,
                   "ref_rollout_score": ref_score,
                   "stages": stage_summaries,
                   "protection_events": protection_events,
                   "best_ckpts": {k: {"f1": v["f1"], "val": v["val"],
                                      "path": str(v["path"])}
                                  for k, v in best_ckpts.items()},
                   "final_ckpt": str(final_ckpt_path(rc))}, f, indent=2)
    print(f"[save] {hp}\n[save] {sp}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    add_rollout_args(parser)
    args = parser.parse_args()
    cfg = get_config(args.scale)
    if args.seed is not None:
        cfg.seed = args.seed
    rc = build_rollout_config(args, cfg)
    print(f"[setup] matrix={rc.version} experiment={rc.experiment} "
          f"model={rc.model} {rc.model_kwargs or ''} "
          f"stages={[s.name for s in rc.stages]}")
    train(rc, args)


if __name__ == "__main__":
    main()
