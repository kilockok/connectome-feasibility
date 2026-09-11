"""Phase-2 rollout-stability trainer.

Trains a phase-1 checkpoint through the progressive-horizon curriculum with
scheduled sampling, DAgger on-policy replay, state-noise injection, mechanistic
state projection and the horizon-weighted macro loss.

    python rollout_train.py --model gnn --experiment G --scale full
    python rollout_train.py --model gnn --experiment G --scale small --smoke

Never touches phase-1 checkpoints; all outputs go through rollout_config path
helpers. Checkpoints include a `mechanistic` flag so rollout_eval.py can
rebuild the right model container.
"""
from __future__ import annotations

import argparse
import csv
import json
import time

import torch

from config import get_config, add_common_args
from connectome import get_connectome
from dataset import (generate_batch, load_or_generate, make_windows,
                     traj_count)
from device import get_device
from evaluate import tune_threshold
from lif import LIFSimulator
from losses import multistep_loss, population_groups
from metrics import compute_loss, compute_metrics
from models import build_model
from models.mechanistic import maybe_wrap
from rollout import run_rollout_eval
from rollout_config import (RolloutConfig, StageConfig, add_rollout_args,
                            build_rollout_config, experiment_index,
                            final_ckpt_path, history_path, phase1_ckpt_path,
                            stage_ckpt_path, summary_path)
from surrogate import compose_step_learned, compose_step_mechanistic


def _empty_cache(device):
    if device.type in ("cuda", "xpu") and hasattr(torch, device.type):
        try:
            getattr(torch, device.type).empty_cache()
        except Exception:
            pass


# ----------------------------------------------------------------------
@torch.no_grad()
def _eval_f1(model, val_data, cfg, device, n_windows, seed=999):
    """Fixed-window validation: mean spike F1 at threshold 0.5."""
    model.eval()
    g = torch.Generator(device="cpu").manual_seed(seed)
    states, stim = val_data["states"], val_data["stimulus"]
    B_total = states.shape[0]
    per = max(1, n_windows // B_total)
    agg, count = 0.0, 0
    with torch.no_grad():
        for i0 in range(0, B_total, 32):
            sl = slice(i0, min(i0 + 32, B_total))
            for _ in range(per):
                x, y, _ = make_windows(states[sl], stim[sl], cfg.K, generator=g)
                out = model(x.to(device))
                m = compute_metrics(out, y.to(device), auroc=False, threshold=0.5)
                agg += m["spike_f1"]
                count += 1
    model.train()
    return agg / count


@torch.no_grad()
def _calibrated_f1(model, val_data, cfg, device, n_windows=512):
    """Threshold tuned on val, then val F1 at that threshold."""
    th = tune_threshold(model, val_data, cfg, device, n_windows=n_windows)
    model.eval()
    g = torch.Generator(device="cpu").manual_seed(999)
    states, stim = val_data["states"], val_data["stimulus"]
    agg, count = 0.0, 0
    with torch.no_grad():
        for i0 in range(0, states.shape[0], 32):
            sl = slice(i0, min(i0 + 32, states.shape[0]))
            for _ in range(max(1, n_windows // states.shape[0])):
                x, y, _ = make_windows(states[sl], stim[sl], cfg.K, generator=g)
                out = model(x.to(device))
                agg += compute_metrics(out, y.to(device), auroc=False,
                                       threshold=th)["spike_f1"]
                count += 1
    model.train()
    return float(agg / count), th


@torch.no_grad()
def _val_rollout_score(model, val_data, cfg, rc):
    """Quick closed-loop val rollout: F1@10 + pop_sim@H - rate_err@H."""
    H = min(rc.val_rollout_horizon, cfg.T - cfg.K - 1)
    m, _, _ = run_rollout_eval(model, val_data, cfg, [10, H],
                               spike_threshold=0.5,
                               n_traj=rc.val_rollout_traj)
    return (m[10]["spike_f1"] + m[H]["pop_similarity"]
            - m[H]["firing_rate_err"]), m[10]["spike_f1"], m[H]["pop_similarity"]


# ----------------------------------------------------------------------
def train_experiment(rc: RolloutConfig, args):
    cfg = rc.base
    torch.manual_seed(cfg.seed)
    device = get_device(override=args.device)
    conn = get_connectome(cfg, device)
    sim = LIFSimulator(conn, cfg, device)
    pos_weight = torch.tensor(cfg.spike_pos_weight, device=device)
    groups = population_groups(cfg.n_neurons, rc.pop_groups, device) \
        if rc.macro_loss else None
    exp_idx = experiment_index(rc.experiment)

    # ---- model: phase-1 base + optional mechanistic wrap ----------------
    base = build_model(rc.model, cfg, conn, device)
    p1_path = phase1_ckpt_path(cfg, rc.model)
    if not p1_path.exists():
        raise SystemExit(f"missing phase-1 checkpoint {p1_path} "
                         f"(train phase 1 first; never train from scratch)")
    blob = torch.load(p1_path, map_location="cpu", weights_only=False)
    base.load_state_dict(blob["state_dict"])
    print(f"[init] {rc.model} from {p1_path.name} "
          f"(epoch {blob['epoch']}, val_loss {blob['val_loss']:.4f})")
    model = maybe_wrap(base, cfg, rc)
    model.to(device)
    mech = bool(getattr(model, "is_mechanistic", False))

    val_data = load_or_generate("val", sim, cfg)

    # ---- reference metrics (before any rollout training) -----------------
    f1_ref, th_ref = _calibrated_f1(model, val_data, cfg, device,
                                    rc.n_val_windows)
    ref_score, _, _ = _val_rollout_score(model, val_data, cfg, rc)
    print(f"[ref] calibrated one-step val F1={f1_ref:.4f} (th={th_ref:.2f}) "
          f"val rollout score={ref_score:.4f}")

    # ---- DAgger warmup ----------------------------------------------------
    buffer = None
    if rc.dagger:
        from dagger import ReplayBuffer, collect_on_policy
        buffer = ReplayBuffer(rc.buffer_capacity)
        c, t, p = collect_on_policy(model, sim, cfg, device,
                                    n_traj=rc.dagger_collect_traj,
                                    horizon=rc.dagger_horizon,
                                    spike_threshold=0.5,
                                    mechanistic=mech,
                                    seed=cfg.seed + 7919 * exp_idx)
        buffer.add(c, t, p)
        print(f"[dagger] warmup buffer size={len(buffer)}")

    # ---- curriculum -------------------------------------------------------
    history = []
    protection_events = []
    prev_best_score = -1e9
    prev_best_state = None
    final_state = None
    t_run0 = time.time()

    for si, stage in enumerate(rc.stages):
        U = stage.unroll
        lr = rc.base_lr * stage.lr_factor
        B = rc.stage_batch_size(stage)
        opt = torch.optim.AdamW(model.parameters(), lr=lr,
                                weight_decay=cfg.weight_decay)
        print(f"[stage {stage.name}] U={U} lr={lr:.2e} B={B} "
              f"teacher={stage.teacher_ratio:.2f} sigma={stage.noise_sigma:.0e} "
              f"dagger={stage.dagger_mix:.2f} grad_ckpt={rc.use_grad_ckpt(stage)}")

        best_score = -1e9
        best_state = None
        best_epoch = 0
        bad_epochs = 0
        gen = torch.Generator(device="cpu")
        gen.manual_seed(cfg.seed + 7919 * exp_idx + 1000 * si)

        for epoch in range(1, stage.epochs + 1):
            t0e = time.time()
            gen.manual_seed(cfg.seed + 7919 * exp_idx + 1000 * si + epoch)
            order = torch.randperm(traj_count(cfg, "train"),
                                   generator=gen)[:stage.n_traj].tolist()
            gen_chunk = max(8 * B, 64)
            ep_loss, ep_parts, nb = 0.0, {}, 0
            dagger_batches = unroll_batches = teacher_steps = 0
            clip_count, clip_total, gn_max = 0, 0, 0.0
            model.train()

            for c0 in range(0, len(order), gen_chunk):
                seeds = [cfg.traj_seed("train", i)
                         for i in order[c0:c0 + gen_chunk]]
                data = generate_batch(seeds, "train", sim, cfg)
                states, stimulus = data["states"], data["stimulus"]
                for b0 in range(0, len(seeds), B):
                    sl = slice(b0, min(b0 + B, len(seeds)))
                    nb_traj = sl.stop - sl.start

                    use_dagger = (buffer is not None and len(buffer) > 0
                                  and stage.dagger_mix > 0
                                  and torch.rand((), generator=gen).item()
                                  < stage.dagger_mix)
                    if use_dagger:
                        # ---- one-step batch from on-policy replay ----
                        ctx, tgt = buffer.sample(nb_traj, gen)
                        ctx, tgt = ctx.to(device), tgt.to(device)
                        out = model(ctx)
                        loss, parts = compute_loss(out, tgt, cfg, pos_weight)
                        dagger_batches += 1
                    else:
                        # ---- unroll batch with scheduled sampling -------
                        t_max = cfg.T - cfg.K - U - 1
                        t0 = torch.randint(0, t_max + 1, (nb_traj,),
                                           generator=gen).to(device)
                        hist, _, _ = make_windows(states[sl], stimulus[sl],
                                                  cfg.K, t0=t0)
                        bidx = torch.arange(nb_traj, device=device)[:, None]
                        targets = torch.empty(nb_traj, U + 1,
                                              cfg.n_neurons, 3,
                                              device=device)
                        targets[:, 0] = states[sl][bidx[:, 0],
                                                   t0 + cfg.K - 1]
                        for u in range(U):
                            targets[:, u + 1] = states[sl][bidx[:, 0],
                                                           t0 + cfg.K + u]

                        step_outs, step_states = [], []
                        for u in range(U):
                            if rc.use_grad_ckpt(stage):
                                out = torch.utils.checkpoint.checkpoint(
                                    model, hist, use_reentrant=False)
                            else:
                                out = model(hist)
                            if mech:
                                v, sp, r, _ = compose_step_mechanistic(
                                    out["dv"], hist[:, -1, :, 0],
                                    hist[:, -1, :, 2], cfg,
                                    mode=rc.surrogate, beta=rc.surrogate_beta)
                            else:
                                v, sp, r = compose_step_learned(
                                    out, cfg, threshold=0.5,
                                    mode=rc.surrogate, beta=rc.surrogate_beta)
                            step_outs.append(out)
                            step_states.append((v, sp, r))

                            tm = torch.rand(nb_traj, generator=gen) \
                                < stage.teacher_ratio
                            teacher_steps += int(tm.sum())
                            use_true = tm.to(device)
                            if use_true.any() and rc.noise \
                                    and stage.noise_sigma > 0:
                                tv = targets[:, u + 1, :, 0].clone()
                                tnoisy_v = tv + torch.randn_like(tv) \
                                    * stage.noise_sigma
                                tsp = targets[:, u + 1, :, 1].clone()
                                trr = targets[:, u + 1, :, 2].clone()
                                flip = torch.rand_like(tsp) < rc.spike_flip_p
                                flip_on = flip & (tsp < 0.5)
                                flip_off = flip & (tsp > 0.5)
                                tsp = torch.where(flip_on, torch.ones_like(tsp),
                                                  tsp)
                                tsp = torch.where(flip_off,
                                                  torch.zeros_like(tsp), tsp)
                                tnoisy_v = torch.where(
                                    flip_on,
                                    torch.full_like(tnoisy_v, cfg.v_reset),
                                    tnoisy_v)
                                trr = torch.where(flip_on,
                                                  torch.ones_like(trr), trr)
                                jit = torch.rand_like(trr) < rc.refrac_jitter_p
                                trr = torch.where(
                                    jit, (trr + (torch.randint_like(
                                        trr, 0, 3) - 1)
                                          / cfg.refractory_period).clamp(0, 1),
                                    trr)
                                fed_v = torch.where(use_true[:, None],
                                                    tnoisy_v, v)
                                fed_sp = torch.where(use_true[:, None],
                                                     tsp, sp)
                                fed_r = torch.where(use_true[:, None],
                                                    trr, r)
                            else:
                                fed_v = torch.where(use_true[:, None],
                                                    targets[:, u + 1, :, 0], v)
                                fed_sp = torch.where(use_true[:, None],
                                                     targets[:, u + 1, :, 1], sp)
                                fed_r = torch.where(use_true[:, None],
                                                    targets[:, u + 1, :, 2], r)

                            u_stim = stimulus[sl][bidx[:, 0], t0 + cfg.K + u]
                            feat = torch.stack([fed_v, fed_sp, fed_r, u_stim],
                                               dim=-1)
                            hist = torch.cat([hist[:, 1:],
                                              feat.unsqueeze(1)], dim=1)
                            if rc.tbptt > 0 and (u + 1) % rc.tbptt == 0:
                                hist = hist.detach()

                        loss, parts = multistep_loss(step_outs, step_states,
                                                     targets, rc, pos_weight,
                                                     groups)
                        unroll_batches += 1

                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    gn = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                        rc.grad_clip)
                    gn_val = float(gn) if torch.is_tensor(gn) else float(gn)
                    gn_max = max(gn_max, gn_val)
                    if gn_val > rc.grad_clip:
                        clip_count += 1
                    clip_total += 1
                    opt.step()
                    ep_loss += parts["loss"]
                    for k, v in parts.items():
                        ep_parts[k] = ep_parts.get(k, 0.0) + v
                    nb += 1
                    del step_outs, step_states, hist, targets, loss
                    if unroll_batches % 20 == 0:
                        _empty_cache(device)

                del data, states, stimulus

            # ---- epoch validation -----------------------------------------
            f1_onestep = _eval_f1(model, val_data, cfg, device,
                                  rc.n_val_windows)
            score, rf1_10, rpop = _val_rollout_score(model, val_data, cfg, rc)
            teacher_frac = teacher_steps / max(1, unroll_batches * U * B)
            clip_frac = clip_count / max(1, clip_total)
            dt = time.time() - t0e

            row = {"stage": stage.name, "unroll": U, "lr": lr,
                   "epoch": epoch, "train_loss": ep_loss / nb,
                   "teacher_target": stage.teacher_ratio,
                   "teacher_actual": teacher_frac,
                   "dagger_batches": dagger_batches,
                   "unroll_batches": unroll_batches,
                   "grad_norm_max": gn_max, "clip_frac": clip_frac,
                   "val_f1_onestep": f1_onestep,
                   "val_rollout_f1_10": rf1_10,
                   "val_rollout_pop_sim": rpop,
                   "val_rollout_score": score, "epoch_time": dt}
            for k, v in ep_parts.items():
                row[f"train_{k}"] = v / nb
            if buffer is not None:
                row["buffer_size"] = len(buffer)
            history.append(row)

            print(f"[{stage.name} {epoch:3d}/{stage.epochs}] "
                  f"loss={ep_loss / nb:.4f} f1@0.5={f1_onestep:.3f} "
                  f"rollout_f1@10={rf1_10:.3f} pop={rpop:.3f} "
                  f"score={score:.4f} teacher={teacher_frac:.2f} "
                  f"gn_max={gn_max:.2f} clip={clip_frac:.2f} {dt:.0f}s")
            if clip_frac > rc.clip_warn_frac:
                print(f"  [warn] grad norm hit clip on {clip_frac:.0%} of "
                      f"batches (threshold {rc.grad_clip})")

            if score > best_score + 1e-3:
                best_score = score
                best_epoch = epoch
                bad_epochs = 0
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
                torch.save({"model": rc.model, "scale": rc.scale,
                            "seed": cfg.seed, "experiment": rc.experiment,
                            "stage": stage.name, "epoch": epoch,
                            "mechanistic": mech, "surrogate": rc.surrogate,
                            "val_score": best_score,
                            "val_f1_onestep": f1_onestep,
                            "state_dict": best_state},
                           stage_ckpt_path(rc, stage))
            else:
                bad_epochs += 1
                if bad_epochs >= rc.patience:
                    print(f"[early stop] {stage.name}: no score improvement "
                          f"for {rc.patience} epochs (best {best_score:.4f})")
                    break

            # ---- DAgger refresh -------------------------------------------
            if buffer is not None and stage.dagger_mix > 0 \
                    and epoch % rc.dagger_collect_every == 0:
                from dagger import collect_on_policy
                c, t, p = collect_on_policy(model, sim, cfg, device,
                                            n_traj=rc.dagger_collect_traj,
                                            horizon=rc.dagger_horizon,
                                            spike_threshold=0.5,
                                            mechanistic=mech,
                                            seed=cfg.seed + 7919 * exp_idx
                                            + 5000 * si + epoch)
                buffer.add(c, t, p)
                print(f"  [dagger] buffer size={len(buffer)}")
                del c, t, p

            _empty_cache(device)

        # ---- stage end: protection check ----------------------------------
        if best_state is not None:
            model.load_state_dict(best_state)
        f1_cal, _ = _calibrated_f1(model, val_data, cfg, device,
                                   rc.n_val_windows)
        print(f"[stage {stage.name} done] best_score={best_score:.4f} "
              f"(ep {best_epoch}) calibrated F1={f1_cal:.4f} "
              f"(ref {f1_ref:.4f})")

        if (f1_cal < f1_ref - rc.protect_drop
                and best_score <= prev_best_score + 1e-3):
            print(f"[PROTECTION] calibrated F1 dropped "
                  f"{f1_ref - f1_cal:.3f} (>{rc.protect_drop}) with no rollout "
                  f"gain vs previous stage — stopping experiment here.")
            protection_events.append(
                {"stage": stage.name, "f1_calibrated": f1_cal,
                 "f1_ref": f1_ref, "best_score": best_score,
                 "prev_best_score": prev_best_score})
            final_state = prev_best_state if prev_best_state is not None \
                else best_state
            break

        prev_best_score = best_score
        prev_best_state = best_state
        final_state = best_state
        _empty_cache(device)

    # ---- save final -------------------------------------------------------
    if final_state is not None:
        model.load_state_dict(final_state)
    final_path = final_ckpt_path(rc)
    torch.save({"model": rc.model, "scale": rc.scale, "seed": cfg.seed,
                "experiment": rc.experiment, "mechanistic": mech,
                "surrogate": rc.surrogate, "final": True,
                "val_score": prev_best_score,
                "state_dict": model.state_dict()}, final_path)
    print(f"[save] final -> {final_path}")

    hp = history_path(rc)
    with open(hp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        w.writeheader()
        w.writerows(history)

    def _prim(o):
        if isinstance(o, (int, float, str, bool)) or o is None:
            return o
        return str(o)

    sp = summary_path(rc)
    with open(sp, "w") as f:
        json.dump({"experiment": rc.experiment, "model": rc.model,
                   "scale": rc.scale, "seed": cfg.seed,
                   "mechanistic": mech, "surrogate": rc.surrogate,
                   "f1_ref": f1_ref, "ref_score": ref_score,
                   "final_val_score": prev_best_score,
                   "protection_events": protection_events,
                   "wall_time_s": time.time() - t_run0}, f, indent=2,
                  default=_prim)
    print(f"[save] {hp}\n[save] {sp}")
    print(f"[done] {rc.experiment} wall {(time.time() - t_run0) / 60:.1f} min")


def main():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    add_rollout_args(parser)
    args = parser.parse_args()
    cfg = get_config(args.scale)
    if args.seed is not None:
        cfg.seed = args.seed
    rc = build_rollout_config(args, cfg)
    if rc.eval_only:
        raise SystemExit("experiment A is eval-only: run rollout_eval.py "
                         "over the phase-1 checkpoint instead")
    print(f"[exp] {rc.experiment} model={rc.model} scale={rc.scale} "
          f"mechanistic={rc.mechanistic} dagger={rc.dagger} "
          f"noise={rc.noise} macro={rc.macro_loss} surrogate={rc.surrogate} "
          f"stages={[s.name for s in rc.stages]}")
    train_experiment(rc, args)


if __name__ == "__main__":
    main()
