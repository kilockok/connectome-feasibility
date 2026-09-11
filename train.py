"""Training entry point.

    python train.py --model gru [--scale small|full] [--epochs N] [--seed S]

Trains one-step prediction: (X[t-K:t], U[t-K:t]) -> X[t+1].
Training trajectories are generated on the fly from a fixed seed pool
(fully reproducible, no gigabyte dataset files). Validation uses a fixed
cached set with the OOD overlap region [0.7N, 0.9N).
"""
from __future__ import annotations

import argparse
import csv
import time

import torch

from config import get_config, RESULTS_DIR, CHECKPOINT_DIR, add_common_args
from connectome import get_connectome
from dataset import (generate_batch, load_or_generate, make_windows,
                     sample_t0, traj_count)
from device import get_device
from lif import LIFSimulator
from metrics import compute_loss, compute_metrics, naive_baseline_metrics
from models import build_model


def evaluate_val(model, val_data, cfg, device, pos_weight,
                 n_windows: int, seed: int = 999) -> dict:
    """Fixed-window validation: identical windows for every epoch/model."""
    model.eval()
    g = torch.Generator(device="cpu").manual_seed(seed)
    states, stim = val_data["states"], val_data["stimulus"]
    B_total = states.shape[0]
    per = max(1, n_windows // B_total)
    agg: dict[str, float] = {}
    naive_agg: dict[str, float] = {}
    count = 0
    with torch.no_grad():
        for i0 in range(0, B_total, 32):
            sl = slice(i0, min(i0 + 32, B_total))
            for _ in range(per):
                x, y, _ = make_windows(states[sl], stim[sl], cfg.K,
                                       generator=g)
                x, y = x.to(device), y.to(device)
                out = model(x)
                _, parts = compute_loss(out, y, cfg, pos_weight)
                m = compute_metrics(out, y, auroc=False)
                nm = naive_baseline_metrics(x, y)
                for k, v in {**parts, **m}.items():
                    agg[k] = agg.get(k, 0.0) + v
                for k, v in nm.items():
                    naive_agg["naive_" + k] = naive_agg.get("naive_" + k, 0.0) + v
                count += 1
    model.train()
    out = {k: v / count for k, v in agg.items()}
    out.update({k: v / count for k, v in naive_agg.items()})
    return out


def gather_states(states: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """states[B,T,N,F] -> [B,N,F] at per-sample times t [B]."""
    B = states.shape[0]
    return states[torch.arange(B, device=states.device), t]


@torch.no_grad()
def _compose_hard(out: dict, cfg, threshold: float = 0.5):
    """Rollout-style state composition (non-differentiable reference)."""
    v = out["v"].clamp(cfg.v_min, cfg.v_th * 3.0)
    sp = (torch.sigmoid(out["s_logits"]) > threshold).to(v.dtype)
    r = out["r"].clamp(0.0, 1.0)
    fired = sp > 0.5
    v = torch.where(fired, torch.full_like(v, cfg.v_reset), v)
    r = torch.where(fired, torch.ones_like(r), r)
    return v, sp, r


def _compose_straight_through(out: dict, cfg, threshold: float = 0.5):
    """Same composition as rollout but differentiable (straight-through)."""
    v = out["v"].clamp(cfg.v_min, cfg.v_th * 3.0)
    p = torch.sigmoid(out["s_logits"])
    sp = p + ((p > threshold).to(p.dtype) - p).detach()
    r = out["r"].clamp(0.0, 1.0)
    fired = (sp > 0.5).to(v.dtype).detach()
    v = v * (1 - fired) + cfg.v_reset * fired
    r = r * (1 - fired) + fired
    return v, sp, r


def finetune_unroll(model, args, cfg, sim, device, pos_weight, ckpt_path):
    """Phase-2 (optional): multi-step unrolled fine-tuning.

    Trains the model through its own predictions (straight-through state
    composition identical to rollout), which directly targets the exposure
    bias that makes one-step-trained rollouts collapse. Loads the phase-1
    checkpoint first and saves to a *_ms checkpoint (never overwrites).
    """
    base_path = (CHECKPOINT_DIR /
                 f"ckpt_{args.model}_{args.scale}_seed{cfg.seed}.pt")
    blob = torch.load(base_path, map_location="cpu", weights_only=False)
    model.load_state_dict(blob["state_dict"])
    print(f"[unroll] initialised from phase-1 checkpoint "
          f"(epoch {blob['epoch']}, val_loss {blob['val_loss']:.4f})")

    opt = torch.optim.AdamW(model.parameters(), lr=args.unroll_lr,
                            weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.unroll_epochs)
    val_data = load_or_generate("val", sim, cfg)
    n_train = args.unroll_traj
    B = cfg.batch_size
    gen_chunk = max(8 * B, 64)
    U = args.unroll
    gen = torch.Generator(device="cpu")
    best_val = float("inf")
    history = []
    for epoch in range(1, args.unroll_epochs + 1):
        t_start = time.time()
        gen.manual_seed(cfg.seed + 10_000 + epoch)
        order = torch.randperm(traj_count(cfg, "train"), generator=gen
                               )[:n_train].tolist()
        ep_loss, nb = 0.0, 0
        model.train()
        for c0 in range(0, n_train, gen_chunk):
            seeds = [cfg.traj_seed("train", i) for i in order[c0:c0 + gen_chunk]]
            data = generate_batch(seeds, "train", sim, cfg)
            states, stimulus = data["states"], data["stimulus"]
            for b0 in range(0, len(seeds), B):
                sl = slice(b0, min(b0 + B, len(seeds)))
                nb_traj = sl.stop - sl.start
                t_max = cfg.T - cfg.K - U - 1
                t0 = torch.randint(0, t_max + 1, (nb_traj,), generator=gen
                                   ).to(device)
                hist = None
                x, _, _ = make_windows(states[sl], stimulus[sl], cfg.K, t0=t0)
                hist = x
                total = 0.0
                for u in range(U):
                    out = model(hist)
                    y_u = gather_states(states[sl], t0 + cfg.K + u)
                    loss_u, _ = compute_loss(out, y_u, cfg, pos_weight)
                    total = total + loss_u
                    v, sp, r = _compose_straight_through(out, cfg)
                    u_stim = gather_states(
                        stimulus[sl].unsqueeze(-1), t0 + cfg.K + u)[..., 0]
                    feat = torch.stack([v, sp, r, u_stim], dim=-1)
                    hist = torch.cat([hist[:, 1:], feat.unsqueeze(1)], dim=1)
                loss = total / U
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                opt.step()
                ep_loss += loss.item()
                nb += 1
        sched.step()
        val = evaluate_val(model, val_data, cfg, device, pos_weight,
                           cfg.n_val_windows)
        history.append({"epoch": epoch, "train_loss": ep_loss / nb,
                        **{f"val_{k}": v for k, v in val.items()}})
        print(f"[unroll {epoch:3d}] train_loss={ep_loss / nb:.4f} "
              f"val_loss={val['loss']:.4f} val_v_mse={val['v_mse']:.4f} "
              f"val_f1={val['spike_f1']:.3f} {time.time() - t_start:.1f}s")
        if val["loss"] < best_val - 1e-4:
            best_val = val["loss"]
            torch.save({"model": args.model, "scale": args.scale,
                        "seed": cfg.seed, "epoch": epoch, "unroll": U,
                        "val_loss": best_val,
                        "state_dict": model.state_dict()}, ckpt_path)
    hist_path = RESULTS_DIR / f"history_{args.model}_{args.scale}_ms.csv"
    with open(hist_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    print(f"[unroll done] best val {best_val:.4f} -> {ckpt_path}")


def main():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--model", required=True,
                        choices=["gru", "transformer", "connectome", "gnn"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--unroll", type=int, default=0,
                        help="phase-2: unrolled multi-step fine-tune steps (0=off)")
    parser.add_argument("--unroll-epochs", type=int, default=10)
    parser.add_argument("--unroll-lr", type=float, default=1e-4)
    parser.add_argument("--unroll-traj", type=int, default=512,
                        help="trajectories per unroll epoch (fine-tune subset)")
    args = parser.parse_args()

    cfg = get_config(args.scale)
    if args.seed is not None:
        cfg.seed = args.seed
    if args.epochs is not None:
        cfg.epochs = args.epochs

    torch.manual_seed(cfg.seed)
    device = get_device(override=args.device)
    conn = get_connectome(cfg, device)
    sim = LIFSimulator(conn, cfg, device)
    model = build_model(args.model, cfg, conn, device)
    pos_weight = torch.tensor(cfg.spike_pos_weight, device=device)

    if args.unroll > 0:
        ms_path = (CHECKPOINT_DIR /
                   f"ckpt_{args.model}_{args.scale}_ms_seed{cfg.seed}.pt")
        finetune_unroll(model, args, cfg, sim, device, pos_weight, ms_path)
        return

    lr = cfg.lr_gru if args.model == "gru" else cfg.lr
    opt = torch.optim.AdamW(model.parameters(), lr=lr,
                            weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)

    print("[data] generating fixed validation set ...")
    val_data = load_or_generate("val", sim, cfg)

    n_train = traj_count(cfg, "train")
    B = cfg.batch_size
    gen_chunk = max(8 * B, 64)      # generate trajectories in large chunks
                                    # (amortises the 256-step python loop)
    ckpt_path = (CHECKPOINT_DIR /
                 f"ckpt_{args.model}_{args.scale}_seed{cfg.seed}.pt")
    hist_path = RESULTS_DIR / f"history_{args.model}_{args.scale}.csv"

    best_val = float("inf")
    bad_epochs = 0
    history = []
    gen = torch.Generator(device="cpu")
    for epoch in range(1, cfg.epochs + 1):
        t_start = time.time()
        gen.manual_seed(cfg.seed + epoch)
        order = torch.randperm(n_train, generator=gen).tolist()
        ep_loss, ep_parts, nb = 0.0, {}, 0
        model.train()
        for c0 in range(0, n_train, gen_chunk):
            seeds = [cfg.traj_seed("train", i) for i in order[c0:c0 + gen_chunk]]
            data = generate_batch(seeds, "train", sim, cfg)
            states, stimulus, metas = data["states"], data["stimulus"], data["meta"]
            for b0 in range(0, len(seeds), B):
                sl = slice(b0, min(b0 + B, len(seeds)))
                t0 = sample_t0(metas[sl], cfg.K, cfg.T,
                               cfg.window_activity_bias, gen, device)
                x, y, _ = make_windows(states[sl], stimulus[sl], cfg.K, t0=t0)
                out = model(x)
                loss, parts = compute_loss(out, y, cfg, pos_weight)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                opt.step()
                ep_loss += parts["loss"]
                for k, v in parts.items():
                    ep_parts[k] = ep_parts.get(k, 0.0) + v
                nb += 1
        sched.step()

        val = evaluate_val(model, val_data, cfg, device, pos_weight,
                           cfg.n_val_windows)
        row = {"epoch": epoch, "train_loss": ep_loss / nb,
               **{f"train_{k}": v / nb for k, v in ep_parts.items()},
               **{f"val_{k}": v for k, v in val.items()}}
        history.append(row)
        dt = time.time() - t_start
        print(f"[epoch {epoch:3d}] train_loss={row['train_loss']:.4f} "
              f"val_loss={val['loss']:.4f} val_v_mse={val['v_mse']:.4f} "
              f"val_spike_f1={val['spike_f1']:.3f} "
              f"(naive_f1={val['naive_spike_f1']:.3f}) {dt:.1f}s")

        if val["loss"] < best_val - 1e-4:
            best_val = val["loss"]
            bad_epochs = 0
            torch.save({"model": args.model, "scale": args.scale,
                        "seed": cfg.seed, "epoch": epoch,
                        "val_loss": best_val,
                        "state_dict": model.state_dict()}, ckpt_path)
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                print(f"[early stop] no val improvement for {cfg.patience} epochs")
                break

    with open(hist_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    print(f"[done] best val loss {best_val:.4f}; checkpoint -> {ckpt_path}")
    print(f"[done] history -> {hist_path}")


if __name__ == "__main__":
    main()
