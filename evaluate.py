"""Full evaluation: one-step, rollout (seen/OOD), perturbation.

    python evaluate.py [--scale small|full] [--models gru transformer ...]

Reads checkpoints produced by train.py, evaluates every available model on
identical fixed data, writes:
  results/metrics_{scale}.json / metrics_{scale}.csv
  results/cache/eval_examples_{scale}.pt   (tensors for plots.py)
Then calls plots.py.
"""
from __future__ import annotations

import argparse
import json

import pandas as pd
import torch

from config import get_config, RESULTS_DIR, CHECKPOINT_DIR, add_common_args
from connectome import get_connectome
from dataset import load_or_generate, make_windows
from device import get_device
from lif import LIFSimulator
from metrics import compute_metrics, naive_baseline_metrics
from models import build_model
from perturbation import run_perturbation_eval
from rollout import naive_rollout, rollout_metrics, run_rollout_eval

MODEL_NAMES = ["gru", "transformer", "connectome", "gnn"]


# ----------------------------------------------------------------------
@torch.no_grad()
def tune_threshold(model, data, cfg, device, n_windows=512) -> float:
    """Pick the spike decision threshold maximising F1 on the val split."""
    g = torch.Generator().manual_seed(777)
    states, stim = data["states"], data["stimulus"]
    probs, trues = [], []
    per = max(1, n_windows // states.shape[0])
    for i0 in range(0, states.shape[0], 32):
        sl = slice(i0, min(i0 + 32, states.shape[0]))
        for _ in range(per):
            x, y, _ = make_windows(states[sl], stim[sl], cfg.K, generator=g)
            out = model(x.to(device))
            probs.append(torch.sigmoid(out["s_logits"]).flatten().cpu())
            trues.append(y[..., 1].flatten().cpu())
    p, t = torch.cat(probs), torch.cat(trues)
    best_t, best_f1 = 0.5, -1.0
    for th in torch.arange(0.05, 0.96, 0.05):
        pred = (p > th).float()
        tp = (pred * t).sum().item()
        fp = (pred * (1 - t)).sum().item()
        fn = ((1 - pred) * t).sum().item()
        f1 = 2 * tp / max(2 * tp + fp + fn, 1.0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(th)
    return best_t


# ----------------------------------------------------------------------
@torch.no_grad()
def tune_rollout_threshold(model, val_data, cfg,
                           candidates=(0.3, 0.5, 0.6, 0.7, 0.8, 0.9)) -> float:
    """Pick the spike threshold for autoregressive rollout on the val split.

    Score per candidate: F1 + population similarity - firing-rate error at
    the longest tuning horizon. Keeps rollouts alive without rewarding
    spike spam.
    """
    horizons = [min(50, cfg.T - cfg.K - 1)]
    best_t, best_score = 0.5, -1e9
    for th in candidates:
        m, _, _ = run_rollout_eval(model, val_data, cfg, horizons,
                                   spike_threshold=th, n_traj=8)
        h = m[horizons[0]]
        score = h["spike_f1"] + h["pop_similarity"] - h["firing_rate_err"]
        print(f"    th={th:.2f}: f1={h['spike_f1']:.3f} "
              f"pop={h['pop_similarity']:.3f} rate_err={h['firing_rate_err']:.4f}")
        if score > best_score:
            best_score, best_t = score, th
    return best_t


@torch.no_grad()
def onestep_eval(model, data, cfg, device, threshold, n_windows=1024,
                 collect_scatter=False):
    g = torch.Generator().manual_seed(888)
    states, stim = data["states"], data["stimulus"]
    per = max(1, n_windows // states.shape[0])
    agg, naive_agg, count = {}, {}, 0
    scatter = None
    for i0 in range(0, states.shape[0], 32):
        sl = slice(i0, min(i0 + 32, states.shape[0]))
        for _ in range(per):
            x, y, _ = make_windows(states[sl], stim[sl], cfg.K, generator=g)
            x, y = x.to(device), y.to(device)
            out = model(x)
            m = compute_metrics(out, y, auroc=False, threshold=threshold)
            nm = naive_baseline_metrics(x, y)
            for k, v in m.items():
                agg[k] = agg.get(k, 0.0) + v
            for k, v in nm.items():
                naive_agg["naive_" + k] = naive_agg.get("naive_" + k, 0.0) + v
            count += 1
            if collect_scatter and scatter is None:
                idx = torch.randperm(y.numel() // 3, generator=g)[:20000]
                scatter = {
                    "v_true": y[..., 0].flatten().cpu()[idx],
                    "v_pred": out["v"].flatten().cpu()[idx],
                    "s_prob": torch.sigmoid(out["s_logits"]).flatten().cpu()[idx],
                    "s_true": y[..., 1].flatten().cpu()[idx]}
    res = {k: v / count for k, v in agg.items()}
    res.update({k: v / count for k, v in naive_agg.items()})
    # AUROC on the collected sample
    if scatter is not None:
        try:
            from sklearn.metrics import roc_auc_score
            st, sp = scatter["s_true"].numpy(), scatter["s_prob"].numpy()
            if st.any() and not st.all():
                res["spike_auroc"] = float(roc_auc_score(st, sp))
        except Exception:
            pass
    return res, scatter


# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--models", nargs="*", default=None)
    args = parser.parse_args()

    cfg = get_config(args.scale)
    torch.manual_seed(cfg.seed)
    device = get_device(override=args.device)
    conn = get_connectome(cfg, device)
    sim = LIFSimulator(conn, cfg, device)

    names = args.models or MODEL_NAMES
    models, thresholds, rollout_thresholds = {}, {}, {}
    for name in names:
        # prefer the multi-step fine-tuned checkpoint when available
        candidates = [CHECKPOINT_DIR /
                      f"ckpt_{name}_{args.scale}_ms_seed{cfg.seed}.pt",
                      CHECKPOINT_DIR /
                      f"ckpt_{name}_{args.scale}_seed{cfg.seed}.pt"]
        ckpt = next((c for c in candidates if c.exists()), None)
        if ckpt is None:
            print(f"[skip] {name}: no checkpoint for scale={args.scale}")
            continue
        blob = torch.load(ckpt, map_location="cpu", weights_only=False)
        model = build_model(name, cfg, conn, device)
        model.load_state_dict(blob["state_dict"])
        model.eval()
        models[name] = model
        print(f"[load] {name}: {ckpt.name} (epoch {blob['epoch']}, "
              f"val_loss {blob['val_loss']:.4f})")
    if not models:
        raise SystemExit("no checkpoints found; train models first")

    print("[data] loading fixed eval splits ...")
    data = {split: load_or_generate(split, sim, cfg)
            for split in ("val", "test_seen", "test_ood")}

    for name, model in models.items():
        thresholds[name] = tune_threshold(model, data["val"], cfg, device)
        print(f"[threshold/onestep] {name}: {thresholds[name]:.2f}")

    # rollout uses its own threshold, tuned on val rollouts (one-step F1
    # calibration is not what keeps long rollouts alive)
    for name, model in models.items():
        rollout_thresholds[name] = tune_rollout_threshold(
            model, data["val"], cfg)
        print(f"[threshold/rollout] {name}: {rollout_thresholds[name]:.2f}")

    results: dict = {"scale": args.scale, "seed": cfg.seed,
                     "n_neurons": cfg.n_neurons, "T": cfg.T, "K": cfg.K,
                     "thresholds": thresholds,
                     "rollout_thresholds": rollout_thresholds, "models": {}}
    examples: dict = {"scatter": {}, "rollout": {}, "perturbation": {}}

    # ---------------- one-step ------------------------------------------
    for name, model in models.items():
        results["models"][name] = {"onestep": {}}
        for split in ("val", "test_seen", "test_ood"):
            res, scatter = onestep_eval(
                model, data[split], cfg, device, thresholds[name],
                collect_scatter=(split == "test_ood"))
            results["models"][name]["onestep"][split] = res
            if scatter is not None:
                examples["scatter"][name] = scatter
            print(f"[onestep] {name:12s} {split:9s} "
                  f"v_mse={res['v_mse']:.4f} (naive {res['naive_v_mse']:.4f}) "
                  f"f1={res['spike_f1']:.3f}")

    # ---------------- rollout -------------------------------------------
    horizons = list(cfg.rollout_horizons)
    for name, model in models.items():
        results["models"][name]["rollout"] = {}
        for split, key in (("test_seen", "seen"), ("test_ood", "ood")):
            m, pred, true = run_rollout_eval(
                model, data[split], cfg, horizons,
                spike_threshold=rollout_thresholds[name])
            results["models"][name]["rollout"][key] = m
            hmax = max(m)
            print(f"[rollout] {name:12s} {key:4s} h={hmax}: "
                  f"v_rmse={m[hmax]['v_rmse']:.3f} f1={m[hmax]['spike_f1']:.3f} "
                  f"pop_sim={m[hmax]['pop_similarity']:.3f}")
            if key == "ood" and len(examples["rollout"]) < 8:
                examples["rollout"][name] = {
                    "pred": pred[:1].cpu(), "true": true[:1].cpu()}

    # naive rollout reference
    K = cfg.K
    for split, key in (("test_seen", "seen"), ("test_ood", "ood")):
        d = data[split]
        n = cfg.n_rollout_traj
        context = torch.cat([d["states"][:n, :K],
                             d["stimulus"][:n, :K].unsqueeze(-1)], dim=-1)
        H = min(max(horizons), cfg.T - K)
        pred = naive_rollout(context.to(device), H, cfg)
        true = d["states"][:n, K:K + H].to(device)
        m = rollout_metrics(pred, true, [h for h in horizons if h <= H])
        results["models"].setdefault("naive", {})[f"rollout_{key}"] = m
        examples["rollout"]["naive"] = {"pred": pred[:1].cpu(),
                                        "true": true[:1].cpu()}

    # ---------------- perturbation --------------------------------------
    print("[perturbation] running intervention tests ...")
    pmetrics, pexamples = run_perturbation_eval(
        models, rollout_thresholds, sim, conn, data["test_ood"], cfg, device)
    results["perturbation"] = pmetrics
    examples["perturbation"] = pexamples
    for name, mm in pmetrics.items():
        for kind, v in mm.items():
            print(f"[perturb] {name:14s} {kind:8s} "
                  f"resp_mse_v={v['resp_mse_v']:.4f} count_corr={v['count_corr']:.3f}")

    # ---------------- save ----------------------------------------------
    def default(o):
        if isinstance(o, (torch.Tensor,)):
            return o.item()
        return str(o)

    json_path = RESULTS_DIR / f"metrics_{args.scale}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=default)

    rows = []
    for name, res in results["models"].items():
        for split, m in res.get("onestep", {}).items():
            for k, v in m.items():
                rows.append({"model": name, "eval": "onestep", "split": split,
                             "horizon": 1, "metric": k, "value": v})

        def emit_rollout(split, horizons_m):
            for h, m in horizons_m.items():
                for k, v in m.items():
                    rows.append({"model": name, "eval": "rollout",
                                 "split": split, "horizon": int(h),
                                 "metric": k, "value": v})

        for key, val in res.items():
            if key == "rollout":                # {split: {h: metrics}}
                for split, horizons_m in val.items():
                    emit_rollout(split, horizons_m)
            elif key.startswith("rollout_"):    # naive: {h: metrics}
                emit_rollout(key.split("_", 1)[1], val)
    for name, kinds in results["perturbation"].items():
        for kind, m in kinds.items():
            for k, v in m.items():
                rows.append({"model": name, "eval": f"perturb_{kind}",
                             "split": "test_ood", "horizon": cfg.perturb_rollout,
                             "metric": k, "value": v})
    csv_path = RESULTS_DIR / f"metrics_{args.scale}.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    ex_path = RESULTS_DIR / "cache" / f"eval_examples_{args.scale}.pt"
    torch.save(examples, ex_path)
    print(f"[save] {json_path}\n[save] {csv_path}\n[save] {ex_path}")

    import plots
    plots.make_all(args.scale)


if __name__ == "__main__":
    main()
