"""Shared loss and metric functions (train.py / evaluate.py / rollout.py)."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_loss(out: dict, y: torch.Tensor, cfg,
                 pos_weight: torch.Tensor) -> tuple[torch.Tensor, dict]:
    lv = F.mse_loss(out["v"], y[..., 0])
    ls = F.binary_cross_entropy_with_logits(
        out["s_logits"], y[..., 1], pos_weight=pos_weight)
    lr_ = F.mse_loss(out["r"], y[..., 2])
    total = cfg.lambda_v * lv + cfg.lambda_s * ls + cfg.lambda_r * lr_
    return total, {"loss": total.item(), "v_loss": lv.item(),
                   "s_loss": ls.item(), "r_loss": lr_.item()}


@torch.no_grad()
def compute_metrics(out: dict, y: torch.Tensor, auroc: bool = True,
                    threshold: float = 0.5) -> dict:
    """One-step metrics against ground-truth next state y [B, N, 3]."""
    v_true, s_true, r_true = y[..., 0], y[..., 1], y[..., 2]
    s_prob = torch.sigmoid(out["s_logits"])
    s_pred = (s_prob > threshold).float()

    m = {}
    m["v_mse"] = F.mse_loss(out["v"], v_true).item()
    m["v_rmse"] = m["v_mse"] ** 0.5
    m["r_mse"] = F.mse_loss(out["r"], r_true).item()

    tp = (s_pred * s_true).sum().item()
    fp = (s_pred * (1 - s_true)).sum().item()
    fn = ((1 - s_pred) * s_true).sum().item()
    m["spike_precision"] = tp / max(tp + fp, 1.0)
    m["spike_recall"] = tp / max(tp + fn, 1.0)
    p, r = m["spike_precision"], m["spike_recall"]
    m["spike_f1"] = 2 * p * r / max(p + r, 1e-9)
    m["spike_rate_true"] = s_true.mean().item()
    m["spike_rate_pred"] = s_pred.mean().item()

    if auroc:
        try:
            from sklearn.metrics import roc_auc_score
            st = s_true.flatten().cpu().numpy()
            sp = s_prob.flatten().cpu().numpy()
            # subsample for speed; AUROC needs both classes present
            if len(st) > 500_000:
                idx = torch.randperm(len(st))[:500_000].numpy()
                st, sp = st[idx], sp[idx]
            m["spike_auroc"] = float(roc_auc_score(st, sp)) if st.any() \
                and not st.all() else float("nan")
        except Exception:
            m["spike_auroc"] = float("nan")
    return m


def naive_baseline_metrics(x: torch.Tensor, y: torch.Tensor) -> dict:
    """x[t+1] = x[t] reference: predict last context state (V,S,R)."""
    out = {"v": x[:, -1, :, 0], "r": x[:, -1, :, 2],
           # map binary last spike to a logit-ish score for shared code
           "s_logits": (x[:, -1, :, 1] * 2 - 1) * 10.0}
    return compute_metrics(out, y, auroc=False)
