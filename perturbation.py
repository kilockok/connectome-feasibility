"""Perturbation / lesion experiments.

Protocol per OOD test trajectory (context ends at t0 = K):
  1. Ground truth: continue the LIF simulation from the true state at t0
     for H steps, once intact and once with an intervention
     (neuron silencing or edge lesion).  Delta_gt = int - base.
  2. Model: autoregressive rollout from the true context, once intact and
     once with the same intervention enforced (silenced neuron's state
     overridden each step; for the connectome transformer an edge lesion
     is applied by rebuilding the attention bias from the lesioned graph).
     Delta_model = int - base.
  3. Compare downstream responses: MSE of Delta V, Pearson correlation of
     per-neuron spike-count deltas.  The zero-delta reference (a model
     that ignores the intervention) is reported for calibration.

Non-connectome models have no way to represent an edge lesion; they score
the zero reference by construction (that is the point of the experiment).
"""
from __future__ import annotations

import torch

from connectome import Connectome
from lif import LIFSimulator
from rollout import rollout


def _pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().float().cpu()
    b = b.flatten().float().cpu()
    a = a - a.mean(); b = b - b.mean()
    den = a.norm() * b.norm()
    return (a @ b / den).item() if den > 1e-9 else 0.0


@torch.no_grad()
def _model_rollout(model, data, b, cfg, H, threshold, silence_extra=None,
                   lesioned_conn=None):
    """One-trajectory rollout for trajectory b, optionally intervened."""
    K, N = cfg.K, cfg.n_neurons
    context = torch.cat([
        data["states"][b:b + 1, :K],
        data["stimulus"][b:b + 1, :K].unsqueeze(-1)], dim=-1)
    future = data["stimulus"][b:b + 1, K:K + H]
    sil = data["silence"][b:b + 1].clone()
    if silence_extra is not None:
        sil = sil.clone()
        sil[0, silence_extra] = True
    if lesioned_conn is not None and hasattr(model, "rebuild_bias_from_connectome"):
        backup = (model.conn_mask.clone(), model.conn_logw.clone(),
                  model.conn_edge.clone())
        model.rebuild_bias_from_connectome(lesioned_conn)
        pred = rollout(model, context, future, cfg, spike_threshold=threshold,
                       silence_mask=sil)
        model.conn_mask, model.conn_logw, model.conn_edge = backup
    else:
        pred = rollout(model, context, future, cfg, spike_threshold=threshold,
                       silence_mask=sil)
    return pred[0]                                       # [H, N, 3]


@torch.no_grad()
def run_perturbation_eval(models: dict[str, torch.nn.Module],
                          thresholds: dict[str, float],
                          sim: LIFSimulator, conn: Connectome,
                          data: dict, cfg, device) -> tuple[dict, dict]:
    """Returns (metrics, examples). See module docstring for the protocol."""
    K, N, H = cfg.K, cfg.n_neurons, cfg.perturb_rollout
    g = torch.Generator().manual_seed(cfg.seed + 555)
    n_traj = min(cfg.n_perturb_traj, data["states"].shape[0])

    # ---------------- pre-compute GT continuations ----------------------
    gt_base, gt_sil, gt_les, tests = [], [], [], []
    for b in range(n_traj):
        st = data["states"][b:b + 1, K]                  # [1, N, 3]
        state0 = (st[..., 0], st[..., 1], st[..., 2] * cfg.refractory_period)
        stim_future = data["stimulus"][b:b + 1, K:K + H]
        sil0 = data["silence"][b:b + 1]
        base = sim.simulate(stim_future, silence_mask=sil0, state0=state0)[0]

        stim_ids = data["meta"][b]["stim_ids"]
        X = int(stim_ids[0])                             # a stimulated neuron
        # strongest target of X that is downstream-active in the baseline
        out_edges = (conn.edge_index[0] == X).nonzero().flatten()
        lesioned_conn, lesion_pair = None, None
        if out_edges.numel() > 0:
            w_abs = conn.edge_weight[out_edges].abs()
            order = torch.argsort(w_abs, descending=True)
            act = base[..., 1].sum(dim=0)                # [N]
            tgt = None
            for e in out_edges[order].tolist():
                j = int(conn.edge_index[1, e])
                if act[j] > 0:
                    tgt, lesion_pair = j, (X, j)
                    break
            if tgt is None:
                e = int(out_edges[order[0]])
                tgt, lesion_pair = int(conn.edge_index[1, e]), (X, int(conn.edge_index[1, e]))
            drop = torch.zeros(conn.n_edges, dtype=torch.bool)
            drop[(conn.edge_index[0] == X) & (conn.edge_index[1] == tgt)] = True
            lesioned_conn = conn.without_edges(drop)

        sil_x = sil0.clone(); sil_x[0, X] = True
        gt_sil.append(sim.simulate(stim_future, silence_mask=sil_x,
                                   state0=state0)[0])
        if lesioned_conn is not None:
            sim_les = sim.with_lesion((conn.edge_index[0] == X) &
                                      (conn.edge_index[1] == tgt))
            gt_les.append(sim_les.simulate(stim_future, silence_mask=sil0,
                                           state0=state0)[0])
        else:
            gt_les.append(None)
        gt_base.append(base)
        tests.append({"traj": b, "silence_x": X, "lesion_pair": lesion_pair,
                      "has_lesion": lesioned_conn is not None})

    # ---------------- model comparison ----------------------------------
    metrics: dict[str, dict] = {}
    examples: dict[str, dict] = {}
    for name, model in models.items():
        model.eval()
        agg = {"silence": [], "lesion": []}
        for t in tests:
            b, X = t["traj"], t["silence_x"]
            base_pred = _model_rollout(model, data, b, cfg, H,
                                       thresholds.get(name, 0.5))
            sil_pred = _model_rollout(model, data, b, cfg, H,
                                      thresholds.get(name, 0.5),
                                      silence_extra=X)
            d_gt = gt_sil[b][..., 0] - gt_base[b][..., 0]          # [H,N] V
            d_md = sil_pred[..., 0] - base_pred[..., 0]
            mse_v = ((d_md - d_gt) ** 2).mean().item()
            c_gt = (gt_sil[b][..., 1] - gt_base[b][..., 1]).sum(dim=0)
            c_md = (sil_pred[..., 1] - base_pred[..., 1]).sum(dim=0)
            agg["silence"].append({"resp_mse_v": mse_v,
                                   "count_corr": _pearson(c_md, c_gt)})
            if name not in examples and b == 0:
                examples[name] = {
                    "silence_x": X,
                    "gt_rate_base": gt_base[b][..., 1].mean(dim=1).cpu(),
                    "gt_rate_int": gt_sil[b][..., 1].mean(dim=1).cpu(),
                    "md_rate_base": base_pred[..., 1].mean(dim=1).cpu(),
                    "md_rate_int": sil_pred[..., 1].mean(dim=1).cpu(),
                    "gt_count_delta": c_gt.cpu(), "md_count_delta": c_md.cpu()}

            if t["has_lesion"] and gt_les[b] is not None:
                if hasattr(model, "rebuild_bias_from_connectome"):
                    ei = conn.edge_index
                    drop = (ei[0] == t["lesion_pair"][0]) & \
                           (ei[1] == t["lesion_pair"][1])
                    les_pred = _model_rollout(
                        model, data, b, cfg, H, thresholds.get(name, 0.5),
                        lesioned_conn=conn.without_edges(drop))
                    d_md_l = les_pred[..., 0] - base_pred[..., 0]
                    c_md_l = (les_pred[..., 1] - base_pred[..., 1]).sum(dim=0)
                else:      # structurally unable to represent the lesion
                    d_md_l = torch.zeros_like(d_gt)
                    c_md_l = torch.zeros_like(c_gt)
                d_gt_l = gt_les[b][..., 0] - gt_base[b][..., 0]
                c_gt_l = (gt_les[b][..., 1] - gt_base[b][..., 1]).sum(dim=0)
                agg["lesion"].append({
                    "resp_mse_v": ((d_md_l - d_gt_l) ** 2).mean().item(),
                    "count_corr": _pearson(c_md_l, c_gt_l)})

        metrics[name] = {}
        for kind in ("silence", "lesion"):
            if agg[kind]:
                metrics[name][kind] = {
                    "resp_mse_v": sum(a["resp_mse_v"] for a in agg[kind]) / len(agg[kind]),
                    "count_corr": sum(a["count_corr"] for a in agg[kind]) / len(agg[kind]),
                    "n_tests": len(agg[kind])}

    # zero-delta reference (model ignores the intervention)
    zero = {"silence": [], "lesion": []}
    for t in tests:
        b = t["traj"]
        d = gt_sil[b][..., 0] - gt_base[b][..., 0]
        zero["silence"].append((d ** 2).mean().item())
        if t["has_lesion"] and gt_les[b] is not None:
            d = gt_les[b][..., 0] - gt_base[b][..., 0]
            zero["lesion"].append((d ** 2).mean().item())
    metrics["zero_reference"] = {
        k: {"resp_mse_v": sum(v) / len(v), "count_corr": 0.0, "n_tests": len(v)}
        for k, v in zero.items() if v}
    return metrics, examples
