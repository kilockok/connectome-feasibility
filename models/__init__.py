"""Model zoo for the connectome dynamics feasibility study."""
from __future__ import annotations

import torch.nn as nn


def build_model(name: str, cfg, connectome=None, device=None,
                model_kwargs: dict | None = None) -> nn.Module:
    """Build a model by name. `model_kwargs` carries architecture options
    for the phase-3 models (gnn_temporal: k_hist/t_layers/...; gnn_wide:
    d_model/gnn_layers). Unknown keys are rejected by the constructors."""
    from models.gru import GRUBaseline
    from models.transformer import VanillaTransformer
    from models.connectome_transformer import ConnectomeTransformer
    from models.gnn import GNNBaseline

    model_kwargs = dict(model_kwargs or {})
    if name == "gru":
        model = GRUBaseline(cfg)
    elif name == "transformer":
        model = VanillaTransformer(cfg)
    elif name == "connectome":
        model = ConnectomeTransformer(cfg, connectome)
    elif name == "gnn":
        model = GNNBaseline(cfg, connectome)
    elif name == "gnn_temporal":
        from models.gnn_temporal import GNNTemporalTransformer
        model = GNNTemporalTransformer(cfg, connectome, **model_kwargs)
    elif name == "gnn_wide":
        # parameter-matched plain GNN control: same class as `gnn`, built
        # with a wider/deeper cfg view (d_model / gnn_layers overrides).
        from dataclasses import replace
        allowed = {"d_model", "gnn_layers"}
        bad = set(model_kwargs) - allowed
        if bad:
            raise ValueError(f"gnn_wide accepts only {allowed}, got {bad}")
        model = GNNBaseline(replace(cfg, **model_kwargs), connectome)
    else:
        raise ValueError(f"unknown model {name}")
    if device is not None:
        model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    suffix = f" {model_kwargs}" if model_kwargs else ""
    print(f"[model] {name}{suffix}: {n_params / 1e6:.3f}M parameters")
    return model


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def match_gnn_wide(cfg, connectome, target_params: int,
                   d_choices=(96, 128, 160, 192, 224, 256, 320),
                   layer_choices=(3, 4, 5, 6, 8)) -> dict:
    """Grid-search (d_model, gnn_layers) for the plain GNNBaseline whose
    parameter count best matches `target_params` (the gnn_temporal model).
    Tie-break: fewer parameters. Builds throwaway CPU models — cheap."""
    from dataclasses import replace
    from models.gnn import GNNBaseline

    best = None
    for d in d_choices:
        for L in layer_choices:
            m = GNNBaseline(replace(cfg, d_model=d, gnn_layers=L), connectome)
            n = count_params(m)
            diff = abs(n - target_params)
            if best is None or diff < best[0] - 1e-9 \
                    or (diff == best[0] and n < best[1]):
                best = (diff, n, {"d_model": d, "gnn_layers": L})
            del m
    diff, n, kw = best
    print(f"[param-match] target={target_params / 1e6:.3f}M -> "
          f"gnn_wide {kw} = {n / 1e6:.3f}M (diff {diff / 1e6:.3f}M)")
    return kw
