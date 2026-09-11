"""Model zoo for the connectome dynamics feasibility study."""
from __future__ import annotations

import torch.nn as nn


def build_model(name: str, cfg, connectome=None, device=None) -> nn.Module:
    from models.gru import GRUBaseline
    from models.transformer import VanillaTransformer
    from models.connectome_transformer import ConnectomeTransformer
    from models.gnn import GNNBaseline

    if name == "gru":
        model = GRUBaseline(cfg)
    elif name == "transformer":
        model = VanillaTransformer(cfg)
    elif name == "connectome":
        model = ConnectomeTransformer(cfg, connectome)
    elif name == "gnn":
        model = GNNBaseline(cfg, connectome)
    else:
        raise ValueError(f"unknown model {name}")
    if device is not None:
        model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] {name}: {n_params / 1e6:.2f}M parameters")
    return model
