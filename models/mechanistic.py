"""Mechanistic wrapper: the network predicts a membrane increment dv and the
LIF spike/reset/refractory mechanics are applied deterministically.

The wrapped phase-1 model's head channel 0 (trained to predict the next
membrane voltage) is reinterpreted as a membrane increment:

    dv = a * (base_v_out - v_t) + b        # a=1, b=0 at init (warm start:
                                           # dv == phase-1 residual V-prediction)

a, b are learnable per-neuron scalars (flag `dv_adapter`, default True) that
let phase-2 fine-tuning recalibrate the increment without moving the
backbone. S, R, resets and the refractory hold then come from
surrogate.compose_step_mechanistic, i.e. exactly the simulator's rules
applied to the last input state (v_t, r_t from x[:, -1]).

state_dict note: the base model is a submodule named `base`, so all of its
keys are prefixed "base." (plus the wrapper's own dv_a/dv_b). Checkpoints
store the wrapper's state_dict together with a "mechanistic" flag;
rollout_eval.py rebuilds the container via maybe_wrap before loading.

forward is a pure function of x (no side effects), so it is safe under
torch.utils.checkpoint.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from surrogate import compose_step_mechanistic

REFR_LOGIT = -20.0   # s_logits forced on refractory neurons -> sp == 0


class MechanisticWrapper(nn.Module):
    """Wraps a phase-1 model (GNN / ConnectomeTransformer) so that head
    channel 0 is reinterpreted as the membrane increment.

    forward(x) -> dict with the SAME keys as any model (v, s_logits, r),
    computed by the deterministic LIF rule applied to the last input state,
    plus extra keys "dv", "v_pre", "s_logits_aux", "r_aux" (the base model's
    raw spike/refractory channels; s_logits_aux is used as an auxiliary
    spike-BCE head in losses.multistep_loss).

    s_logits = (v_pre - v_th) * logit_scale (default logit_scale=8.0), so
    sigmoid(.) > 0.5  <=>  v_pre >= v_th: all phase-1 eval/threshold code
    keeps working unchanged. Refractory neurons get s_logits = REFR_LOGIT,
    hence sp = 0.
    """

    def __init__(self, base: nn.Module, cfg, logit_scale: float = 8.0,
                 dv_adapter: bool = True):
        super().__init__()
        self.base = base
        self.cfg = cfg
        self.logit_scale = float(logit_scale)
        self.is_mechanistic = True
        if dv_adapter:
            self.dv_a = nn.Parameter(torch.ones(cfg.n_neurons))
            self.dv_b = nn.Parameter(torch.zeros(cfg.n_neurons))
        else:
            # fixed identity adapter (buffers: they follow .to(device) but
            # receive no gradients)
            self.register_buffer("dv_a", torch.ones(cfg.n_neurons))
            self.register_buffer("dv_b", torch.zeros(cfg.n_neurons))

    def forward(self, x: torch.Tensor) -> dict:
        cfg = self.cfg
        out = self.base(x)
        v_t = x[:, -1, :, 0]
        r_t = x[:, -1, :, 2]
        dv = self.dv_a * (out["v"] - v_t) + self.dv_b
        # mode is irrelevant for the forward values (hard threshold); "ste"
        # only decides the surrogate gradient if this path is ever backpropagated.
        v_next, _sp, r_next, v_pre = compose_step_mechanistic(
            dv, v_t, r_t, cfg, mode="ste")
        s_logits = (v_pre - cfg.v_th) * self.logit_scale
        refr = r_t > 1e-6
        s_logits = torch.where(refr, torch.full_like(s_logits, REFR_LOGIT),
                               s_logits)
        return {"v": v_next, "s_logits": s_logits, "r": r_next,
                "dv": dv, "v_pre": v_pre,
                "s_logits_aux": out["s_logits"], "r_aux": out["r"]}


def maybe_wrap(model: nn.Module, cfg, rc) -> nn.Module:
    """MechanisticWrapper(model, cfg) if rc.mechanistic else model.
    Sets attribute `is_mechanistic` on the returned module."""
    if getattr(rc, "mechanistic", False):
        wrapped = MechanisticWrapper(model, cfg)
        wrapped.is_mechanistic = True
        return wrapped
    model.is_mechanistic = False
    return model
