"""v7 residual models: shared spatial encoder, additive correction over base.

Every learned model predicts next state as (base-LIF pre-reset computation)
+ (general head over its history features). No adaptation formula in any
learned head; no a labels in the main loss. K2 tests whether the last
transition suffices; SetHistory tests unordered history; Ordered reuses the
verified global temporal path. The true-a oracle is privileged.
"""
import torch
from torch import nn
from models.latent_temporal_v2 import SpatialEncoderV2, GlobalTemporalPredictorV2
from models.history_set_encoder import SetHistoryPredictor


def base_pre(x_last, cfg, W, i_bias):
    """Differentiable base-LIF update of the last token [V,S,R,U]."""
    c = cfg
    v, s, r, u = x_last.unbind(-1)
    current = (s @ W) + u + i_bias
    refr = r > 0
    vn = torch.where(refr, torch.full_like(v, c.v_reset),
                     v + c.alpha * (-(v - c.v_rest) + current)).clamp(min=c.v_min)
    p = torch.sigmoid((vn - c.v_th) / 0.1)
    v_base = p * c.v_reset + (1 - p) * vn
    r_base = p * 1.0 + (1 - p) * (r * c.refractory_period - 1).clamp(min=0) / c.refractory_period
    return dict(v_base=v_base, logit_base=(vn - c.v_th) / 0.1, r_base=r_base)


class ResidualModel(nn.Module):
    """Shared encoder + additive residual head over the base computation."""

    def __init__(self, conn, cfg, kind, d=64, layers=2):
        super().__init__()
        self.kind = kind
        self.cfg = cfg
        self.oracle = kind == 'oracle'
        if kind == 'k1':
            self.k = 1
            self.spatial = SpatialEncoderV2(conn, d, layers)
            feat = d
        elif kind == 'k2':
            self.k = 2
            self.spatial = SpatialEncoderV2(conn, d, layers)
            feat = 2 * d
        elif kind == 'set':
            self.k = 32
            self.set = SetHistoryPredictor(conn, k=32, d=d, layers=layers)
            feat = 2 * d
        elif kind == 'ordered':
            self.k = 32
            self.g = GlobalTemporalPredictorV2(conn, k=32, d=d, layers=layers)
            feat = 2 * d
        elif kind == 'oracle':
            self.k = 1
            self.spatial = SpatialEncoderV2(conn, d, layers)
            feat = d + 1
        else:
            raise ValueError(kind)
        self.head = nn.Sequential(nn.Linear(feat, d), nn.GELU(), nn.Linear(d, 3))
        W = conn.dense_weight('cpu')
        self.register_buffer('W', W)
        self.register_buffer('i_bias', conn.i_bias.cpu()
                             if conn.i_bias is not None else torch.zeros(conn.n_neurons))

    def features(self, x, a=None):
        x = x[:, -self.k:]
        if self.kind == 'k1':
            return self.spatial(x[:, -1])
        if self.kind == 'k2':
            return torch.cat((self.spatial(x[:, -1]), self.spatial(x[:, -2])), -1)
        if self.kind == 'set':
            fused, _ = self.set.encode(x)
            return fused
        if self.kind == 'ordered':
            fused, _ = self.g.encode(x)
            return fused
        if self.kind == 'oracle':
            if a is None:
                raise ValueError('Oracle requires true a')
            base = self.spatial(x[:, -1])
            return torch.cat((base, a[..., None]), -1)
        raise ValueError(self.kind)

    def forward(self, x, a=None):
        if a is not None and self.kind != 'oracle':
            raise ValueError('Hidden a may not enter a non-oracle model')
        f = self.features(x, a)
        corr = self.head(f)
        base = base_pre(x[:, -1], self.cfg, self.W, self.i_bias)
        return dict(v=base['v_base'] + corr[..., 0],
                    s_logits=base['logit_base'] + corr[..., 1],
                    r=base['r_base'] + corr[..., 2],
                    corr=corr)


def build_v7(kind, conn, cfg):
    return ResidualModel(conn, cfg, kind)
