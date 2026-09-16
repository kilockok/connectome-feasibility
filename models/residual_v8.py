"""v8 residual models: M0 base, M1 K1, M2 K2, M3 Set, M4 Ordered,
M5a EventSimple, M5b EventRich, M6 Oracle-STP (privileged).

All learned models share SpatialEncoderV2 + additive correction over the
differentiable base-LIF update; event features are computed from observed
history only (legal); the oracle receives the true STP-modulated current
sufficient statistic per postsynaptic neuron (privileged, separate).
"""
import torch
from torch import nn
from models.latent_temporal_v2 import SpatialEncoderV2, GlobalTemporalPredictorV2
from models.history_set_encoder import SetHistoryPredictor
from models.residual_v7 import base_pre


def event_features(x, kind):
    """Legal presynaptic event summaries from the observed window [B,K,N,4].

    EventSimple: per-neuron last-spike age (normalized by K) + recent count.
    EventRich: + last-2 ISI, spike count in last 4/8/32, and exponential
    traces with fixed time constants {2,8,32}.
    """
    s = x[..., 1]                                   # [B,K,N]
    if x.shape[1] < 2:
        s = torch.cat((s, torch.zeros_like(s[:, :1])), 1)
    B, K, N = s.shape
    dev = s.device
    steps = torch.arange(K, device=dev, dtype=s.dtype)
    active = s > .5
    # age since last spike (K if none)
    last_idx = torch.where(active, steps[None, :, None], torch.full_like(steps[None, :, None].expand(B, -1, N), -1))
    age = (K - 1) - last_idx.amax(1)
    age = age.clamp(min=0, max=K) / K
    count_recent = active[:, -8:].float().mean(1)
    if kind == 'simple':
        per_neuron = torch.stack((age, count_recent), -1)               # [B,N,2]
        return per_neuron
    # rich
    def trace(tau):
        w = torch.exp(-(K - 1 - steps) / tau)
        return (active.float() * w[None, :, None]).sum(1) / w.sum().clamp(min=1e-9)
    t2, t8, t32 = trace(2.0), trace(8.0), trace(32.0)
    # last two inter-spike intervals (approximate, K-capped)
    idx = torch.where(active, steps[None, :, None], torch.full_like(steps[None, :, None].expand(B, -1, N), -10**9))
    top2 = idx.topk(2, dim=1).values                                    # [B,2,N]
    isi1 = (top2[:, 0] - top2[:, 1]).clamp(0, K) / K
    isi2_raw = top2[:, 1]
    isi2 = torch.where(isi2_raw > -10**8, (top2[:, 1] - top2[:, 0]).clamp(0, K) / K,
                       torch.ones_like(isi2_raw))
    c4 = active[:, -4:].float().mean(1)
    c8 = active[:, -8:].float().mean(1)
    c32 = active.float().mean(1)
    per_neuron = torch.stack((age, isi1, isi2, c4, c8, c32, t2, t8, t32), -1)   # [B,N,9]
    return per_neuron


class ResidualModelV8(nn.Module):
    def __init__(self, conn, cfg, kind, d=64, layers=2):
        super().__init__()
        self.kind = kind
        self.cfg = cfg
        self.oracle = kind == 'oracle'
        if kind in ('k1', 'oracle'):
            self.k = 1
            self.spatial = SpatialEncoderV2(conn, d, layers)
            feat = d + (1 if kind == 'oracle' else 0)
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
        elif kind in ('event_simple', 'event_rich'):
            self.k = 32
            self.spatial = SpatialEncoderV2(conn, d, layers)
            nf = 2 if kind == 'event_simple' else 9
            self.ef = nn.Sequential(nn.Linear(nf, d), nn.GELU(), nn.Linear(d, d))
            feat = 2 * d
        else:
            raise ValueError(kind)
        self.head = nn.Sequential(nn.Linear(feat, d), nn.GELU(), nn.Linear(d, 3))
        W = conn.dense_weight('cpu')
        self.register_buffer('W', W)
        self.register_buffer('i_bias', conn.i_bias.cpu()
                             if conn.i_bias is not None else torch.zeros(conn.n_neurons))

    def features(self, x, stp_current=None):
        x = x[:, -self.k:]
        if self.kind == 'k1':
            return self.spatial(x[:, -1])
        if self.kind == 'oracle':
            if stp_current is None:
                raise ValueError('Oracle requires the STP sufficient statistic')
            base = self.spatial(x[:, -1])
            return torch.cat((base, stp_current[..., None]), -1)
        if self.kind == 'k2':
            return torch.cat((self.spatial(x[:, -1]), self.spatial(x[:, -2])), -1)
        if self.kind == 'set':
            fused, _ = self.set.encode(x)
            return fused
        if self.kind == 'ordered':
            fused, _ = self.g.encode(x)
            return fused
        if self.kind in ('event_simple', 'event_rich'):
            ef = self.ef(event_features(x, 'simple' if self.kind == 'event_simple' else 'rich'))
            return torch.cat((self.spatial(x[:, -1]), ef), -1)
        raise ValueError(self.kind)

    def forward(self, x, stp_current=None):
        if stp_current is not None and self.kind != 'oracle':
            raise ValueError('Hidden STP state may not enter a non-oracle model')
        f = self.features(x, stp_current)
        corr = self.head(f)
        base = base_pre(x[:, -1], self.cfg, self.W, self.i_bias)
        return dict(v=base['v_base'] + corr[..., 0],
                    s_logits=base['logit_base'] + corr[..., 1],
                    r=base['r_base'] + corr[..., 2],
                    corr=corr)


def build_v8(kind, conn, cfg):
    return ResidualModelV8(conn, cfg, kind)
