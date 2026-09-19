"""v11b sparse/gated correction models (Part B).

    vn_base = exact hard-LIF pre-reset membrane potential (known formula)
    vn_pred = vn_base + g_t * deltaV_t            (0 <= g <= 1, sigmoid gate)
    spike logit = (vn_pred - v_th) / 0.1          (teacher event convention,
                                                   PRE-reset - the v11 fix)
    training: soft blend p*v_reset + (1-p)*vn_pred; eval: hard reset.

Gate inputs (deployment-legal only): current V, base distance-to-threshold,
|I_syn|, refractory, stimulus, spatial features e_t, latent z_t (unless
kind='nolatent'). No hidden teacher state / residual / spike / mechanism
mask inputs anywhere.

Variants (single-factor ablation ladder, same budget):
  m3a         plain v11 loss (no null penalty)
  m3b         + L_null: (g*deltaV)^2 on NULL-teacher windows, lambda=1
  m3c         + event-weighted V loss (w = 1 + 4*{teacher spike or
              |vn_base - v_th| < 0.2555}; threshold from TRAIN p1, frozen in
              protocol)
  m3_nolatent M3b with the k0 encoder (z = 0) - C3 control
"""
import torch
from torch import nn

from models.latent_hybrid_v11 import LatentHybridV11


class GatedHybridV11b(nn.Module):
    def __init__(self, conn, cfg, latent=True, d=64, dz=64, layers=2, heads=4):
        super().__init__()
        self.cfg = cfg
        self.backbone = LatentHybridV11(conn, cfg, kind='full' if latent else 'k0', k=32,
                                        d=d, dz=dz, layers=layers, heads=heads)
        self.latent = latent
        self.delta_head = nn.Sequential(nn.Linear(d + dz, d), nn.GELU(), nn.Linear(d, 1))
        self.gate_head = nn.Sequential(nn.Linear(d + dz + 5, d), nn.GELU(), nn.Linear(d, 1))
        W = conn.dense_weight('cpu')
        self.register_buffer('W', W)
        self.register_buffer('i_bias', conn.i_bias.cpu()
                             if conn.i_bias is not None else torch.zeros(conn.n_neurons))

    def base_vn(self, x_last):
        c = self.cfg
        v, s, r, u = x_last.unbind(-1)
        current = (s @ self.W) + u + self.i_bias
        refr = r > 0
        vn = torch.where(refr, torch.full_like(v, c.v_reset),
                         v + c.alpha * (-(v - c.v_rest) + current)).clamp(min=c.v_min)
        isyn = s @ self.W
        return vn, refr, isyn

    def forward(self, x, hard=False):
        c = self.cfg
        x = x[:, -32:]
        z_seq, e = self.backbone.latent(x)
        if not self.latent:
            z = torch.zeros(x.shape[0], self.backbone.dz, device=x.device)
        else:
            z = z_seq[:, -1]
        vn_base, refr, isyn = self.base_vn(x[:, -1])
        v, s, r, u = x[:, -1].unbind(-1)
        gfeat = torch.stack((v, vn_base - c.v_th, isyn.abs(), r, u), -1)   # [B,N,5]
        fused = torch.cat((e[:, -1], z[:, None, :].expand(-1, x.shape[2], -1)), -1)
        delta = self.delta_head(fused)[..., 0]                             # [B,N]
        g = torch.sigmoid(self.gate_head(torch.cat((fused, gfeat), -1)))[..., 0]
        corr = g * delta
        vn = vn_base + corr
        logit = (vn - c.v_th) / 0.1
        if hard:
            fire = (~refr) & (vn >= c.v_th)
            v_next = torch.where(fire, torch.full_like(vn, c.v_reset), vn)
            r_next = torch.where(fire, torch.ones_like(r),
                                 (r * c.refractory_period - 1).clamp(min=0)
                                 / c.refractory_period)
            v_next = torch.where(refr, torch.full_like(vn, c.v_reset), v_next)
        else:
            p = torch.sigmoid(logit)
            v_next = p * c.v_reset + (1 - p) * vn
            r_next = p * 1.0 + (1 - p) * (r * c.refractory_period - 1).clamp(min=0) \
                / c.refractory_period
        return dict(v=v_next, s_logits=logit, r=r_next, corr=corr, gate=g,
                    delta=delta, vn_base=vn_base, z=z_seq)


def build_v11b(variant, conn, cfg):
    return GatedHybridV11b(conn, cfg, latent=(variant != 'm3_nolatent'))
