"""Per-edge short-term synaptic plasticity teacher (latent_state_v8).

ONE mechanism: per-edge STP state (u, x) following a discrete
Tsodyks-Markram-style rule. Between presynaptic spikes the state recovers
toward (U, 1) with edge-specific timescales; on a presynaptic spike the edge
contributes w*u*x*dt_current, then u facilitates and x depletes. The state is
hidden from learners; it is invisible between edge events and re-enters the
observable current only when the edge fires again. Edge heterogeneity comes
in three recorded clusters (depression / facilitation / mixed). Effective
weights are normalized per cluster so the mean current matches the base LIF
manifold (the residual is temporal, not a static rescaling).

Discrete update per step (documented as a discrete rule, not exact
continuous-time integration):
  I_i(t)     = sum_j w_ji * u_ji(t) * x_ji(t) * s_j(t) / m_cluster(ji)
  after use: u_ji <- u_ji + U_ji * (1 - u_ji)          (on spike)
             x_ji <- x_ji - u_ji * x_ji                (on spike)
  recovery:  u_ji <- U_ji + (u_ji - U_ji) * rho_fac_ji (every step)
             x_ji <- 1 + (x_ji - 1) * rho_rec_ji       (every step)
"""
from dataclasses import dataclass
import math
import torch
from lif import LIFSimulator
from dataset import sample_traj_params, build_stimulus

CLUSTERS = (
    dict(name='depression', U=0.50, tau_rec=16.0, tau_fac=2.0),
    dict(name='facilitation', U=0.08, tau_rec=4.0, tau_fac=24.0),
    dict(name='mixed', U=0.25, tau_rec=8.0, tau_fac=8.0),
)


@dataclass(frozen=True)
class STPConfig:
    enabled: bool = True
    n_clusters: int = 3


class STPLIFSimulator(LIFSimulator):
    def __init__(self, connectome, cfg, device, stp=STPConfig(), norm=None):
        super().__init__(connectome, cfg, device)
        self.stp = stp
        N = self.N
        W = self.W  # [src, dst]
        mask = (W != 0)
        self.registered_mask = mask
        g = torch.Generator().manual_seed(cfg.seed + 991)
        cluster = torch.randint(0, len(CLUSTERS), (N, N), generator=g)
        self.cluster = torch.where(mask.cpu(), cluster, torch.full_like(cluster, -1)).to(device)
        U = torch.zeros(N, N)
        tr = torch.ones(N, N)
        tf = torch.ones(N, N)
        for ci, cl in enumerate(CLUSTERS):
            m = self.cluster.cpu() == ci
            U[m] = cl['U']
            tr[m] = cl['tau_rec']
            tf[m] = cl['tau_fac']
        self.U = U.to(device)
        self.rho_rec = torch.exp(-1.0 / tr).to(device)
        self.rho_fac = torch.exp(-1.0 / tf).to(device)
        self.norm = torch.ones(N, N, device=device) if norm is None else norm.to(device)

    def stp_init(self):
        u = self.U.clone()
        x = torch.ones_like(u)
        return u, x

    @torch.no_grad()
    def step(self, x, u_stim, stp_state, silence=None):
        c = self.cfg
        v, s, r = x.unbind(-1)
        uu, xx = stp_state
        if self.stp.enabled:
            g = uu * xx * self.norm
        else:
            g = torch.ones_like(uu)
        current = torch.einsum('bj,bji->bi', s, self.W * g) + u_stim + self.i_bias
        refr = r > 0
        vn = torch.where(refr, torch.full_like(v, c.v_reset),
                         v + c.alpha * (-(v - c.v_rest) + current)).clamp(min=c.v_min)
        fire = (~refr) & (vn >= c.v_th)
        vn = torch.where(fire, torch.full_like(vn, c.v_reset), vn)
        rn = torch.where(fire, torch.full_like(r, float(c.refractory_period)),
                         (r * c.refractory_period - 1).clamp(min=0)) / c.refractory_period
        result = torch.stack((vn, fire.float(), rn), -1)
        if silence is not None:
            resting = torch.zeros_like(result)
            resting[..., 0] = c.v_rest
            result = torch.where(silence[..., None], resting, result)
        if self.stp.enabled:
            # fire_mat[b,j,i] = 1 where presynaptic neuron j fired and edge j->i exists.
            fire_mat = fire.float()[:, :, None] * self.registered_mask[None].float()
            u_new = (uu + self.U * (1 - uu) * fire_mat).clamp(max=1.0)
            x_new = (xx - u_new * xx * fire_mat).clamp(min=0.0, max=1.0)
            uu = self.U + (u_new - self.U) * self.rho_fac
            xx = 1.0 + (x_new - 1.0) * self.rho_rec
        return result, (uu, xx)

    @torch.no_grad()
    def generate(self, seeds, split, probe=None):
        """Same visible-RNG layout as the v2/v7 teachers. probe: optional dict
        with extra targeted stimulus pulses {t: {neuron: amplitude}} for the
        controlled-probe experiments (dynamics-legal spike induction)."""
        c = self.cfg
        stimuli, initial, silenced = [], [], []
        for seed in seeds:
            g = torch.Generator().manual_seed(int(seed))
            p = sample_traj_params(g, c, split)
            u = build_stimulus(p, c)
            if probe is not None:
                for t, pulses in probe.items():
                    for neuron, amp in pulses.items():
                        u[t, neuron] += amp
            stimuli.append(u)
            initial.append(torch.stack((torch.rand(c.n_neurons, generator=g) * c.v_th,
                                        torch.zeros(c.n_neurons), torch.zeros(c.n_neurons)), -1))
            silenced.append(p['silenced'])
        u = torch.stack(stimuli).to(self.device)
        x = torch.stack(initial).to(self.device)
        sil = torch.stack(silenced).to(self.device)
        uu, xx = self.stp_init()
        uu, xx = uu.expand(x.shape[0], -1, -1).clone(), xx.expand(x.shape[0], -1, -1).clone()
        states, upath, xpath, gpath = [x], [], [], []
        for t in range(c.T):
            x, (uu, xx) = self.step(x, u[:, t], (uu, xx), sil)
            states.append(x)
            upath.append(uu.clone()); xpath.append(xx.clone())
            gpath.append((uu * xx * self.norm * self.registered_mask.float()).clone())
        return dict(states=torch.stack(states, 1), stimulus=u,
                    u_path=torch.stack(upath, 1), x_path=torch.stack(xpath, 1),
                    g_path=torch.stack(gpath, 1), silence=sil,
                    cluster=self.cluster.cpu(),
                    seeds=torch.tensor(seeds, device=self.device))


def calibrate_norm(sim, cfg, seeds=(70_000_000, 70_000_001, 70_000_002, 70_000_003),
                   iterations=0):
    """norm = 1/U per edge: at sparse events (x ~ 1, u ~ U) the event-mean
    STP current matches the base manifold (mean g ~ 1), with temporal
    modulation from event history. Frozen; no rate-dependent iteration."""
    return (1.0 / sim.U.clamp(min=1e-9)).cpu()
