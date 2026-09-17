"""latent_state_v9: unified parameterized candidate-mechanism teacher.

ONE simulator composing the candidate omitted mechanisms with independent
enable flags and per-family parameters:

  GAIN   global multiplicative gain from a damped noisy 2-D oscillator
         (identical equation/RNG layout to lif_latent_v2, v5/v6);
  ADAPT  per-neuron spike-triggered adaptation current (lif_adapt_v7, v7);
  STP    per-edge (u,x) short-term plasticity, 3 clusters, norm=1/U, plus
         GLOBAL scaling knobs strength s and tau_scale m:
             g = 1 + s*(u*x*norm - 1),   tau_rec/fac *= m
         (s=1, m=1 reproduces lif_stp_v8 exactly);
  OU     additive global colored latent current (stationary AR(1)) - the
         structurally distinct UNKNOWN family for open-set tests;
  NULL   everything off == base LIF exactly.

Mixtures (phase 2) enable pairs; single-mechanism configs reproduce the
legacy v7/v8 teachers bitwise (test_v9.py). Per-edge STP state stays
per-step only (never stored as a path) - the v8 15GB GPU lesson.

Step order per transition (matches lif_stp_v8/lif_adapt_v7/lif_latent_v2):
  1. g = STP edge gain (or 1)
  2. I = (S @ (W*g)) * gain + U + i_bias + I_ou
  3. vn = V + alpha*(-(V - v_rest) + I) (non-refractory), clamp v_min
  4. vn -= c_adapt * a (non-refractory)
  5. fire/reset/refractory (+silencing)
  6. hidden updates: z oscillator, a, (u,x), ou current

Hidden summaries stored per trajectory: hidden_summary [B,T,2]
(mean, std of the mechanism's hidden drive per step; zeros for NULL).
RNG layout: visible draws identical per seed across all mechanisms;
hidden noise streams use per-mechanism offsets (+91M gain / +92M adapt /
+93M ou) so mixtures never share noise.
"""
from dataclasses import dataclass, field
import math
import torch
from lif import LIFSimulator
from dataset import sample_traj_params, build_stimulus

GAIN_OFFSET = 91_000_000
ADAPT_OFFSET = 92_000_000
OU_OFFSET = 93_000_000

STP_CLUSTERS = (
    dict(name='depression', U=0.50, tau_rec=16.0, tau_fac=2.0),
    dict(name='facilitation', U=0.08, tau_rec=4.0, tau_fac=24.0),
    dict(name='mixed', U=0.25, tau_rec=8.0, tau_fac=8.0),
)


@dataclass(frozen=True)
class GainParams:
    alpha: float = 0.6
    omega: float = 0.08
    beta_vel: float = 0.99
    sigma: float = 0.01
    burn_in: int = 256
    rng_offset: int = GAIN_OFFSET


@dataclass(frozen=True)
class AdaptParams:
    beta: float = 0.3
    tau_a: float = 20.0
    c: float = 0.5
    rng_offset: int = ADAPT_OFFSET

    @property
    def rho(self):
        return math.exp(-1.0 / self.tau_a)


@dataclass(frozen=True)
class STPParams:
    strength: float = 1.0
    tau_scale: float = 1.0


@dataclass(frozen=True)
class OUParams:
    sigma: float = 0.0
    tau: float = 16.0
    burn_in: int = 256
    rng_offset: int = OU_OFFSET

    @property
    def rho(self):
        return math.exp(-1.0 / self.tau)


@dataclass(frozen=True)
class MechSpec:
    """One candidate mechanism configuration (or a mixture)."""
    name: str = 'null'
    gain: GainParams | None = None
    adapt: AdaptParams | None = None
    stp: STPParams | None = None
    ou: OUParams | None = None

    def family(self):
        fams = []
        if self.gain is not None:
            fams.append('gain')
        if self.adapt is not None:
            fams.append('adapt')
        if self.stp is not None:
            fams.append('stp')
        if self.ou is not None:
            fams.append('ou')
        return '+'.join(fams) if fams else 'null'


class MechanismLIFSimulator(LIFSimulator):
    def __init__(self, connectome, cfg, device, spec: MechSpec):
        super().__init__(connectome, cfg, device)
        self.spec = spec
        N = self.N
        if spec.stp is not None:
            W = self.W
            mask = (W != 0)
            self.edge_mask = mask
            g = torch.Generator().manual_seed(cfg.seed + 991)
            cluster = torch.randint(0, len(STP_CLUSTERS), (N, N), generator=g)
            self.cluster = torch.where(mask.cpu(), cluster,
                                       torch.full_like(cluster, -1)).to(device)
            U = torch.zeros(N, N)
            tr = torch.ones(N, N)
            tf = torch.ones(N, N)
            for ci, cl in enumerate(STP_CLUSTERS):
                m = self.cluster.cpu() == ci
                U[m] = cl['U']
                tr[m] = cl['tau_rec'] * spec.stp.tau_scale
                tf[m] = cl['tau_fac'] * spec.stp.tau_scale
            self.U = U.to(device)
            self.rho_rec = torch.exp(-1.0 / tr).to(device)
            self.rho_fac = torch.exp(-1.0 / tf).to(device)
            self.stp_norm = (1.0 / self.U.clamp(min=1e-9))

    # ---------------- hidden initial states ----------------
    def z_init_path(self, seed, gp: GainParams):
        """Exact lif_latent_v2.z_path layout: [T,2], separate RNG stream."""
        c = self.cfg
        zg = torch.Generator().manual_seed(int(seed) + gp.rng_offset)
        noise = torch.randn(c.T + gp.burn_in, generator=zg)
        z = torch.zeros(2)
        for t in range(gp.burn_in):
            z = self._z_step(z, noise[t], gp)
        path = []
        for t in range(c.T):
            z = self._z_step(z, noise[gp.burn_in + t], gp)
            path.append(z.clone())
        return torch.stack(path)

    @staticmethod
    def _z_step(z, eps, gp: GainParams):
        pos, vel = z[0], z[1]
        new_pos = pos + vel
        new_vel = gp.beta_vel * vel - gp.omega ** 2 * pos + gp.sigma * eps
        return torch.stack((new_pos.clamp(-4.0, 4.0), new_vel))

    def a_init(self, seed, ap: AdaptParams):
        g = torch.Generator().manual_seed(int(seed) + ap.rng_offset)
        rate0 = 0.02
        mean = ap.beta * rate0 / (1 - ap.rho)
        return (torch.randn(self.N, generator=g) * (mean / 2) + mean).clamp(min=0)

    def ou_path(self, seed, op: OUParams):
        c = self.cfg
        g = torch.Generator().manual_seed(int(seed) + op.rng_offset)
        eps = torch.randn(c.T + op.burn_in, generator=g)
        x = torch.zeros(())
        sd = op.sigma * math.sqrt(1 - op.rho ** 2)
        for t in range(op.burn_in):
            x = op.rho * x + sd * eps[t]
        path = []
        for t in range(c.T):
            x = op.rho * x + sd * eps[op.burn_in + t]
            path.append(x.clone())
        return torch.stack(path)

    # ---------------- core step ----------------
    @torch.no_grad()
    def step(self, x, u_stim, hidden, silence=None):
        c = self.cfg
        spec = self.spec
        v, s, r = x.unbind(-1)
        z, a, stp_state, ou = hidden
        current = s @ self.W
        uu, xx = stp_state if spec.stp is not None else (None, None)
        if spec.stp is not None and spec.stp.strength != 0:
            g_edge = (1.0 - spec.stp.strength) + spec.stp.strength * (uu * xx * self.stp_norm)
            current = torch.einsum('bj,bji->bi', s, self.W * g_edge)
        if spec.gain is not None:
            gain = 1.0 + spec.gain.alpha * torch.tanh(z[:, 0])
            current = current * gain[:, None]
        current = current + u_stim + self.i_bias
        if spec.ou is not None:
            current = current + ou[:, None]
        refr = r > 0
        vn = torch.where(refr, torch.full_like(v, c.v_reset),
                         v + c.alpha * (-(v - c.v_rest) + current)).clamp(min=c.v_min)
        if spec.adapt is not None:
            vn = vn - spec.adapt.c * a * (~refr).float()
        fire = (~refr) & (vn >= c.v_th)
        vn = torch.where(fire, torch.full_like(vn, c.v_reset), vn)
        rn = torch.where(fire, torch.full_like(r, float(c.refractory_period)),
                         (r * c.refractory_period - 1).clamp(min=0)) / c.refractory_period
        result = torch.stack((vn, fire.float(), rn), -1)
        if silence is not None:
            resting = torch.zeros_like(result)
            resting[..., 0] = c.v_rest
            result = torch.where(silence[..., None], resting, result)
        # hidden updates
        if spec.adapt is not None:
            a = spec.adapt.rho * a + spec.adapt.beta * result[..., 1]
        if spec.stp is not None:
            fire_mat = result[..., 1][:, :, None] * self.edge_mask[None].float()
            u_new = (uu + self.U * (1 - uu) * fire_mat).clamp(max=1.0)
            x_new = (xx - u_new * xx * fire_mat).clamp(min=0.0, max=1.0)
            uu = self.U + (u_new - self.U) * self.rho_fac
            xx = 1.0 + (x_new - 1.0) * self.rho_rec
            stp_state = (uu, xx)
        return result, (z, a, stp_state, ou)

    # ---------------- generation ----------------
    @torch.no_grad()
    def generate(self, seeds, split, extra_stim=None):
        """Same visible-RNG layout as the legacy teachers. extra_stim:
        optional [B,T,N] additional stimulus (controlled probe protocols)."""
        c = self.cfg
        spec = self.spec
        stimuli, initial, silenced = [], [], []
        zpaths, a0s, oupaths = [], [], []
        for seed in seeds:
            g = torch.Generator().manual_seed(int(seed))
            p = sample_traj_params(g, c, split)
            u = build_stimulus(p, c)
            stimuli.append(u)
            initial.append(torch.stack((torch.rand(c.n_neurons, generator=g) * c.v_th,
                                        torch.zeros(c.n_neurons), torch.zeros(c.n_neurons)), -1))
            silenced.append(p['silenced'])
            if spec.gain is not None:
                zpaths.append(self.z_init_path(seed, spec.gain))
            if spec.adapt is not None:
                a0s.append(self.a_init(seed, spec.adapt))
            if spec.ou is not None:
                oupaths.append(self.ou_path(seed, spec.ou))
        u = torch.stack(stimuli).to(self.device)
        if extra_stim is not None:
            u = u + extra_stim.to(self.device)
        x = torch.stack(initial).to(self.device)
        sil = torch.stack(silenced).to(self.device)
        B = len(seeds)
        z = torch.stack(zpaths).to(self.device) if spec.gain is not None else None
        a = torch.stack(a0s).to(self.device) if spec.adapt is not None else None
        ou = torch.stack(oupaths).to(self.device) if spec.ou is not None else None
        if spec.stp is not None:
            uu = self.U.clone().expand(B, -1, -1).clone()
            xx = torch.ones(B, self.N, self.N, device=self.device)
            stp_state = (uu, xx)
        else:
            stp_state = None
        states = [x]
        summary = torch.zeros(B, c.T, 2)
        for t in range(c.T):
            zt = z[:, t] if z is not None else None
            out = ou[:, t] if ou is not None else None
            x, (_, a, stp_state, _) = self.step(x, u[:, t], (zt, a, stp_state, out), sil)
            states.append(x)
            summary[:, t] = self._summary(len(seeds), zt, a, stp_state, out)
        return dict(states=torch.stack(states, 1), stimulus=u, silence=sil,
                    hidden_summary=summary,
                    seeds=torch.tensor(seeds, device=self.device))

    def _summary(self, B, z, a, stp_state, ou):
        """[B,2] per-trajectory (mean, std) of the mechanism's hidden drive."""
        spec = self.spec
        if spec.gain is not None:
            g = spec.gain.alpha * torch.tanh(z[:, 0])          # [B] (global)
            return torch.stack((g, torch.zeros_like(g)), -1).cpu()
        if spec.adapt is not None:
            return torch.stack((a.mean(1), a.std(1)), -1).cpu()
        if spec.stp is not None:
            uu, xx = stp_state
            dev = (uu * xx * self.stp_norm - 1.0)[:, self.edge_mask]  # [B,E]
            return torch.stack((dev.mean(1), dev.std(1)), -1).cpu()
        if spec.ou is not None:
            return torch.stack((ou, torch.zeros_like(ou)), -1).cpu()
        return torch.zeros(B, 2)


FAMILIES = ('null', 'gain', 'adapt', 'stp', 'ou')





