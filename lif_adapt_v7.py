"""Minimal spike-triggered adaptation teacher (latent_state_v7).

ONE mechanism only: per-neuron adaptation current a (hidden), decaying with
one timescale, incremented by the neuron's own spikes, subtracting from the
pre-threshold voltage of non-refractory neurons. The v2-v6 gain latent is
OFF. Synthetic mechanism test; not a validated biological model.

Ordering per step (explicit):
  1. current = (s @ W) + u + i_bias  (observed spikes, known graph)
  2. refractory: pre-threshold value = v_reset; non-refractory:
     vn = V + alpha*(-(V - v_rest) + current), clamped at v_min
  3. adaptation: vn -= c * a   (non-refractory neurons only)
  4. threshold: fire = (~refr) & (vn >= v_th); fired -> V = v_reset, R refills
  5. a <- rho * a + beta * fire   (hidden update, autonomous)
c is in voltage units. The post-reset state (V,S,R) is the only observation.
"""
from dataclasses import dataclass
import math
import torch
from lif import LIFSimulator
from dataset import sample_traj_params, build_stimulus


@dataclass(frozen=True)
class AdaptConfig:
    c: float = 0.5          # voltage per unit adaptation
    tau_a: float = 20.0     # steps
    beta: float = 0.3       # increment per spike

    def __post_init__(self):
        if self.c < 0 or self.tau_a <= 0 or self.beta < 0:
            raise ValueError('Invalid adaptation parameters')

    @property
    def rho(self):
        return math.exp(-1.0 / self.tau_a)


class AdaptationLIFSimulator(LIFSimulator):
    def __init__(self, connectome, cfg, device, adapt=AdaptConfig()):
        super().__init__(connectome, cfg, device)
        self.adapt = adapt

    @torch.no_grad()
    def step(self, x, u, a, silence=None):
        c = self.cfg
        ad = self.adapt
        v, s, r = x.unbind(-1)
        current = (s @ self.W) + u + self.i_bias
        refr = r > 0
        vn = torch.where(refr, torch.full_like(v, c.v_reset),
                         v + c.alpha * (-(v - c.v_rest) + current)).clamp(min=c.v_min)
        vn = vn - ad.c * a * (~refr).float()
        fire = (~refr) & (vn >= c.v_th)
        vn = torch.where(fire, torch.full_like(vn, c.v_reset), vn)
        rn = torch.where(fire, torch.full_like(r, float(c.refractory_period)),
                         (r * c.refractory_period - 1).clamp(min=0)) / c.refractory_period
        result = torch.stack((vn, fire.float(), rn), -1)
        if silence is not None:
            resting = torch.zeros_like(result)
            resting[..., 0] = c.v_rest
            result = torch.where(silence[..., None], resting, result)
            fire_out = result[..., 1]
        else:
            fire_out = fire.float()
        a_new = ad.rho * a + ad.beta * fire_out
        return result, a_new

    def a_init(self, seed):
        """Approximate steady-state init, per (seed, neuron), separate RNG."""
        g = torch.Generator().manual_seed(int(seed) + 91_000_000)
        rate0 = 0.02
        mean = self.adapt.beta * rate0 / (1 - self.adapt.rho)
        a0 = torch.randn(self.N, generator=g) * (mean / 2) + mean
        return a0.clamp(min=0)

    @torch.no_grad()
    def generate(self, seeds, split, intervention=None):
        """Same visible-RNG layout as the v2 teacher: c=0 reproduces the
        original LIF data exactly (a consumes a separate generator)."""
        c = self.cfg
        stimuli, initial, silenced, a0s = [], [], [], []
        for seed in seeds:
            g = torch.Generator().manual_seed(int(seed))
            p = sample_traj_params(g, c, split)
            stimuli.append(build_stimulus(p, c))
            initial.append(torch.stack((torch.rand(c.n_neurons, generator=g) * c.v_th,
                                        torch.zeros(c.n_neurons), torch.zeros(c.n_neurons)), -1))
            silenced.append(p['silenced'])
            a0s.append(self.a_init(seed))
        u = torch.stack(stimuli).to(self.device)
        x0 = torch.stack(initial).to(self.device)
        sil = torch.stack(silenced).to(self.device)
        a = torch.stack(a0s).to(self.device)
        states, a_path = [x0], []
        for t in range(c.T):
            if intervention is not None and t == intervention[0]:
                kind = intervention[1]
                if kind == 'a_set':
                    a = torch.full_like(a, float(intervention[2]))
                elif kind == 'a_scale':
                    a = a * float(intervention[2])
                elif kind == 'sham':
                    pass
                else:
                    raise ValueError(kind)
            x, a = self.step(states[-1], u[:, t], a, sil)
            states.append(x)
            a_path.append(a)
        return dict(states=torch.stack(states, 1), stimulus=u, a=torch.stack(a_path, 1),
                    silence=sil, seeds=torch.tensor(seeds, device=self.device))


@torch.no_grad()
def adaptation_reference(data, sim):
    """Mechanism-known reference (declared advantage: exact equation + params).

    Back out a[t] from each completed non-reset transition: e[t] = V[t+1] -
    F_base_pre(x_t,u_t) = -c*a[t] on free neurons; recurse a_hat with the
    known rho/beta; on reset or boundary steps carry the decayed estimate.
    Uses only completed transitions plus declared mechanism knowledge.
    """
    c = sim.cfg
    ad = sim.adapt
    S = data['states'][:, :-1, :, 1]
    V = data['states'][:, :-1, :, 0]
    R = data['states'][:, :-1, :, 2]
    U = data['stimulus']
    current = (S @ sim.W) + U + sim.i_bias
    refr = R > 0
    vn = torch.where(refr, torch.full_like(V, c.v_reset),
                     V + c.alpha * (-(V - c.v_rest) + current)).clamp(min=c.v_min)
    fire_base = (~refr) & (vn >= c.v_th)
    free = (~refr) & (~fire_base)
    e = data['states'][:, 1:, :, 0] - vn  # ~ -c*a on free neurons
    B, T, N = e.shape
    a_hat = torch.zeros(B, T, N, device=e.device)
    a_prev = torch.zeros(B, N, device=e.device)
    for t in range(T):
        est = (-e[:, t] / ad.c)
        a_now = torch.where(free[:, t], est, a_prev)
        a_hat[:, t] = a_now
        a_prev = ad.rho * a_now + ad.beta * data['states'][:, t + 1, :, 1]
    return a_hat
