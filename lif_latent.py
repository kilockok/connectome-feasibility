"""One hidden mechanism: stationary AR(1) global synaptic gain.

Stored X[t] is the state BEFORE U[t]; X[t+1] = F(X[t],U[t],z[t]).
The public observations contain only V, spike, refractory. z is diagnostic
teacher state and may enter an explicitly labelled oracle only.
"""
from dataclasses import dataclass
import math
import torch
from lif import LIFSimulator
from dataset import sample_traj_params, build_stimulus


@dataclass(frozen=True)
class LatentConfig:
    rho: float = .99
    alpha: float = .3
    sigma_z: float = .10

    def __post_init__(self):
        if not 0 <= self.rho < 1 or not 0 <= self.alpha < 1 or self.sigma_z < 0:
            raise ValueError('Invalid AR/gain parameters')


class HiddenStateLIFSimulator(LIFSimulator):
    def __init__(self, connectome, cfg, device, latent=LatentConfig()):
        super().__init__(connectome, cfg, device)
        self.latent = latent

    @torch.no_grad()
    def simulate(self, stimulus, v0=None, silence_mask=None, state0=None, *, z_path=None):
        """Legacy-style post-transition array, but require an explicit hidden path.

        Refuse silent fallback to the parent's unmodulated simulate method.
        Use generate() for deterministic seeds and pre-transition storage.
        """
        if z_path is None or z_path.shape != stimulus.shape[:2]:
            raise ValueError('Hidden simulation requires z_path[B,T]; use generate() for seeded trajectories')
        if state0 is None:
            v = torch.zeros(stimulus.shape[0], self.N, device=self.device) if v0 is None else v0.to(self.device)
            x = torch.stack((v, torch.zeros_like(v), torch.zeros_like(v)), -1)
        else:
            v,s,r = (t.to(self.device) for t in state0)
            x = torch.stack((v,s,r/self.cfg.refractory_period),-1)
        values=[]
        for t in range(stimulus.shape[1]):
            x=self.step(x,stimulus[:,t].to(self.device),z_path[:,t].to(self.device),silence_mask)
            values.append(x)
        return torch.stack(values,1)

    def with_lesion(self, drop_edge_mask):
        return HiddenStateLIFSimulator(self.connectome.without_edges(drop_edge_mask),
                                       self.cfg,self.device,self.latent)

    @torch.no_grad()
    def step(self, x, u, z, silence=None):
        c = self.cfg
        v, s, r = x.unbind(-1)
        gain = 1 + self.latent.alpha * torch.tanh(z)
        current = (s @ self.W) * gain[:, None] + u + self.i_bias
        refr = r > 0
        vn = torch.where(refr, torch.full_like(v, c.v_reset),
                         v + c.alpha * (-(v-c.v_rest) + current)).clamp(min=c.v_min)
        fire = (~refr) & (vn >= c.v_th)
        vn = torch.where(fire, torch.full_like(vn, c.v_reset), vn)
        # Convert to counter, exactly as the original simulator does.
        rn = torch.where(fire, torch.full_like(r, float(c.refractory_period)),
                         (r*c.refractory_period-1).clamp(min=0)) / c.refractory_period
        result = torch.stack((vn, fire.float(), rn), -1)
        if silence is not None:
            resting = torch.zeros_like(result)
            resting[..., 0] = c.v_rest
            result = torch.where(silence[..., None], resting, result)
        return result

    @torch.no_grad()
    def generate(self, seeds, split, intervention=None):
        c, lc = self.cfg, self.latent
        stimuli, initial, silenced, paths = [], [], [], []
        for seed in seeds:
            g = torch.Generator().manual_seed(int(seed))
            p = sample_traj_params(g, c, split)
            stimuli.append(build_stimulus(p, c))
            initial.append(torch.stack((torch.rand(c.n_neurons, generator=g)*c.v_th,
                                        torch.zeros(c.n_neurons), torch.zeros(c.n_neurons)), -1))
            silenced.append(p['silenced'])
            # Separate RNG: alpha=0 leaves every original visible random draw intact.
            zg = torch.Generator().manual_seed(int(seed)+91_000_000)
            noise = torch.randn(c.T+1, generator=zg)
            z = torch.empty(c.T)
            z[0] = noise[0]*lc.sigma_z/math.sqrt(1-lc.rho**2)
            for t in range(1, c.T):
                z[t] = lc.rho*z[t-1] + lc.sigma_z*noise[t]
            if intervention is not None:
                at, low, high = intervention
                z[:at] = low
                z[at:] = high
            paths.append(z)
        u = torch.stack(stimuli).to(self.device)
        x0 = torch.stack(initial).to(self.device)
        sil = torch.stack(silenced).to(self.device)
        z = torch.stack(paths).to(self.device)
        states = [x0]
        for t in range(c.T):
            states.append(self.step(states[-1], u[:, t], z[:, t], sil))
        return dict(states=torch.stack(states, 1), stimulus=u, z=z,
                    silence=sil, seeds=torch.tensor(seeds, device=self.device))
