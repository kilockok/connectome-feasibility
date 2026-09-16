"""Second-order hidden-state teacher (latent_state_v2).

ONE hidden mechanism only: a global synaptic gain driven by a damped, noisy
two-dimensional oscillator (z_pos, z_vel).

    z_pos[t+1] = z_pos[t] + dt_z * z_vel[t]
    z_vel[t+1] = beta * z_vel[t] - omega^2 * dt_z * z_pos[t] + sigma * eps[t]
    gain[t]    = 1 + alpha * tanh(z_pos[t])
    W_eff[t]   = gain[t] * W

The velocity component makes temporal ORDER causally relevant at the teacher
level: two branches sharing the same observable state and the same z_pos but
opposite z_vel have the same instantaneous gain yet diverge in the near
future. Unordered window statistics cannot capture the sign of z_vel.

Stored X[t] is the state BEFORE U[t]; X[t+1] = F(X[t], U[t], z_pos[t]).
The public observations contain only V, spike, refractory and the stimulus.
z is diagnostic teacher state and may enter an explicitly labelled oracle only.
"""
from dataclasses import dataclass
import torch
from lif import LIFSimulator
from dataset import sample_traj_params, build_stimulus


@dataclass(frozen=True)
class LatentV2Config:
    alpha: float = .6       # gain = 1 + alpha*tanh(z_pos)
    beta: float = .99       # velocity damping (0<beta<1 keeps the oscillator bounded)
    omega: float = .08      # oscillator rate, rad/step
    sigma: float = .01      # innovation std on z_vel
    dt_z: float = 1.0
    z_clip: float = 4.0     # numerical safety bound; essentially never active when damped
    burn_in: int = 256      # oscillator burn-in so z starts near its stationary law
    random_init: bool = False   # deterministic-ablation mode: randomized initial z,
                                # then noise-free evolution (use with sigma=0)
    init_std_pos: float = 0.77  # ~ stationary std of the stochastic teacher
    init_std_vel: float = 0.046

    def __post_init__(self):
        if not 0 <= self.alpha < 1.5 or not 0 < self.beta < 1 or self.omega <= 0 \
                or self.sigma < 0 or self.dt_z <= 0:
            raise ValueError('Invalid second-order latent parameters')
        # Jury stability of A=[[1,dt],[-omega^2*dt,beta]]: det=beta+omega^2*dt<1.
        if self.omega ** 2 * self.dt_z >= 1 - self.beta:
            raise ValueError('Unstable oscillator: need omega^2*dt_z < 1-beta')


class HiddenStateLIFSimulatorV2(LIFSimulator):
    def __init__(self, connectome, cfg, device, latent=LatentV2Config()):
        super().__init__(connectome, cfg, device)
        self.latent = latent

    def z_step(self, z, noise):
        """One exact oscillator update; noise is eps ~ N(0,1) matching z[..., 0]."""
        lc = self.latent
        pos, vel = z[..., 0], z[..., 1]
        new_pos = pos + lc.dt_z * vel
        new_vel = lc.beta * vel - lc.omega ** 2 * lc.dt_z * pos + lc.sigma * noise
        new_pos = new_pos.clamp(-lc.z_clip, lc.z_clip)
        return torch.stack((new_pos, new_vel), -1)

    def z_path(self, seed, intervention=None):
        """Deterministic oscillator path [T,2] from its own RNG stream.

        alpha=0 leaves every visible random draw of the original simulator
        untouched because the oscillator consumes a separate generator.
        """
        lc, c = self.latent, self.cfg
        zg = torch.Generator().manual_seed(int(seed) + 91_000_000)
        noise = torch.randn(c.T + lc.burn_in, generator=zg)
        if lc.random_init:
            z = torch.tensor([torch.randn((), generator=zg) * lc.init_std_pos,
                              torch.randn((), generator=zg) * lc.init_std_vel])
        else:
            z = torch.zeros(2)
        for t in range(0 if lc.random_init else lc.burn_in):
            z = self.z_step(z, noise[t])
        path = []
        for t in range(c.T):
            z = self.z_step(z, noise[lc.burn_in + t])
            if intervention is not None and t == intervention[0]:
                z = apply_intervention(z, intervention)
            path.append(z.clone())
        return torch.stack(path)

    def gain(self, z):
        return 1 + self.latent.alpha * torch.tanh(z[..., 0])

    @torch.no_grad()
    def step(self, x, u, z, silence=None):
        """Exact same transition as latent_state_v1, gain driven by z[..., 0]."""
        c = self.cfg
        v, s, r = x.unbind(-1)
        gain = self.gain(z)
        current = (s @ self.W) * gain[:, None] + u + self.i_bias
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
        return result

    @torch.no_grad()
    def simulate(self, stimulus, v0=None, silence_mask=None, state0=None, *, z_path=None):
        """Legacy-style post-transition array; requires an explicit [B,T,2] z path."""
        if z_path is None or z_path.shape[:2] != stimulus.shape[:2] or z_path.shape[-1] != 2:
            raise ValueError('Hidden simulation requires z_path[B,T,2]; use generate() for seeded trajectories')
        if state0 is None:
            v = torch.zeros(stimulus.shape[0], self.N, device=self.device) if v0 is None else v0.to(self.device)
            x = torch.stack((v, torch.zeros_like(v), torch.zeros_like(v)), -1)
        else:
            v, s, r = (t.to(self.device) for t in state0)
            x = torch.stack((v, s, r / self.cfg.refractory_period), -1)
        values = []
        for t in range(stimulus.shape[1]):
            x = self.step(x, stimulus[:, t].to(self.device), z_path[:, t].to(self.device), silence_mask)
            values.append(x)
        return torch.stack(values, 1)

    def with_lesion(self, drop_edge_mask):
        return HiddenStateLIFSimulatorV2(self.connectome.without_edges(drop_edge_mask),
                                         self.cfg, self.device, self.latent)

    @torch.no_grad()
    def generate(self, seeds, split, intervention=None):
        """Same visible-RNG layout as latent_state_v1.generate; z is [B,T,2]."""
        c = self.cfg
        stimuli, initial, silenced, paths = [], [], [], []
        for seed in seeds:
            g = torch.Generator().manual_seed(int(seed))
            p = sample_traj_params(g, c, split)
            stimuli.append(build_stimulus(p, c))
            initial.append(torch.stack((torch.rand(c.n_neurons, generator=g) * c.v_th,
                                        torch.zeros(c.n_neurons), torch.zeros(c.n_neurons)), -1))
            silenced.append(p['silenced'])
            paths.append(self.z_path(seed, intervention))
        u = torch.stack(stimuli).to(self.device)
        x0 = torch.stack(initial).to(self.device)
        sil = torch.stack(silenced).to(self.device)
        z = torch.stack(paths).to(self.device)
        states = [x0]
        for t in range(c.T):
            states.append(self.step(states[-1], u[:, t], z[:, t], sil))
        return dict(states=torch.stack(states, 1), stimulus=u, z=z,
                    silence=sil, seeds=torch.tensor(seeds, device=self.device))


def apply_intervention(z, intervention):
    """In-place-time latent intervention applied at a single step.

    kind        effect at the intervention step
    vel_flip    z_vel <- -z_vel                     (velocity sign flip)
    phase_jump  z_pos <- z_pos + value              (phase jump)
    regime      z <- (value_pos, value_vel)         (latent regime switch)
    """
    _, kind, *value = intervention
    z = z.clone()
    if kind == 'vel_flip':
        z[1] = -z[1]
    elif kind == 'phase_jump':
        z[0] = z[0] + float(value[0])
    elif kind == 'regime':
        z[0], z[1] = float(value[0]), float(value[1])
    else:
        raise ValueError(f'Unknown intervention kind: {kind}')
    return z
