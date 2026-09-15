"""latent_state_v2 invariants. Run before main training."""
import torch
from dataclasses import replace
from config import Config
from connectome import Connectome
from lif import LIFSimulator
from dataset import generate_batch
from lif_latent_v2 import HiddenStateLIFSimulatorV2, LatentV2Config, apply_intervention
from latent_data import windows, history_control
from models.latent_temporal_v2 import build_v2


def main():
    cfg = Config(seed=1234, n_neurons=100, T=64)
    conn = Connectome.generate(cfg)
    dev = torch.device('cuda')
    seeds = [101, 202, 303]

    sim = HiddenStateLIFSimulatorV2(conn, cfg, dev, LatentV2Config())
    a, b = sim.generate(seeds, 'train'), sim.generate(seeds, 'train')
    assert torch.equal(a['states'], b['states']) and torch.equal(a['z'], b['z'])
    assert a['z'].shape == (3, 64, 2)
    print('[PASS] v2 trajectory reproducibility; z is [B,T,2]')

    zero = HiddenStateLIFSimulatorV2(conn, cfg, dev, LatentV2Config(alpha=0.))
    old = generate_batch(seeds, 'train', LIFSimulator(conn, cfg, dev), cfg)
    assert torch.allclose(zero.generate(seeds, 'train')['states'][:, 1:], old['states'], atol=1e-6)
    print('[PASS] alpha=0 exactly equals the original LIF')

    try:
        LatentV2Config(omega=0.2, beta=0.99)
        raise SystemExit('[FAIL] unstable oscillator accepted')
    except ValueError:
        pass
    z = a['z']
    assert z[..., 0].abs().max() <= 4.0 and z[..., 0].std() > 0.3
    print('[PASS] unstable oscillator rejected; z bounded and non-degenerate')

    # z enters only through gain: same z_pos, different z_vel -> identical one-step
    x = a['states'][:2, 10]
    u = a['stimulus'][:2, 10]
    z1 = torch.tensor([[0.5, 1.0], [0.5, -1.0]], device=dev)
    z2 = torch.tensor([[0.5, -1.0], [0.5, 1.0]], device=dev)
    xa = torch.stack([sim.step(x[i:i+1], u[i:i+1], z1[i:i+1])[0] for i in range(2)])
    xb = torch.stack([sim.step(x[i:i+1], u[i:i+1], z2[i:i+1])[0] for i in range(2)])
    # same (x,u), same z_pos, different z_vel -> identical immediate transition
    assert torch.allclose(xa, xb)
    print('[PASS] one-step transition depends on z_pos only (z_vel acts on future)')

    # but the future diverges under opposite velocities
    za = torch.tensor([0.0, 0.5], device=dev).expand(2, 2)
    zb = torch.tensor([0.0, -0.5], device=dev).expand(2, 2)
    xa, xb = x, x
    for h in range(1, 24):
        za, zb = sim.z_step(za, torch.zeros(2, device=dev)), sim.z_step(zb, torch.zeros(2, device=dev))
        xa, xb = sim.step(xa, a['stimulus'][:2, 10 + h], za), sim.step(xb, a['stimulus'][:2, 10 + h], zb)
    assert not torch.allclose(xa[..., 0], xb[..., 0], atol=1e-4)
    print('[PASS] opposite z_vel branches diverge within 24 steps')

    # interventions
    assert torch.equal(apply_intervention(torch.tensor([1.0, 2.0]), (0, 'vel_flip')), torch.tensor([1.0, -2.0]))
    assert torch.equal(apply_intervention(torch.tensor([1.0, 2.0]), (0, 'phase_jump', 0.5)), torch.tensor([1.5, 2.0]))
    assert torch.equal(apply_intervention(torch.tensor([1.0, 2.0]), (0, 'regime', -1.0, -2.0)), torch.tensor([-1.0, -2.0]))
    d = sim.generate(seeds, 'train', (32, 'vel_flip'))
    assert torch.equal(d['z'][:, :32], a['z'][:, :32]) and not torch.equal(d['z'][:, 32:], a['z'][:, 32:])
    print('[PASS] vel_flip / phase_jump / regime interventions apply at the right step')

    data = sim.generate(seeds[:2], 'train')
    bidx, tidx = torch.tensor([0, 1]), torch.tensor([32, 40])
    x, y = windows(data, bidx, tidx, 32)
    altered = dict(data, z=data['z'] * 100 + 9)
    x2, y2 = windows(altered, bidx, tidx, 32)
    assert torch.equal(x, x2) and torch.equal(y, y2)
    print('[PASS] windows never read z; targets are X[t+1]')

    for lab in ('gnn_k1', 'local_k16', 'local_k32', 'global_k16', 'global_k32', 'wide', 'gshuffle', 'glast'):
        m, _ = build_v2(lab, conn)
        m = m.to(dev).eval()
        with torch.no_grad():
            out = m(x)
        assert all(torch.isfinite(v).all() for v in out.values())
        try:
            m(x, z=torch.randn(2, 2, device=dev))
            raise SystemExit(f'[FAIL] {lab} accepted z')
        except ValueError:
            pass
    print('[PASS] all non-oracle models reject z; finite forward for every label')

    mo, _ = build_v2('oracle', conn)
    mo = mo.to(dev).eval()
    try:
        mo(x)
        raise SystemExit('[FAIL] oracle ran without z')
    except ValueError:
        pass
    with torch.no_grad():
        out = mo(x, z=torch.randn(2, 2, device=dev))
    assert torch.isfinite(out['v']).all()
    print('[PASS] oracle requires and uses explicit 2-D z')

    # causality for both temporal paths
    for lab in ('local_k32', 'global_k32'):
        m, _ = build_v2(lab, conn)
        m = m.to(dev).eval()
        x1 = torch.randn(1, 32, 100, 4, device=dev)
        x2 = x1.clone(); x2[:, 16:] = 99.
        with torch.no_grad():
            _, s1 = m.encode(x1, return_sequence=True)
            _, s2 = m.encode(x2, return_sequence=True)
        assert torch.allclose(s1[:, :16], s2[:, :16], atol=1e-5)
    print('[PASS] causal mask: future tokens cannot affect earlier positions')

    # global pooling makes the global context identical-per-neuron by construction
    m, _ = build_v2('global_k32', conn)
    m = m.to(dev).eval()
    with torch.no_grad():
        fused, zctx = m.encode(x)
    assert zctx.shape == (2, 64)
    print('[PASS] global context shape [B,D]; broadcast fused with per-neuron codes')

    print('ALL V2 INVARIANTS PASSED')


if __name__ == '__main__':
    main()
