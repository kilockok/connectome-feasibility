"""Scientific invariants; run on the remote host before training."""
from dataclasses import replace
import torch
from config import get_config
from connectome import Connectome
from dataset import generate_batch
from lif import LIFSimulator
from lif_latent import HiddenStateLIFSimulator, LatentConfig
from latent_data import windows, history_control, sample_indices
from models.latent_temporal import LatentPredictor


def main():
    torch.set_num_threads(2)
    cfg = replace(get_config('small'), n_neurons=20, T=64, K=32, silence_prob=0)
    conn = Connectome.generate(cfg)
    dev = torch.device('cpu')
    sim = HiddenStateLIFSimulator(conn, cfg, dev)
    seeds = [cfg.traj_seed('train', i) for i in range(3)]
    data = sim.generate(seeds, 'train')
    assert torch.equal(data['states'], sim.generate(seeds, 'train')['states'])
    print('[PASS] trajectory reproducibility')
    zero = HiddenStateLIFSimulator(conn, cfg, dev, LatentConfig(alpha=0))
    old = generate_batch(seeds, 'train', LIFSimulator(conn, cfg, dev), cfg)
    assert torch.allclose(zero.generate(seeds, 'train')['states'][:, 1:], old['states'], atol=1e-6)
    print('[PASS] alpha=0 agrees with original LIF')
    silcfg=replace(cfg,silence_prob=1.)
    silsim=HiddenStateLIFSimulator(conn,silcfg,dev,LatentConfig(alpha=0))
    silold=generate_batch(seeds,'train',LIFSimulator(conn,silcfg,dev),silcfg)
    assert torch.allclose(silsim.generate(seeds,'train')['states'][:,1:],silold['states'],atol=1e-6)
    try:
        sim.simulate(data['stimulus'])
        raise AssertionError('Hidden simulator silently discarded z')
    except ValueError:
        pass
    b, t = torch.tensor([0, 1]), torch.tensor([32, 40])
    x, y = windows(data, b, t, 32)
    altered = dict(data, z=data['z']*100+9)
    assert torch.equal(x, windows(altered, b, t, 32)[0])
    assert x.shape == (2, 32, 20, 4)
    assert torch.equal(sim.step(x[:, -1, :, :3], x[:, -1, :, 3], data['z'][b, t]), y)
    print('[PASS] z does not enter input; exact transition/stimulus alignment')
    for k in (1, 8, 16, 32):
        m = LatentPredictor(conn, k=k, d=16).eval()
        out = m(x)
        assert all(v.shape == (2, 20) and torch.isfinite(v).all() for v in out.values())
    print('[PASS] K=1/8/16/32 finite forward and correct shape')
    m = LatentPredictor(conn, d=16).eval()
    xp = x.clone()
    xp[:, 16:] += 7
    with torch.no_grad():
        a = m.encode(x, return_sequence=True)[1]
        bseq = m.encode(xp, return_sequence=True)[1]
    assert torch.allclose(a[:, :16], bseq[:, :16], atol=1e-6)
    print('[PASS] future observations cannot affect earlier temporal tokens')
    ordered = torch.arange(2*32*20*4).reshape(2, 32, 20, 4)
    shuffled = history_control(ordered, 'shuffle', torch.Generator().manual_seed(2))
    assert torch.equal(ordered[:, -1], shuffled[:, -1])
    assert not torch.equal(ordered, shuffled)
    assert sorted(ordered[0, :, 0, 0].tolist()) == sorted(shuffled[0, :, 0, 0].tolist())
    for j in range(32):
        src = int(shuffled[0, j, 0, 0]//80)
        assert torch.equal(shuffled[:, j], ordered[:, src])
    print('[PASS] shuffle preserves current token and whole timestep content')
    assert all(torch.equal(a, b) for a, b in zip(sample_indices(data, 20, 17), sample_indices(data, 20, 17)))
    pools = [{cfg.traj_seed(sp, i) for i in range(512)} for sp in ('train','val','test_seen','test_ood')]
    assert all(not a.intersection(b) for i,a in enumerate(pools) for b in pools[i+1:])
    print('[PASS] identical comparison sampling; disjoint trajectory splits')
    with torch.no_grad():
        feature = m(x, return_features=True)[1].mean(1)
    before = {k: v.clone() for k,v in m.state_dict().items()}
    probe = torch.nn.Linear(feature.shape[-1], 1)
    probe(feature).square().mean().backward()
    assert all(p.grad is None for p in m.parameters())
    assert all(torch.equal(v, before[k]) for k,v in m.state_dict().items())
    try:
        m(x, torch.zeros(2))
        raise AssertionError('z accepted')
    except ValueError:
        pass
    print('[PASS] frozen probe features and non-oracle rejection of z')
    # Identical observations, distinct hidden states, distinct futures.
    xx = data['states'][:, 10].clone()
    xx[..., 1] = 1
    xx[..., 2] = 0
    lo = sim.step(xx, torch.zeros_like(xx[..., 0]), torch.full((3,), -1.))
    hi = sim.step(xx, torch.zeros_like(xx[..., 0]), torch.full((3,), 1.))
    assert not torch.equal(lo, hi)
    print('[PASS] same observation, different hidden state, different future')
    from eval_latent import rollout_prediction, rollout_summary
    class ExactTeacher(torch.nn.Module):
        k=8
        oracle=True
        def forward(self,x,z=None):
            y=sim.step(x[:,-1,:,:3],x[:,-1,:,3],z)
            return dict(v=y[...,0],s_logits=y[...,1]*40-20,r=y[...,2])
    prediction,truth=rollout_prediction(ExactTeacher(),data,cfg,.5,n=3,horizon=20)
    assert torch.equal(prediction,truth)
    print('[PASS] exact-teacher closed loop matches trajectory (future stimulus alignment)')
    truth=torch.zeros(2,10,20,3);prediction=truth.clone()
    truth[0,:,0,1]=1
    prediction[0,:2,0,1]=1
    prediction[1,4:7,:,1]=1
    result=rollout_summary(prediction,truth,(5,10))
    assert result['failures'][0]['low']==3 and result['failures'][1]['high']==5
    assert result['failures'][0]['high'] is None
    print('[PASS] low/high sustained failure onsets and zero-activity conventions')
    from latent_probe import fit_probe,apply_probe
    f=torch.randn(200,8,generator=torch.Generator().manual_seed(11))
    z=2*f[:,0]-f[:,1]+.7
    probe=fit_probe((f[:100],z[:100]),(f[100:150],z[100:150]))
    assert (apply_probe(probe,f[150:])-z[150:]).abs().mean()<.01
    print('[PASS] frozen ridge probe recovers a known held-out linear latent')
    print('ALL LATENT INVARIANTS PASSED')


if __name__ == '__main__':
    main()
