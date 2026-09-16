"""latent_state_v4 driver: N scaling and N_obs observability sweep."""
import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
import torch
from run_latent_v2 import setup as setup_v2
from train_latent import atomic_save
from train_latent_v4 import train_one_v4
from connectome import Connectome
from lif_latent_v2 import HiddenStateLIFSimulatorV2

ROOT = Path('results/latent_state_v4')
CKPTS = ROOT / 'checkpoints'
CORE = ['gnn_k1', 'set_k32', 'global_k32', 'oracle']
SECONDARY = ['gshuffle', 'wide', 'stats_k32', 'deriv']


def base_data(n, cfg, lc):
    """N=100 reuses the v2 cache; N=1000 the v3 scale cache; others generate."""
    if n == 100:
        blob = torch.load('results/latent_state_v2/data/hidden.pt', map_location='cpu', weights_only=False)
        return {sp: {k: v.to('cuda') for k, v in d.items()} for sp, d in blob['data'].items()}, 'v2-cache'
    if n == 1000:
        blob = torch.load('results/latent_state_v3/scale_n1000/data_hidden.pt',
                          map_location='cpu', weights_only=False)
        return {sp: {k: v.to('cuda') for k, v in d.items()} for sp, d in blob['data'].items()}, 'v3-scale-cache'
    cfgN = replace(cfg, n_neurons=n)
    conn = Connectome.generate(cfgN)
    path = ROOT / 'data' / f'n{n}.pt'
    key = dict(config=asdict(cfgN), latent=asdict(lc), protocol='pretransition_v4')
    if path.exists():
        blob = torch.load(path, map_location='cpu', weights_only=False)
        if blob['key'] != key:
            raise ValueError(f'cache mismatch n={n}')
        return {sp: {k: v.to('cuda') for k, v in d.items()} for sp, d in blob['data'].items()}, 'v4-cache'
    sim = HiddenStateLIFSimulatorV2(conn, cfgN, torch.device('cuda'), lc)
    all_data = {}
    for sp, count in (('train', cfgN.n_train_traj), ('val', cfgN.n_val_traj),
                      ('test_seen', cfgN.n_test_seen_traj), ('test_ood', cfgN.n_test_traj)):
        chunks = []
        for i in range(0, count, 64):
            d = sim.generate([cfgN.traj_seed(sp, j) for j in range(i, min(i + 64, count))], sp)
            chunks.append({k: v.cpu() for k, v in d.items()})
        all_data[sp] = {k: torch.cat([d[k] for d in chunks]) for k in chunks[0]}
        print('v4-data', n, sp, count, 'rate', float(all_data[sp]['states'][..., 1].mean()), flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_save(dict(key=key, data=all_data), path)
    return {sp: {k: v.to('cuda') for k, v in d.items()} for sp, d in all_data.items()}, 'v4-generated'


def obs_subset(conn, m, protocol, seed=0):
    if m >= conn.n_neurons:
        return torch.arange(conn.n_neurons)
    if protocol == 'random':
        return torch.randperm(conn.n_neurons, generator=torch.Generator().manual_seed(seed))[:m].sort().values
    deg = torch.zeros(conn.n_neurons)
    deg.index_add_(0, conn.edge_index[0], torch.ones(conn.n_edges))
    deg.index_add_(0, conn.edge_index[1], torch.ones(conn.n_edges))
    order = deg.argsort()
    ranks = torch.linspace(0, conn.n_neurons - 1, m).round().long()
    return order[ranks].sort().values


def subset_connectome(conn, obs):
    obs = torch.as_tensor(obs, dtype=torch.long)
    keep = torch.isin(conn.edge_index[0], obs) & torch.isin(conn.edge_index[1], obs)
    ei = conn.edge_index[:, keep]
    remap = torch.full((conn.n_neurons,), -1, dtype=torch.long)
    remap[obs] = torch.arange(len(obs))
    ei = remap[ei]
    return Connectome(len(obs), ei, conn.edge_weight[keep], conn.neuron_type[obs],
                      conn.i_bias[obs] if conn.i_bias is not None else None)


def slice_split(d, obs):
    return dict(states=d['states'][..., obs, :], stimulus=d['stimulus'][..., obs],
                z=d['z'], silence=d['silence'][:, obs], seeds=d['seeds'])


def slice_data(data, obs):
    """Accepts either a single split dict or the full {split: dict} mapping."""
    dev = data['states'].device if 'states' in data else next(iter(data.values()))['states'].device
    obs = obs.to(dev)
    if 'states' in data:
        return slice_split(data, obs)
    return {sp: slice_split(d, obs) for sp, d in data.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, required=True)
    p.add_argument('--n-obs', type=int, default=None)
    p.add_argument('--protocol', choices=['degree', 'random'], default='degree')
    p.add_argument('--labels', nargs='+')
    p.add_argument('--seeds', nargs='+', type=int, default=[1234, 1235, 1236])
    p.add_argument('--epochs', type=int, default=24)
    p.add_argument('--steps', type=int, default=48)
    p.add_argument('--batch', type=int, default=None)
    args = p.parse_args()
    torch.set_num_threads(2)
    cfg, lc, _ = setup_v2('hidden')
    batch = args.batch if args.batch is not None else (8 if args.n >= 1000 else 16)
    data, source = base_data(args.n, cfg, lc)
    cfgN = replace(cfg, n_neurons=args.n)
    conn = Connectome.generate(cfgN)
    tag = f'n{args.n}'
    if args.n_obs is not None:
        obs = obs_subset(conn, args.n_obs, args.protocol)
        conn = subset_connectome(conn, obs)
        data = slice_data(data, obs)
        tag = f'n1000obs{args.n_obs}_{args.protocol}'
        (ROOT / 'data').mkdir(parents=True, exist_ok=True)
        torch.save(obs.cpu(), ROOT / 'data' / f'obs_{tag}.pt')
    out = ROOT / tag
    out.mkdir(parents=True, exist_ok=True)
    snapshot = dict(n=args.n, n_obs=args.n_obs, protocol=args.protocol, data_source=source,
                    labels=args.labels or CORE, seeds=args.seeds, epochs=args.epochs,
                    steps=args.steps, batch=batch, teacher='v2 hidden unchanged')
    (out / 'config_snapshot.json').write_text(json.dumps(snapshot, indent=2))
    for seed in args.seeds:
        for label in (args.labels or CORE):
            train_one_v4(label, seed, conn, cfgN, lc, data, out, CKPTS / tag,
                         args.epochs, args.steps, batch)
    print('V4 STAGE COMPLETE', tag, flush=True)


if __name__ == '__main__':
    main()
