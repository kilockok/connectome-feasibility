"""latent_state_v2 orchestration. All compute runs locally on the CUDA host."""
import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
import torch
from connectome import Connectome
from lif_latent_v2 import HiddenStateLIFSimulatorV2, LatentV2Config
from calibrate_latent_v2 import base_config
from train_latent import atomic_save
from train_latent_v2 import train_one_v2

ROOT = Path('results/latent_state_v2')
CKPTS = ROOT / 'checkpoints'
HIDDEN_LABELS = ['gnn_k1', 'local_k16', 'local_k32', 'global_k16', 'global_k32',
                 'wide', 'gshuffle', 'glast', 'oracle']
MARKOV_LABELS = ['gnn_k1', 'global_k32', 'gshuffle', 'wide']


def setup(regime):
    selected = json.loads((ROOT / 'teacher_calibration/selection.json').read_text())
    lat = selected['latent']
    lc = LatentV2Config(alpha=lat['alpha'], beta=lat['beta'], omega=lat['omega'], sigma=lat['sigma'])
    if regime == 'markov':
        lc = replace(lc, alpha=0.)
    cfg = base_config()
    conn = Connectome.generate(cfg)
    return cfg, lc, conn


def datasets(regime, cfg, lc, conn):
    path = ROOT / 'data' / f'{regime}.pt'
    key = dict(config=asdict(cfg), latent=asdict(lc), protocol='pretransition_v2')
    if path.exists():
        blob = torch.load(path, map_location='cpu', weights_only=False)
        if blob['key'] != key:
            raise ValueError('Dataset cache config mismatch; use a new experiment namespace')
        return {sp: {k: v.to('cuda') for k, v in d.items()} for sp, d in blob['data'].items()}
    sim = HiddenStateLIFSimulatorV2(conn, cfg, torch.device('cuda'), lc)
    all_data = {}
    for sp, count in (('train', cfg.n_train_traj), ('val', cfg.n_val_traj),
                      ('test_seen', cfg.n_test_seen_traj), ('test_ood', cfg.n_test_traj)):
        chunks = []
        for i in range(0, count, 64):
            d = sim.generate([cfg.traj_seed(sp, j) for j in range(i, min(i + 64, count))], sp)
            chunks.append({k: v.cpu() for k, v in d.items()})
        all_data[sp] = {k: torch.cat([d[k] for d in chunks]) for k in chunks[0]}
        print('data', regime, sp, count, 'rate', float(all_data[sp]['states'][..., 1].mean()), flush=True)
    atomic_save(dict(key=key, data=all_data), path)
    return {sp: {k: v.to('cuda') for k, v in d.items()} for sp, d in all_data.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=['markov', 'hidden'], required=True)
    p.add_argument('--labels', nargs='+')
    p.add_argument('--seeds', nargs='+', type=int, default=[1234, 1235, 1236])
    p.add_argument('--epochs', type=int, default=24)
    p.add_argument('--steps', type=int, default=48)
    args = p.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False  # P100 has no TF32; flag kept for parity
    cfg, lc, conn = setup(args.stage)
    data = datasets(args.stage, cfg, lc, conn)
    labels = args.labels or (MARKOV_LABELS if args.stage == 'markov' else HIDDEN_LABELS)
    out = ROOT / args.stage
    out.mkdir(parents=True, exist_ok=True)
    snapshot = dict(config=asdict(cfg), latent=asdict(lc), model_seeds=args.seeds,
                    epochs=args.epochs, steps=args.steps, batch=16, labels=labels,
                    training='teacher forced, tangent off, DAgger off, long BPTT off',
                    input='X[t],U[t] before transition; target X[t+1]; same endpoint distribution for all K',
                    primary_checkpoint='best_val.pt (minimum validation state loss)',
                    hidden_mechanism='ONE global gain = 1+alpha*tanh(z_pos); z_vel affects future only',
                    z_supervision='forbidden outside the labelled oracle',
                    precision='float32', scope='N=100 controlled feasibility experiment')
    (out / 'config_snapshot.json').write_text(json.dumps(snapshot, indent=2))
    for seed in args.seeds:
        for label in labels:
            train_one_v2(label, seed, conn, cfg, lc, data, out, CKPTS / args.stage,
                         args.epochs, args.steps)
    print('TRAINING STAGE COMPLETE', args.stage, flush=True)


if __name__ == '__main__':
    main()
