"""Stage 11: N=1000 scaling of the core comparison (teacher mechanism unchanged).

Only gnn_k1 / global_k32 / gshuffle / oracle x 3 seeds. Allowed adjustments per
protocol: stimulus scale (activity normalization) and batch size only. Same
alpha/beta/omega/sigma, same T=256, same split protocol.
"""
import json
from dataclasses import asdict, replace
from pathlib import Path
import torch
from run_latent_v3 import ROOT
from run_latent_v2 import setup as setup_v2
from train_latent import atomic_save
from train_latent_v2 import train_one_v2
from connectome import Connectome
from lif_latent_v2 import HiddenStateLIFSimulatorV2

LABELS = ['gnn_k1', 'global_k32', 'gshuffle', 'oracle']
SEEDS = [1234, 1235, 1236]


def scale_data(cfg, lc, conn, tag):
    path = ROOT / 'scale_n1000' / f'data_{tag}.pt'
    key = dict(config=asdict(cfg), latent=asdict(lc), protocol='pretransition_v2_n1000')
    if path.exists():
        blob = torch.load(path, map_location='cpu', weights_only=False)
        if blob['key'] != key:
            raise ValueError('scale cache mismatch')
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
        print('scale-data', tag, sp, count, 'rate', float(all_data[sp]['states'][..., 1].mean()), flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_save(dict(key=key, data=all_data), path)
    return {sp: {k: v.to('cuda') for k, v in d.items()} for sp, d in all_data.items()}


def main():
    torch.set_num_threads(2)
    cfg, lc, conn100 = setup_v2('hidden')
    # Activity pre-check on 8 trajectories with the unchanged stimulus scale.
    cfg1000 = replace(cfg, n_neurons=1000)
    conn = Connectome.generate(cfg1000)
    sim = HiddenStateLIFSimulatorV2(conn, cfg1000, torch.device('cuda'), lc)
    probe = sim.generate([cfg1000.traj_seed('val', j) for j in range(8)], 'val')
    rate = float(probe['states'][..., 1].mean())
    print('N1000 unchanged-stimulus rate:', rate, flush=True)
    stim_note = 'unchanged'
    if not 0.001 < rate < 0.15:
        # Only allowed adjustment: stimulus scale to restore activity.
        cfg1000 = replace(cfg1000, stim_min_neurons=10, stim_max_neurons=50)
        conn = Connectome.generate(cfg1000)
        sim = HiddenStateLIFSimulatorV2(conn, cfg1000, torch.device('cuda'), lc)
        probe = sim.generate([cfg1000.traj_seed('val', j) for j in range(8)], 'val')
        rate = float(probe['states'][..., 1].mean())
        stim_note = 'stim neurons x10 (10-50) to restore activity'
        print('N1000 adjusted-stimulus rate:', rate, flush=True)
    (ROOT / 'scale_n1000').mkdir(parents=True, exist_ok=True)
    (ROOT / 'scale_n1000' / 'stim_note.txt').write_text(
        f'teacher mechanism unchanged; stimulus {stim_note}; rate={rate:.5f}\n')
    data = scale_data(cfg1000, lc, conn, 'hidden')
    out = ROOT / 'scale_n1000'
    for seed in SEEDS:
        for label in LABELS:
            train_one_v2(label, seed, conn, cfg1000, lc, data, out, ROOT / 'checkpoints' / 'scale_n1000',
                         epochs=24, steps=48, batch=8)
    print('SCALE N1000 TRAINING COMPLETE', flush=True)


if __name__ == '__main__':
    main()
