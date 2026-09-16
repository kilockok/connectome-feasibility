"""v8 controlled probe: burst vs sparse history, matched silence, identical probe.

Branch A (burst): targeted pulses drive presyn neuron j at tau-24, tau-20, tau-16.
Branch B (sparse): no extra pulses.
Both then receive the same probe pulse on j at tau+d (separate branch per d).
The postsyn response to the probe carries the STP state imprint; learners
must predict it from observed history only.
"""
import csv
import json
from pathlib import Path
import numpy as np
import torch
from run_v8 import ROOT
from run_v7 import setup_cfg
from lif_stp_v8 import STPLIFSimulator, STPConfig, calibrate_norm
from latent_data import windows
from models.residual_v8 import build_v8
from calibrate_latent import write_csv

TAU = 128
DS = (0, 2, 4, 8, 16, 32)
N_TRAJ = 32
PROBE_AMP = 6.0
SEEDS = (1234, 1235, 1236, 1237, 1238)


def with_stp_current(d):
    d['stp_current'] = (d['g_path'] * d['states'][:, :-1, :, 1][..., None]).sum(2)
    return d


def build_branches(sim, cfg, j=0):
    seeds = list(range(80_000_000, 80_000_000 + N_TRAJ))
    branches = {}
    for d in DS:
        probe = {TAU + d: {j: PROBE_AMP}}
        branches[('sparse', d)] = with_stp_current(sim.generate(seeds, 'test_seen', probe=probe))
        burst_probe = dict(probe)
        for tb in (TAU - 24, TAU - 20, TAU - 16):
            burst_probe.setdefault(tb, {})[j] = PROBE_AMP
        branches[('burst', d)] = with_stp_current(sim.generate(seeds, 'test_seen', probe=burst_probe))
    return branches


@torch.no_grad()
def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', nargs='+', type=int, default=list(SEEDS))
    p.add_argument('--j', type=int, default=0)
    args = p.parse_args()
    torch.set_num_threads(2)
    cfg = setup_cfg()
    conn = __import__('connectome', fromlist=['Connectome']).Connectome.generate(cfg)
    norm = calibrate_norm(STPLIFSimulator(conn, cfg, torch.device('cuda'), STPConfig(enabled=False)), cfg)
    sim = STPLIFSimulator(conn, cfg, torch.device('cuda'), STPConfig(enabled=True), norm=norm)
    branches = build_branches(sim, cfg, args.j)
    # postsyn targets of j
    postsyn = (conn.dense_weight('cpu')[args.j] != 0).nonzero().flatten().tolist()
    rows = []
    for seed in args.seeds:
        for kind in ('k1', 'k2', 'set', 'ordered', 'event_simple', 'event_rich', 'oracle'):
            summary = json.loads((ROOT / 'natural' / 'training' / f'{kind}_seed{seed}.json').read_text())
            model = build_v8(kind, conn, cfg).cuda()
            model.load_state_dict(torch.load(summary['checkpoint'], map_location='cuda',
                                             weights_only=False)['state_dict'])
            model = model.eval()
            for (hist, d), data in branches.items():
                t = TAU + d
                sc = data['stp_current'][:, t]
                true_resp = data['states'][:, t + 1, postsyn, 0]
                for traj in range(N_TRAJ):
                    x, _ = windows(data, torch.tensor([traj], device='cuda'),
                                   torch.tensor([t], device='cuda'), model.k)
                    sc_in = sc[traj:traj + 1] if model.oracle else None
                    out = model(x, stp_current=sc_in)
                    pred_resp = out['v'][0, postsyn].cpu()
                    rows.append(dict(seed=seed, model=kind, history=hist, d=d, traj=traj,
                                     resp_rmse=float((pred_resp - true_resp[traj].cpu()).square().mean().sqrt()),
                                     resp_bias=float((pred_resp - true_resp[traj].cpu()).mean()),
                                     true_resp_mean=float(true_resp[traj].mean()),
                                     pred_resp_mean=float(pred_resp.mean())))
            print('probe done', kind, seed, flush=True)
            del model
            torch.cuda.empty_cache()
    write_csv(ROOT / 'metrics' / 'probe.csv', rows)
    print('ROWS', len(rows), flush=True)


if __name__ == '__main__':
    main()
