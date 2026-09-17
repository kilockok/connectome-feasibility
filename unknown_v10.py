"""v10 Section 17 sanity check: unknown (OU) under optimized probes.

OU trajectories (never a candidate): passive fits -> optimized_adaptive
probe choice (shortlist) -> execute on the OU teacher -> 4-candidate NLL.
Question: does the unknown still get force-adsorbed to a candidate with
high confidence? (v9 showed yes, mostly to GAIN.)
"""
import csv
import json
from pathlib import Path
import numpy as np
import torch
from connectome import Connectome
from run_v7 import setup_cfg
from protocol_v9 import FAMILIES, OU_TEST
from teachers_v9 import MechanismLIFSimulator, MechSpec
from common_v10 import CANDIDATES
from library_v10 import build_stim, response_window
from intervention_v9 import out_neighbors, J, TAU
from design_v10 import posterior
from design_v10b import fits_one, adaptive_choices, teacher_branch, score_windows

ROOT = Path('results/latent_state_v10')
NCTX = 32


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    cfg = setup_cfg()
    conn = Connectome.generate(cfg)
    dev = torch.device('cuda')
    lib = json.loads((ROOT / 'protocol' / 'intervention_library.json').read_text())
    short_ids = json.loads((ROOT / 'protocol' / 'configs' / 'shortlist.json').read_text())['ids']
    short_entries = [e for e in lib['entries'] if e['id'] in short_ids]
    entry_map = {e['id']: e for e in lib['entries']}
    Inb_full = out_neighbors(conn, J)
    Wcpu = conn.dense_weight('cpu')
    strength = Wcpu[J].abs().sum(0)[Inb_full]
    Inb = Inb_full[strength.argsort(descending=True)].numpy()
    g_amp = float(lib['g_amp'])
    rows = []
    for pid, p in OU_TEST:
        spec = MechSpec(name=f'ou_{pid}', ou=p)
        sim = MechanismLIFSimulator(conn, cfg, dev, spec)
        seeds = [cfg.traj_seed('test_seen', 600 + (0 if pid == 'u1' else 32) + i) for i in range(NCTX)]
        d = sim.generate(seeds, 'test_seen')
        states, stim = d['states'], d['stimulus']
        del sim
        torch.cuda.empty_cache()
        fits_list = [fits_one(states[i:i + 1], stim[i:i + 1], cfg, conn, dev) for i in range(NCTX)]
        # passive scoring
        for bi in range(NCTX):
            bank = {c: fits_list[bi][c]['bank'][0] for c in CANDIDATES}
            nll = score_windows(states[bi:bi + 1], stim[bi:bi + 1], cfg, conn, bank,
                                [(TAU, TAU + 32)], dev)
            p_post, ent = posterior(nll[None])
            win = int(nll.argmin())
            rows.append(dict(unknown=pid, policy='passive', traj=bi, probe='',
                             winner=CANDIDATES[win], conf_winner=float(p_post[0, win]),
                             entropy=float(ent[0])))
        # optimized probe per context
        chosen = adaptive_choices(states, stim, fits_list, cfg, conn, short_entries,
                                  short_ids, Inb, g_amp, dev)
        for bi in range(NCTX):
            e = entry_map[chosen[bi]]
            es = build_stim(e, cfg, Inb, g_amp)[None]
            sim = MechanismLIFSimulator(conn, cfg, dev, spec)
            dd = sim.generate([seeds[bi]], 'test_seen', extra_stim=es)
            bank = {c: fits_list[bi][c]['bank'][0] for c in CANDIDATES}
            nll = score_windows(dd['states'], dd['stimulus'], cfg, conn, bank,
                                [response_window(e)], dev)
            p_post, ent = posterior(nll[None])
            win = int(nll.argmin())
            rows.append(dict(unknown=pid, policy='optimized', traj=bi,
                             winner=CANDIDATES[win], conf_winner=float(p_post[0, win]),
                             entropy=float(ent[0]), probe=chosen[bi]))
            del sim, dd
            torch.cuda.empty_cache()
        print('ou', pid, 'done', flush=True)
    path = ROOT / 'metrics' / 'unknown_v10.csv'
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    from collections import Counter
    for pid, _ in OU_TEST:
        for pol in ('passive', 'optimized'):
            rs = [r for r in rows if r['unknown'] == pid and r['policy'] == pol]
            print(pid, pol, 'forced:', dict(Counter(r['winner'] for r in rs)),
                  'conf', np.mean([float(r['conf_winner']) for r in rs]).round(3),
                  'ent', np.mean([float(r['entropy']) for r in rs]).round(3), flush=True)
    print('UNKNOWN V10 COMPLETE')


if __name__ == '__main__':
    main()
