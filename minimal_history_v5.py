"""Stage 8: minimal history on the strict-alias subset."""
import csv
from pathlib import Path
import numpy as np
import torch
from run_latent_v3 import datasets_v3
from run_latent_v2 import setup as setup_v2
from alias_eval_v5 import load_checked
from latent_data import windows
from lif_latent_v2 import HiddenStateLIFSimulatorV2

ROOT = Path('results/latent_state_v5')
SEEDS = (1234, 1235, 1236, 1237, 1238)
KS = (1, 2, 4, 8, 16, 32)


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    cfg, lc, conn = setup_v2('hidden')
    data = datasets_v3('hidden')['test_seen']
    sim = HiddenStateLIFSimulatorV2(conn, cfg, torch.device('cuda'), lc)
    pairs_blob = torch.load(ROOT / 'alias_pairs' / 'pairs.pt', weights_only=False)
    pairs = [r for r in pairs_blob['rows'] if r['tier'] == 'strict']
    rows = []
    for seed in SEEDS:
        model, th = load_checked('global_k32', seed, conn, torch.device('cuda'))
        for k_eff in KS:
            correct = []
            gains = []
            for rec in pairs:
                (ta, t) = map(int, rec['A'].split(':'))
                (tb, s) = map(int, rec['B'].split(':'))
                out = {}
                for tag, (tr, tt) in (('A', (ta, t)), ('B', (tb, s))):
                    x, _ = windows(data, torch.tensor([tr], device='cuda'), torch.tensor([tt], device='cuda'), model.k)
                    x = x[:, -k_eff:]
                    xa, ua = data['states'][tr, tt], data['stimulus'][tr, tt]
                    za = data['z'][tr, tt]
                    Da = sim.step(xa[None], ua[None], za[None], data['silence'])[0]
                    Ba = sim.step(xa[None], ua[None], torch.zeros_like(za[None]), data['silence'])[0]
                    isyn = xa[:, 1] @ sim.W
                    free = (xa[:, 2] <= 0) & (Ba[:, 1] <= .5) & (Da[:, 1] <= .5)
                    v = model(x)['v'][0]
                    dh = v - Ba[:, 0]
                    ge = float((dh[free] * isyn[free]).sum() / (isyn[free] * isyn[free]).sum().clamp(min=1e-12)) / cfg.alpha + 1.0
                    out[tag] = (ge, float(sim.gain(za[None])))
                (geA, gA), (geB, gB) = out['A'], out['B']
                correct.append(float(abs(geA - gA) + abs(geB - gB) < abs(geA - gB) + abs(geB - gA)))
                gains.append((geA, gA, geB, gB))
            rows.append(dict(seed=seed, k_eff=k_eff, gain_acc=float(np.mean(correct)),
                             gain_mae=float(np.mean([abs(a - b) + abs(c - d) for a, b, c, d in gains])) / 2))
        print('minimal-K done seed', seed, flush=True)
        del model
        torch.cuda.empty_cache()
    with (ROOT / 'table7_minimal_history.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print('ROWS', len(rows), flush=True)


if __name__ == '__main__':
    main()
