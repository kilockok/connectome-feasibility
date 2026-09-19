"""v11b M2: residual shrinkage - val-selected scaling of M1's corrections.

(gamma_v, gamma_s) from the frozen grid {0,.25,.5,.75,1}^2, selected to
minimize VAL V-RMSE (seed 8001 windows). Applied to M1 checkpoints
post-hoc: v = v_base + gamma_v*corr_v; s_logits = logit_base + gamma_s*corr_s;
r = r_base + gamma_v*corr_r. Writes configs/m2_shrink.json.
"""
import json
from pathlib import Path
import numpy as np
import torch
from connectome import Connectome
from run_v7 import setup_cfg
from latent_data import sample_indices, windows
from models.latent_hybrid_v11 import build_v11
from data_v11 import load_pool

ROOT = Path('results/latent_state_v11b')
SEEDS = (1234, 1235, 1236, 1237, 1238)
GRID = (0.0, 0.25, 0.5, 0.75, 1.0)


@torch.no_grad()
def main():
    cfg = setup_cfg()
    conn = Connectome.generate(cfg)
    dev = torch.device('cuda')
    pool = load_pool(cfg)
    val = {k: v.to(dev) for k, v in pool['val'].items()}
    out_json = {}
    for seed in SEEDS:
        s11 = json.loads(Path(f'results/latent_state_v11/metrics/training/full_seed{seed}.json').read_text())
        m1 = build_v11('full', conn, cfg).to(dev)
        m1.load_state_dict(torch.load(s11['checkpoint'], map_location=dev, weights_only=False)['state_dict'])
        m1 = m1.eval()
        bi, ti = sample_indices(val, 1024, 8001)
        xs, ys = [], []
        for off in range(0, len(bi), 64):
            x, y = windows(val, bi[off:off + 64], ti[off:off + 64], 32)
            xs.append(x); ys.append(y)
        best = None
        for gv in GRID:
            for gs in GRID:
                errs = []
                for x, y in zip(xs, ys):
                    out = m1(x)
                    v = (out['v'] - out['corr'][..., 0]) + gv * out['corr'][..., 0]
                    errs.append(float((v - y[..., 0]).square().mean()))
                rmse = float(np.mean(errs)) ** 0.5
                if best is None or rmse < best[0]:
                    best = (rmse, gv, gs)
        out_json[str(seed)] = dict(val_v_rmse=best[0], gamma_v=best[1], gamma_s=best[2])
        print('seed', seed, 'selected', out_json[str(seed)], flush=True)
        del m1
        torch.cuda.empty_cache()
    (ROOT / 'configs').mkdir(parents=True, exist_ok=True)
    (ROOT / 'configs' / 'm2_shrink.json').write_text(json.dumps(out_json, indent=1))
    print('M2 SELECT COMPLETE', flush=True)


if __name__ == '__main__':
    main()
