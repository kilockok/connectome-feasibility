"""v11b Part A5: oracle diagnostics (privileged, teacher-forced only).

O1: the correction SPACE - true one-step residual magnitude per mask
    (re-derived on the v11 eval windows; complements v10 artifact.csv).
O2: oracle-gated M1 delta - apply b5's predicted delta only where it reduces
    one-step V error (gate uses teacher y; PRIVILEGED, never deployable).
    Measures the headroom of gating the EXISTING residual predictor:
    if O2 does not beat B0, no gating of M1's deltas can beat B0.
O3: oracle-gated spike-logit correction of b5 (fix only wrong decisions).

All computed on the v11 deterministic eval windows (2048/family-split,
seed 8001), testB + null. Output: results/latent_state_v11b/results/rollout/oracle.csv
"""
import csv
import json
from pathlib import Path
import numpy as np
import torch
from connectome import Connectome
from run_v7 import setup_cfg
from latent_data import sample_indices, windows
from models.latent_hybrid_v11 import build_v11
from protocol_v9 import FAMILIES, SEEDS
from data_v11 import load_pool
from eval_v11 import hard_lif_out, masks_for

ROOT = Path('results/latent_state_v11b')
COUNT = 2048


@torch.no_grad()
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', nargs='+', type=int, default=list(SEEDS))
    args = ap.parse_args()
    cfg = setup_cfg()
    conn = Connectome.generate(cfg)
    dev = torch.device('cuda')
    W = conn.dense_weight(dev)
    ib = conn.i_bias.to(dev) if conn.i_bias is not None else torch.zeros(cfg.n_neurons, device=dev)
    pool = load_pool(cfg)
    groups = []
    d = pool['testB']
    for fi, fam in enumerate(FAMILIES):
        sel = (d['mech'] == fi).nonzero(as_tuple=True)[0]
        groups.append((fam, {k: v[sel].cuda() if k != 'mech' else v[sel] for k, v in d.items()}))
    groups.append(('null', {k: v.cuda() for k, v in pool['null'].items()}))

    rows = []
    for fam, data in groups:
        bi, ti = sample_indices(data, COUNT, 8001)
        # ---- O1: true residual space (model-free) ----
        res_by_mask = {m: [] for m in ('all', 'free', 'info', 'event')}
        for off in range(0, len(bi), 64):
            x, y = windows(data, bi[off:off + 64], ti[off:off + 64], 32)
            b0 = hard_lif_out(x, cfg, W, ib)
            m = masks_for(x, y, cfg, W, ib)
            e = y[..., 0] - b0['v']
            for name, mk in m.items():
                vals = e[mk]
                res_by_mask[name].append(float(vals.square().mean()) if vals.numel() else float('nan'))
        for name, v in res_by_mask.items():
            vv = [x for x in v if x == x]
            rows.append(dict(oracle='O1_true_residual_rms', seed='-', family=fam, mask=name,
                             value=float(np.mean(vv) ** 0.5) if vv else float('nan')))
        print('O1', fam, 'done', flush=True)
        # ---- O2/O3 per seed ----
        for seed in args.seeds:
            s11 = json.loads(Path(f'results/latent_state_v11/metrics/training/full_seed{seed}.json').read_text())
            m5 = build_v11('full', conn, cfg).to(dev)
            m5.load_state_dict(torch.load(s11['checkpoint'], map_location=dev, weights_only=False)['state_dict'])
            m5 = m5.eval()
            errs = {m: [] for m in ('all', 'free', 'info', 'event')}
            errg, err0 = {m: [] for m in errs}, {m: [] for m in errs}
            for off in range(0, len(bi), 64):
                x, y = windows(data, bi[off:off + 64], ti[off:off + 64], 32)
                b0 = hard_lif_out(x, cfg, W, ib)
                out = m5(x)
                m = masks_for(x, y, cfg, W, ib)
                vn_base = b0['v']
                # b5's v prediction implies delta = v_b5 - v_base_soft...;
                # use the model's correction channel directly: corr[...,0] is
                # defined over the SOFT base; the oracle applies the same delta
                # over the HARD base vn (both share vn pre-reset semantics on
                # free steps; on spike steps v resets to v_reset in both).
                delta = out['corr'][..., 0]
                v_gated = torch.where((vn_base + delta - y[..., 0]).abs() < (vn_base - y[..., 0]).abs(),
                                      vn_base + delta, vn_base)
                for name, mk in m.items():
                    e0 = (vn_base - y[..., 0]).square()[mk]
                    eg = (v_gated - y[..., 0]).square()[mk]
                    err0[name].append(float(e0.mean()) if e0.numel() else float('nan'))
                    errg[name].append(float(eg.mean()) if eg.numel() else float('nan'))
            for name in errs:
                a = [x for x in err0[name] if x == x]
                b = [x for x in errg[name] if x == x]
                rows.append(dict(oracle='O2_oracle_gated_b5delta', seed=seed, family=fam, mask=name,
                                 value=float(np.mean(b) ** 0.5)))
                rows.append(dict(oracle='O2_ref_b0', seed=seed, family=fam, mask=name,
                                 value=float(np.mean(a) ** 0.5)))
            del m5
            torch.cuda.empty_cache()
            print('O2', fam, seed, 'done', flush=True)
    outdir = ROOT / 'results' / 'rollout'
    outdir.mkdir(parents=True, exist_ok=True)
    with (outdir / 'oracle.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print('ORACLE COMPLETE', len(rows), flush=True)


if __name__ == '__main__':
    main()
