"""v11b C4: paired correction attribution on one-step windows (testD final,
testB diagnostic). For each model vs M0 (exact hard LIF):

Spike 2x2 (per-neuron-step, per family):
  m0 wrong -> model correct (fixed) | m0 correct -> model wrong (introduced)
V paired (per mask): frac with |err_model| < |err_m0| - eps (reduced) and
  reverse (increased), with mean magnitudes.

Outputs results/ablation/attribution.csv.
"""
import csv
import json
from pathlib import Path
import numpy as np
import torch
from connectome import Connectome
from run_v7 import setup_cfg
from latent_data import sample_indices, windows
from eval_latent import calibrate_threshold
from models.latent_hybrid_v11 import build_v11
from models.latent_hybrid_v11b import build_v11b
from protocol_v9 import FAMILIES, SEEDS
from data_v11 import load_pool
from eval_v11 import hard_lif_out, masks_for

ROOT = Path('results/latent_state_v11b')
COUNT = 2048
VARIANTS = ('m3a', 'm3b', 'm3c', 'm3_nolatent')


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
    ib = conn.i_bias.to(dev)
    pool = load_pool(cfg)
    td = torch.load(ROOT / 'configs' / 'testd_data.pt', map_location='cpu', weights_only=False)['store']
    m2sel = json.loads((ROOT / 'configs' / 'm2_shrink.json').read_text())
    val = {k: v.cuda() for k, v in pool['val'].items()}

    # m0 threshold on val
    bi, ti = sample_indices(val, 512, 8002)
    outs, ys = [], []
    for off in range(0, len(bi), 64):
        x, y = windows(val, bi[off:off + 64], ti[off:off + 64], 32)
        outs.append(hard_lif_out(x, cfg, W, ib)); ys.append(y)
    o = {k: torch.cat([a[k] for a in outs]) for k in outs[0]}
    thr0 = calibrate_threshold(o, torch.cat(ys))

    groups = []
    for fi, fam in enumerate(FAMILIES):
        groups.append(('testD', fam, {k: v.cuda() for k, v in td[f'{fam}/testD'].items()}))
        sel = (pool['testB']['mech'] == fi).nonzero(as_tuple=True)[0]
        groups.append(('testB', fam, {k: v[sel].cuda() if k != 'mech' else v[sel]
                                      for k, v in pool['testB'].items()}))
    groups.append(('testD', 'null', {k: v.cuda() for k, v in td['null/testD'].items()}))

    rows = []
    for seed in args.seeds:
        fns = []
        s11 = json.loads(Path(f'results/latent_state_v11/metrics/training/full_seed{seed}.json').read_text())
        m1 = build_v11('full', conn, cfg).to(dev)
        m1.load_state_dict(torch.load(s11['checkpoint'], map_location=dev, weights_only=False)['state_dict'])
        m1 = m1.eval()
        fns.append(('m1_v11', lambda x, _m=m1: _m(x), s11['threshold']))
        gv, gs = m2sel[str(seed)]['gamma_v'], m2sel[str(seed)]['gamma_s']
        def m2fn(x, _m=m1, _gv=gv, _gs=gs):
            out = _m(x)
            return dict(v=(out['v'] - out['corr'][..., 0]) + _gv * out['corr'][..., 0],
                        s_logits=(out['s_logits'] - out['corr'][..., 1]) + _gs * out['corr'][..., 1],
                        r=(out['r'] - out['corr'][..., 2]) + _gv * out['corr'][..., 2])
        fns.append(('m2_shrink', m2fn, s11['threshold']))
        for variant in VARIANTS:
            spath = ROOT / 'results' / 'training' / f'{variant}_seed{seed}.json'
            if not spath.exists():
                continue
            sv = json.loads(spath.read_text())
            mv = build_v11b(variant, conn, cfg).to(dev)
            mv.load_state_dict(torch.load(sv['checkpoint'], map_location=dev, weights_only=False)['state_dict'])
            mv = mv.eval()
            fns.append((variant, lambda x, _m=mv: _m(x, hard=True), sv['threshold']))
        for split, fam, d in groups:
            bi, ti = sample_indices(d, COUNT, 8001)
            # accumulate per-window tensors
            b0_sp, b0_ve = [], []
            for off in range(0, len(bi), 64):
                x, y = windows(d, bi[off:off + 64], ti[off:off + 64], 32)
                b0 = hard_lif_out(x, cfg, W, ib)
                b0_sp.append((torch.sigmoid(b0['s_logits']) > thr0).float().cpu())
                b0_ve.append((b0['v'] - y[..., 0]).abs().cpu())
            b0_sp = torch.cat(b0_sp); b0_ve = torch.cat(b0_ve)
            for name, fn, thr in fns:
                m_sp, m_ve, masks_all = [], [], {m: [] for m in ('free', 'info', 'event')}
                for off in range(0, len(bi), 64):
                    x, y = windows(d, bi[off:off + 64], ti[off:off + 64], 32)
                    out = fn(x)
                    m_sp.append((torch.sigmoid(out['s_logits']) > thr).float().cpu())
                    m_ve.append((out['v'] - y[..., 0]).abs().cpu())
                    mk = masks_for(x, y, cfg, W, ib)
                    for mname in masks_all:
                        masks_all[mname].append(mk[mname].cpu())
                m_sp = torch.cat(m_sp); m_ve = torch.cat(m_ve)
                # spikes vs teacher
                # teacher spikes come from windows y
                # recompute teacher spike tensor
                yt = []
                for off in range(0, len(bi), 64):
                    x, y = windows(d, bi[off:off + 64], ti[off:off + 64], 32)
                    yt.append(y[..., 1].cpu())
                yt = torch.cat(yt)
                w0 = (b0_sp != yt); wm = (m_sp != yt)
                row = dict(model=name, seed=seed, split=split, family=fam,
                           spike_fixed=int((w0 & ~wm).sum()), spike_introduced=int((~w0 & wm).sum()),
                           spike_bothwrong=int((w0 & wm).sum()), spike_bothok=int((~w0 & ~wm).sum()),
                           n=int(w0.numel()))
                for mname, mlist in masks_all.items():
                    mk = torch.cat(mlist)
                    e0 = b0_ve[mk]; em = m_ve[mk]
                    red = em < e0 - 1e-6; inc = em > e0 + 1e-6
                    row[f'{mname}_v_reduced_frac'] = float(red.float().mean())
                    row[f'{mname}_v_increased_frac'] = float(inc.float().mean())
                    row[f'{mname}_v_reduced_mag'] = float((e0 - em)[red].mean()) if red.any() else 0.0
                    row[f'{mname}_v_increased_mag'] = float((em - e0)[inc].mean()) if inc.any() else 0.0
                rows.append(row)
            print('attrib', seed, split, fam, 'done', flush=True)
        del m1
        torch.cuda.empty_cache()
    outdir = ROOT / 'results' / 'ablation'
    outdir.mkdir(parents=True, exist_ok=True)
    with (outdir / 'attribution.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print('ATTRIB COMPLETE', len(rows), flush=True)


if __name__ == '__main__':
    main()
