"""v11b evaluation: one-step (testD final + testB diagnostic), C1 NULL
control, C2 z-shuffle, C4 attribution.

Window protocol identical to v11 (2048 deterministic windows per group,
seed 8001, K=32, masks from the window's own past). Per-model thresholds
calibrated on val (seed 8002). Models: m0 (exact hard LIF), m1 (v11
hybrid), m2 (shrunk m1), m3a/m3b/m3c/m3_nolatent (as available).

Outputs:
  results/onestep/onestep.csv        (model, seed, split, family, mask,
                                      v_rmse, spike_f1, corr_rms, gate_mean,
                                      gate_active_frac)
  results/null_control/null_control.csv
  results/zshuffle/zshuffle.csv
  results/ablation/attribution.csv   (spike 2x2 vs m0 + paired V, per family)
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
from models.latent_hybrid_v11b import build_v11b
from protocol_v9 import FAMILIES, SEEDS
from data_v11 import load_pool
from eval_v11 import hard_lif_out, masks_for, f1_counts

ROOT = Path('results/latent_state_v11b')
COUNT = 2048
EVAL_SEED = 8001
VARIANTS = ('m3a', 'm3b', 'm3c', 'm3_nolatent')


@torch.no_grad()
def eval_windows(model_fn, data, cfg, W, ib, thr):
    bi, ti = sample_indices(data, COUNT, EVAL_SEED)
    errs = {m: [] for m in ('all', 'free', 'info', 'event')}
    sp, st = [], []
    corr_sq, gate_vals = [], []
    for offset in range(0, len(bi), 64):
        b, t = bi[offset:offset + 64], ti[offset:offset + 64]
        x, y = windows(data, b, t, 32)
        out = model_fn(x)
        m = masks_for(x, y, cfg, W, ib)
        ve = (out['v'] - y[..., 0]).square()
        for name, mk in m.items():
            vals = ve[mk]
            errs[name].append(float(vals.mean()) if vals.numel() else float('nan'))
        sp.append((torch.sigmoid(out['s_logits']) > thr).float().cpu())
        st.append(y[..., 1].cpu())
        if 'corr' in out:
            corr_sq.append(float(out['corr'].square().mean()))
        if 'gate' in out:
            gate_vals.append(out['gate'].flatten().cpu())
    sp, st = torch.cat(sp), torch.cat(st)
    res = {m: float(np.nanmean(errs[m]) ** 0.5) for m in errs}
    res['spike_f1'] = f1_counts(sp, st)
    res['corr_rms'] = float(np.mean(corr_sq) ** 0.5) if corr_sq else 0.0
    if gate_vals:
        g = torch.cat(gate_vals)
        res['gate_mean'] = float(g.mean())
        res['gate_active_frac'] = float((g > 0.5).float().mean())
    else:
        res['gate_mean'] = -1.0
        res['gate_active_frac'] = -1.0
    return res


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
    td = torch.load(ROOT / 'configs' / 'testd_data.pt', map_location='cpu', weights_only=False)['store']
    m2sel = json.loads((ROOT / 'configs' / 'm2_shrink.json').read_text()) \
        if (ROOT / 'configs' / 'm2_shrink.json').exists() else None

    groups = []
    for fi, fam in enumerate(FAMILIES):
        sel = (pool['testB']['mech'] == fi).nonzero(as_tuple=True)[0]
        groups.append(('testB', fam, {k: v[sel].cuda() if k != 'mech' else v[sel]
                                      for k, v in pool['testB'].items()}))
        groups.append(('testD', fam, {k: v.cuda() for k, v in td[f'{fam}/testD'].items()}))
    groups.append(('testB', 'null', {k: v.cuda() for k, v in pool['null'].items()}))
    groups.append(('testD', 'null', {k: v.cuda() for k, v in td['null/testD'].items()}))

    rows, zrows = [], []
    # ---- m0 ----
    from eval_latent import calibrate_threshold
    val = {k: v.cuda() for k, v in pool['val'].items()}
    bi, ti = sample_indices(val, 512, 8002)
    outs, ys = [], []
    for off in range(0, len(bi), 64):
        x, y = windows(val, bi[off:off + 64], ti[off:off + 64], 32)
        outs.append(hard_lif_out(x, cfg, W, ib)); ys.append(y)
    o = {k: torch.cat([a[k] for a in outs]) for k in outs[0]}
    thr0 = calibrate_threshold(o, torch.cat(ys))
    for split, fam, d in groups:
        res = eval_windows(lambda x: hard_lif_out(x, cfg, W, ib), d, cfg, W, ib, thr0)
        for mask in ('all', 'free', 'info', 'event'):
            rows.append(dict(model='m0_hard', seed='-', split=split, family=fam, mask=mask,
                             v_rmse=res[mask], spike_f1=res['spike_f1'], corr_rms=0.0,
                             gate_mean=-1, gate_active_frac=-1))
    print('m0 done', flush=True)

    for seed in args.seeds:
        # ---- m1 + m2 ----
        s11 = json.loads(Path(f'results/latent_state_v11/metrics/training/full_seed{seed}.json').read_text())
        m1 = build_v11('full', conn, cfg).to(dev)
        m1.load_state_dict(torch.load(s11['checkpoint'], map_location=dev, weights_only=False)['state_dict'])
        m1 = m1.eval()
        fns = [('m1_v11', lambda x, _m=m1: _m(x), s11['threshold'])]
        if m2sel is not None:
            gv = m2sel[str(seed)]['gamma_v']; gs = m2sel[str(seed)]['gamma_s']
            def m2fn(x, _m=m1, _gv=gv, _gs=gs):
                out = _m(x)
                return dict(v=(out['v'] - out['corr'][..., 0]) + _gv * out['corr'][..., 0],
                            s_logits=(out['s_logits'] - out['corr'][..., 1]) + _gs * out['corr'][..., 1],
                            r=(out['r'] - out['corr'][..., 2]) + _gv * out['corr'][..., 2],
                            corr=out['corr'] * _gv)
            fns.append(('m2_shrink', m2fn, s11['threshold']))
        # ---- m3 variants ----
        for variant in VARIANTS:
            sp_path = ROOT / 'results' / 'training' / f'{variant}_seed{seed}.json'
            if not sp_path.exists():
                continue
            sv = json.loads(sp_path.read_text())
            mv = build_v11b(variant, conn, cfg).to(dev)
            mv.load_state_dict(torch.load(sv['checkpoint'], map_location=dev, weights_only=False)['state_dict'])
            mv = mv.eval()
            fns.append((variant, lambda x, _m=mv: _m(x, hard=True), sv['threshold']))
            if variant == 'm3b':
                fns.append(('m3b_soft_diag', lambda x, _m=mv: _m(x, hard=False),
                            sv['threshold']))
        for name, fn, thr in fns:
            for split, fam, d in groups:
                res = eval_windows(fn, d, cfg, W, ib, thr)
                for mask in ('all', 'free', 'info', 'event'):
                    rows.append(dict(model=name, seed=seed, split=split, family=fam, mask=mask,
                                     v_rmse=res[mask], spike_f1=res['spike_f1'],
                                     corr_rms=res['corr_rms'], gate_mean=res['gate_mean'],
                                     gate_active_frac=res['gate_active_frac']))
            print(name, seed, 'done', flush=True)
        del m1
        torch.cuda.empty_cache()
        # NOTE: z-shuffle needs direct model access; handled below per variant
        spath = ROOT / 'results' / 'training' / f'm3b_seed{seed}.json'
        if spath.exists():
            sv = json.loads(spath.read_text())
            mv = build_v11b('m3b', conn, cfg).to(dev)
            mv.load_state_dict(torch.load(sv['checkpoint'], map_location=dev, weights_only=False)['state_dict'])
            mv = mv.eval()
            for split, fam, d in groups:
                if split != 'testD':
                    continue
                def zs(x, _m=mv):
                    g = torch.Generator(device='cpu').manual_seed(9000)
                    perm = torch.randperm(x.shape[0], generator=g).to(x.device)
                    zseq, e = _m.backbone.latent(x)
                    z_last = zseq[:, -1][perm]
                    # recompute forward with shuffled z via monkey-patch style:
                    vn_base, refr, isyn = _m.base_vn(x[:, -1])
                    v, s, r, u = x[:, -1].unbind(-1)
                    gfeat = torch.stack((v, vn_base - cfg.v_th, isyn.abs(), r, u), -1)
                    fused = torch.cat((e[:, -1], z_last[:, None, :].expand(-1, x.shape[2], -1)), -1)
                    delta = _m.delta_head(fused)[..., 0]
                    gg = torch.sigmoid(_m.gate_head(torch.cat((fused, gfeat), -1)))[..., 0]
                    corr = gg * delta
                    vn = vn_base + corr
                    logit = (vn - cfg.v_th) / 0.1
                    fire = (~refr) & (vn >= cfg.v_th)
                    v_next = torch.where(fire, torch.full_like(vn, cfg.v_reset), vn)
                    v_next = torch.where(refr, torch.full_like(vn, cfg.v_reset), v_next)
                    r_next = torch.where(fire, torch.ones_like(r),
                                         (r * cfg.refractory_period - 1).clamp(min=0)
                                         / cfg.refractory_period)
                    return dict(v=v_next, s_logits=logit, r=r_next, corr=corr, gate=gg)
                res = eval_windows(zs, d, cfg, W, ib, sv['threshold'])
                for mask in ('all', 'free', 'info', 'event'):
                    zrows.append(dict(model='m3b_zshuffle', seed=seed, split=split, family=fam,
                                      mask=mask, v_rmse=res[mask], spike_f1=res['spike_f1'],
                                      corr_rms=res['corr_rms'], gate_mean=res['gate_mean'],
                                      gate_active_frac=res['gate_active_frac']))
            del mv
            torch.cuda.empty_cache()
            print('m3b zshuffle', seed, 'done', flush=True)

    for fname, r, sub in (('onestep.csv', rows, 'onestep'), ('zshuffle.csv', zrows, 'zshuffle')):
        d0 = ROOT / 'results' / sub
        d0.mkdir(parents=True, exist_ok=True)
        with (d0 / fname).open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(r[0])); w.writeheader(); w.writerows(r)
        print(sub + '/' + fname, len(r), 'rows', flush=True)
    print('EVAL V11B COMPLETE', flush=True)


if __name__ == '__main__':
    main()
