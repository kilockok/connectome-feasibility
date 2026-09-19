"""v11 Stage 1 evaluation: residual prediction baseline + not-input-copy controls.

Models:
  b0  hard LIF transition (exact, zero parameters)  - the artifact-free prior
  b1  soft base_pre formula                          - artifact floor reference
  b2  v9 OrderedHistory corrector (checkpoint)       - continuity reference
  b5  v11 full hybrid (per seed)
Controls:
  zshuffle : b5 with z permuted across trajectories within the eval batch
  nullctl  : b5 trained on the NULL-teacher pool (artifact-only reference)

Splits: testA / testB per family + null/test. Deterministic eval windows
(count=2048 per family-split, seed 8001). Masks: all / free / info-bearing /
event, all derived from the window's OWN past K frames (legal).

Outputs: metrics/onestep.csv, zshuffle.csv, null_control.csv
"""
import csv
import json
from pathlib import Path
import numpy as np
import torch
from connectome import Connectome
from latent_data import sample_indices, windows
from eval_latent import calibrate_threshold
from run_v7 import setup_cfg
from models.residual_v7 import base_pre
from models.residual_v8 import build_v8
from models.latent_hybrid_v11 import build_v11
from protocol_v9 import ROOT as ROOT9, FAMILIES, SEEDS
from data_v11 import load_pool

ROOT = Path('results/latent_state_v11')
COUNT = 2048
EVAL_SEED = 8001


@torch.no_grad()
def hard_lif_out(x, cfg, W, ib):
    """Exact hard-LIF one-step prediction from the last frame (b0)."""
    v, s, r, u = x[:, -1, :, 0], x[:, -1, :, 1], x[:, -1, :, 2], x[:, -1, :, 3]
    current = s @ W + u + ib
    refr = r > 0
    vn = torch.where(refr, torch.full_like(v, cfg.v_reset),
                     v + cfg.alpha * (-(v - cfg.v_rest) + current)).clamp(min=cfg.v_min)
    fire = (~refr) & (vn >= cfg.v_th)
    logits = (vn - cfg.v_th) / 0.1          # PRE-reset vn drives the spike logit
    vn = torch.where(fire, torch.full_like(vn, cfg.v_reset), vn)
    rn = torch.where(fire, torch.ones_like(r),
                     (r * cfg.refractory_period - 1).clamp(min=0) / cfg.refractory_period)
    return dict(v=vn, s_logits=logits, r=rn)


@torch.no_grad()
def soft_base_out(x, cfg, W, ib):
    base = base_pre(x[:, -1], cfg, W, ib)
    return dict(v=base['v_base'], s_logits=base['logit_base'], r=base['r_base'])


@torch.no_grad()
def masks_for(x, y, cfg, W, ib):
    """Masks for the target transition, from the window's own past only."""
    s_hist = x[..., 1]
    isyn_hist = torch.einsum('bkj,ji->bki', s_hist, W)
    v, s, r, u = x[:, -1, :, 0], x[:, -1, :, 1], x[:, -1, :, 2], x[:, -1, :, 3]
    isyn = s @ W
    refr = r > 0
    vn = torch.where(refr, torch.full_like(v, cfg.v_reset),
                     v + cfg.alpha * (-(v - cfg.v_rest) + isyn + u + ib)).clamp(min=cfg.v_min)
    fire_base = (~refr) & (vn >= cfg.v_th)
    free = (~refr) & (~fire_base)
    med = isyn_hist.abs().flatten(1).median(1).values[:, None]
    info = free & (isyn.abs() > med)
    event = (y[..., 1] > 0.5)
    B, N = v.shape
    return dict(all=torch.ones(B, N, dtype=torch.bool, device=v.device),
                free=free, info=info, event=event)


def f1_counts(pred, true):
    tp = float((pred * true).sum()); fp = float((pred * (1 - true)).sum())
    fn = float(((1 - pred) * true).sum())
    p = tp / max(tp + fp, 1.); r = tp / max(tp + fn, 1.)
    return 2 * p * r / max(p + r, 1e-9)


@torch.no_grad()
def eval_model(model_fn, data, cfg, W, ib, thr, zshuffle=False, seed_perm=0):
    """model_fn(x) -> out dict. Returns per-mask V RMSE + spike F1."""
    bi, ti = sample_indices(data, COUNT, EVAL_SEED)
    errs = {m: [] for m in ('all', 'free', 'info', 'event')}
    sp, st = [], []
    for offset in range(0, len(bi), 64):
        b, t = bi[offset:offset + 64], ti[offset:offset + 64]
        x, y = windows(data, b, t, 32)
        out = model_fn(x)
        if zshuffle:
            g = torch.Generator(device='cpu').manual_seed(9000 + seed_perm)
            perm = torch.randperm(x.shape[0], generator=g).to(x.device)
            out = model_fn(x, z_perm=perm)
        m = masks_for(x, y, cfg, W, ib)
        ve = (out['v'] - y[..., 0]).square()
        for name, mk in m.items():
            vals = ve[mk]
            errs[name].append(float(vals.mean()) if vals.numel() else float('nan'))
        sp.append((torch.sigmoid(out['s_logits']) > thr).float().cpu())
        st.append(y[..., 1].cpu())
    sp, st = torch.cat(sp), torch.cat(st)
    return {m: float(np.nanmean([e for e in errs[m] if e == e]) ** 0.5) for m in errs} | \
           dict(spike_f1=f1_counts(sp, st))


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
    for split in ('testA', 'testB'):
        d = pool[split]
        for fi, fam in enumerate(FAMILIES):
            sel = (d['mech'] == fi).nonzero(as_tuple=True)[0]
            groups.append((split, fam, {k: v[sel].cuda() if k != 'mech' else v[sel]
                                        for k, v in d.items()}))
    groups.append(('null', 'null', {k: v.cuda() for k, v in pool['null'].items()}))

    # ---- analytic models: one threshold each (deterministic) ----
    rows, zrows, crows = [], [], []
    val = {k: v.cuda() for k, v in pool['val'].items()}
    for name, fn in (('b0_hard', lambda x: hard_lif_out(x, cfg, W, ib)),
                     ('b1_soft', lambda x: soft_base_out(x, cfg, W, ib))):
        bi, ti = sample_indices(val, 512, 8002)
        outs, ys = [], []
        for offset in range(0, len(bi), 64):
            x, y = windows(val, bi[offset:offset + 64], ti[offset:offset + 64], 32)
            outs.append(fn(x)); ys.append(y)
        out_c = {k2: torch.cat([o[k2] for o in outs]) for k2 in outs[0]}
        thr = calibrate_threshold(out_c, torch.cat(ys))
        for split, fam, d in groups:
            res = eval_model(fn, d, cfg, W, ib, thr)
            for mask, v in res.items():
                if mask == 'spike_f1':
                    continue
                rows.append(dict(model=name, seed='-', split=split, family=fam, mask=mask,
                                 v_rmse=v, spike_f1=res['spike_f1'], n_windows=COUNT))
        print(name, 'done', flush=True)

    # ---- learned models per seed ----
    for seed in args.seeds:
        models = []
        # b2: v9 ordered checkpoint (continuity)
        s9 = json.loads((ROOT9 / 'metrics' / 'training' / f'ordered_seed{seed}.json').read_text())
        m2 = build_v8('ordered', conn, cfg).cuda()
        m2.load_state_dict(torch.load(s9['checkpoint'], map_location='cuda',
                                      weights_only=False)['state_dict'])
        models.append(('b2_v9ordered', m2, 0.5))
        # b5: v11 full
        s11 = json.loads((ROOT / 'metrics' / 'training' / f'full_seed{seed}.json').read_text())
        m5 = build_v11('full', conn, cfg).cuda()
        m5.load_state_dict(torch.load(s11['checkpoint'], map_location='cuda',
                                      weights_only=False)['state_dict'])
        models.append(('b5_hybrid', m5, s11['threshold']))
        # null-trained control
        npath = ROOT / 'metrics' / 'training' / f'full_seed{seed}_null.json'
        mn = None
        if npath.exists():
            sn = json.loads(npath.read_text())
            mn = build_v11('full', conn, cfg).cuda()
            mn.load_state_dict(torch.load(sn['checkpoint'], map_location='cuda',
                                          weights_only=False)['state_dict'])
        for name, m, thr in models:
            m = m.eval()
            for split, fam, d in groups:
                res = eval_model(lambda x, _m=m: _m(x), d, cfg, W, ib, thr)
                for mask, v in res.items():
                    if mask == 'spike_f1':
                        continue
                    rows.append(dict(model=name, seed=seed, split=split, family=fam, mask=mask,
                                     v_rmse=v, spike_f1=res['spike_f1'], n_windows=COUNT))
                # z-shuffle control (b5 only)
                if name == 'b5_hybrid':
                    def zs_fn(x, _m=m):
                        g = torch.Generator(device='cpu').manual_seed(9000)
                        perm = torch.randperm(x.shape[0], generator=g).to(x.device)
                        with torch.no_grad():
                            z = _m.latent(x)[0][:, -1]
                        return _m(x, z_override=z[perm])
                    res = eval_model(zs_fn, d, cfg, W, ib, thr)
                    for mask, v in res.items():
                        if mask == 'spike_f1':
                            continue
                        zrows.append(dict(model='b5_zshuffle', seed=seed, split=split,
                                          family=fam, mask=mask, v_rmse=v,
                                          spike_f1=res['spike_f1'], n_windows=COUNT))
            print(name, seed, 'done', flush=True)
        # null-trained control rows: eval on every group
        if mn is not None:
            mn = mn.eval()
            for split, fam, d in groups:
                res = eval_model(lambda x, _m=mn: _m(x), d, cfg, W, ib, sn['threshold'])
                # also residual RMS of the correction channel
                bi, ti = sample_indices(d, 512, 8001)
                b, t = bi[:64], ti[:64]
                x, y = windows(d, b, t, 32)
                out = mn(x)
                rr = float(out['corr'][..., 0].square().mean().sqrt())
                for mask, v in res.items():
                    if mask == 'spike_f1':
                        continue
                    crows.append(dict(model='b5_nulltrained', seed=seed, split=split,
                                      family=fam, mask=mask, v_rmse=v,
                                      spike_f1=res['spike_f1'], resid_rms=rr,
                                      n_windows=COUNT))
            del mn
            torch.cuda.empty_cache()
        del m2, m5
        torch.cuda.empty_cache()

    (ROOT / 'metrics').mkdir(parents=True, exist_ok=True)
    for fname, r in (('onestep.csv', rows), ('zshuffle.csv', zrows),
                     ('null_control.csv', crows)):
        with (ROOT / 'metrics' / fname).open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(r[0]))
            w.writeheader(); w.writerows(r)
        print(fname, len(r), 'rows', flush=True)
    print('EVAL V11 COMPLETE', flush=True)


if __name__ == '__main__':
    main()
