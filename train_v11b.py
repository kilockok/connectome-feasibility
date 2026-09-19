"""v11b Part B training: gated hybrid variants on the v9 mixed pool.

Batch composition (all variants): 3/4 mechanism-pool windows + 1/4
NULL-teacher windows (null train split, paired seeds). Only the LOSS differs:
  m3a         : compute_loss only (lambda_null = 0)
  m3b         : + mean((g*delta)^2) on null windows (lambda_null = 1)
  m3c         : m3b + event-weighted V MSE (w = 1 + 4*{teacher spike or
                |vn_base - v_th| < 0.2555})
  m3_nolatent : m3b with the k0 encoder (C3 control)
24 epochs x 48 steps x batch 16, AdamW 3e-4, wd 1e-4, clip 1.0, val 512
windows (seed 8001), early stop patience 6, best_val checkpoint, threshold
calibrated on val (seed 8002). Identical data order for every variant.
"""
import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
import torch
import torch.nn.functional as F
from connectome import Connectome
from latent_data import sample_indices, windows
from metrics import compute_loss
from eval_latent import calibrate_threshold
from train_latent import atomic_save
from calibrate_latent import write_csv
from run_v7 import setup_cfg
from models.latent_hybrid_v11b import build_v11b
from data_v11 import load_pool, null_train

ROOT = Path('results/latent_state_v11b')
CKPTS = ROOT / 'results' / 'checkpoints'
SEEDS = (1234, 1235, 1236, 1237, 1238)
EVENT_DELTA = 0.2555     # protocol-frozen (train free-step p1 of |vn-v_th|)
EVENT_W = 4.0
LAM_NULL = 1.0


def loss_variant(out, y, cfg, pw, variant, null_mask):
    """compute_loss + variant terms. null_mask: [B] bool (null windows)."""
    if variant == 'm3c':
        ve = (out['v'] - y[..., 0]).square()
        w = 1.0 + EVENT_W * ((y[..., 1] > 0.5) |
                             ((out['vn_base'] - cfg.v_th).abs() < EVENT_DELTA)).float()
        lv = (ve * w).sum() / w.sum()
        ls = F.binary_cross_entropy_with_logits(out['s_logits'], y[..., 1], pos_weight=pw)
        lr_ = F.mse_loss(out['r'], y[..., 2])
        base = cfg.lambda_v * lv + cfg.lambda_s * ls + cfg.lambda_r * lr_
    else:
        base, _ = compute_loss(out, y, cfg, pw)
    if variant in ('m3b', 'm3c', 'm3_nolatent') and null_mask.any():
        pen = (out['corr'][null_mask]).square().mean()
        base = base + LAM_NULL * pen
    return base


def train_one(variant, seed, conn, cfg, pool, nullpool, epochs=24, steps=48, batch=16):
    tag = f'{variant}_seed{seed}'
    directory = CKPTS / tag
    directory.mkdir(parents=True, exist_ok=True)
    summary_path = ROOT / 'results' / 'training' / f'{tag}.json'
    if summary_path.exists():
        print('skip', tag, flush=True)
        return json.loads(summary_path.read_text())
    torch.manual_seed(seed)
    model = build_v11b(variant, conn, cfg).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    pw = torch.tensor(cfg.spike_pos_weight, device='cuda')
    best, bad, history = float('inf'), 0, []
    t0 = time.monotonic()
    for epoch in range(1, epochs + 1):
        model.train()
        bi, ti = sample_indices(pool['train'], steps * batch * 3 // 4,
                                100_000 + seed * 100 + epoch)
        nb, nt = sample_indices(nullpool['train'], steps * batch // 4,
                                200_000 + seed * 100 + epoch)
        losses = []
        for offset in range(0, len(bi), batch * 3 // 4):
            bm, tm = bi[offset:offset + batch * 3 // 4], ti[offset:offset + batch * 3 // 4]
            bn, tn = nb[offset:offset + batch // 4], nt[offset:offset + batch // 4]
            xm, ym = windows(pool['train'], bm, tm, 32)
            xn, yn = windows(nullpool['train'], bn, tn, 32)
            x = torch.cat((xm, xn)); y = torch.cat((ym, yn))
            nm = torch.zeros(len(x), dtype=torch.bool, device=x.device)
            nm[len(xm):] = True
            opt.zero_grad(set_to_none=True)
            out = model(x)
            loss = loss_variant(out, y, cfg, pw, variant, nm)
            if not torch.isfinite(loss):
                raise RuntimeError('non-finite loss')
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if not torch.isfinite(grad):
                raise RuntimeError('non-finite grad')
            opt.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            bi, ti = sample_indices(pool['val'], 512, 8001)
            tot, cnt = 0., 0
            for offset in range(0, len(bi), 32):
                b, t = bi[offset:offset + 32], ti[offset:offset + 32]
                x, y = windows(pool['val'], b, t, 32)
                out = model(x)
                loss, _ = compute_loss(out, y, cfg, pw)
                tot += float(loss) * len(b); cnt += len(b)
        vl = tot / cnt
        history.append(dict(epoch=epoch, train_loss=sum(losses) / len(losses), val_loss=vl,
                            elapsed=time.monotonic() - t0))
        improved = vl < best - 1e-6
        if improved:
            best = vl
            atomic_save(dict(state_dict=model.state_dict(), variant=variant, seed=seed,
                             epoch=epoch, config=asdict(cfg)), directory / 'best_val.pt')
        bad = 0 if improved else bad + 1
        write_csv(ROOT / 'results' / 'training' / f'{tag}.csv', history)
        print(f'{tag} ep={epoch} loss={vl:.5f}', flush=True)
        if bad >= 6:
            break
    model.load_state_dict(torch.load(directory / 'best_val.pt', map_location='cuda',
                                     weights_only=False)['state_dict'])
    model = model.eval()
    with torch.no_grad():
        bi, ti = sample_indices(pool['val'], 512, 8002)
        outs, ys = [], []
        for offset in range(0, len(bi), 32):
            b, t = bi[offset:offset + 32], ti[offset:offset + 32]
            x, y = windows(pool['val'], b, t, 32)
            outs.append({k2: v.cpu() for k2, v in model(x).items() if k2 in ('v', 's_logits', 'r')})
            ys.append(y.cpu())
    out_cat = {k2: torch.cat([o[k2] for o in outs]) for k2 in outs[0]}
    thr = calibrate_threshold(out_cat, torch.cat(ys))
    summary = dict(tag=tag, variant=variant, seed=seed, epochs=len(history), best=best,
                   params=sum(p.numel() for p in model.parameters()), threshold=thr,
                   checkpoint=str(directory / 'best_val.pt'))
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))
    del model, opt
    torch.cuda.empty_cache()
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variants', nargs='+', default=['m3a', 'm3b'])
    ap.add_argument('--seeds', nargs='+', type=int, default=list(SEEDS))
    ap.add_argument('--epochs', type=int, default=24)
    ap.add_argument('--steps', type=int, default=48)
    args = ap.parse_args()
    cfg = setup_cfg()
    conn = Connectome.generate(cfg)
    pool0 = load_pool(cfg)
    pool = dict(train={k: v.cuda() for k, v in pool0['train'].items()},
                val={k: v.cuda() for k, v in pool0['val'].items()})
    nd = null_train(cfg, conn)
    nullpool = dict(train={k: v.cuda() for k, v in nd['train'].items()},
                    val={k: v.cuda() for k, v in nd['val'].items()})
    for variant in args.variants:
        for seed in args.seeds:
            print('train', variant, seed, flush=True)
            train_one(variant, seed, conn, cfg, pool, nullpool,
                      epochs=args.epochs, steps=args.steps)
    print('TRAIN V11B COMPLETE', flush=True)


if __name__ == '__main__':
    main()
