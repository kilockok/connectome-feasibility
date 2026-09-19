"""v11 Stage 1 training: LatentHybridV11 on the v9 mixed mechanism pool
(mechanism-blind, no labels), or on the NULL pool (artifact control).

Mirrors run_v9.train_one: 24 epochs x 48 steps x batch 16, AdamW 3e-4,
wd 1e-4, grad clip 1.0, val 512 windows/epoch (fixed seed 8001), early stop
(patience 6), checkpoint best_val + last, summary json with val-calibrated
spike threshold.
"""
import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
import torch
from connectome import Connectome
from latent_data import sample_indices, windows
from metrics import compute_loss
from eval_latent import calibrate_threshold
from train_latent import atomic_save
from calibrate_latent import write_csv
from run_v7 import setup_cfg
from models.latent_hybrid_v11 import build_v11
from data_v11 import load_pool, null_train

ROOT = Path('results/latent_state_v11')
CKPTS = ROOT / 'checkpoints'
SEEDS = (1234, 1235, 1236, 1237, 1238)


def tag_for(kind, k, seed, null):
    base = f'{kind}_k{k}_seed{seed}' if kind != 'full' or k != 32 else f'full_seed{seed}'
    return base + ('_null' if null else '')


def train_one(kind, k, seed, conn, cfg, pool, null=False, epochs=24, steps=48, batch=16):
    tag = tag_for(kind, k, seed, null)
    directory = CKPTS / tag
    directory.mkdir(parents=True, exist_ok=True)
    summary_path = ROOT / 'metrics' / 'training' / f'{tag}.json'
    if summary_path.exists():
        print('skip', tag, flush=True)
        return json.loads(summary_path.read_text())
    torch.manual_seed(seed)
    model = build_v11(kind, conn, cfg, k=k).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    pw = torch.tensor(cfg.spike_pos_weight, device='cuda')
    best, bad, history = float('inf'), 0, []
    last = directory / 'last.pt'
    start_epoch = 1
    if last.exists():
        resume = torch.load(last, map_location='cpu', weights_only=False)
        model.load_state_dict(resume['state_dict']); opt.load_state_dict(resume['optimizer'])
        history = resume['history']; best = resume['best']; bad = resume['bad']
        start_epoch = resume['epoch'] + 1
        torch.set_rng_state(resume['rng']); torch.cuda.set_rng_state_all(resume['cuda_rng'])
    t0 = time.monotonic()
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        bi, ti = sample_indices(pool['train'], steps * batch, 100_000 + seed * 100 + epoch)
        losses = []
        for offset in range(0, len(bi), batch):
            b, t = bi[offset:offset + batch], ti[offset:offset + batch]
            x, y = windows(pool['train'], b, t, model.k if model.k >= 1 else 1)
            opt.zero_grad(set_to_none=True)
            out = model(x)
            loss, _ = compute_loss(out, y, cfg, pw)
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
                x, y = windows(pool['val'], b, t, model.k if model.k >= 1 else 1)
                out = model(x)
                loss, _ = compute_loss(out, y, cfg, pw)
                tot += float(loss) * len(b); cnt += len(b)
        vl = tot / cnt
        history.append(dict(epoch=epoch, train_loss=sum(losses) / len(losses), val_loss=vl,
                            elapsed=time.monotonic() - t0))
        improved = vl < best - 1e-6
        if improved:
            best = vl
            atomic_save(dict(state_dict=model.state_dict(), kind=kind, k=k, seed=seed,
                             epoch=epoch, config=asdict(cfg)), directory / 'best_val.pt')
        bad = 0 if improved else bad + 1
        atomic_save(dict(state_dict=model.state_dict(), optimizer=opt.state_dict(),
                         history=history, best=best, bad=bad, epoch=epoch,
                         rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all()), last)
        write_csv(ROOT / 'metrics' / 'training' / f'{tag}.csv', history)
        print(f'{tag} ep={epoch} loss={vl:.5f}', flush=True)
        if bad >= 6:
            break
    # val-calibrated spike threshold
    model.load_state_dict(torch.load(directory / 'best_val.pt', map_location='cuda',
                                     weights_only=False)['state_dict'])
    model = model.eval()
    with torch.no_grad():
        bi, ti = sample_indices(pool['val'], 512, 8002)
        outs, ys = [], []
        for offset in range(0, len(bi), 32):
            b, t = bi[offset:offset + 32], ti[offset:offset + 32]
            x, y = windows(pool['val'], b, t, model.k if model.k >= 1 else 1)
            outs.append({k2: v.cpu() for k2, v in model(x.cuda() if not x.is_cuda else x).items()
                         if k2 in ('v', 's_logits', 'r')})
            ys.append(y.cpu())
    out_cat = {k2: torch.cat([o[k2] for o in outs]) for k2 in outs[0]}
    thr = calibrate_threshold(out_cat, torch.cat(ys))
    summary = dict(tag=tag, kind=kind, k=k, seed=seed, epochs=len(history), best=best,
                   params=sum(p.numel() for p in model.parameters()), threshold=thr,
                   checkpoint=str(directory / 'best_val.pt'))
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))
    del model, opt
    torch.cuda.empty_cache()
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--kinds', nargs='+', default=['full'])
    ap.add_argument('--ks', nargs='+', type=int, default=[32])
    ap.add_argument('--seeds', nargs='+', type=int, default=list(SEEDS))
    ap.add_argument('--null', action='store_true', help='train on NULL-teacher pool')
    args = ap.parse_args()
    cfg = setup_cfg()
    conn = Connectome.generate(cfg)
    if args.null:
        nd = null_train(cfg, conn)
        pool = dict(train={k: v.cuda() for k, v in nd['train'].items()},
                    val={k: v.cuda() for k, v in nd['val'].items()})
    else:
        pool0 = load_pool(cfg)
        pool = dict(train={k: v.cuda() for k, v in pool0['train'].items()},
                    val={k: v.cuda() for k, v in pool0['val'].items()})
    for kind in args.kinds:
        for k in args.ks:
            for seed in args.seeds:
                tag = tag_for(kind, k, seed, args.null)
                print('train', tag, flush=True)
                train_one(kind, k, seed, conn, cfg, pool, null=args.null)
    print('TRAIN V11 COMPLETE', flush=True)


if __name__ == '__main__':
    main()
