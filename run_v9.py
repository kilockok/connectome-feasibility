"""v9 Stage 4: Phase A mechanism-blind unified corrector + Gate B eval.

ONE OrderedHistory residual corrector (v8 architecture, K=32) trained on the
mixed GAIN+ADAPT+STP train pool WITHOUT mechanism labels or any
classification loss. 5 paired seeds. Gate B: improvement over the base_pre
formula predictor must be > 0 on every family, on testA (seen params) and
testB (held-out params). NULL split reported as the artifact reference.
"""
import argparse
import csv
import json
import time
from dataclasses import asdict
from pathlib import Path
import torch
from connectome import Connectome
from latent_data import sample_indices, windows
from metrics import compute_loss
from eval_latent import calibrate_threshold, summarize
from train_latent import atomic_save
from calibrate_latent import write_csv
from models.residual_v8 import build_v8
from models.residual_v7 import base_pre
from run_v7 import setup_cfg
from protocol_v9 import ROOT, CKPTS, FAMILIES, SEEDS

KIND = 'ordered'
GPU_KEYS = ('states', 'stimulus', 'silence')


def load_pool(cfg):
    blob = torch.load(ROOT / 'data' / 'v9_data.pt', map_location='cpu', weights_only=False)
    store = blob['store']
    pool = {}
    for split in ('train', 'val', 'testA', 'testB', 'testC'):
        parts, mech = [], []
        for fi, fam in enumerate(FAMILIES):
            d = store[f'{fam}/{split}']
            parts.append(d)
            mech += [fi] * len(d['states'])
        pool[split] = {k: torch.cat([p[k] for p in parts]).cuda() for k in GPU_KEYS}
        pool[split]['mech'] = torch.tensor(mech).cuda()
    pool['null'] = {k: v.cuda() for k, v in store['null/test'].items() if k in GPU_KEYS}
    pool['null']['mech'] = torch.full((len(pool['null']['states']),), -1).cuda()
    return pool


def train_one(seed, conn, cfg, pool, epochs=24, steps=48, batch=16):
    directory = CKPTS / f'{KIND}_seed{seed}'
    directory.mkdir(parents=True, exist_ok=True)
    summary_path = ROOT / 'metrics' / 'training' / f'{KIND}_seed{seed}.json'
    if summary_path.exists():
        return json.loads(summary_path.read_text())
    torch.manual_seed(seed)
    model = build_v8(KIND, conn, cfg).cuda()
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
            x, y = windows(pool['train'], b, t, model.k)
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
                x, y = windows(pool['val'], b, t, model.k)
                out = model(x)
                loss, _ = compute_loss(out, y, cfg, pw)
                tot += float(loss) * len(b); cnt += len(b)
        vl = tot / cnt
        history.append(dict(epoch=epoch, train_loss=sum(losses) / len(losses), val_loss=vl,
                            elapsed=time.monotonic() - t0))
        improved = vl < best - 1e-6
        if improved:
            best = vl
            atomic_save(dict(state_dict=model.state_dict(), kind=KIND, seed=seed, epoch=epoch,
                             config=asdict(cfg)), directory / 'best_val.pt')
        bad = 0 if improved else bad + 1
        atomic_save(dict(state_dict=model.state_dict(), optimizer=opt.state_dict(),
                         history=history, best=best, bad=bad, epoch=epoch,
                         rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all()), last)
        write_csv(ROOT / 'metrics' / 'training' / f'{KIND}_seed{seed}.csv', history)
        print(f'{KIND} s={seed} ep={epoch} loss={vl:.5f}', flush=True)
        if bad >= 6:
            break
    summary = dict(kind=KIND, seed=seed, epochs=len(history), best=best,
                   params=sum(p.numel() for p in model.parameters()),
                   checkpoint=str(directory / 'best_val.pt'))
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))
    del model, opt
    torch.cuda.empty_cache()
    return summary


@torch.no_grad()
def gate_b(seed, conn, cfg, pool, count=768):
    """Per-family correction metrics on testA/testB + null reference."""
    summary = json.loads((ROOT / 'metrics' / 'training' / f'{KIND}_seed{seed}.json').read_text())
    model = build_v8(KIND, conn, cfg).cuda()
    model.load_state_dict(torch.load(summary['checkpoint'], map_location='cuda',
                                     weights_only=False)['state_dict'])
    model = model.eval()
    rows = []
    for split in ('testA', 'testB', 'testC', 'null'):
        d = pool[split]
        fams = [None] if split == 'null' else list(FAMILIES)
        for fi, fam in enumerate(fams):
            if split == 'null':
                mask = torch.ones(len(d['states']), dtype=torch.bool, device='cuda')
                fam_name = 'null'
            else:
                mask = d['mech'] == fi
                fam_name = FAMILIES[fi]
            idx = mask.nonzero(as_tuple=True)[0]
            g = torch.Generator().manual_seed(8001)
            bi = idx[torch.randint(len(idx), (count,), generator=g)].cuda()
            ti = torch.randint(31, d['stimulus'].shape[1], (count,), generator=g).cuda()
            outs, ys, bases = [], [], []
            for off in range(0, len(bi), 32):
                b, t = bi[off:off + 32], ti[off:off + 32]
                x, y = windows(d, b, t, model.k)
                out = model(x)
                xw, _ = windows(d, b, t, 1)
                base = base_pre(xw[:, -1], cfg, model.W, model.i_bias)
                outs.append({k: v.cpu() for k, v in out.items() if k in ('v', 's_logits', 'r')})
                ys.append(y.cpu())
                bases.append({k: v.cpu() for k, v in base.items()})
            o = {k: torch.cat([a[k] for a in outs]) for k in outs[0]}
            y = torch.cat(ys)
            bs = {k: torch.cat([a[k] for a in bases]) for k in bases[0]}
            th = calibrate_threshold(o, y)
            m = summarize(o, y, th)
            bm = summarize(dict(v=bs['v_base'], s_logits=bs['logit_base'], r=bs['r_base']), y, th)
            base_v = float((bs['v_base'] - y[..., 0]).square().mean().sqrt())
            rows.append(dict(seed=seed, split=split, family=fam_name,
                             spike_f1=m['spike_f1'], v_rmse=m['v_rmse'],
                             base_v_rmse=base_v, improve_over_base=base_v - m['v_rmse'],
                             base_spike_f1=bm['spike_f1'], f1_gain=m['spike_f1'] - bm['spike_f1'],
                             n=count))
    del model
    torch.cuda.empty_cache()
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', nargs='+', type=int, default=list(SEEDS))
    p.add_argument('--eval-only', action='store_true')
    args = p.parse_args()
    torch.set_num_threads(2)
    cfg = setup_cfg()
    conn = Connectome.generate(cfg)
    pool = load_pool(cfg)
    if not args.eval_only:
        for seed in args.seeds:
            train_one(seed, conn, cfg, pool)
        print('V9 PHASE-A TRAIN COMPLETE', flush=True)
    rows = []
    for seed in args.seeds:
        rows += gate_b(seed, conn, cfg, pool)
        print('gateB done seed', seed, flush=True)
    path = ROOT / 'metrics' / 'correction_per_seed.csv'
    old = []
    if path.exists():
        old = list(csv.DictReader(path.open()))
    done = {(r['seed'], r['split'], r['family']) for r in old}
    rows = [r for r in rows if (str(r['seed']), r['split'], r['family']) not in done]
    with path.open('a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        if f.tell() == 0:
            w.writeheader()
        for r in rows:
            w.writerow(r)
    print('GATE B ROWS WRITTEN', len(rows), flush=True)


if __name__ == '__main__':
    main()
