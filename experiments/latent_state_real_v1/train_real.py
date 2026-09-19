"""Training for latent_state_real_v1. One model per (kind, graph, seed, h).
Same budget everywhere: d=64, AdamW 3e-4, 30 epochs x 100 steps x batch 64,
early stop patience 6 on val RMSE. Windows: (session, t) with t in [L, T-h];
train sessions only. Checkpoint: results_real/checkpoints/{tag}/best.pt.
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from models_real import build_real

ROOT = Path('.')
CK = ROOT / 'results_real' / 'checkpoints'
L = 32
SEEDS = (1234, 1235, 1236, 1237, 1238)


def load(tag_graph='A_real'):
    d = torch.load(ROOT / 'data_real.pt', weights_only=False)
    return d, d[tag_graph]


def sample_windows(d, split, count, seed, L, h):
    g = np.random.default_rng(seed)
    fids = [f for f, s in d['split_of'].items() if s == split]
    picks = []
    for _ in range(count):
        fid = fids[int(g.integers(len(fids)))]
        T = d['sessions'][fid]['y'].shape[0]
        t = int(g.integers(L, T - h))
        picks.append((fid, t))
    return picks


def batch(d, picks, h):
    xs, ys = [], []
    for fid, t in picks:
        y = d['sessions'][fid]['y']
        xs.append(y[t - L:t])
        ys.append(y[t + h - 1])
    return (torch.tensor(np.stack(xs), dtype=torch.float32),
            torch.tensor(np.stack(ys), dtype=torch.float32))


def train_one(kind, graph, seed, h, epochs=30, steps=100, batch_size=64):
    tag = f'{kind}_{graph}_h{h}_seed{seed}'
    out = CK / tag
    summ = out / 'summary.json'
    if summ.exists():
        print('skip', tag, flush=True)
        return
    d, A = load('A_real' if graph == 'real' else ('A_edges_shuf' if graph == 'edges' else 'A_weights_shuf'))
    dev = torch.device('cuda')
    torch.manual_seed(seed)
    base_kind = {'m1': 'm1', 'm2': 'm2', 'm2_identity': 'm2', 'm3': 'm3',
                 'm3_nolatent': 'm3_nolatent', 'b1': 'b1'}[kind]
    model = build_real(base_kind, A, 37, latent=(kind != 'm3_nolatent')).to(dev)
    if kind == 'm2_identity':
        I = torch.eye(37, device=dev)
        model.An = I
        model.enc.A = I
    out.mkdir(parents=True, exist_ok=True)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    best, bad, hist = float('inf'), 0, []
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        all_picks = sample_windows(d, 'train', steps * batch_size,
                                   50_000 + seed * 100 + ep, L, h)
        for step_i in range(steps):
            picks = all_picks[step_i * batch_size:(step_i + 1) * batch_size]
            x, y = batch(d, picks, h)
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            errs = []
            vpicks = sample_windows(d, 'val', 1024, 8001, L, h)
            for vi in range(16):
                picks = vpicks[vi * 64:(vi + 1) * 64]
                x, y = batch(d, picks, h)
                errs.append(float(torch.nn.functional.mse_loss(model(x.to(dev)), y.to(dev))))
            vl = float(np.mean(errs))
        hist.append(dict(epoch=ep, val_rmse=vl, elapsed=time.time() - t0))
        if vl < best - 1e-6:
            best = vl
            torch.save(dict(state_dict=model.state_dict(), kind=kind, graph=graph, h=h,
                            seed=seed), out / 'best.pt')
        bad = 0 if vl < best - 1e-6 + 1e-12 else bad + 1
        if bad >= 6:
            break
    summ.write_text(json.dumps(dict(tag=tag, kind=kind, graph=graph, h=h, seed=seed,
                                    best_val_rmse=best, epochs=len(hist),
                                    elapsed_s=hist[-1]['elapsed'],
                                    params=sum(p.numel() for p in model.parameters()))))
    print(tag, 'done best_val', round(best, 5), 'epochs', len(hist), flush=True)
    del model, opt
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--kinds', nargs='+', default=['m1', 'm2', 'm3'])
    ap.add_argument('--graphs', nargs='+', default=['real'])
    ap.add_argument('--hs', nargs='+', type=int, default=[1])
    ap.add_argument('--seeds', nargs='+', type=int, default=list(SEEDS))
    args = ap.parse_args()
    for kind in args.kinds:
        for graph in args.graphs:
            for h in args.hs:
                for seed in args.seeds:
                    train_one(kind, graph, seed, h)
    print('TRAIN REAL COMPLETE', flush=True)


if __name__ == '__main__':
    main()
