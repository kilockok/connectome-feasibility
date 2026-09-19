"""Stage 6 evaluation for latent_state_real_v1.

A. one-step + teacher-forced multi-horizon (h in 1,2,4,8): RMSE/MAE/R2,
   deterministic eval windows (512 per split, seed 9001), per-seed models.
B. autonomous rollout from the h=1 models (200 frames, own predictions
   fed back; NO future true activity): long-term statistics vs real data.
C. controls: m3 z-shuffle (eval-time latent permutation).

Outputs: results_real/onestep.csv, results_real/horizon.csv,
results_real/rollout_stats.csv, results_real/zshuffle.csv
"""
import csv, json
from pathlib import Path
import numpy as np
import torch
from models_real import build_real

ROOT = Path('.')
L = 32
HS = (1, 2, 4, 8)
SEEDS = (1234, 1235, 1236, 1237, 1238)
ROLLOUT_T = 200


def eval_windows_real(model, d, fids, h, dev, count=512, seed=9001, zshuffle=False):
    g = np.random.default_rng(seed)
    errs, maes, res, targ = [], [], [], []
    for _ in range(count):
        fid = fids[int(g.integers(len(fids)))]
        y = d['sessions'][fid]['y']
        t = int(g.integers(L, y.shape[0] - h - 1))
        x = torch.tensor(y[t - L + 1:t + 1][None], dtype=torch.float32, device=dev)
        yt = y[t + h]
        with torch.no_grad():
            if zshuffle:
                # two windows: swap their latents (pairwise z-shuffle)
                fid2 = fids[int(g.integers(len(fids)))]
                y2 = d['sessions'][fid2]['y']
                t2 = int(g.integers(L, y2.shape[0] - h - 1))
                x2 = torch.tensor(y2[t2 - L + 1:t2 + 1][None], dtype=torch.float32, device=dev)
                zb = torch.cat([model.z_last(x)[:, None], model.z_last(x2)[:, None]], 0)
                yh = model(x, z_override=zb[1:2])[0].cpu().numpy()
            else:
                yh = model(x)[0].cpu().numpy()
        errs.append(float(np.mean((yh - yt) ** 2)))
        maes.append(float(np.mean(np.abs(yh - yt))))
        res.append(yh); targ.append(yt)
    res = np.stack(res); targ = np.stack(targ)
    ss_res = ((res - targ) ** 2).sum()
    ss_tot = ((targ - targ.mean(0)) ** 2).sum()
    return dict(rmse=float(np.mean(errs) ** 0.5), mae=float(np.mean(maes)),
                r2=float(1 - ss_res / max(ss_tot, 1e-9)))


@torch.no_grad()
def rollout(model, y0, T, dev):
    """Autonomous: own predictions fed back. y0 [L,R] context."""
    hist = [torch.tensor(v, dtype=torch.float32, device=dev) for v in y0]
    out = []
    for _ in range(T):
        x = torch.stack(hist[-L:])[None]
        nxt = model(x)[0]
        out.append(nxt.cpu().numpy())
        hist.append(nxt.detach())
    return np.stack(out)


def acf(x, lags):
    """mean per-region autocorrelation at given lags. x [T,R]"""
    x = x - x.mean(0)
    den = (x ** 2).sum(0).clip(min=1e-9)
    return np.array([np.mean([(x[:len(x) - l] * x[l:]).sum(0) / den for _ in [0]][0] if False
                             else np.sum(x[:-l] * x[l:], 0) / den) for l in lags])


def stats_pack(y):
    """long-term statistics pack for a [T,R] series."""
    lags = np.arange(1, 51)
    a = acf(y, lags)
    fr = np.fft.rfft(y - y.mean(0), axis=0)
    ps = (np.abs(fr) ** 2).mean(1)
    amp = y.std(0)
    fc = np.corrcoef(y.T)
    iu = np.triu_indices(y.shape[1], 1)
    return dict(acf=a, lags=lags, ps=ps, amp=amp, fc_off=fc[iu])


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', nargs='+', type=int, default=list(SEEDS))
    args = ap.parse_args()
    d = torch.load(ROOT / 'data_real.pt', weights_only=False)
    dev = torch.device('cuda')
    A_real = torch.tensor(d['A_real'], dtype=torch.float32)
    fids = {sp: [f for f, s in d['split_of'].items() if s == sp] for sp in ('val', 'test')}

    ck = ROOT / 'results_real' / 'checkpoints'
    rows, hrows, zrows = [], [], []
    for seed in args.seeds:
        for kind in ('b1', 'm1', 'm2', 'm3', 'm3_nolatent', 'm2_identity'):
            for graph in ('real', 'edges', 'weights'):
                if graph != 'real' and kind not in ('m2', 'm3'):
                    continue
                if kind in ('m3_nolatent', 'm2_identity') and graph != 'real':
                    continue
                sp = ck / f'{kind}_{graph}_h1_seed{seed}' / 'summary.json'
                if not sp.exists():
                    continue
                meta = json.loads(sp.read_text())
                A = A_real if graph == 'real' else torch.tensor(
                    d['A_edges_shuf' if graph == 'edges' else 'A_weights_shuf'], dtype=torch.float32)
                model = build_real(kind if kind != 'm2_identity' else 'm2', A, 37,
                                   latent=(kind != 'm3_nolatent')).to(dev)
                model.load_state_dict(torch.load(ck / meta['tag'] / 'best.pt',
                                                 map_location=dev, weights_only=False)['state_dict'])
                if kind == 'm2_identity':
                    I = torch.eye(37, device=dev)
                    model.An = I; model.enc.A = I
                model = model.eval()
                for split in ('val', 'test'):
                    r = eval_windows_real(model, d, fids[split], 1, dev)
                    rows.append(dict(model=kind, graph=graph, seed=seed, split=split, h=1, **r))
                print(meta['tag'], 'eval done', flush=True)
                # z-shuffle control for m3 real
                if kind == 'm3' and graph == 'real':
                    for split in ('val', 'test'):
                        r = eval_windows_real(model, d, fids[split], 1, dev, zshuffle=True)
                        zrows.append(dict(model='m3_zshuffle', graph=graph, seed=seed,
                                          split=split, h=1, **r))
                del model
                torch.cuda.empty_cache()
        # horizon models (teacher-forced)
        for kind in ('m1', 'm2', 'm3'):
            for h in (2, 4, 8):
                sp = ck / f'{kind}_real_h{h}_seed{seed}' / 'summary.json'
                if not sp.exists():
                    continue
                meta = json.loads(sp.read_text())
                model = build_real(kind, A_real, 37).to(dev)
                model.load_state_dict(torch.load(ck / meta['tag'] / 'best.pt',
                                                 map_location=dev, weights_only=False)['state_dict'])
                model = model.eval()
                for split in ('val', 'test'):
                    r = eval_windows_real(model, d, fids[split], h, dev)
                    hrows.append(dict(model=kind, graph='real', seed=seed, split=split, h=h, **r))
                del model
                torch.cuda.empty_cache()
        print('seed', seed, 'horizons done', flush=True)

    out = ROOT / 'results_real'
    out.mkdir(exist_ok=True)
    for fname, rr in (('onestep.csv', rows), ('horizon.csv', hrows), ('zshuffle.csv', zrows)):
        if rr:
            with (out / fname).open('w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(rr[0])); w.writeheader(); w.writerows(rr)
            print(fname, len(rr), 'rows', flush=True)
    print('EVAL CORE COMPLETE', flush=True)


if __name__ == '__main__':
    main()
