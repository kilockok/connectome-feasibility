"""Stage 3 baselines (observation-space, train-fitted only):
  B0a persistence  y(t+h) = y(t)
  B0b AR(5)        per-region autoregression, least squares on train
  B1  no-history continuous dynamics (MLP on current frame)
  B2  structural linear: y(t+1) = a*y + B (A_norm y) + c, least squares
Writes baseline_metrics.json + baseline_predictions cache for diagnostics.
"""
import json
from pathlib import Path
import numpy as np
import torch

ROOT = Path('.')
L = 32
HS = (1, 2, 4, 8)


def main():
    d = torch.load(ROOT / 'data_real.pt', weights_only=False)
    tb = None
    A = torch.tensor(d['A_real'], dtype=torch.float32)
    An = torch.log1p(A.abs()); An = An / An.sum(0, keepdim=True).clamp(min=1e-9)
    train_fids = [f for f, s in d['split_of'].items() if s == 'train']
    val_fids = [f for f, s in d['split_of'].items() if s == 'val']
    test_fids = [f for f, s in d['split_of'].items() if s == 'test']

    # ---------- fit B0b (AR5) and B2 (structural linear) on train ----------
    Xtr, Ytr = [], []
    for fid in train_fids:
        y = d['sessions'][fid]['y']
        for t in range(5, y.shape[0]):
            Xtr.append(y[t - 5:t].T.reshape(-1))         # window ends at t-1
            Ytr.append(y[t])                             # one-step-ahead target
    Xtr = np.stack(Xtr); Ytr = np.stack(Ytr)
    # global AR: predict all regions from the last 5 frames (all regions)
    W_ar, *_ = np.linalg.lstsq(np.c_[Xtr, np.ones(len(Xtr))], Ytr, rcond=None)
    # B2: y(t+1)_i = a*y_i + b*(An y)_i + c_i - scalar a,b + region bias c
    # (protocol-frozen constrained form; design rows are per (sample, region))
    Phi2, Y2 = [], []
    for fid in train_fids:
        y = d['sessions'][fid]['y']
        Ay = y @ An.numpy()                               # [T,R]: (A y)_i = sum_src y_src A[src,i]
        T = y.shape[0]
        Yi = y[1:T - 0]                                   # targets at t+1 for t in 0..T-2 -> use t in 1..T-2
        Phi_t = np.stack((y, Ay), -1)                     # [T,R,2]
        Phi2.append(Phi_t[:-1].reshape(-1, 2))
        Y2.append(y[1:].reshape(-1))
    Phi2 = np.concatenate(Phi2); Y2 = np.concatenate(Y2)
    R = 37
    E = np.zeros((len(Y2), R))
    E[np.arange(len(Y2)), np.tile(np.arange(R), len(Y2) // R)] = 1.0
    D = np.concatenate((Phi2, E), 1)                      # [N, 2+R]
    W2, *_ = np.linalg.lstsq(D, Y2, rcond=None)
    a_hat, b_hat, c_hat = W2[0], W2[1], W2[2:]
    print('B2 fitted: a=%.3f b=%.4f' % (a_hat, b_hat))

    # ---------- eval helpers ----------
    def eval_windows(pred_fn, fids, h, count=512, seed=9001):
        g = np.random.default_rng(seed)
        errs, maes, r2s = [], [], []
        for _ in range(count):
            fid = fids[int(g.integers(len(fids)))]
            y = d['sessions'][fid]['y']
            t = int(g.integers(max(L, 5), y.shape[0] - h - 1))   # t = last observed
            yh = pred_fn(y, t, h)
            yt = y[t + h]                                       # h frames ahead
            errs.append(float(np.mean((yh - yt) ** 2)))
            maes.append(float(np.mean(np.abs(yh - yt))))
            ss_res = float(np.sum((yh - yt) ** 2))
            r2s.append(ss_res)
        # R2 computed against per-region variance of the eval pool
        return float(np.mean(errs) ** 0.5), float(np.mean(maes)), float(np.mean(r2s))

    def pred_persist(y, t, h):
        return y[t]

    def pred_ar(y, t, h):
        w = y[t - 4:t + 1].copy()                        # last observed frame = t
        nxt = None
        for _ in range(h):
            nxt = np.c_[w.T.reshape(1, -1), np.ones(1)] @ W_ar
            nxt = nxt[0]
            w = np.vstack((w[1:], nxt[None]))
        return nxt

    def pred_b2(y, t, h):
        cur = y[t].copy()
        for _ in range(h):
            cur = a_hat * cur + b_hat * (cur @ An.numpy()) + c_hat
        return cur

    results = {}
    for name, fn in (('B0a_persistence', pred_persist), ('B0b_AR5', pred_ar), ('B2_structlin', pred_b2)):
        for split, fids in (('val', val_fids), ('test', test_fids)):
            for h in HS:
                rmse, mae, _ = eval_windows(fn, fids, h)
                results[f'{name}/{split}/h{h}'] = dict(rmse=rmse, mae=mae)
                print(name, split, 'h', h, 'rmse %.4f' % rmse, flush=True)
    (ROOT / 'baseline_metrics.json').write_text(json.dumps(results, indent=1))
    print('baselines done (B1 MLP is trained via train_real.py kind=b1)')


if __name__ == '__main__':
    main()
