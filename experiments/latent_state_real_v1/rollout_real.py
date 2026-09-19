"""Stage 6C: autonomous rollout statistics (200 frames from test-session
contexts; own predictions fed back; no future true activity).

Models: B0a persistence, B0b AR(5), B2 structural-linear, M2, M3.
Statistics per rollout: ACF (lags 1..50), mean power spectrum, region
amplitude std, FC matrix, stability (frac frames |y|>5*train std).
Real-data reference computed per test session.
Output: results_real/rollout_stats.csv + rollout diagnostic arrays .npz
"""
import json
from pathlib import Path
import numpy as np
import torch
from models_real import build_real

ROOT = Path('.')
L = 32
T_ROLL = 200
SEEDS = (1234, 1235, 1236, 1237, 1238)


def acf(x, lags):
    x = x - x.mean(0)
    den = (x ** 2).sum(0).clip(min=1e-9)
    return np.array([np.sum(x[:-l] * x[l:], 0) / den for l in lags]).mean(1)


def stats_pack(y, amp_ref):
    lags = np.arange(1, 51)
    a = acf(y, lags)
    ps = (np.abs(np.fft.rfft(y - y.mean(0), axis=0)) ** 2).mean(1)
    fc = np.corrcoef(y.T)
    iu = np.triu_indices(y.shape[1], 1)
    return dict(acf=a, ps=ps, amp=y.std(0), fc_off=fc[iu],
                unstable=float((np.abs(y) > 5 * amp_ref[None]).mean()))


def main():
    d = torch.load(ROOT / 'data_real.pt', weights_only=False)
    dev = torch.device('cuda')
    A = torch.tensor(d['A_real'], dtype=torch.float32)
    An = torch.log1p(A.abs()); An = An / An.sum(0, keepdim=True).clamp(min=1e-9)
    An = An.numpy()
    test_fids = [f for f, s in d['split_of'].items() if s == 'test']
    amp_ref = np.concatenate([d['sessions'][f]['y'] for f in test_fids]).std(0)

    # fit AR5/B2 exactly as baselines_real (train-only)
    train_fids = [f for f, s in d['split_of'].items() if s == 'train']
    Xtr, Ytr = [], []
    for fid in train_fids:
        y = d['sessions'][fid]['y']
        for t in range(5, y.shape[0]):
            Xtr.append(y[t - 5:t].T.reshape(-1)); Ytr.append(y[t])
    W_ar, *_ = np.linalg.lstsq(np.c_[np.stack(Xtr), np.ones(len(Xtr))], np.stack(Ytr), rcond=None)
    Phi2, Y2 = [], []
    for fid in train_fids:
        y = d['sessions'][fid]['y']
        Ay = y @ An
        Phi2.append(np.stack((y, Ay), -1)[:-1].reshape(-1, 2))
        Y2.append(y[1:].reshape(-1))
    Phi2 = np.concatenate(Phi2); Y2 = np.concatenate(Y2)
    R = 37
    E = np.zeros((len(Y2), R)); E[np.arange(len(Y2)), np.tile(np.arange(R), len(Y2) // R)] = 1.0
    W2, *_ = np.linalg.lstsq(np.concatenate((Phi2, E), 1), Y2, rcond=None)
    a_hat, b_hat, c_hat = W2[0], W2[1], W2[2:]

    def roll_ar(y0, T):
        w = y0[-5:].copy(); out = []
        for _ in range(T):
            nxt = (np.c_[w.T.reshape(1, -1), np.ones(1)] @ W_ar)[0]
            out.append(nxt); w = np.vstack((w[1:], nxt[None]))
        return np.stack(out)

    def roll_b2(y0, T):
        cur = y0[-1].copy(); out = []
        for _ in range(T):
            cur = a_hat * cur + b_hat * (cur @ An) + c_hat
            out.append(cur)
        return np.stack(out)

    models = {}
    for kind in ('m2', 'm3'):
        for seed in SEEDS:
            meta = json.loads((ROOT / 'results_real' / 'checkpoints' / f'{kind}_real_h1_seed{seed}' / 'summary.json').read_text())
            m = build_real(kind, A, 37).to(dev)
            m.load_state_dict(torch.load(ROOT / 'results_real' / 'checkpoints' / meta['tag'] / 'best.pt',
                                         map_location=dev, weights_only=False)['state_dict'])
            models[f'{kind}_s{seed}'] = m.eval()

    rows = []
    packs = {}
    for fid in test_fids:
        y = d['sessions'][fid]['y']
        y0 = y[:L]
        real_pack = stats_pack(y[L:L + T_ROLL], amp_ref)
        packs[f'{fid}/real'] = real_pack
        cands = {'b0a_persist': np.repeat(y0[-1:], T_ROLL, 0),
                 'b0b_ar5': roll_ar(y0, T_ROLL),
                 'b2_structlin': roll_b2(y0, T_ROLL)}
        for name, m in models.items():
            with torch.no_grad():
                hist = [torch.tensor(v, dtype=torch.float32, device=dev) for v in y0]
                out = []
                for _ in range(T_ROLL):
                    x = torch.stack(hist[-L:])[None]
                    nxt = m(x)[0]
                    out.append(nxt.cpu().numpy())
                    hist.append(nxt.detach())
            cands[name] = np.stack(out)
        for name, yr in cands.items():
            pk = stats_pack(yr, amp_ref)
            packs[f'{fid}/{name}'] = pk
            acf_err = float(np.abs(pk['acf'] - real_pack['acf']).mean())
            ps_err = float(np.abs(np.log(pk['ps'] + 1e-9) - np.log(real_pack['ps'] + 1e-9)).mean())
            fc_corr = float(np.corrcoef(pk['fc_off'], real_pack['fc_off'])[0, 1])
            amp_ratio = float(pk['amp'].mean() / real_pack['amp'].mean())
            rows.append(dict(session=fid, model=name, acf_err=acf_err, ps_err=ps_err,
                             fc_corr=fc_corr, amp_ratio=amp_ratio, unstable_frac=pk['unstable']))
        print(fid, 'done', flush=True)
    np.savez(ROOT / 'results_real' / 'rollout_packs.npz', **{k: json.dumps(
        {kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv) for kk, vv in v.items()})
        for k, v in packs.items()})
    import csv
    with (ROOT / 'results_real' / 'rollout_stats.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print('ROLLOUT STATS COMPLETE', len(rows), flush=True)


if __name__ == '__main__':
    main()
