"""Stage 7: predictive-state sufficiency of the ordered representation."""
import csv
from pathlib import Path
import numpy as np
import torch
from torch import nn
from run_latent_v3 import datasets_v3
from run_latent_v2 import setup as setup_v2
from alias_eval_v5 import load_checked
from latent_data import sample_indices, windows

ROOT = Path('results/latent_state_v5')
SEEDS = (1234, 1235, 1236, 1237, 1238)
K = 32
HORIZONS = (1, 2, 4, 8, 16)


class Pred(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, 128), nn.GELU(), nn.Linear(128, 128), nn.GELU(),
                                 nn.Linear(128, d_out))

    def forward(self, f):
        return self.net(f)


def fit(model, fx, y, fv, yv, epochs=200):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    mean, std = fx.mean(0), fx.std(0).clamp(min=1e-5)
    ym = y.mean(0)
    best, wait, state = float('inf'), 0, None
    for ep in range(epochs):
        model.train(); opt.zero_grad(set_to_none=True)
        loss = (model((fx - mean) / std) - (y - ym)).square().mean()
        loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = (model((fv - mean) / std) - (yv - ym)).square().mean()
        if vl < best - 1e-7:
            best, wait = float(vl), 0
            state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= 25:
                break
    model.load_state_dict(state)
    return model.eval(), mean, std, ym


@torch.no_grad()
def features(model, data, h, count):
    dev = next(model.parameters()).device
    bi, ti = sample_indices(data, count, 4242)
    ti = ti.clamp(max=data['stimulus'].shape[1] - h - 1)
    xs, hs, curs, olds, ys = [], [], [], [], []
    for st in range(0, count, 32):
        b, t = bi[st:st + 32], ti[st:st + 32]
        x, _ = windows(data, b, t, K)
        _, hh = model.encode(x)
        hs.append(hh.cpu())
        pop = x.mean(2)
        curs.append(torch.cat((pop[:, -1], x[:, -1, :, 1].mean(1, keepdim=True)), -1).cpu())
        olds.append(torch.cat((pop[:, :-1].mean(1), pop[:, :-1].std(1, unbiased=False),
                               pop[:, -2] - pop[:, -8:-1].mean(1) if K >= 8 else pop[:, -2] - pop[:, -2],
                               x[:, :-1, :, 1].mean((1, 2))[:, None]), -1).cpu())
        ys.append(data['states'][b, t + h, :, 0].cpu())
    return (torch.cat(hs), torch.cat(curs), torch.cat(olds), torch.cat(ys))


def r2(pred, true):
    pred, true = pred.double(), true.double()
    den = (true - true.mean(0)).square().sum()
    return float(1 - (pred - true).square().sum() / den.clamp(min=1e-12))


def main():
    torch.set_num_threads(2)
    cfg, lc, conn = setup_v2('hidden')
    data = datasets_v3('hidden')
    rows = []
    for seed in SEEDS:
        model, th = load_checked('global_k32', seed, conn, torch.device('cuda'))
        for h in HORIZONS:
            tr = features(model, data['train'], h, 2048)
            va = features(model, data['val'], h, 1024)
            te = features(model, data['test_seen'], h, 1024)
            for name, key in (('x+h', lambda c: torch.cat((c[1], c[0]), -1)),
                              ('x+h+raw', lambda c: torch.cat((c[1], c[0], c[2]), -1))):
                fx = key(tr); fv = key(va); ft = key(te)
                mdl, mean, std, ym = fit(Pred(fx.shape[1], tr[3].shape[1]).cuda(), fx.cuda(), tr[3].cuda(),
                                         fv.cuda(), va[3].cuda())
                with torch.no_grad():
                    pred = mdl((ft.cuda() - mean.cuda()) / std.cuda()) + ym.cuda()
                rows.append(dict(seed=seed, horizon=h, condition=name, v_r2=r2(pred.cpu(), te[3])))
        print('sufficiency done seed', seed, flush=True)
        del model
        torch.cuda.empty_cache()
    with (ROOT / 'table6_predictive_sufficiency.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print('ROWS', len(rows), flush=True)


if __name__ == '__main__':
    main()
