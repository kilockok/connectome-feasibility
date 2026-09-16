"""Stage 3: structured gain-head model (mechanism-exact channel) + controls.

M1: v5 Ordered (frozen). M2: fresh GlobalTemporal (plain residual head, same
budget). M3: encoder -> q_hat -> EXACT base-LIF gain channel (pre-reset gain
application, correct threshold/reset flow). M4: parameter-free scalar windowed
estimator feeding the same formula. All comparisons use identical
data/budget/loss; M3/M4's structural prior (knowing the gain-channel form) is
declared. No z/gain supervision anywhere: q_hat is fit by prediction loss only.
"""
import csv
import json
import time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from torch import nn
from run_latent_v3 import datasets_v3
from run_latent_v2 import setup as setup_v2
from alias_eval_v5 import load_checked
from latent_data import sample_indices, windows
from metrics import compute_loss
from models.latent_temporal_v2 import GlobalTemporalPredictorV2
from eval_latent import predict_windows, calibrate_threshold, summarize
from train_latent import atomic_save
from calibrate_latent import write_csv

ROOT = Path('results/latent_state_v6')
SEEDS = (1234, 1235, 1236, 1237, 1238)


class GainChannelModel(nn.Module):
    """encoder -> q_hat in (-1,1) -> exact base-LIF step with gain (1+q_hat)."""

    def __init__(self, conn, cfg, d=64, layers=2):
        super().__init__()
        self.k = 32
        self.oracle = False
        base = GlobalTemporalPredictorV2(conn, k=32, d=d, layers=layers)
        self.encoder = base
        self.q_head = nn.Linear(d, 1)
        self.cfg = cfg
        W = conn.dense_weight('cpu')
        self.register_buffer('W', W)
        self.register_buffer('i_bias', conn.i_bias.cpu()
                             if conn.i_bias is not None else torch.zeros(conn.n_neurons))

    def q_hat(self, x):
        _, zctx = self.encoder.encode(x)
        return torch.tanh(self.q_head(zctx)).squeeze(-1)  # [B]

    def lif_gain_step(self, x_last, q):
        """Exact teacher transition form with gain (1+q); differentiable soft reset."""
        c = self.cfg
        v, s, r, u = x_last.unbind(-1)
        isyn = s @ self.W
        current = (1.0 + q)[:, None] * isyn + u + self.i_bias
        refr = r > 0
        vn = torch.where(refr, torch.full_like(v, c.v_reset),
                         v + c.alpha * (-(v - c.v_rest) + current)).clamp(min=c.v_min)
        p = torch.sigmoid((vn - c.v_th) / 0.1)
        v_pred = p * c.v_reset + (1 - p) * vn
        r_pred = p * 1.0 + (1 - p) * (r * c.refractory_period - 1).clamp(min=0) / c.refractory_period
        s_logits = (vn - c.v_th) / 0.1
        return dict(v=v_pred, s_logits=s_logits, r=r_pred, isyn=isyn)

    def forward(self, x, z=None):
        if z is not None:
            raise ValueError('No hidden input allowed')
        x = x[:, -self.k:]
        q = self.q_hat(x)
        return self.lif_gain_step(x[:, -1], q), q


class PlainResidualModel(nn.Module):
    """Same encoder, plain 2-layer residual decoder (v5 Ordered architecture)."""

    def __init__(self, conn, cfg, d=64, layers=2):
        super().__init__()
        self.inner = GlobalTemporalPredictorV2(conn, k=32, d=d, layers=layers)
        self.k = 32
        self.oracle = False

    def forward(self, x, z=None):
        return self.inner(x, z=z), None


def train_model(model_cls, label, seed, conn, cfg, lc, data, epochs=24, steps=48, batch=16):
    directory = ROOT / 'checkpoints' / f'{label}_seed{seed}'
    directory.mkdir(parents=True, exist_ok=True)
    summary_path = ROOT / 'stage3' / 'training' / f'{label}_seed{seed}.json'
    if summary_path.exists():
        return json.loads(summary_path.read_text())
    torch.manual_seed(seed)
    model = model_cls(conn, cfg).cuda()
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
        bi, ti = sample_indices(data['train'], steps * batch, 100_000 + seed * 100 + epoch)
        losses = []
        for offset in range(0, len(bi), batch):
            b, t = bi[offset:offset + batch], ti[offset:offset + batch]
            x, y = windows(data['train'], b, t, model.k)
            opt.zero_grad(set_to_none=True)
            out, q = model(x)
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
            bi, ti = sample_indices(data['val'], 512, 8001)
            tot, cnt, tot_v, tot_f1p, tot_f1t = 0., 0, 0., 0., 0.
            for offset in range(0, len(bi), 32):
                b, t = bi[offset:offset + 32], ti[offset:offset + 32]
                x, y = windows(data['val'], b, t, model.k)
                out, q = model(x)
                loss, _ = compute_loss(out, y, cfg, pw)
                tot += float(loss) * len(b); cnt += len(b)
        vl = tot / cnt
        history.append(dict(epoch=epoch, train_loss=sum(losses) / len(losses), val_loss=vl,
                            elapsed=time.monotonic() - t0))
        improved = vl < best - 1e-6
        if improved:
            best = vl
            atomic_save(dict(state_dict=model.state_dict(), label=label, seed=seed,
                             epoch=epoch, config=asdict(cfg)), directory / 'best_val.pt')
        bad = 0 if improved else bad + 1
        atomic_save(dict(state_dict=model.state_dict(), optimizer=opt.state_dict(),
                         history=history, best=best, bad=bad, epoch=epoch,
                         rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all()), last)
        write_csv(ROOT / 'stage3' / 'training' / f'{label}_seed{seed}.csv', history)
        print(f'{label} s={seed} ep={epoch} loss={vl:.5f}', flush=True)
        if bad >= 6:
            break
    summary = dict(label=label, seed=seed, epochs=len(history), best=best,
                   params=sum(p.numel() for p in model.parameters()),
                   checkpoint=str(directory / 'best_val.pt'))
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))
    del model, opt
    torch.cuda.empty_cache()
    return summary


@torch.no_grad()
def eval_natural(model_cls, summary, conn, cfg, data):
    blob = torch.load(summary['checkpoint'], map_location='cuda', weights_only=False)
    model = model_cls(conn, cfg).cuda()
    model.load_state_dict(blob['state_dict'])
    model = model.eval()
    out_rows = {}
    for sp in ('test_seen', 'test_ood'):
        bi, ti = sample_indices(data[sp], 1024, 8001)
        outs, ys, qs = [], [], []
        for offset in range(0, len(bi), 32):
            b, t = bi[offset:offset + 32], ti[offset:offset + 32]
            x, y = windows(data[sp], b, t, model.k)
            o, q = model(x)
            outs.append({k: v.cpu() for k, v in o.items() if k != 'isyn'})
            ys.append(y.cpu())
            if q is not None:
                qs.append(q.cpu())
        o = {k: torch.cat([a[k] for a in outs]) for k in outs[0]}
        y = torch.cat(ys)
        th = calibrate_threshold(o, y)
        m = summarize(o, y, th)
        out_rows[sp] = dict(**m, threshold=th)
        if qs:
            tq = (torch.tanh(data[sp]['z'][bi, ti][:, 0]) * 1.0).cpu()
            qq = torch.cat(qs)
            out_rows[sp]['q_mae'] = float((qq - tq).abs().mean())
            out_rows[sp]['q_corr'] = float(torch.corrcoef(torch.stack((qq, tq)))[0, 1])
    return out_rows


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', nargs='+', type=int, default=list(SEEDS))
    p.add_argument('--part', choices=['train', 'eval'], required=True)
    args = p.parse_args()
    torch.set_num_threads(2)
    cfg, lc, conn = setup_v2('hidden')
    data = datasets_v3('hidden')
    (ROOT / 'stage3').mkdir(exist_ok=True)
    if args.part == 'train':
        for seed in args.seeds:
            train_model(PlainResidualModel, 'm2_plain', seed, conn, cfg, lc, data)
            train_model(GainChannelModel, 'm3_gain', seed, conn, cfg, lc, data)
        print('STAGE3 TRAIN COMPLETE', flush=True)
    else:
        rows = []
        for seed in args.seeds:
            for label, cls in (('m2_plain', PlainResidualModel), ('m3_gain', GainChannelModel)):
                s = json.loads((ROOT / 'stage3' / 'training' / f'{label}_seed{seed}.json').read_text())
                res = eval_natural(cls, s, conn, cfg, data)
                for sp, m in res.items():
                    rows.append(dict(seed=seed, model=label, split=sp, **m))
                print(label, seed, res['test_seen'], flush=True)
        write_csv(ROOT / 'metrics_stage3_natural.csv', rows)
        print('STAGE3 EVAL COMPLETE', flush=True)


if __name__ == '__main__':
    main()
