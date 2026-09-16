"""v7 training + natural evaluation driver."""
import argparse
import csv
import json
import time
from dataclasses import asdict, replace
from pathlib import Path
import torch
from config import Config
from connectome import Connectome
from lif_adapt_v7 import AdaptationLIFSimulator, AdaptConfig, adaptation_reference
from latent_data import sample_indices, windows
from metrics import compute_loss
from eval_latent import calibrate_threshold, summarize
from train_latent import atomic_save
from calibrate_latent import write_csv
from models.residual_v7 import build_v7

ROOT = Path('results/latent_state_v7')
CKPTS = ROOT / 'checkpoints'
LABELS = ('k1', 'k2', 'set', 'ordered', 'oracle')


def setup_cfg():
    sel = json.loads((Path('results/latent_state_v1/teacher_calibration/selection.json')).read_text())
    cfg = Config(**sel['config'])
    return replace(cfg, n_train_traj=512, n_val_traj=64, n_test_seen_traj=64, n_test_traj=64)


def datasets(regime, cfg, adapt):
    conn = Connectome.generate(cfg)
    path = ROOT / 'data' / f'{regime}.pt'
    key = dict(config=asdict(cfg), adapt=asdict(adapt), protocol='pretransition_v7')
    if path.exists():
        blob = torch.load(path, map_location='cpu', weights_only=False)
        if blob['key'] != key:
            raise ValueError('cache mismatch')
        return conn, {sp: {k: v.to('cuda') for k, v in d.items()} for sp, d in blob['data'].items()}
    sim = AdaptationLIFSimulator(conn, cfg, torch.device('cuda'), adapt)
    all_data = {}
    for sp, count in (('train', cfg.n_train_traj), ('val', cfg.n_val_traj),
                      ('test_seen', cfg.n_test_seen_traj), ('test_ood', cfg.n_test_traj)):
        chunks = []
        for i in range(0, count, 64):
            d = sim.generate([cfg.traj_seed(sp, j) for j in range(i, min(i + 64, count))], sp)
            chunks.append({k: v.cpu() for k, v in d.items()})
        all_data[sp] = {k: torch.cat([d[k] for d in chunks]) for k in chunks[0]}
        print('v7-data', regime, sp, count, 'rate', float(all_data[sp]['states'][..., 1].mean()), flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_save(dict(key=key, data=all_data), path)
    return conn, {sp: {k: v.to('cuda') for k, v in d.items()} for sp, d in all_data.items()}


def train_one(kind, seed, conn, cfg, data, epochs=24, steps=48, batch=16):
    directory = CKPTS / f'{kind}_seed{seed}'
    directory.mkdir(parents=True, exist_ok=True)
    summary_path = ROOT / 'natural' / 'training' / f'{kind}_seed{seed}.json'
    if summary_path.exists():
        return json.loads(summary_path.read_text())
    torch.manual_seed(seed)
    model = build_v7(kind, conn, cfg).cuda()
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
            a = data['train']['a'][b, t] if model.oracle else None
            opt.zero_grad(set_to_none=True)
            out = model(x, a=a)
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
            tot, cnt = 0., 0
            for offset in range(0, len(bi), 32):
                b, t = bi[offset:offset + 32], ti[offset:offset + 32]
                x, y = windows(data['val'], b, t, model.k)
                a = data['val']['a'][b, t] if model.oracle else None
                out = model(x, a=a)
                loss, _ = compute_loss(out, y, cfg, pw)
                tot += float(loss) * len(b); cnt += len(b)
        vl = tot / cnt
        history.append(dict(epoch=epoch, train_loss=sum(losses) / len(losses), val_loss=vl,
                            elapsed=time.monotonic() - t0))
        improved = vl < best - 1e-6
        if improved:
            best = vl
            atomic_save(dict(state_dict=model.state_dict(), kind=kind, seed=seed, epoch=epoch,
                             config=asdict(cfg)), directory / 'best_val.pt')
        bad = 0 if improved else bad + 1
        atomic_save(dict(state_dict=model.state_dict(), optimizer=opt.state_dict(),
                         history=history, best=best, bad=bad, epoch=epoch,
                         rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all()), last)
        write_csv(ROOT / 'natural' / 'training' / f'{kind}_seed{seed}.csv', history)
        print(f'{kind} s={seed} ep={epoch} loss={vl:.5f}', flush=True)
        if bad >= 6:
            break
    summary = dict(kind=kind, seed=seed, epochs=len(history), best=best,
                   params=sum(p.numel() for p in model.parameters()),
                   checkpoint=str(directory / 'best_val.pt'))
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))
    del model, opt
    torch.cuda.empty_cache()
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--regime', choices=['adapt', 'negctrl'], default='adapt')
    p.add_argument('--seeds', nargs='+', type=int, default=[1234, 1235, 1236, 1237, 1238])
    p.add_argument('--labels', nargs='+', default=list(LABELS))
    args = p.parse_args()
    torch.set_num_threads(2)
    cfg = setup_cfg()
    adapt = AdaptConfig() if args.regime == 'adapt' else AdaptConfig(beta=0.0)
    conn, data = datasets(args.regime, cfg, adapt)
    for seed in args.seeds:
        for label in args.labels:
            train_one(label, seed, conn, cfg, data)
    print('V7 TRAIN COMPLETE', args.regime, flush=True)


if __name__ == '__main__':
    main()
