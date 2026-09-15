"""Conditional U=4/U=8 unroll finetuning (core v2 hypothesis passed).

From each base model's best_val checkpoint, short scheduled-unroll finetuning
under the v2 pretransition contract: feed the model's OWN soft next-state
(sigmoid spikes keep the chain differentiable) with the true future stimulus,
loss on every unrolled step. No DAgger, no tangent, U<=8 per protocol.
"""
import json
import time
from dataclasses import asdict
from pathlib import Path
import torch
from run_latent_v2 import ROOT, CKPTS, setup, datasets
from train_latent import atomic_save
from train_latent_v2 import load_model_v2
from models.latent_temporal_v2 import build_v2
from latent_data import sample_indices, windows
from metrics import compute_loss
from eval_latent import predict_windows, calibrate_threshold, summarize
from calibrate_latent import write_csv

PLAN = [('global_k32', 4), ('global_k32', 8), ('gnn_k1', 4), ('gnn_k1', 8)]


@torch.no_grad()
def val_score(model, data, cfg):
    o, y, _ = predict_windows(model, data, 512, seed=8001)
    th = calibrate_threshold(o, y)
    return summarize(o, y, th), th


def soft_state(out, cfg):
    v = out['v'].clamp(cfg.v_min, cfg.v_th * 3)
    s = out['s_logits'].sigmoid()
    r = out['r'].clamp(0, 1)
    return torch.stack((v, s, r), -1)


def train_unroll(base_label, U, seed, conn, cfg, lc, data, epochs=4, steps=48, batch=16):
    label = f'{base_label}_u{U}'
    outdir = ROOT / 'unroll'
    directory = ROOT / 'checkpoints' / 'unroll' / f'{label}_seed{seed}'
    directory.mkdir(parents=True, exist_ok=True)
    summary_path = outdir / 'training' / f'{label}_seed{seed}.json'
    if summary_path.exists():
        return json.loads(summary_path.read_text())
    base_summary = json.loads((ROOT / 'hidden' / 'training' / f'{base_label}_seed{seed}.json').read_text())
    model, base_blob = load_model_v2(base_summary, conn, torch.device('cuda'))
    spec = dict(base_blob['spec'])
    model = model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    pw = torch.tensor(cfg.spike_pos_weight, device='cuda')
    K = model.k
    best = dict(one_step=-float('inf'), rollout=None, combined=None)
    history, bad, start_epoch = [], 0, 1
    last = directory / 'last.pt'
    if last.exists():
        resume = torch.load(last, map_location='cpu', weights_only=False)
        model.load_state_dict(resume['state_dict']); opt.load_state_dict(resume['optimizer'])
        history = resume['history']; best = resume['best']; bad = resume['bad']
        start_epoch = resume['epoch'] + 1
        torch.set_rng_state(resume['rng']); torch.cuda.set_rng_state_all(resume['cuda_rng'])
    start_time = time.monotonic()
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        bi, ti = sample_indices(data['train'], steps * batch, 300_000 + seed * 100 + epoch)
        ti = ti.clamp(max=cfg.T - U - 1)
        losses = []
        for offset in range(0, len(bi), batch):
            b, t = bi[offset:offset + batch], ti[offset:offset + batch]
            x, _ = windows(data['train'], b, t, K)
            z = data['train']['z'][b.to(x.device), t.to(x.device)] if model.oracle else None
            total = 0.
            for u in range(U):
                pred = model(x, z=z)
                y = data['train']['states'][b.to(x.device), t.to(x.device) + u + 1]
                loss, _ = compute_loss(pred, y, cfg, pw)
                total = total + loss
                if u + 1 < U:
                    s = soft_state(pred, cfg)
                    stim = data['train']['stimulus'][b.to(x.device), t.to(x.device) + u + 1]
                    x = torch.cat((x[:, 1:], torch.cat((s, stim[..., None]), -1)[:, None]), 1)
            loss = total / U
            if not torch.isfinite(loss):
                raise RuntimeError(f'Non-finite unroll loss {label}/{seed}/{epoch}')
            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if not torch.isfinite(grad):
                raise RuntimeError('Non-finite gradient')
            opt.step()
            losses.append(float(loss.detach()))
        model.eval()
        vm, th = val_score(model, data['val'], cfg)
        vl = float(vm['spike_f1'])
        row = dict(epoch=epoch, train_loss=sum(losses) / len(losses), val_f1=vl,
                   val_v_rmse=vm['v_rmse'], threshold=th,
                   elapsed_seconds=time.monotonic() - start_time)
        history.append(row)
        blob = dict(state_dict=model.state_dict(), spec=spec, label=label, seed=seed, epoch=epoch,
                    config=asdict(cfg), latent=asdict(lc), threshold=th, control='ordered',
                    validation=row, unroll=U, base=base_summary['checkpoint'],
                    protocol='latent_state_v2_unroll')
        improved = vl > best['one_step'] + 1e-6
        if improved:
            best['one_step'] = vl
            atomic_save(blob, directory / 'best_val.pt')
        bad = 0 if improved else bad + 1
        atomic_save(dict(**blob, optimizer=opt.state_dict(), history=history, best=best, bad=bad,
                         rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all()), last)
        write_csv(outdir / 'training' / f'{label}_seed{seed}.csv', history)
        print(f'{label} seed={seed} ep={epoch} unroll_loss={row["train_loss"]:.5f} '
              f'valF1={vl:.4f} V={row["val_v_rmse"]:.4f} sec={row["elapsed_seconds"]:.1f}', flush=True)
        if bad >= 3:
            break
    summary = dict(label=label, seed=seed, spec=spec, control='ordered',
                   params=sum(p.numel() for p in model.parameters()), epochs=len(history),
                   training_seconds=history[-1]['elapsed_seconds'] if history else 0,
                   best=best, checkpoint=str(directory / 'best_val.pt'),
                   unroll=U, base=base_summary['checkpoint'])
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))
    del model, opt
    torch.cuda.empty_cache()
    return summary


def main():
    torch.set_num_threads(2)
    cfg, lc, conn = setup('hidden')
    data = datasets('hidden', cfg, lc, conn)
    (ROOT / 'unroll').mkdir(exist_ok=True)
    (ROOT / 'unroll' / 'plan.json').write_text(json.dumps(
        dict(plan=[dict(base=b, unroll=u) for b, u in PLAN], epochs=4, steps=48, lr=1e-4,
             feeding='own soft next-state (sigmoid spikes) + true future stimulus',
             note='conditional stage entered because core v2 temporal gates passed'), indent=2))
    for seed in (1234, 1235, 1236):
        for base_label, U in PLAN:
            train_unroll(base_label, U, seed, conn, cfg, lc, data)
    print('UNROLL STAGE COMPLETE', flush=True)


if __name__ == '__main__':
    main()
