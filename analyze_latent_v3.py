"""v3 per-entry evaluation: pooled + per-trajectory one-step, rollout, raw blobs."""
import json
from pathlib import Path
import torch
from run_latent_v3 import ROOT, datasets_v3
from run_latent_v2 import setup as setup_v2
from train_latent import atomic_save
from train_latent_v2 import load_model_v2
from latent_data import sample_indices, windows, history_control
from eval_latent import summarize, rollout_prediction, rollout_summary


@torch.no_grad()
def onestep_full(model, data, threshold, count=1024, seed=8001, control='ordered'):
    """Pooled summarize() plus per-trajectory macro F1 (both-empty F1=0) and active-only."""
    model.eval()
    bi, ti = sample_indices(data, count, seed)
    sg = torch.Generator().manual_seed(seed + 19)
    outs, tgts = [], []
    for start in range(0, count, 32):
        b, t = bi[start:start + 32], ti[start:start + 32]
        x, y = windows(data, b, t, model.k)
        x = history_control(x, control, sg)
        z = data['z'][b.to(x.device), t.to(x.device)] if model.oracle else None
        outs.append({k: v.cpu() for k, v in model(x, z=z).items()})
        tgts.append(y.cpu())
    out = {k: torch.cat([o[k] for o in outs]) for k in outs[0]}
    y = torch.cat(tgts)
    pooled = summarize(out, y, threshold)
    p = (out['s_logits'].sigmoid() > threshold).float()
    t = y[..., 1]
    tp = (p * t).sum(-1); fp = (p * (1 - t)).sum(-1); fn = ((1 - p) * t).sum(-1)
    ntraj = len(data['states'])
    totals = torch.zeros(ntraj, 3).index_add_(0, bi, torch.stack((tp, fp, fn), -1))
    f1 = 2 * totals[:, 0] / (2 * totals[:, 0] + totals[:, 1] + totals[:, 2]).clamp(min=1)
    active = (t.sum(-1) > 0).float()
    act = torch.zeros(ntraj).index_add_(0, bi, active) > 0
    return dict(pooled=pooled,
                macro_f1=float(f1.mean()),
                macro_f1_active=float(f1[act].mean()) if act.any() else None,
                n_windows=count), out, y


def evaluate_replication_entry(summary, regime, cfg, lc, conn, data):
    label, seed = summary['label'], summary['seed']
    outpath = ROOT / 'eval' / 'entries' / f'{regime}_{label}_{seed}.json'
    if outpath.exists():
        return json.loads(outpath.read_text())
    model, blob = load_model_v2(summary, conn, torch.device('cuda'))
    control, threshold = summary['control'], blob['threshold']
    result = dict(regime=regime, label=label, seed=seed, params=summary['params'],
                  epoch=blob['epoch'], threshold=threshold, one_step={}, rollout={})
    raw = {}
    for sp in ('val', 'test_seen', 'test_ood'):
        m, o, y = onestep_full(model, data[sp], threshold, control=control)
        result['one_step'][sp] = m
        rp, rt = rollout_prediction(model, data[sp], cfg, threshold, n=32, horizon=200, control=control)
        result['rollout'][sp] = rollout_summary(rp, rt)
        raw[sp] = dict(out=o, target=y, rollout_pred=rp, rollout_true=rt)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    atomic_save(raw, outpath.with_suffix('.pt'))
    tmp = outpath.with_suffix('.tmp')
    tmp.write_text(json.dumps(result, indent=2, allow_nan=False))
    tmp.replace(outpath)
    print('V3-EVAL', regime, label, seed, result['one_step']['test_seen']['pooled'], flush=True)
    del model, raw
    torch.cuda.empty_cache()
    return result


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--regime', choices=['hidden', 'markov'], required=True)
    args = p.parse_args()
    torch.set_num_threads(2)
    cfg, lc, conn = setup_v2(args.regime)
    data = datasets_v3(args.regime)
    for path in sorted((ROOT / 'replication' / args.regime / 'training').glob('*.json')):
        evaluate_replication_entry(json.loads(path.read_text()), args.regime, cfg, lc, conn, data)


if __name__ == '__main__':
    main()
