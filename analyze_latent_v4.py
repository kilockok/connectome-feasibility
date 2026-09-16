"""v4 evaluation: one-step only (rollout attribution was closed in v3)."""
import argparse
import json
from dataclasses import replace
from pathlib import Path
import torch
from run_latent_v4 import ROOT, base_data, obs_subset, subset_connectome, slice_data
from run_latent_v2 import setup as setup_v2
from train_latent import atomic_save
from models.history_set_encoder import load_model_v4
from analyze_latent_v3 import onestep_full


def evaluate_v4_entry(summary, tag, conn, data):
    label, seed = summary['label'], summary['seed']
    outpath = ROOT / 'eval' / 'entries' / f'{tag}_{label}_{seed}.json'
    if outpath.exists():
        return json.loads(outpath.read_text())
    model, blob = load_model_v4(summary, conn, torch.device('cuda'))
    control, threshold = summary['control'], blob['threshold']
    result = dict(tag=tag, label=label, seed=seed, params=summary['params'],
                  epoch=blob['epoch'], threshold=threshold, one_step={})
    raw = {}
    for sp in ('val', 'test_seen', 'test_ood'):
        m, o, y = onestep_full(model, data[sp], threshold, control=control)
        result['one_step'][sp] = m
        raw[sp] = dict(out=o, target=y)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    atomic_save(raw, outpath.with_suffix('.pt'))
    tmp = outpath.with_suffix('.tmp')
    tmp.write_text(json.dumps(result, indent=2, allow_nan=False))
    tmp.replace(outpath)
    print('V4-EVAL', tag, label, seed, result['one_step']['test_seen']['pooled'], flush=True)
    del model, raw
    torch.cuda.empty_cache()
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tag', required=True, help="e.g. n100, n250, n1000obs50_degree")
    args = p.parse_args()
    torch.set_num_threads(2)
    cfg, lc, _ = setup_v2('hidden')
    if 'obs' in args.tag:
        import re
        m = re.match(r'n(\d+)obs(\d+)_(\w+)', args.tag)
        n, n_obs, protocol = int(m.group(1)), int(m.group(2)), m.group(3)
        data, _ = base_data(n, cfg, lc)
        cfgN = replace(cfg, n_neurons=n)
        conn = Connectome.generate(cfgN)
        obs = torch.load(ROOT / 'data' / f'obs_{args.tag}.pt')
        conn = subset_connectome(conn, obs)
        data = slice_data(data, obs.cuda())
    else:
        n = int(args.tag[1:])
        data, _ = base_data(n, cfg, lc)
        cfgN = replace(cfg, n_neurons=n)
        conn = Connectome.generate(cfgN)
    for path in sorted((ROOT / args.tag / 'training').glob('*.json')):
        evaluate_v4_entry(json.loads(path.read_text()), args.tag, conn, data)


if __name__ == '__main__':
    from connectome import Connectome
    main()
