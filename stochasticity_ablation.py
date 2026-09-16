"""Stage 7-9: stochasticity decomposition.

S0: original stochastic teacher (sigma=0.008) - existing replication models.
S1: deterministic hidden teacher (sigma=0, randomized initial z) - recalibrate
    activity, retrain gnn_k1/global_k32/gshuffle/oracle x 3 seeds, rollout.
S2: diagnostic-only full-future-z oracle (sees z_pos/z_vel[t..t+8] as input);
    trained and rolled out on S0 to test whether future latent innovations are
    the binding constraint on rollout. Never a formal model.
"""
import json
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
from run_latent_v3 import ROOT
from run_latent_v2 import setup as setup_v2
from train_latent import atomic_save
from train_latent_v2 import train_one_v2, load_model_v2
from lif_latent_v2 import LatentV2Config
from models.latent_temporal_v2 import SpatialEncoderV2
from torch import nn

DETERMINISTIC_LABELS = ['gnn_k1', 'global_k32', 'gshuffle', 'oracle']
SEEDS = [1234, 1235, 1236]


def det_data(cfg, lc0, conn):
    """Generate the sigma=0 teacher dataset (own namespace, own cache key)."""
    from lif_latent_v2 import HiddenStateLIFSimulatorV2
    path = ROOT / 'deterministic_teacher' / 'data.pt'
    key = dict(latent={'alpha': lc0.alpha, 'beta': lc0.beta, 'omega': lc0.omega, 'sigma': 0.0, 'beta': 0.995, 'random_init': True},
               protocol='pretransition_v2_deterministic')
    if path.exists():
        blob = torch.load(path, map_location='cpu', weights_only=False)
        if blob['key'] != key:
            raise ValueError('deterministic cache mismatch')
        return {sp: {k: v.to('cuda') for k, v in d.items()} for sp, d in blob['data'].items()}
    lc = replace(lc0, sigma=0.0, beta=0.995, random_init=True)
    sim = HiddenStateLIFSimulatorV2(conn, cfg, torch.device('cuda'), lc)
    all_data = {}
    for sp, count in (('train', cfg.n_train_traj), ('val', cfg.n_val_traj),
                      ('test_seen', cfg.n_test_seen_traj), ('test_ood', cfg.n_test_traj)):
        chunks = []
        for i in range(0, count, 64):
            d = sim.generate([cfg.traj_seed(sp, j) for j in range(i, min(i + 64, count))], sp)
            chunks.append({k: v.cpu() for k, v in d.items()})
        all_data[sp] = {k: torch.cat([d[k] for d in chunks]) for k in chunks[0]}
        print('det-data', sp, count, 'rate', float(all_data[sp]['states'][..., 1].mean()), flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_save(dict(key=key, data=all_data), path)
    return {sp: {k: v.to('cuda') for k, v in d.items()} for sp, d in all_data.items()}


class FutureZOracle(nn.Module):
    """Diagnostic-only: spatial encoder + current AND future z path (z[t..t+H]).

    Tests whether knowing the future latent innovations suffices for stable
    rollout. Labelled diagnostic; never counts as a formal model.
    """

    def __init__(self, conn, horizon=8, d=64, layers=2):
        super().__init__()
        self.k = 1
        self.horizon = horizon
        self.oracle = True
        self.spatial = SpatialEncoderV2(conn, d, layers, 4 + 2 * (horizon + 1))
        self.decoder = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 3))

    def forward(self, x, z_future=None, return_features=False):
        if z_future is None or z_future.shape[-1] != 2 * (self.horizon + 1):
            raise ValueError('FutureZOracle requires z_future [B, 2*(H+1)]')
        b, n, _ = x[:, -1].shape
        feat = torch.cat((x[:, -1], z_future[:, None, :].expand(b, n, z_future.shape[-1])), -1)
        h = self.spatial(feat)
        y = self.decoder(h)
        out = dict(v=y[..., 0], s_logits=y[..., 1], r=y[..., 2])
        return (out, h) if return_features else out


def z_future_path(data, b, t, H):
    """[B, 2*(H+1)]: z[b, t..t+H] flattened."""
    idx = t[:, None] + torch.arange(H + 1, device=t.device)
    zz = data['z'][b[:, None].to(t.device), idx]
    return zz.flatten(1)


@torch.no_grad()
def rollout_future_oracle(model, data, cfg, threshold, n=32, horizon=200, start=32):
    from eval_latent import state_from_output, rollout_summary
    model.eval()
    n = min(n, len(data['states']))
    H = model.horizon
    b = torch.arange(n, device=data['states'].device)
    from latent_data import windows
    x, _ = windows(data, b, torch.full((n,), start, device=b.device), 1)
    pred = []
    for step in range(horizon):
        t = start + step
        zf = z_future_path(data, b, torch.full((n,), t, device=b.device), min(H, cfg.T - t - 1))
        if zf.shape[-1] != 2 * (H + 1):  # pad tail with last z
            need = 2 * (H + 1) - zf.shape[-1]
            zf = torch.cat((zf, zf[:, -2:].expand(-1, -1, -1).reshape(n, -1)[:, :need]), -1) if need > 0 else zf
        out = model(x, z_future=zf)
        state = state_from_output(out, cfg, threshold)
        pred.append(state.cpu())
        if step + 1 < horizon:
            stim = data['stimulus'][:n, t + 1]
            x = torch.cat((x[:, 1:], torch.cat((state, stim[..., None]), -1)[:, None]), 1)
    return torch.stack(pred, 1), data['states'][:n, start + 1:start + horizon + 1].cpu()


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--part', choices=['det_teacher', 'future_oracle'], required=True)
    args = p.parse_args()
    torch.set_num_threads(2)
    cfg, lc, conn = setup_v2('hidden')
    if args.part == 'det_teacher':
        data = det_data(cfg, lc, conn)
        out = ROOT / 'deterministic_teacher'
        for seed in SEEDS:
            for label in DETERMINISTIC_LABELS:
                train_one_v2(label, seed, conn, cfg, lc, data, out, ROOT / 'checkpoints' / 'deterministic')
        print('DET TEACHER TRAINING COMPLETE', flush=True)
    else:
        from run_latent_v3 import datasets_v3
        data = datasets_v3('hidden')
        out = ROOT / 'stochasticity' / 'future_oracle'
        out.mkdir(parents=True, exist_ok=True)
        for seed in SEEDS:
            summary_path = out / 'training' / f'fzoracle_seed{seed}.json'
            if summary_path.exists():
                continue
            torch.manual_seed(seed)
            model = FutureZOracle(conn).cuda()
            opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
            pw = torch.tensor(cfg.spike_pos_weight, device='cuda')
            from latent_data import sample_indices, windows
            from metrics import compute_loss
            best, bad = float('inf'), 0
            directory = ROOT / 'checkpoints' / 'future_oracle' / f'fzoracle_seed{seed}'
            directory.mkdir(parents=True, exist_ok=True)
            for epoch in range(1, 13):
                model.train()
                bi, ti = sample_indices(data['train'], 48 * 16, 500_000 + seed * 100 + epoch)
                ti = ti.clamp(max=cfg.T - 10)
                for off in range(0, len(bi), 16):
                    b, t = bi[off:off + 16].cuda(), ti[off:off + 16].cuda()
                    x, y = windows(data['train'], b, t, 1)
                    zf = z_future_path(data['train'], b, t, model.horizon)
                    opt.zero_grad(set_to_none=True)
                    loss, _ = compute_loss(model(x, z_future=zf), y, cfg, pw)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                    opt.step()
                model.eval()
                with torch.no_grad():
                    bi, ti = sample_indices(data['val'], 512, 8001)
                    ti = ti.clamp(max=cfg.T - 10)
                    tot, cnt = 0., 0
                    for off in range(0, len(bi), 32):
                        b, t = bi[off:off + 32].cuda(), ti[off:off + 32].cuda()
                        x, y = windows(data['val'], b, t, 1)
                        zf = z_future_path(data['val'], b, t, model.horizon)
                        loss, _ = compute_loss(model(x, z_future=zf), y, cfg, pw)
                        tot += float(loss) * len(b); cnt += len(b)
                vl = tot / cnt
                print(f'fzoracle seed={seed} ep={epoch} val={vl:.5f}', flush=True)
                if vl < best - 1e-6:
                    best, bad = vl, 0
                    torch.save(dict(state_dict=model.state_dict(), horizon=model.horizon),
                               directory / 'best_val.pt')
                else:
                    bad += 1
                    if bad >= 4:
                        break
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(json.dumps(dict(label='fzoracle', seed=seed, best_val=best)))
        print('FUTURE ORACLE TRAINING COMPLETE', flush=True)


def eval_part():
    """Stage 9: rollout decomposition across S0 / S1 / oracles."""
    from analyze_latent_v3 import evaluate_replication_entry
    from eval_latent import rollout_summary, rollout_prediction
    from calibrate_latent import write_csv
    cfg, lc, conn = setup_v2('hidden')
    rows = []
    # S1 deterministic teacher entries
    det = det_data(cfg, lc, conn)
    for p in sorted((ROOT / 'deterministic_teacher' / 'training').glob('*.json')):
        s = json.loads(p.read_text())
        s2 = dict(s)
        s2['checkpoint'] = s['checkpoint']
        evaluate_replication_entry(s2, 'det', cfg, lc, conn, det)
        rows.append(dict(condition='S1_deterministic', label=s['label'], seed=s['seed']))
    # S0 current-z oracle + base rollouts are already in eval entries.
    # S0 future-z oracle rollout
    from run_latent_v3 import datasets_v3
    data = datasets_v3('hidden')
    for seed in SEEDS:
        blob = torch.load(ROOT / 'checkpoints' / 'future_oracle' / f'fzoracle_seed{seed}' / 'best_val.pt',
                          map_location='cuda', weights_only=False)
        model = FutureZOracle(conn, horizon=blob['horizon']).cuda()
        model.load_state_dict(blob['state_dict'])
        model = model.eval()
        from latent_data import sample_indices, windows
        from eval_latent import calibrate_threshold
        bi, ti = sample_indices(data['val'], 512, 8001)
        ti = ti.clamp(max=cfg.T - 10)
        outs, ys = [], []
        with torch.no_grad():
            for off in range(0, len(bi), 32):
                b, t = bi[off:off + 32].cuda(), ti[off:off + 32].cuda()
                x, y = windows(data['val'], b, t, 1)
                zf = z_future_path(data['val'], b, t, model.horizon)
                outs.append({k: v.cpu() for k, v in model(x, z_future=zf).items()})
                ys.append(y.cpu())
        o = {k: torch.cat([a[k] for a in outs]) for k in outs[0]}
        th = calibrate_threshold(o, torch.cat(ys))
        for sp in ('test_seen', 'test_ood'):
            rp, rt = rollout_future_oracle(model, data[sp], cfg, th, n=32, horizon=200)
            rr = rollout_summary(rp, rt)
            out = ROOT / 'stochasticity' / 'future_oracle' / f'rollout_fzoracle_{sp}_{seed}.json'
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(rr, indent=2, allow_nan=False))
        del model
        torch.cuda.empty_cache()
    write_csv(ROOT / 'stochasticity' / 'conditions.csv', rows)
    print('STOCHASTICITY EVAL COMPLETE', flush=True)


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'eval':
        eval_part()
    else:
        main()
