"""v11b Part A: independent rollout error attribution (DIAGNOSTIC ONLY).

Does not reinterpret v11 gates. Compares closed-loop rollouts of:
  b0   exact hard LIF (lif.py semantics)
  b2   v9 OrderedHistory corrector (checkpoint)
  b5   v11 hybrid (checkpoint)
  b5n  v11 hybrid trained on NULL (correction-noise behavior in closed loop)

Rollout protocol (frozen here):
  branch at t0=96 from the teacher trajectory; ALL models receive the same
  teacher context [t0-32, t0] (encoder init for b2/b5; initial state for b0);
  then fully autonomous: each model's OWN predicted states feed the encoder
  and I_syn (predicted spikes feed S@W). Stimulus = true future stimulus
  (known external input). Learned models use state_from_output
  (v9-lineage hard-reset convention, per-model val-calibrated threshold,
  val seed 8002 - no test-set threshold tuning).
  Teacher-forced one-step references come from v11 metrics (not recomputed).

A3 error propagation: first spike-divergence time/neuron, V error before/
after, I_syn error after, per-horizon spike P/R/F1, corrector-introduced
error pre-divergence, paired spike attribution (b0 wrong/right x b5
wrong/right), correction magnitude in residual-active vs null regions.

A4 causal branches: replicate the teacher generation loop exactly (same
seeds/RNG layout as teachers_v9.generate), keep the hidden state at the
branch point, then: (a) teacher continuation; (b) single-spike flip;
(c) single-neuron residual correction. Measures whether one spike error
propagates, and whether one correction helps - under teacher dynamics.

A5 oracle (privileged, teacher-forced only, labelled): O1 perfect
free-step residual; O2 oracle-gated M1 delta (apply b5's delta only where
it reduces one-step error) - the gating headroom measurement.

Outputs: results/latent_state_v11b/results/rollout/*.csv + branch/*.csv
"""
import csv
import json
from pathlib import Path
import numpy as np
import torch
from connectome import Connectome
from run_v7 import setup_cfg
from latent_data import sample_indices, windows
from eval_latent import calibrate_threshold, state_from_output
from models.residual_v8 import build_v8
from models.latent_hybrid_v11 import build_v11
from teachers_v9 import MechanismLIFSimulator
from protocol_v9 import ROOT as ROOT9, FAMILIES, SEEDS, SELECTED, spec_for
from data_v11 import load_pool

ROOT = Path('results/latent_state_v11b')
T0 = 96
K = 32
HMAX = 100
HS = (1, 5, 10, 25, 50, 100)
NTRAJ = 32


# ------------------------------------------------------------------ models
def load_models(conn, cfg, seed, dev):
    out = {}
    s9 = json.loads((ROOT9 / 'metrics' / 'training' / f'ordered_seed{seed}.json').read_text())
    m2 = build_v8('ordered', conn, cfg).to(dev)
    m2.load_state_dict(torch.load(s9['checkpoint'], map_location=dev, weights_only=False)['state_dict'])
    out['b2'] = m2.eval()
    s11 = json.loads(Path(f'results/latent_state_v11/metrics/training/full_seed{seed}.json').read_text())
    m5 = build_v11('full', conn, cfg).to(dev)
    m5.load_state_dict(torch.load(s11['checkpoint'], map_location=dev, weights_only=False)['state_dict'])
    out['b5'] = m5.eval()
    s11n = json.loads(Path(f'results/latent_state_v11/metrics/training/full_seed{seed}_null.json').read_text())
    m5n = build_v11('full', conn, cfg).to(dev)
    m5n.load_state_dict(torch.load(s11n['checkpoint'], map_location=dev, weights_only=False)['state_dict'])
    out['b5n'] = m5n.eval()
    return out, {'b5': s11['threshold'], 'b5n': s11n['threshold']}


@torch.no_grad()
def calibrate_on_val(model, pool, dev):
    val = {k: v.to(dev) for k, v in pool['val'].items()}
    bi, ti = sample_indices(val, 512, 8002)
    outs, ys = [], []
    for off in range(0, len(bi), 32):
        x, y = windows(val, bi[off:off + 32], ti[off:off + 32], K)
        outs.append({k: v.cpu() for k, v in model(x.to(dev)).items() if k in ('v', 's_logits', 'r')})
        ys.append(y.cpu())
    o = {k: torch.cat([a[k] for a in outs]) for k in outs[0]}
    return calibrate_threshold(o, torch.cat(ys))


# ------------------------------------------------------------------ rollout
@torch.no_grad()
def rollout_b0(states0, stim_future, cfg, conn, dev):
    """Exact hard LIF, autonomous. states0 [B,N,3] teacher state at t0."""
    from lif import LIFSimulator
    sim = LIFSimulator(conn, cfg, dev)
    V, S, R = states0[..., 0], states0[..., 1], states0[..., 2] * cfg.refractory_period
    return sim.simulate(stim_future, state0=(V.to(dev), S.to(dev), R.to(dev)))


@torch.no_grad()
def rollout_model(model, thr, ctx_states, ctx_stim, states0, stim_future, cfg, conn, dev):
    """Autonomous rollout with the v9-lineage state_from_output convention.
    ctx_*: teacher context [B,K,N,*]; the encoder consumes the model's OWN
    predicted frames after branching."""
    B = states0.shape[0]
    H = stim_future.shape[1]
    hist = torch.cat((ctx_states.to(dev), states0.to(dev)[:, None]), 1)   # [B,K+1,N,3]
    stim_hist = ctx_stim.to(dev)                                          # [B,K,N]
    preds = [states0.to(dev)]
    for t in range(H):
        x = torch.cat((hist[:, -K:], stim_hist[:, -K:][..., None]), -1)
        out = model(x)
        nxt = state_from_output(out, cfg, thr)
        preds.append(nxt)
        hist = torch.cat((hist, nxt[:, None]), 1)
        stim_hist = torch.cat((stim_hist, stim_future[:, t:t + 1].to(dev)), 1)
    return torch.stack(preds[1:], 1)                                     # [B,H,N,3]


def f1_pr(tp, fp, fn):
    p = tp / max(tp + fp, 1.0); r = tp / max(tp + fn, 1.0)
    return p, r, 2 * p * r / max(p + r, 1e-9)


# ------------------------------------------------------------------ main
@torch.no_grad()
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', nargs='+', type=int, default=list(SEEDS))
    args = ap.parse_args()
    torch.set_num_threads(2)
    cfg = setup_cfg()
    conn = Connectome.generate(cfg)
    dev = torch.device('cuda')
    pool = load_pool(cfg)

    # cohort: testB 32 trajs/family + null/test 32 (diagnostic cohort)
    cohorts = {}
    d = pool['testB']
    for fi, fam in enumerate(FAMILIES):
        sel = (d['mech'] == fi).nonzero(as_tuple=True)[0][:NTRAJ]
        cohorts[fam] = {k: v[sel] for k, v in d.items() if k in ('states', 'stimulus')}
    cohorts['null'] = {k: v[:NTRAJ] for k, v in pool['null'].items() if k in ('states', 'stimulus')}

    rows, div_rows, arows = [], [], []
    for seed in args.seeds:
        models, thrs = load_models(conn, cfg, seed, dev)
        thrs['b2'] = calibrate_on_val(models['b2'], pool, dev)
        for fam, d in cohorts.items():
            states, stim = d['states'].to(dev), d['stimulus'].to(dev)
            ctx_states = states[:, T0 - K:T0]
            ctx_stim = stim[:, T0 - K:T0]
            s0 = states[:, T0]
            stim_fut = stim[:, T0:T0 + HMAX]
            teacher = states[:, T0 + 1:T0 + HMAX + 1]
            preds = {'b0': rollout_b0(s0, stim_fut, cfg, conn, dev)}
            for name in ('b2', 'b5', 'b5n'):
                preds[name] = rollout_model(models[name], thrs[name], ctx_states, ctx_stim,
                                            s0, stim_fut, cfg, conn, dev)
            tv, ts = teacher[..., 0], teacher[..., 1]
            for name, pr in preds.items():
                pv, ps = pr[..., 0], pr[..., 1]
                verr = (pv - tv).square().mean(2).sqrt()          # [B,H]
                agree = (ps == ts).float().mean(2)                # [B,H]
                # first spike divergence
                divmask = (ps != ts)
                anydiv = divmask.any(2)                           # [B,H]
                for H in HS:
                    for b in range(pv.shape[0]):
                        rows.append(dict(seed=seed, family=fam, model=name, traj=b, horizon=H,
                                         v_rmse=float(verr[b, :H].mean()),
                                         spike_agree=float(agree[b, :H].mean())))
                # per-trajectory divergence detail
                for b in range(pv.shape[0]):
                    idx = anydiv[b].nonzero(as_tuple=True)[0]
                    if len(idx) == 0:
                        div_rows.append(dict(seed=seed, family=fam, model=name, traj=b,
                                             div_t=-1, div_neuron=-1, v_err_pre=0.0,
                                             v_err_post=float(verr[b, :8].mean()),
                                             isyn_err_post=0.0, n_div=0))
                        continue
                    tstar = int(idx[0])
                    neurons = divmask[b, tstar].nonzero(as_tuple=True)[0]
                    pre = verr[b, max(0, tstar - 8):tstar].mean() if tstar > 0 else torch.tensor(0.)
                    post = verr[b, tstar:tstar + 8].mean()
                    # I_syn error after divergence (own spikes vs teacher spikes)
                    isyn_pred = torch.einsum('tj,ji->ti', ps[b], conn.dense_weight(dev))
                    isyn_true = torch.einsum('tj,ji->ti', ts[b], conn.dense_weight(dev))
                    isyn_err = (isyn_pred - isyn_true).square().mean(1).sqrt()
                    post_isyn = isyn_err[tstar:tstar + 8].mean() if tstar < HMAX else torch.tensor(0.)
                    div_rows.append(dict(seed=seed, family=fam, model=name, traj=b, div_t=tstar,
                                         div_neuron=int(neurons[0]),
                                         v_err_pre=float(pre), v_err_post=float(post),
                                         isyn_err_post=float(post_isyn),
                                         n_div=int(anydiv[b].sum())))
            print(seed, fam, 'rollout done', flush=True)
            # ---- paired spike attribution vs b0 (C4-style, per horizon) ----
            tt = ts                                            # [B,H,N] teacher spikes
            for H in (10, 50, 100):
                a = preds['b0'][..., 1][:, :H]
                wa = (a != tt[:, :H])                          # b0 wrong
                for nb in ('b5', 'b2', 'b5n'):
                    b_ = preds[nb][..., 1][:, :H]
                    wb = (b_ != tt[:, :H])
                    arows.append(dict(
                        seed=seed, family=fam, model=nb, horizon=H,
                        b0w_mw=int((wa & wb).sum()),       # both wrong
                        b0w_mc=int((wa & ~wb).sum()),      # b0 wrong -> model correct (fixed)
                        b0c_mw=int((~wa & wb).sum()),      # b0 correct -> model wrong (introduced)
                        b0c_mc=int((~wa & ~wb).sum()),     # both correct
                        n=int(wa.numel())))
                # paired V-error analysis on the same windows
                eb0 = (preds['b0'][..., 0][:, :H] - tv[:, :H]).abs()
                for nb in ('b5', 'b2', 'b5n'):
                    eb = (preds[nb][..., 0][:, :H] - tv[:, :H]).abs()
                    red = (eb < eb0 - 1e-6)
                    inc = (eb > eb0 + 1e-6)
                    arows.append(dict(
                        seed=seed, family=fam, model=nb, horizon=1000 + H,
                        b0w_mw=-1, b0w_mc=-1, b0c_mw=-1, b0c_mc=-1, n=int(eb0.numel())))
                    arows[-1].update(dict(v_reduced_frac=float(red.float().mean()),
                                          v_increased_frac=float(inc.float().mean()),
                                          v_reduced_mag=float((eb0 - eb)[red].mean()) if red.any() else 0.0,
                                          v_increased_mag=float((eb - eb0)[inc].mean()) if inc.any() else 0.0))
            del preds
            torch.cuda.empty_cache()
        del models
        torch.cuda.empty_cache()

    outdir = ROOT / 'results' / 'rollout'
    outdir.mkdir(parents=True, exist_ok=True)
    with (outdir / 'rollout.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with (outdir / 'divergence.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(div_rows[0])); w.writeheader(); w.writerows(div_rows)
    # attribution rows have two schemas (spike 2x2 and V paired); split by horizon marker
    sp = [r for r in arows if r['horizon'] < 1000]
    vr = [r for r in arows if r['horizon'] >= 1000]
    with (outdir / 'attribution_spike.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(sp[0])); w.writeheader(); w.writerows(sp)
    with (outdir / 'attribution_v.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(vr[0])); w.writeheader(); w.writerows(vr)
    print('ROLLOUT PART-A CORE COMPLETE', len(rows), len(div_rows), flush=True)


if __name__ == '__main__':
    main()
