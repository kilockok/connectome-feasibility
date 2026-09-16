"""Stage 2: alias discrimination + residual prediction on the alias benchmark."""
import csv
import json
from pathlib import Path
import numpy as np
import torch
from run_latent_v3 import ROOT as V3ROOT, datasets_v3
from run_latent_v4 import ROOT as V4ROOT
from run_latent_v2 import setup as setup_v2
from train_latent_v2 import load_model_v2
from models.history_set_encoder import load_model_v4, SetHistoryPredictor, StatsHistoryPredictor, DerivativeBaselinePredictor
from models.latent_temporal_v2 import GlobalTemporalPredictorV2, LocalTemporalPredictorV2
from latent_data import windows
from lif_latent_v2 import HiddenStateLIFSimulatorV2

ROOT = Path('results/latent_state_v5')
SEEDS = (1234, 1235, 1236, 1237, 1238)
EXPECTED = {
    'gnn_k1': (LocalTemporalPredictorV2, 69443),
    'stats_k32': (StatsHistoryPredictor, 83523),
    'set_k32': (SetHistoryPredictor, 172483),
    'deriv': (DerivativeBaselinePredictor, 78403),
    'global_k32': (GlobalTemporalPredictorV2, 173635),
    'oracle': (LocalTemporalPredictorV2, 69571),
}


def summary_path(label, seed):
    if label in ('gnn_k1', 'global_k32', 'oracle'):
        return V3ROOT / 'replication' / 'hidden' / 'training' / f'{label}_seed{seed}.json'
    return V4ROOT / 'n100' / 'training' / f'{label}_seed{seed}.json'


def load_checked(label, seed, conn, device):
    cls, nparams = EXPECTED[label]
    summary = json.loads(summary_path(label, seed).read_text())
    if label in ('gnn_k1', 'global_k32', 'oracle'):
        model, blob = load_model_v2(summary, conn, device)
    else:
        model, blob = load_model_v4(summary, conn, device)
    assert isinstance(model, cls), f'{label}: wrong class {type(model)}'
    n = sum(p.numel() for p in model.parameters())
    assert n == nparams == summary['params'], f'{label}: param mismatch {n} vs {nparams} vs {summary["params"]}'
    x = torch.randn(2, 32, conn.n_neurons, 4, device=device)
    with torch.no_grad():
        out = model(x[:, -model.k:], z=torch.randn(2, 2, device=device) if model.oracle else None)
    assert out['v'].shape == (2, conn.n_neurons)
    assert torch.isfinite(out['v']).all()
    return model.eval(), blob['threshold']


@torch.no_grad()
def evaluate_model(model, label, data, pairs_blob, sim, threshold):
    rows = []
    d = data
    dev = next(model.parameters()).device
    for i, rec in enumerate(pairs_blob['rows']):
        (ta, t) = map(int, rec['A'].split(':'))
        (tb, s) = map(int, rec['B'].split(':'))
        xa, ua = d['states'][ta, t], d['stimulus'][ta, t]
        xb, ub = d['states'][tb, s], d['stimulus'][tb, s]
        za, zb = d['z'][ta, t], d['z'][tb, s]
        sil = d['silence']
        Da = sim.step(xa[None], ua[None], za[None], sil)[0]
        Ba = sim.step(xa[None], ua[None], torch.zeros_like(za[None]), sil)[0]
        Db = sim.step(xb[None], ub[None], zb[None], sil)[0]
        Bb = sim.step(xb[None], ub[None], torch.zeros_like(zb[None]), sil)[0]
        resA, resB = (Da - Ba)[:, 0], (Db - Bb)[:, 0]
        freeA = (xa[:, 2] <= 0) & (Ba[:, 1] <= .5) & (Da[:, 1] <= .5)
        freeB = (xb[:, 2] <= 0) & (Bb[:, 1] <= .5) & (Db[:, 1] <= .5)
        isynA = xa[:, 1] @ sim.W
        isynB = xb[:, 1] @ sim.W
        aLIF = sim.cfg.alpha
        xwA, _ = windows(d, torch.tensor([ta], device=dev), torch.tensor([t], device=dev), model.k)
        xwB, _ = windows(d, torch.tensor([tb], device=dev), torch.tensor([s], device=dev), model.k)
        zinA = za[None] if model.oracle else None
        zinB = zb[None] if model.oracle else None
        pA = model(xwA, z=zinA)
        pB = model(xwB, z=zinB)
        vA, vB = pA['v'][0], pB['v'][0]
        # assignment: does A's prediction fit A's future better than B's?
        eAA = (vA - Da[:, 0]).square().mean()
        eAB = (vA - Db[:, 0]).square().mean()
        eBA = (vB - Da[:, 0]).square().mean()
        eBB = (vB - Db[:, 0]).square().mean()
        correct = float((eAA + eBB) < (eAB + eBA))
        margin = float((eAB + eBA) - (eAA + eBB))
        # residual-channel discrimination: which gain does the model believe?
        dA = (vA - Ba[:, 0])
        dB = (vB - Bb[:, 0])
        rAA = (dA - resA).square().mean()
        rAB = (dA - resB).square().mean()
        rBA = (dB - resA).square().mean()
        rBB = (dB - resB).square().mean()
        res_correct = float((rAA + rBB) < (rAB + rBA))
        res_margin = float((rAB + rBA) - (rAA + rBB))
        # scalar gain estimate: project the model's implied residual onto the
        # observable synaptic-current direction (Delta_true = a*(gain-1)*I_syn).
        def gain_est(d_hat, isyn, free):
            ii = isyn[free]
            dd = d_hat[free]
            den = (ii * ii).sum().clamp(min=1e-12)
            return float((dd * ii).sum() / den) / aLIF + 1.0
        gA = float(sim.gain(za[None])); gB = float(sim.gain(zb[None]))
        geA = gain_est(vA - Ba[:, 0], isynA, freeA)
        geB = gain_est(vB - Bb[:, 0], isynB, freeB)
        gain_correct = float((abs(geA - gA) + abs(geB - gB)) < (abs(geA - gB) + abs(geB - gA)))
        gain_margin = float((abs(geA - gB) + abs(geB - gA)) - (abs(geA - gA) + abs(geB - gB)))
        # residual on free neurons
        def resid(pred_v, base_v, res_v, free):
            d_hat = (pred_v - base_v)[free]
            d_true = res_v[free]
            den = d_true.square().sum().clamp(min=1e-12)
            r2 = float(1 - (d_hat - d_true).square().sum() / den)
            rmse = float((d_hat - d_true).square().mean().sqrt())
            cs = float(torch.nn.functional.cosine_similarity(d_hat, d_true, dim=0)) if free.sum() > 1 and d_true.norm() > 1e-9 else None
            return r2, rmse, cs
        r2A, rmseA, csA = resid(vA, Ba[:, 0], resA, freeA)
        r2B, rmseB, csB = resid(vB, Bb[:, 0], resB, freeB)
        rows.append(dict(tier=rec['tier'], pair=i, correct=correct, margin=margin,
                         res_correct=res_correct, res_margin=res_margin,
                         gain_est_A=geA, gain_est_B=geB, gain_A=gA, gain_B=gB,
                         gain_correct=gain_correct, gain_margin=gain_margin,
                         rAA=float(rAA), rAB=float(rAB), rBA=float(rBA), rBB=float(rBB),
                         err_aa=float(eAA), err_ab=float(eAB), err_ba=float(eBA), err_bb=float(eBB),
                         res_r2_A=r2A, res_r2_B=r2B, res_rmse_A=rmseA, res_rmse_B=rmseB,
                         res_cos_A=csA, res_cos_B=csB,
                         free_A=int(freeA.sum()), free_B=int(freeB.sum())))
    return rows


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', nargs='+', type=int, default=list(SEEDS))
    args = p.parse_args()
    torch.set_num_threads(2)
    cfg, lc, conn = setup_v2('hidden')
    data = datasets_v3('hidden')['test_seen']
    sim = HiddenStateLIFSimulatorV2(conn, cfg, torch.device('cuda'), lc)
    pairs_blob = torch.load(ROOT / 'alias_pairs' / 'pairs.pt', weights_only=False)
    all_rows = []
    for seed in args.seeds:
        for label in ('gnn_k1', 'stats_k32', 'set_k32', 'deriv', 'global_k32', 'oracle'):
            model, th = load_checked(label, seed, conn, torch.device('cuda'))
            rows = evaluate_model(model, label, data, pairs_blob, sim, th)
            for r in rows:
                r.update(model=label, seed=seed)
            all_rows.extend(rows)
            acc = np.mean([r['correct'] for r in rows if r['tier'] == 'strict'])
            print(f'{label} s={seed} strict_acc={acc:.3f}', flush=True)
            del model
            torch.cuda.empty_cache()
    with (ROOT / 'table1_alias_discrimination.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        w.writeheader(); w.writerows(all_rows)
    print('ROWS', len(all_rows), flush=True)


if __name__ == '__main__':
    main()
