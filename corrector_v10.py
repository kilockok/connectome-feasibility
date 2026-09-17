"""v10 Stage 11: corrector secondary analysis.

Two questions only (per v10 scope):
A. Under optimized interventions, does the frozen v9 corrector's one-step
   prediction improve over the (hard) base more than passive?
B. Under high-SNR probe windows, does FrozenZ contain mechanism information
   (linear probe balanced accuracy vs passive windows)? No new label loss.
"""
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from connectome import Connectome
from run_v7 import setup_cfg
from protocol_v9 import ROOT as ROOT9, FAMILIES, SELECTED, spec_for, SEEDS
from teachers_v9 import MechanismLIFSimulator
from models.residual_v8 import build_v8
from library_v10 import build_stim, response_window
from intervention_v9 import out_neighbors, J, TAU
from latent_data import windows as lw

ROOT = Path('results/latent_state_v10')
NCTX = 32


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    cfg = setup_cfg()
    conn = Connectome.generate(cfg)
    dev = torch.device('cuda')
    blob = torch.load(ROOT9 / 'data' / 'v9_data.pt', map_location='cpu', weights_only=False)
    store = blob['store']
    lib = json.loads((ROOT / 'protocol' / 'intervention_library.json').read_text())
    short_ids = json.loads((ROOT / 'protocol' / 'configs' / 'shortlist.json').read_text())['ids']
    entry_map = {e['id']: e for e in lib['entries']}
    Inb_full = out_neighbors(conn, J)
    Wcpu = conn.dense_weight('cpu')
    strength = Wcpu[J].abs().sum(0)[Inb_full]
    Inb = Inb_full[strength.argsort(descending=True)].numpy()
    g_amp = float(lib['g_amp'])
    rows = []

    # ---- build optimized-probe branches (top-3 shortlist entries) ----
    probe_entries = short_ids[:3] + ['passive']
    branches = {}
    for fam in FAMILIES:
        d = store[f'{fam}/testB']
        sel = np.arange(NCTX)
        seeds = [cfg.traj_seed('test_seen', 64 + int(i)) for i in sel]
        pid = SELECTED[fam]['splitB'][0]
        for eid in probe_entries:
            if eid == 'passive':
                branches[(fam, eid)] = (d['states'][sel].to(dev), d['stimulus'][sel].to(dev),
                                        (TAU, TAU + 32))
                continue
            e = entry_map[eid]
            es = torch.stack([build_stim(e, cfg, Inb, g_amp)] * len(sel))
            sim = MechanismLIFSimulator(conn, cfg, dev, spec_for(fam, pid))
            dd = sim.generate(seeds, 'test_seen', extra_stim=es)
            branches[(fam, eid)] = (dd['states'], dd['stimulus'], response_window(e))
            del sim
            torch.cuda.empty_cache()
    # ---- A: corrector one-step prediction under probes ----
    from models.residual_v7 import base_pre
    for cseed in SEEDS:
        summary = json.loads((ROOT9 / 'metrics' / 'training' / f'ordered_seed{cseed}.json').read_text())
        corr = build_v8('ordered', conn, cfg).cuda()
        corr.load_state_dict(torch.load(summary['checkpoint'], map_location='cuda',
                                        weights_only=False)['state_dict'])
        corr = corr.eval()
        Zs, ys_c, conds = [], [], []
        for fam in FAMILIES:
            for eid in probe_entries:
                st_, stm, w = branches[(fam, eid)]
                # teacher-forced one-step predictions on the response window
                outs, base_v, ys = [], [], []
                for t in range(w[0], w[1]):
                    x = torch.cat((st_[:, t - 32 + 1:t + 1], stm[:, t - 32 + 1:t + 1].unsqueeze(-1)), -1)
                    out = corr(x)
                    outs.append(out['v'])
                    xw = x[:, -1]
                    base = base_pre(xw, cfg, corr.W, corr.i_bias)
                    base_v.append(base['v_base'])
                    ys.append(st_[:, t + 1, :, 0])
                o = torch.stack(outs, 1)
                bv = torch.stack(base_v, 1)
                yv = torch.stack(ys, 1)
                info_mask = None
                err_model = float((o - yv).square().mean().sqrt())
                err_base = float((bv - yv).square().mean().sqrt())
                rows.append(dict(stage='corrector_A', seed=cseed, family=fam, probe=eid,
                                 v_rmse_model=err_model, v_rmse_base=err_base,
                                 improve=err_base - err_model))
                # FrozenZ features for probe B
                with torch.no_grad():
                    z = torch.cat([corr.g.encode(
                        torch.cat((st_[:, t - 32 + 1:t + 1],
                                   stm[:, t - 32 + 1:t + 1].unsqueeze(-1)), -1))[0].mean(1).cpu()
                        for t in range(w[0], w[1], 4)])
                Zs.append(z)
                ys_c += [FAMILIES.index(fam)] * len(z)
                conds += [eid] * len(z)
        # ---- B: FrozenZ linear probe, per probe condition ----
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        Z = torch.cat(Zs).numpy()
        ynp = np.array(ys_c)
        conds = np.array(conds)
        tr = conds == 'passive'
        for cond in probe_entries:
            te = conds == cond
            sc = StandardScaler().fit(Z[tr])
            clf = LogisticRegression(max_iter=3000).fit(sc.transform(Z[tr]), ynp[tr])
            from sklearn.metrics import balanced_accuracy_score
            acc = float(balanced_accuracy_score(ynp[te], clf.predict(sc.transform(Z[te]))))
            rows.append(dict(stage='corrector_B', seed=cseed, family='all', probe=cond,
                             v_rmse_model=float('nan'), v_rmse_base=float('nan'),
                             improve=acc))
        print('corrector seed', cseed, 'done', flush=True)
        del corr
        torch.cuda.empty_cache()
    path = ROOT / 'metrics' / 'corrector_secondary.csv'
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print('CORRECTOR SECONDARY COMPLETE', len(rows), flush=True)


if __name__ == '__main__':
    main()
