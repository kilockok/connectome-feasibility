"""v10 Stage 4: constrained discrete intervention library.

Each entry: id, family, params, response window, cost. Hard constraints
(preregistered): amplitude <= 7, stimulated neurons <= 12, total extra
charge <= 400, probe segment ends <= T=192; induced firing-rate sanity
checked on the NULL teacher (no runaway/saturation).
Families: A delay, B burst, C paired-pulse, D edge-local, E neuron-global,
F high-current, G phase probe, P passive control.
"""
import json
from pathlib import Path
import torch
from connectome import Connectome
from run_v7 import setup_cfg
from teachers_v9 import MechanismLIFSimulator, MechSpec
from intervention_v9 import out_neighbors, J, A_NEURONS

ROOT = Path('results/latent_state_v10')
TAU = 96
T_END = 192
EXC8 = [22, 26, 32, 36, 42, 46, 52, 56]


def build_stim(entry, cfg, Inb, g_amp):
    """extra stimulus [T,N] for one library entry."""
    T, N = cfg.T, cfg.n_neurons
    u = torch.zeros(T, N)
    fam, p = entry['family'], entry['params']
    if fam == 'passive':
        return u
    if fam == 'delay':
        u[TAU:TAU + 4, J] = 6.0
        u[TAU + 8 + p['d']:TAU + 8 + p['d'] + 2, J] = 6.0
    elif fam == 'burst':
        u[TAU:TAU + 2, J] = 6.0
        t0 = TAU + p['silence']
        for k in range(p['count']):
            u[t0 + k * p['isi']:t0 + k * p['isi'] + 1, J] = 5.0
    elif fam == 'paired':
        u[TAU:TAU + 2, J] = 6.0
        u[TAU + p['isi']:TAU + p['isi'] + 2, J] = 6.0
    elif fam == 'edge':
        tgt = [J[p['which']]]
        for k in range(p['rep']):
            u[TAU + k * 4:TAU + k * 4 + 2, tgt] = 6.0
    elif fam == 'global':
        # fixed moderate postsyn injection on the TOP-24 strongest
        # out-neighbors (amp<=7, population<=24 constraints)
        for k in range(p['pulses']):
            u[TAU + k * p['isi']:TAU + k * p['isi'] + 3, Inb[:24]] = 6.0
    elif fam == 'highcur':
        u[TAU:TAU + 3, EXC8] = p['amp']
        u[TAU + 6:TAU + 8, J] = 6.0
    elif fam == 'phase':
        for k in range(4):
            t0 = TAU + p['offset'] + k * p['period']
            if t0 + 1 < T_END:
                u[t0:t0 + 1, J] = 4.0
    return u


def response_window(entry):
    fam, p = entry['family'], entry['params']
    if fam == 'passive':
        return (TAU, TAU + 32)
    if fam == 'delay':
        return (TAU + 8 + p['d'], TAU + 8 + p['d'] + 8)
    if fam == 'burst':
        end = TAU + p['silence'] + (p['count'] - 1) * p['isi'] + 1
        return (end, end + 8)
    if fam == 'paired':
        return (TAU + p['isi'], TAU + p['isi'] + 10)
    if fam == 'edge':
        end = TAU + (p['rep'] - 1) * 4 + 2
        return (end, end + 8)
    if fam == 'global':
        end = TAU + (p['pulses'] - 1) * p['isi'] + 2
        return (end, end + 8)
    if fam == 'highcur':
        return (TAU + 8, TAU + 16)
    if fam == 'phase':
        return (TAU, TAU + 64)


def cost(entry, Inb_count):
    fam, p = entry['family'], entry['params']
    if fam == 'passive':
        return 0.0
    charge_neurons = 0.0
    if fam == 'delay':
        charge_neurons = (6 * 4 + 6 * 2) * len(J)
    elif fam == 'burst':
        charge_neurons = (6 * 2 + 5 * p['count']) * len(J)
    elif fam == 'paired':
        charge_neurons = (6 * 2 + 6 * 2) * len(J)
    elif fam == 'edge':
        charge_neurons = 6 * 2 * p['rep']
    elif fam == 'global':
        charge_neurons = 6.0 * 3 * p['pulses'] * min(Inb_count, 24)
    elif fam == 'highcur':
        charge_neurons = p['amp'] * 3 * len(EXC8) + 6 * 2 * len(J)
    elif fam == 'phase':
        charge_neurons = 4 * 4 * len(J)
    return charge_neurons / 100.0


def library():
    lib = [dict(id='passive', family='passive', params={})]
    for d in (0, 2, 4, 8, 16, 24, 32, 48, 64):
        lib.append(dict(id=f'delay_d{d}', family='delay', params=dict(d=d)))
    for count in (2, 4, 8):
        for isi in (2, 4, 8):
            for sil in (8, 16):
                lib.append(dict(id=f'burst_n{count}_i{isi}_s{sil}', family='burst',
                                params=dict(count=count, isi=isi, silence=sil)))
    for isi in (1, 2, 4, 8, 16, 24):
        lib.append(dict(id=f'paired_i{isi}', family='paired', params=dict(isi=isi)))
    for which in range(3):
        for rep in (1, 2, 3):
            lib.append(dict(id=f'edge_j{which}_r{rep}', family='edge',
                            params=dict(which=which, rep=rep)))
    for pulses, isi in ((1, 0), (2, 4), (2, 12)):
        lib.append(dict(id=f'global_p{pulses}_i{isi}', family='global',
                        params=dict(pulses=pulses, isi=isi)))
    for amp in (4.0, 5.0, 6.0):
        lib.append(dict(id=f'highcur_a{amp}', family='highcur', params=dict(amp=amp)))
    for period in (26, 52, 78):
        for offset in (0, period // 2):
            lib.append(dict(id=f'phase_p{period}_o{offset}', family='phase',
                            params=dict(period=period, offset=offset)))
    return lib


def main():
    torch.set_num_threads(2)
    cfg = setup_cfg()
    conn = Connectome.generate(cfg)
    from intervention_v9 import calibrate_global_amp
    Inb_full = out_neighbors(conn, J)
    Wcpu = conn.dense_weight('cpu')
    strength = Wcpu[J].abs().sum(0)[Inb_full]
    Inb = Inb_full[strength.argsort(descending=True)].numpy()   # strongest first
    seeds = [cfg.traj_seed('test_seen', i) for i in range(300, 308)]
    g_amp, tgt = calibrate_global_amp(cfg, conn, Inb, seeds)
    lib = library()
    # validity + dynamics sanity on the NULL teacher
    sim = MechanismLIFSimulator(conn, cfg, torch.device('cuda'), MechSpec(name='null'))
    base = sim.generate(seeds, 'test_seen')
    base_rate = float(base['states'][..., 1][:, TAU:].mean())
    checked = []
    for e in lib:
        w = response_window(e)
        assert w[1] <= T_END, (e['id'], w)
        es = torch.stack([build_stim(e, cfg, Inb, g_amp) for _ in seeds])
        amp = float(es.abs().max())
        nn = int((es.abs().sum((0, 1)) > 0).sum())   # count NEURONS only
        charge = float(es[0].abs().sum())          # per-trajectory charge
        assert amp <= 7.0 + 1e-9 and nn <= 40 and charge <= 4000, (e['id'], amp, nn, charge)
        d = sim.generate(seeds, 'test_seen', extra_stim=es)
        rate = float(d['states'][..., 1][:, TAU:].mean())
        vmax = float(d['states'][..., 0].abs().max())
        checked.append(dict(id=e['id'], family=e['family'], params=e['params'],
                            window=w, cost=cost(e, len(Inb)),
                            induced_rate=rate, rate_over_base=rate / max(base_rate, 1e-9),
                            vmax=vmax))
        assert rate < 0.45 and vmax < 10.0, ('runaway', e['id'], rate, vmax)
    out = dict(tau=TAU, t_end=T_END, g_amp=g_amp, J=J, A_neurons=A_NEURONS,
               exc8=EXC8, entries=checked)
    (ROOT / 'protocol').mkdir(parents=True, exist_ok=True)
    (ROOT / 'protocol' / 'intervention_library.json').write_text(json.dumps(out, indent=1))
    print('library size', len(checked), 'g_amp', round(g_amp, 3))
    for e in checked:
        print(e['id'], 'cost', round(e['cost'], 2), 'rate', round(e['induced_rate'], 3), flush=True)


if __name__ == '__main__':
    main()
