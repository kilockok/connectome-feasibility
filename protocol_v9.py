"""v9 shared protocol: effect-matched selection (frozen), splits, paths."""
from pathlib import Path
from teachers_v9 import MechSpec, GainParams, AdaptParams, STPParams, OUParams

ROOT = Path('results/latent_state_v9')
CKPTS = ROOT / 'checkpoints'
FAMILIES = ('gain', 'adapt', 'stp')
SEEDS = (1234, 1235, 1236, 1237, 1238)

# Frozen 2026-09-17 (revision 2, low-effect band) from
# results/latent_state_v9/metrics/effect_pool.csv. Train residual-V-RMS
# [0.128, 0.153]; spike disagreement [0.011, 0.017]; see
# audit/effect_matching_audit.md. Split B = unseen interpolated parameters;
# Split C = extrapolated parameters.
SELECTED = {
    'gain': dict(
        train=[('g1', GainParams(alpha=0.04, omega=0.05)),
               ('g2', GainParams(alpha=0.04, omega=0.12)),
               ('g3', GainParams(alpha=0.08, omega=0.08))],
        splitB=('gB', GainParams(alpha=0.06, omega=0.08)),
        splitC=('gC', GainParams(alpha=0.12, omega=0.16))),
    'adapt': dict(
        train=[('a1', AdaptParams(beta=0.2, tau_a=8.0, c=0.15)),
               ('a2', AdaptParams(beta=0.15, tau_a=20.0, c=0.25)),
               ('a3', AdaptParams(beta=0.35, tau_a=40.0, c=0.15))],
        splitB=('aB', AdaptParams(beta=0.25, tau_a=14.0, c=0.2)),
        splitC=('aC', AdaptParams(beta=0.3, tau_a=20.0, c=0.35))),
    'stp': dict(
        train=[('s1', STPParams(strength=0.015, tau_scale=0.5)),
               ('s2', STPParams(strength=0.015, tau_scale=2.0)),
               ('s3', STPParams(strength=0.025, tau_scale=1.0))],
        splitB=('sB', STPParams(strength=0.02, tau_scale=1.5)),
        splitC=('sC', STPParams(strength=0.04, tau_scale=4.0))),
}
OU_TEST = [('u1', OUParams(sigma=0.03, tau=8.0)),
           ('u2', OUParams(sigma=0.02, tau=32.0))]

MECH_NAMES = {fam: dict(
    train=[f'{fam}_{pid}' for pid, _ in SELECTED[fam]['train']],
    splitB=f"{fam}_{SELECTED[fam]['splitB'][0]}",
    splitC=f"{fam}_{SELECTED[fam]['splitC'][0]}") for fam in FAMILIES}

# split layout (trajectory indices into cfg.traj_seed(split, idx))
LAYOUT = dict(train_per_config=170, val_per_config=21, testA_per_config=21,
              testB=64, testC=32, null_test=64)
TESTB_IDX0 = 64
TESTC_IDX0 = 128
NULL_IDX0 = 160


def spec_for(family, pid):
    sel = SELECTED[family]
    for name, p in sel['train']:
        if name == pid:
            return MechSpec(name=f'{family}_{pid}', **{family: p})
    for key in ('splitB', 'splitC'):
        name, p = sel[key]
        if name == pid:
            return MechSpec(name=f'{family}_{pid}', **{family: p})
    raise KeyError((family, pid))


def all_mechanism_specs():
    out = []
    for fam in FAMILIES:
        sel = SELECTED[fam]
        for pid, p in sel['train'] + [sel['splitB'], sel['splitC']]:
            out.append(MechSpec(name=f'{fam}_{pid}', **{fam: p}))
    return out
