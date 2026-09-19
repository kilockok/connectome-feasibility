"""Stage 2 data prep: processed tensors + frozen splits + graph variants.

Output: data_real.pt {sessions: {fid: {y [T,R], date, n_frames}},
        regions: 37 ROI nums, A_real/edges_shuf/weights_shuf [R,R] (raw tbar
        level), splits: {train/val/test session ids}}
Preprocessing: authors' high-pass (butter1, 0.01 Hz, fs=1.2) + trim map;
z-score with statistics from TRAIN SESSIONS only (per region, pooled).
"""
import sys, json
sys.path.insert(0, '.')
from compat_pickle import load_pickle
import numpy as np, pandas as pd, torch, pathlib
from scipy import signal

FS = 1.2
HP = 0.01
SPLIT_DATES = dict(
    train=['2017-10-26', '2017-10-30', '2017-11-16', '2018-10-19', '2018-10-31', '2018-11-03'],
    val=['2017-11-08', '2018-12-14'],
    test=['2018-10-20', '2018-12-12'])
TRIM = {'2018-10-19_1': np.array(list(range(100, 900)) + list(range(1100, 2000))),
        '2017-11-08_1': np.array(list(range(100, 1900)) + list(range(2000, 4000))),
        '2018-10-20_1': np.array(list(range(100, 1000)))}

dv = json.loads(pathlib.Path('dataset_validation.json').read_text())
COMMON = dv['common_region_nums']

def preprocess(fid, df):
    x = df.loc[COMMON].to_numpy(dtype=np.float64)          # [R, T] (rows in COMMON order)
    sos = signal.butter(1, HP, 'hp', fs=FS, output='sos')
    x = signal.sosfilt(sos, x, axis=1)
    date_run = fid.replace('ito_', '')
    keep = TRIM.get(date_run)
    x = x[:, keep] if keep is not None else x[:, 100:]
    return x.T                                            # [T, R]

sessions = {}
dates = {}
for p in sorted(pathlib.Path('data/data/ito_responses').glob('*.pkl')):
    fid = p.stem
    date = fid.replace('ito_', '').rsplit('_', 1)[0]
    x = preprocess(fid, load_pickle(p))
    sessions[fid] = dict(y=x, date=date, n_frames=int(x.shape[0]))
    dates.setdefault(date, []).append(fid)
    print(fid, x.shape, 'date', date, flush=True)

split_of = {}
for sp, ds in SPLIT_DATES.items():
    for d in ds:
        for fid in dates[d]:
            split_of[fid] = sp
counts = {sp: sum(1 for v in split_of.values() if v == sp) for sp in ('train', 'val', 'test')}
print('sessions per split:', counts)

# z-score stats from TRAIN sessions only
train_cat = np.concatenate([sessions[f]['y'] for f, s in split_of.items() if s == 'train'], 0)
mu = train_cat.mean(0); sd = train_cat.std(0) + 1e-9
for fid in sessions:
    sessions[fid]['y'] = (sessions[fid]['y'] - mu) / sd

# --- graphs (raw tbar level) ---
tb = pd.read_csv('data/data/hemi_2_atlas/JRC2018_ito_tbar_matrix.csv', index_col=0)
tb.columns = [int(c[1:]) if str(c).startswith('V') else int(c) for c in tb.columns]
tb.index = tb.index.astype(int)
A_real = tb.loc[COMMON, COMMON].to_numpy(dtype=np.float64)     # [R,R] A[src,dst]

rng = np.random.default_rng(4242)
# The tbar graph is near-complete (99.9% off-diagonal nonzero), so
# degree-preserving rewiring is degenerate. Frozen fake-graph controls:
#  A_edges_shuf  : node relabeling P A P^T (destroys region identity <->
#                  structure correspondence; preserves ALL graph statistics)
#  A_weights_shuf: fixed topology, uniformly permuted weights
perm = rng.permutation(37)
A_edges = A_real[np.ix_(perm, perm)].copy()
Aw = A_real.copy()
wp = A_real[A_real > 0].copy()
rng.shuffle(wp)
Aw[A_real > 0] = wp
print('graphs: real; node-relabel (P A P^T); weights-shuffled (same topology)')

torch.save(dict(sessions=sessions, split_of=split_of, regions=COMMON,
                A_real=A_real, A_edges_shuf=A_edges, A_weights_shuf=Aw,
                mu=mu, sd=sd, fs=FS, split_dates=SPLIT_DATES),
           'data_real.pt')
pathlib.Path('splits.json').write_text(json.dumps(dict(split_of=split_of, split_dates=SPLIT_DATES), indent=1))
print('data_real.pt + splits.json written')
