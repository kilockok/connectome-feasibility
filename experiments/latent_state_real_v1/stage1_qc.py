"""Stage 1 QC: validate the SC-FC functional + structural data for modeling.

Produces dataset_validation.json with per-fly QC, the common region set,
and structural-matrix statistics on that set.
"""
import sys, json, hashlib
sys.path.insert(0, '.')
from compat_pickle import load_pickle
import pandas as pd, numpy as np, pathlib

# --- canonical 38 central-brain regions (from SC-FC bridge.getItoNames) ---
INCLUDE38 = ['AL_R', 'AOTU_R', 'ATL_R', 'ATL_L', 'AVLP_R', 'BU_R', 'BU_L', 'CAN_R', 'CRE_R', 'CRE_L',
             'EB', 'EPA_R', 'FB', 'GOR_R', 'GOR_L', 'IB_R', 'IB_L', 'ICL_R', 'LAL_R', 'LH_R',
             'MB_CA_R', 'MB_ML_R', 'MB_ML_L', 'MB_PED_R', 'MB_VL_R', 'NO', 'PB', 'PLP_R', 'PVLP_R',
             'SCL_R', 'SIP_R', 'SLP_R', 'SMP_R', 'SMP_L', 'SPS_R', 'VES_R', 'WED_R']
ix = pd.read_csv('data/data/ito_68_atlas/Original_Index_panda_full.csv')
name2num = dict(zip(ix['name'], ix['num']))
num2name = dict(zip(ix['num'], ix['name']))
inc_nums = sorted(name2num[n] for n in INCLUDE38 if n in name2num)
print('38-region num list len:', len(inc_nums))

TRIM = {'ito_2018-10-19_1': np.array(list(range(100, 900)) + list(range(1100, 2000))),
        'ito_2017-11-08_1': np.array(list(range(100, 1900)) + list(range(2000, 4000))),
        'ito_2018-10-20_1': np.array(list(range(100, 1000)))}
# note: trim keys in SC-FC code are brain-file ids like 2018-10-19_1

flies = {}
for p in sorted(pathlib.Path('data/data/ito_responses').glob('*.pkl')):
    fid = p.stem
    d = load_pickle(p)
    present38 = [n for n in inc_nums if n in set(d.index)]
    flies[fid] = dict(df=d, present=present38)
    print(fid, d.shape, 'present-of-38:', len(present38))

common = sorted(set.intersection(*[set(f['present']) for f in flies.values()]))
print('common regions across all 20 flies:', len(common))
print('common names:', [num2name[n] for n in common][:40])

# --- per-fly QC on the common region set ---
qc = {}
for fid, f in flies.items():
    d = f['df']
    sub = d.loc[common].to_numpy(dtype=np.float64)   # [R, T]
    nan_frac = float(np.isnan(sub).mean())
    flat = int((np.nanstd(sub, axis=1) < 1e-9).sum())
    T = sub.shape[1]
    fid_short = fid.replace('ito_', '')
    trim = TRIM.get(fid_short)
    if trim is not None:
        T_eff = int(len(trim))
    else:
        T_eff = T - 100  # default: drop first 100 frames
    qc[fid] = dict(n_regions_present=int(d.shape[0]), n_frames_raw=int(T),
                   n_frames_after_trim=T_eff, nan_frac=nan_frac, flatline_regions=flat,
                   mean_fluorescence=float(np.nanmean(sub)),
                   std_fluorescence=float(np.nanstd(sub)))

# --- structural matrix on the common set ---
tb = pd.read_csv('data/data/hemi_2_atlas/JRC2018_ito_tbar_matrix.csv', index_col=0)
tb.columns = [int(c[1:]) if str(c).startswith('V') else int(c) for c in tb.columns]; tb.index = tb.index.astype(int)
S = tb.loc[common, common].to_numpy(dtype=np.float64)
cc = pd.read_csv('data/data/hemi_2_atlas/JRC2018_ito_cellcount_matrix.csv', index_col=0)
cc.columns = [int(c[1:]) if str(c).startswith('V') else int(c) for c in cc.columns]; cc.index = cc.index.astype(int)
C = cc.loc[common, common].to_numpy(dtype=np.float64)
off = ~np.eye(len(common), dtype=bool)
struct = dict(n_regions=len(common),
              tbar=dict(frac_zero=float((S[off] == 0).mean()),
                        median_nonzero=float(np.median(S[off][S[off] > 0])),
                        max=float(S.max()), directed=not np.allclose(S, S.T)),
              cellcount=dict(frac_zero=float((C[off] == 0).mean()),
                             median_nonzero=float(np.median(C[off][C[off] > 0])) if (C[off] > 0).any() else 0.0))

out = dict(fs_hz=1.2, highpass_cutoff_hz=0.01, default_drop_first_frames=100,
           trim_map={k: [int(v[0]), int(v[-1]), int(len(v))] for k, v in TRIM.items()},
           n_flies=len(flies), common_region_nums=common,
           common_region_names=[num2name[n] for n in common],
           per_fly=qc, structural=struct)
pathlib.Path('dataset_validation.json').write_text(json.dumps(out, indent=1))
print(json.dumps(struct, indent=1))
print('dataset_validation.json written')
