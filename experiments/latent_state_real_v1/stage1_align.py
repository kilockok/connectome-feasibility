import sys; sys.path.insert(0, '.')
from compat_pickle import load_pickle
import pandas as pd, numpy as np, pathlib
# response row ids across all ito flies
unions = set()
for p in sorted(pathlib.Path('data/data/ito_responses').glob('*.pkl')):
    d = load_pickle(p)
    unions |= set(d.index.tolist())
print('ito response row-id union:', sorted(unions))
print('min', min(unions), 'max', max(unions), 'count', len(unions))
tb = pd.read_csv('data/data/hemi_2_atlas/JRC2018_ito_tbar_matrix.csv', index_col=0)
print('tbar matrix:', tb.shape, 'index range:', tb.index.min(), tb.index.max())
print('diag sample:', np.diag(tb.values)[:5])
print('symmetric?', np.allclose(tb.values, tb.values.T))
print('value stats: max %.1f median-off-diag %.1f frac-zero %.3f' % (
    tb.values.max(), np.median(tb.values[~np.eye(86, dtype=bool)]), (tb.values == 0).mean()))
cc = pd.read_csv('data/data/hemi_2_atlas/JRC2018_ito_cellcount_matrix.csv', index_col=0)
print('cellcount:', cc.shape, 'diag:', np.diag(cc.values)[:5])
ix = pd.read_csv('data/data/ito_68_atlas/Original_Index_panda_full.csv')
print('atlas num range:', ix['num'].min(), ix['num'].max(), 'n rows:', len(ix))
print('names:', ix['name'].dropna().tolist()[:20])
