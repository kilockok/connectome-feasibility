import sys; sys.path.insert(0, '.')
from compat_pickle import load_pickle
import pandas as pd, numpy as np
d = load_pickle('data/data/connectome_connectivity/Connectivity_computed_20210114.pkl')
print('Connectivity shape:', d.shape)
print('index:', list(d.index)[:10])
print(d.iloc[:4, :4])
print()
ws = load_pickle('data/data/connectome_connectivity/WeightedSynapseNumber_computed_20210114.pkl')
print('WeightedSynapseNumber:'); print(ws.iloc[:3, :3])
print()
for f in ('JRC2018_ito_tbar_matrix.csv', 'JRC2018_ito_cellcount_matrix.csv'):
    df = pd.read_csv('data/data/hemi_2_atlas/' + f, index_col=0)
    print(f, df.shape, 'index sample:', list(df.index[:5]))
print()
ix = pd.read_csv('data/data/ito_68_atlas/Original_Index_panda_full.csv')
print('ito atlas index csv:', ix.shape, list(ix.columns))
print(ix.head(8))
