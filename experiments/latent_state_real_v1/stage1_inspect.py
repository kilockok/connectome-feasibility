import sys; sys.path.insert(0, '.')
from compat_pickle import load_pickle
import pandas as pd, numpy as np, pathlib
print('=== ito_responses shapes ===')
for p in sorted(pathlib.Path('data/data/ito_responses').glob('*.pkl')):
    d = load_pickle(p)
    print('%-32s %s' % (p.name, d.shape))
print('=== branson_responses shapes ===')
for p in sorted(pathlib.Path('data/data/branson_responses').glob('*.pkl')):
    d = load_pickle(p)
    print('%-32s %s' % (p.name, d.shape))
print('=== connectivity ===')
for p in sorted(pathlib.Path('data/data/connectome_connectivity').glob('*')):
    try:
        d = load_pickle(p)
        shp = getattr(d, 'shape', '')
        print('%-52s %s %s' % (p.name, type(d).__name__, shp))
    except Exception as ex:
        print(p.name, 'ERR', ex)
