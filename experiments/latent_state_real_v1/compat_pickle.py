"""Pandas 2.x-compatible unpickler for the SC-FC legacy pickles."""
import pickle
import pandas as pd
import numpy as np


class CompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'pandas.core.indexes.numeric':
            module = 'pandas.core.indexes.api'
            if name in ('Int64Index', 'UInt64Index', 'Float64Index'):
                return getattr(pd, name) if hasattr(pd, name) else pd.Index
        if module == 'numpy.core.multiarray' or module.startswith('numpy.core'):
            module = module.replace('numpy.core', 'numpy._core')
        return super().find_class(module, name)


def load_pickle(path):
    with open(path, 'rb') as f:
        return CompatUnpickler(f).load()
