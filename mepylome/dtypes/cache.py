from functools import lru_cache, wraps

import numpy as np
import pandas as pd

# from mepylome.dtypes import Manifest


class Hash:
    def __init__(self, value):
        self.value = value
        self.key = Hash.get_key(value)

    def __hash__(self):
        return hash(self.key)

    @staticmethod
    def get_key(value):
        if hasattr(value, "__class__") and value.__class__.__name__ == "Manifest":
        # if isinstance(value, Manifest):
            return value.array_type.value
        if isinstance(value, pd.Index):
            return value.values.tobytes()
        if isinstance(value, np.ndarray):
            if value.dtype != np.dtype(object):
                return value.tobytes()
            else:
                N = len(value)
                L = 57
                idx_left = [i * N // L for i in range(L)] + [-1]
                key = (
                    tuple(value[x] for x in idx_left),
                    N,
                )
                return key

    def __eq__(self, __value):
        return __value.key == self.key


def cache(function):
    @lru_cache()
    def cached_wrapper(*hashed_arg):
        return function(*[x.value for x in hashed_arg])

    @wraps(function)
    def wrapper(*array):
        return cached_wrapper(*[Hash(x) for x in array])

    return wrapper
