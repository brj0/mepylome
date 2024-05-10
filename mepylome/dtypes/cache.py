import inspect
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
        if (
            hasattr(value, "__class__")
            and value.__class__.__name__ == "Manifest"
        ):
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
        if isinstance(value, list):
            return tuple(value)

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


def get_id_tuple(f, args, kwargs):
    id_list = [id(f)]
    id_list.extend(id(arg) for arg in args)
    # Sort keyword arguments by key to ensure consistent order
    sorted_kwargs = sorted(kwargs.items())
    id_list.extend((key, id(value)) for key, value in sorted_kwargs)
    return tuple(id_list)


class Memoize:
    def __init__(self, cls):
        self.cls = cls
        self._cache = {}
        init_signature = inspect.signature(cls.__init__)
        self.init_defaults = {
            key: val.default
            for key, val in init_signature.parameters.items()
            if val.default is not inspect.Parameter.empty
        }
        self.__dict__.update(cls.__dict__)
        # This bit allows staticmethods to work as you would expect.
        for attr_name, attr_value in cls.__dict__.items():
            if isinstance(attr_value, staticmethod):
                self.__dict__[attr_name] = attr_value.__func__

    def __call__(self, *args, **kwargs):
        """Generate key"""
        complete_kwargs = {**self.init_defaults, **kwargs}
        key = get_id_tuple(self.cls, args, complete_kwargs)
        if key not in self._cache:
            self._cache[key] = self.cls(*args, **kwargs)
        return self._cache[key]

    def __instancecheck__(self, other):
        """Make isinstance() work"""
        return isinstance(other, self.cls)


def memoize(f):
    return Memoize(f)
