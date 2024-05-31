import inspect

import numpy as np


def np_hash(array):
    if array.dtype != np.dtype(object):
        return array.tobytes()
    full_len = len(array)
    sample_len = 57
    idx_left = [i * full_len // sample_len for i in range(sample_len)] + [-1]
    key = (
        tuple(array[x] for x in idx_left),
        full_len,
    )
    return key


def cache_key(arg):
    type_map = {
        "ArrayType": str,
        "Manifest": lambda x: x.array_type.value,
        "PosixPath": str,
        "bool": str,
        "int": str,
        "NoneType": str,
        "str": str,
        "RangeIndex": lambda x: x.values.tobytes(),
        "Index": lambda x: x.values.tobytes(),
        "ndarray": np_hash,
    }
    arg_type = arg.__class__.__name__ if hasattr(arg, "__class__") else None
    return type_map.get(arg_type, id)(arg)


def get_id_tuple(f, args, kwargs):
    id_list = [cache_key(f)]
    id_list.extend(cache_key(arg) for arg in args)
    # Sort keyword arguments by key to ensure consistent order
    sorted_kwargs = sorted((k, v) for k, v in kwargs.items() if k != "verbose")
    id_list.extend((key, cache_key(value)) for key, value in sorted_kwargs)
    return tuple(id_list)


def memoize(f):
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
            """Generate key."""
            complete_kwargs = {**self.init_defaults, **kwargs}
            key = get_id_tuple(self.cls, args, complete_kwargs)
            if key not in self._cache:
                self._cache[key] = self.cls(*args, **kwargs)
            return self._cache[key]

        def __instancecheck__(self, other):
            """Make isinstance() work"""
            return isinstance(other, self.cls)

    return Memoize(f)
