"""Utility functions for caching and memoization.

Utility functions and classes for caching and memoization to optimize
performance by storing and reusing computed results.
"""

import inspect

import numpy as np


def np_hash(array):
    """Generates a hashable key for a NumPy array.

    This function creates a hashable key from a NumPy array by converting it to
    bytes if the data type is not an object. If the array's data type is an
    object (used for IlmnID), it samples 57 evenly spaced elements from the
    array along with the last element and combines them into a tuple with the
    array's length to form the key.

    Args:
        array (numpy.ndarray): The input array for which to generate the
            hashable key.

    Returns:
        tuple: A hashable key representing the array.

    Note:
        If 'array' is an object, collisions are possible.

    Example:
        >>> arr = np.array([1, 2, 3, 4, 5])
        >>> np_hash(arr)
    """
    if array.dtype != np.dtype(object):
        return array.tobytes()
    full_len = len(array)
    sample_len = 57
    idx_left = [i * full_len // sample_len for i in range(sample_len)] + [-1]
    return (
        tuple(array[x] for x in idx_left),
        full_len,
    )


def cache_key(*args):
    """Generates a cache key for arguments based on their type.

    Args:
        arg: The input arguments, which can be of various types such as
            ArrayType, Manifest, PosixPath, bool, int, NoneType, str,
            RangeIndex, Index, or ndarray.

    Returns:
        The cache key for the arguments.

    Warning:
        If an argument is a numpy array of object type, the key may not be
        unique.
    """
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
    keys = []
    for arg in args:
        arg_type = arg.__class__.__name__ if hasattr(arg, "__class__") else None
        keys.append(type_map.get(arg_type, id)(arg))
    if len(args) == 1:
        return keys[0]
    return tuple(keys)


def get_id_tuple(f, args, kwargs):
    """Generates a identifier tuple for a class/function and its arguments.

    This function creates a tuple that uniquely identifies a function call
    based on the function reference, its positional arguments, and keyword
    arguments. The keyword arguments are sorted by key to ensure a consistent
    order.

    Args:
        f: The function reference.
        args: A list of positional arguments passed to the function.
        kwargs: A dictionary of keyword arguments passed to the function. The
            keyword argument 'verbose' is excluded from the identifier.

    Returns:
        tuple: A tuple representing the unique identifier for the function
            call.

    Warning:
        If arg is a numpy array of object type (used in IlmnID), the key may
        not be unique.
    """
    id_list = [cache_key(f)]
    id_list.extend(cache_key(arg) for arg in args)
    # Sort keyword arguments by key to ensure consistent order
    sorted_kwargs = sorted((k, v) for k, v in kwargs.items() if k != "verbose")
    id_list.extend((key, cache_key(value)) for key, value in sorted_kwargs)
    return tuple(id_list)


def memoize(f):
    """Memoization decorator for classes and functions.

    Description:
        The `memoize` function is a decorator that provides memoization for
        class instantiation and functions. It caches instances of the decorated
        class or function based on the arguments provided. If an instance or
        function result with the same arguments already exists in the cache, it
        returns the cached instance or result instead of creating a new one.

    Args:
        f (Union[class, function]): The class or function to be decorated with
        memoization.

    Returns:
        Memoize: A memoized version of the input class or function.

    Note:
        Adapted from:
        https://stackoverflow.com/questions/10879137/how-can-i-memoize-a-class-instantiation-in-python
    """

    class Memoize:
        def __init__(self, cls):
            self.cls = cls
            self._cache = {}
            self.__name__ = cls.__name__
            self.__doc__ = cls.__doc__
            self.__module__ = cls.__module__
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
            """Make isinstance() work."""
            return isinstance(other, self.cls)

    return Memoize(f)
