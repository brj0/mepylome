"""mepylome package.

This package provides tools for handling Illumina methylation arrays and
performing methylation analysis.
"""

import warnings
from contextlib import suppress

# If C++ parser should be added, before installation do: export MEPYLOME_CPP=1
with suppress(ModuleNotFoundError):
    from _mepylome import IdatParser as _IdatParser

from mepylome.dtypes import (
    CNV,
    Annotation,
    ArrayType,
    IdatParser,
    Manifest,
    MethylData,
    RawData,
    ReferenceMethylData,
    idat_basepaths,
)

# Suppress pyranges warnings
warnings.simplefilter(action="ignore", category=FutureWarning)


__all__ = [
    "ArrayType",
    "Annotation",
    "CNV",
    "IdatParser",
    "Manifest",
    "MethylData",
    "RawData",
    "ReferenceMethylData",
    "idat_basepaths",
]

# Conditionally add _IdatParser to __all__ if it exists
try:
    _ = _IdatParser
    __all__ += ["_IdatParser"]
except NameError:
    pass
