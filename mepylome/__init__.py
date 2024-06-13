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
    CHROMOSOME_DATA,
    CNV,
    GAPS,
    GENES,
    IMPORTANT_GENES,
    ZIP_ENDING,
    Annotation,
    ArrayType,
    Channel,
    Chromosome,
    CNVPlot,
    IdatParser,
    InfiniumDesignType,
    Manifest,
    MethylData,
    ProbeType,
    RawData,
    ReferenceMethylData,
    cache,
    cnv_plot_from_data,
    get_id_tuple,
    idat_basepaths,
    memoize,
    read_cnv_data_from_disk,
)

# Suppress pyranges warnings
warnings.simplefilter(action="ignore", category=FutureWarning)


__all__ = [
    "Annotation",
    "ArrayType",
    "CHROMOSOME_DATA",
    "CNV",
    "CNVPlot",
    "Channel",
    "Chromosome",
    "GAPS",
    "GENES",
    "IMPORTANT_GENES",
    "IdatParser",
    "InfiniumDesignType",
    "Manifest",
    "MethylData",
    "ProbeType",
    "RawData",
    "ReferenceMethylData",
    "ZIP_ENDING",
    "cache",
    "cnv_plot_from_data",
    "get_id_tuple",
    "idat_basepaths",
    "memoize",
    "read_cnv_data_from_disk",
]

# Conditionally add _IdatParser to __all__ if it exists
try:
    _ = _IdatParser
    __all__ += ["_IdatParser"]
except NameError:
    pass
