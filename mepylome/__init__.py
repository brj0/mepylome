"""mepylome package.

This package provides tools for handling Illumina methylation arrays and
performing methylation analysis.
"""

import logging
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
from mepylome.utils import make_log_file

# Suppress pyranges warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

LOG_FILE = make_log_file("stdout")


def setup_logging():
    logger = logging.getLogger()

    if logger.hasHandlers():
        return

    logger.setLevel(logging.DEBUG)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # File handler
    file_handler = logging.FileHandler(LOG_FILE, mode="a")
    file_handler.setLevel(logging.DEBUG)

    # Formatter with no milliseconds
    log_format = "%(asctime)s [%(module)s] %(message)s"
    formatter = logging.Formatter(log_format, "%H:%M:%S")
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # Don't show logging statements of other libraries.
    flask_logger = logging.getLogger("werkzeug")
    flask_logger.setLevel(logging.ERROR)
    numba_logger = logging.getLogger("numba")
    numba_logger.setLevel(logging.ERROR)


setup_logging()

__all__ = [
    "ArrayType",
    "Annotation",
    "CNV",
    "IdatParser",
    "Manifest",
    "MethylData",
    "LOG_FILE",
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
