"""mepylome.utils package.

This package provides utility functions and classes for file handling and
miscellaneous operations.
"""

from .downloader import download_idats, setup_tutorial_files
from .files import (
    download_file,
    download_files,
    ensure_directory_exists,
    get_csv_file,
    get_resource_path,
    reset_file,
)
from .varia import (
    CONFIG,
    MEPYLOME_TMP_DIR,
    Timer,
    get_free_port,
    make_log_file,
    normexp_get_xs,
)

__all__ = [
    "download_file",
    "download_files",
    "download_idats",
    "ensure_directory_exists",
    "get_csv_file",
    "make_log_file",
    "reset_file",
    "get_resource_path",
    "setup_tutorial_files",
    "get_free_port",
    "Timer",
    "normexp_get_xs",
    "CONFIG",
    "MEPYLOME_TMP_DIR",
]
