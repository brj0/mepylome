"""Utilities for handling file operations.

This module provides utilities such as downloading files (used to download
manifest files from Illumina), ensuring directories exist, and working with
file-like objects and archives.
"""

import gzip
import logging
import shutil
import ssl
import zipfile
from pathlib import Path, PurePath
from urllib.error import URLError
from urllib.request import urlopen

__all__ = [
    "download_file",
    "ensure_directory_exists",
    "get_file_object",
    "get_csv_file",
    "reset_file",
]


logger = logging.getLogger(__name__)


def make_path_like(path_like):
    """Attempts to convert a string to a Path instance."""
    if isinstance(path_like, Path):
        return path_like

    try:
        return Path(path_like)
    except TypeError as exc:
        msg = f"could not convert to Path: {path_like}"
        raise TypeError(msg) from exc


def require_path(inner):
    """Decorator that ensures the provided argument is a path."""

    def wrapped(orig_path, *args, **kwargs):
        path_like = make_path_like(orig_path)
        return inner(path_like, *args, **kwargs)

    return wrapped


@require_path
def ensure_directory_exists(path_like):
    """Ensures the ancestor directories of the provided path exist."""
    if path_like.exists():
        return

    parent_dir = path_like
    if path_like.suffix:
        parent_dir = path_like.parent

    parent_dir.mkdir(parents=True, exist_ok=True)


def download_file(src_url, dest, overwrite=False):
    """Download a file from a URL and save it to a destination directory.

    Args:
        src_url (str): The URL from which the file will be downloaded.
        dest (path_like): The path where the file will be saved.
        overwrite (bool, optional): If True, overwrite the file if it already
            exists. Defaults to False.
    """
    dest_path = make_path_like(dest)
    dest_dir = dest_path.parent
    if not dest_path.exists():
        ensure_directory_exists(dest_dir)
    elif not overwrite:
        # Check if file already exists, and return if it is there.
        logger.info(
            "File exists: %s. To overwrite, set overwrite=True.", dest_path
        )
        return
    try:
        print(
            f"Downloading manifest from {src_url} to {dest_dir}.\n"
            "Can take several minutes..."
        )
        # Skip the SSL certificate check
        context = ssl._create_unverified_context()
        with urlopen(src_url, context=context) as response, dest_path.open(
            "wb"
        ) as out_file:
            shutil.copyfileobj(response, out_file)
    except URLError as e:
        logger.exception(e)
        logger.info(
            "Downloading manifest from %s failed. Please correct "
            "url in source code",
            src_url,
        )
        raise URLError(e) from e


def is_file_like(obj):
    """Check if the object is a file-like object.

    For objects to be considered file-like, they must be an iterator AND have
    either a `read` and/or `write` method as an attribute.  Note: file-like
    objects must be iterable, but iterable objects need not be file-like.

    Arguments:
        obj {any} --The object to check.

    Returns:
        [boolean] -- [description]

    Examples:
    --------
    >>> buffer(StringIO("data"))
    >>> is_file_like(buffer)
    True
    >>> is_file_like([1, 2, 3])
    False
    """
    if not (hasattr(obj, "read") or hasattr(obj, "write")):
        return False

    if not hasattr(obj, "__iter__"):
        return False

    return True


def get_file_object(filepath_or_buffer):
    """Returns a file-like object for the given input.

    Args:
        filepath_or_buffer (str or file-like object): The file path or
            file-like object to be processed. Can be a gz-archived file.

    Returns:
        file-like object: A file-like object for reading the file.
    """
    if is_file_like(filepath_or_buffer):
        return filepath_or_buffer

    if PurePath(filepath_or_buffer).suffix == ".gz":
        return gzip.open(filepath_or_buffer, "rb")

    return open(filepath_or_buffer, "rb")


def get_csv_file(file_or_archive, filename):
    """Retrieve a CSV file from a regular file or a ZIP archive.

    This function extracts a specific CSV file from either a regular CSV file
    or a ZIP archive and returns it as a file-like object.

    Examples:
        >>> get_csv_file('archive.zip', 'example.csv')
        >>> get_csv_file('/path/to/example.csv', 'example.csv')
    """
    file_or_archive = make_path_like(file_or_archive)
    if file_or_archive.suffix == ".csv":
        return open(file_or_archive, "rb")
    if file_or_archive.suffix == ".zip":
        with zipfile.ZipFile(file_or_archive, "r") as archive:
            file_list = archive.namelist()
            file_match = next(
                (f for f in file_list if f.endswith(filename)), None
            )
            if file_match:
                return archive.open(file_match, "r")
            msg = f"File '{filename}' not found in the ZIP archive."
            raise FileNotFoundError(msg)
    else:
        msg = "Unsupported file type. Only '.csv' and '.zip' are supported."
        raise ValueError(msg)


def reset_file(filepath_or_buffer):
    """Attempts to return the open file to the beginning if it is seekable."""
    if not hasattr(filepath_or_buffer, "seek"):
        return

    filepath_or_buffer.seek(0)
