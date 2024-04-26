# Lib
from pathlib import Path, PurePath
from urllib.error import URLError
from urllib.request import urlopen
import gzip
import logging
import shutil
import zipfile


__all__ = [
    "download_file",
    "ensure_directory_exists",
    "get_file_object",
    "get_file_from_archive",
    "is_file_like",
    "read_and_reset",
    "reset_file",
]


logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO)


def read_and_reset(inner):
    """Decorator that resets a file-like object back to the original
    position after the function has been called."""

    def wrapper(infile, *args, **kwargs):
        current_position = infile.tell()
        rval = inner(infile, *args, **kwargs)
        infile.seek(current_position)
        return rval

    return wrapper


def make_path_like(path_like):
    """Attempts to convert a string to a Path instance."""

    if isinstance(path_like, Path):
        return path_like

    try:
        return Path(path_like)
    except TypeError:
        raise TypeError(f"could not convert to Path: {path_like}")


def require_path(inner):
    """Decorator that ensure the argument provided to the inner function
    is a Path instance."""

    def wrapped(orig_path, *args, **kwargs):
        path_like = make_path_like(orig_path)
        return inner(path_like, *args, **kwargs)

    return wrapped


@require_path
def ensure_directory_exists(path_like):
    """Ensures the ancestor directories of the provided path
    exist, making them if they do not."""
    if path_like.exists():
        return

    parent_dir = path_like
    if path_like.suffix:
        parent_dir = path_like.parent

    parent_dir.mkdir(parents=True, exist_ok=True)


def download_file(filename, src_url, dest_dir, overwrite=False):
    """download_file now defaults to non-SSL if SSL fails, with warning to
    user.
    MacOS doesn't have ceritifi installed by default."""
    dir_path = make_path_like(dest_dir)
    dest_path = dir_path.joinpath(filename)
    if not dest_path.exists():
        ensure_directory_exists(dest_dir)
    elif not overwrite:
        # check if file already exists, and return if it is there.
        logger.info(
            f"File exists: {dest_path}. Set overwrite=True to overwrite the file."
        )
        return
    try:
        print(
            f"Downloading manifest from {src_url} to {dest_dir}. "
            "Can take several minutes..."
        )
        with urlopen(src_url) as response:
            with open(dest_path, "wb") as out_file:
                shutil.copyfileobj(response, out_file)
    except URLError as e:
        logger.error(e)
        logger.info(
            f"Downloading manifest from {src_url} failed. Please correct "
            "url in source code"
        )
        raise URLError(e)


def is_file_like(obj):
    """Check if the object is a file-like object.  For objects to be considered
    file-like, they must be an iterator AND have either a
    `read` and/or `write` method as an attribute.
    Note: file-like objects must be iterable, but iterable objects need not be file-like.

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
    """Returns a file-like object based on the provided input.
    If the input argument is a string, it will attempt to open the file
    in 'rb' mode.
    """
    if is_file_like(filepath_or_buffer):
        return filepath_or_buffer

    if PurePath(filepath_or_buffer).suffix == ".gz":
        return gzip.open(filepath_or_buffer, "rb")

    return open(filepath_or_buffer, "rb")


def get_file_from_archive(file_or_archive, filename):
    """Retrieve a file object from a regular file or a ZIP archive."""
    if isinstance(file_or_archive, str):
        file_or_archive = Path(file_or_archive)
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
            raise ValueError(
                f"File '{filename}' not found in the ZIP archive."
            )
    else:
        raise ValueError(
            "Unsupported file type. Only '.csv' and '.zip' are supported."
        )


def reset_file(filepath_or_buffer):
    """Attempts to return the open file to the beginning if it is seekable."""
    if not hasattr(filepath_or_buffer, "seek"):
        return

    filepath_or_buffer.seek(0)

def idat_pair_basepaths(dir_):
    pass
