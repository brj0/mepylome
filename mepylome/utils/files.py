"""Utilities for handling file operations.

This module provides utilities such as downloading files (used to download
manifest files from Illumina), ensuring directories exist, and working with
file-like objects and archives.
"""

import gzip
import logging
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePath
from typing import IO, Any, Iterable, Union

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

try:
    from importlib.resources import files
except (ImportError, ModuleNotFoundError):
    from importlib_resources import files


__all__ = [
    "download_file",
    "download_files",
    "ensure_directory_exists",
    "get_file_object",
    "get_resource_path",
    "get_csv_file",
    "reset_file",
]

GEO_BASE_URL = (
    "https://www.ncbi.nlm.nih.gov/geo/download/"
    "?acc={acc}&format=file&file={filename}"
)


def get_resource_path(package: str, resource_name: str = "") -> Path:
    """Returns the full path to the resource within the specified package."""
    package_path = files(package)
    return package_path.joinpath(resource_name)


def ensure_directory_exists(path_like: Union[str, Path]) -> None:
    """Ensures the ancestor directories of the provided path exist."""
    Path(path_like).mkdir(parents=True, exist_ok=True)


def download_file(
    url: str,
    save_path: Union[str, Path],
    overwrite: bool = False,
    show_progress: bool = True,
    chunk_size: int = 8192,
    max_attempts: int = 5,
    retry_delay: float = 3.0,
) -> None:
    """Download a file from a URL and save it to a destination directory.

    Retries the download up to `max_attempts` times if it fails.

    Args:
        url (str): The URL from which the file will be downloaded.
        save_path (path_like): The path where the file will be saved.
        overwrite (bool): If True, overwrite existing file. Defaults to False.
        show_progress (bool): Display logs and progress bar. Defaults to True.
        chunk_size (int): Chunk size in bytes. Defaults to 8192.
        max_attempts (int): Number of times to retry on failure. Defaults to 3.
        retry_delay (float): Seconds to wait between retries. Defaults to 3.0.
    """
    save_path = Path(save_path)
    ensure_directory_exists(save_path.parent)

    if save_path.exists() and not overwrite:
        if show_progress:
            logger.info(
                "File already exists at %s. Skipping download.", save_path
            )
        return

    def _single_download(temp_path: Path) -> None:
        if temp_path.exists() and not overwrite:
            mode = "ab"  # resume partial download
            resume_size = temp_path.stat().st_size
        else:
            mode = "wb"
            resume_size = 0

        headers = {"Range": f"bytes={resume_size}-"} if resume_size > 0 else {}

        with requests.get(
            url, stream=True, headers=headers, timeout=10
        ) as response:
            response.raise_for_status()

            total_size = (
                int(response.headers.get("content-length", 0)) + resume_size
            )
            progress_bar = (
                tqdm(
                    total=total_size,
                    initial=resume_size,
                    unit="iB",
                    unit_scale=True,
                    desc="Downloading",
                )
                if show_progress
                else None
            )

            with temp_path.open(mode) as file:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        file.write(chunk)
                        if progress_bar:
                            progress_bar.update(len(chunk))
            if progress_bar:
                progress_bar.close()

    attempt = 0
    success = False

    while attempt < max_attempts and not success:
        attempt += 1

        try:
            if show_progress:
                logger.info("Downloading from %s to %s...", url, save_path)

            temp_path = save_path.with_suffix(save_path.suffix + ".part")
            _single_download(temp_path)
            temp_path.rename(save_path)

            success = True

            if show_progress:
                logger.info("Download completed: %s", save_path)

        except requests.RequestException as e:
            logger.warning(
                "Download attempt %d/%d failed: %s", attempt, max_attempts, e
            )
            if temp_path.exists():
                logger.warning("Partial download saved: %s", temp_path)

            if attempt < max_attempts:
                logger.warning("Retrying in %.1f seconds...", retry_delay)
                time.sleep(retry_delay)
            else:
                logger.warning(
                    "All %d download attempts failed for %s", max_attempts, url
                )

    if not success:
        raise RuntimeError(
            f"Failed to download {url} after {max_attempts} attempts."
        )


def download_files(
    urls: Iterable[str],
    save_paths: Iterable[Union[str, Path]],
    overwrite: bool = False,
    show_progress: bool = True,
    max_workers: Union[int, None] = None,
) -> None:
    """Download multiple files in parallel with optional progress bar.

    Args:
        urls (Iterable[str]): URLs to download.
        save_paths (Iterable[str | Path]): Corresponding save paths.
        overwrite (bool): Overwrite existing files.
        show_progress (bool): Show progress bars.
        max_workers (int | None): Number of parallel downloads.
    """
    # Convert to list to allow len() and zip()
    urls = list(urls)
    save_paths = [Path(p) for p in save_paths]

    if len(urls) != len(save_paths):
        raise ValueError("urls and save_paths must have the same length")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(download_file, url, path, overwrite, False): url
            for url, path in zip(urls, save_paths)
        }

        progress = tqdm(
            total=len(futures),
            desc="Downloading (parallel)",
            unit="file",
            disable=not show_progress,
        )
        try:
            for future in as_completed(futures):
                error = future.exception()
                if error is not None:
                    logger.error(
                        "Error downloading %s: %s", futures[future], error
                    )
                progress.update(1)
        finally:
            progress.close()


def is_file_like(obj: Any) -> bool:
    """Check if the object is a file-like object.

    For objects to be considered file-like, they must be an iterator AND have
    either a `read` and/or `write` method as an attribute.  Note: file-like
    objects must be iterable, but iterable objects need not be file-like.

    Arguments:
        obj (any): The object to check.

    Returns:
        boolean: description

    Examples:
        >>> buffer(StringIO("data"))
        >>> is_file_like(buffer)
        True
        >>> is_file_like([1, 2, 3])
        False
    """
    return all(hasattr(obj, attr) for attr in ("read", "write", "__iter__"))


def get_file_object(filepath_or_buffer: Union[str, Path, IO]) -> IO:
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


def get_csv_file(
    file_or_archive: Union[str, Path], filename: str
) -> IO[bytes]:
    """Retrieve a CSV file from a regular file or a ZIP archive.

    This function extracts a specific CSV file from either a regular CSV file
    or a ZIP archive and returns it as a file-like object.

    Examples:
        >>> get_csv_file('archive.zip', 'example.csv')
        >>> get_csv_file('/path/to/example.csv', 'example.csv')
    """
    file_or_archive = Path(file_or_archive)

    if file_or_archive.suffix == ".csv":
        return file_or_archive.open("rb")

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


def reset_file(filepath_or_buffer: IO) -> None:
    """Attempts to return the open file to the beginning if it is seekable."""
    if hasattr(filepath_or_buffer, "seek"):
        filepath_or_buffer.seek(0)
