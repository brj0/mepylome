"""Utilities for handling file operations.

This module provides utilities such as downloading files (used to download
manifest files from Illumina), ensuring directories exist, and working with
file-like objects and archives.
"""

import gzip
import logging
import shutil
import zipfile
from itertools import repeat
from multiprocessing import Pool
from pathlib import Path, PurePath

import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

try:
    from importlib.resources import files
except (ImportError, ModuleNotFoundError):
    from importlib_resources import files


__all__ = [
    "download_file",
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


def get_resource_path(package, resource_name=""):
    """Returns the full path to the resource within the specified package."""
    package_path = files(package)
    return package_path.joinpath(resource_name)


def ensure_directory_exists(path_like):
    """Ensures the ancestor directories of the provided path exist."""
    Path(path_like).mkdir(parents=True, exist_ok=True)


def download_file(url, save_path, overwrite=False, show_progress=True):
    """Download a file from a URL and save it to a destination directory.

    Args:
        url (str): The URL from which the file will be downloaded.
        save_path (path_like): The path where the file will be saved.
        overwrite (bool, optional): If True, overwrite the file if it already
            exists. Defaults to False.
        show_progress (bool, optional): If True, displays logging messages and
            progress bar during download. Defaults to True.
    """
    save_path = Path(save_path)
    ensure_directory_exists(save_path.parent)

    if save_path.exists() and not overwrite:
        if show_progress:
            logger.info(
                "File already exists at %s. Skipping download.", save_path
            )
        return

    if show_progress:
        logger.info("Downloading from %s to %s...", url, save_path)

    try:
        import requests

        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        block_size = 8192
        progress_bar = (
            tqdm(
                total=total_size,
                unit="iB",
                unit_scale=True,
                desc="Downloading",
            )
            if show_progress and total_size > 0
            else None
        )

        with save_path.open("wb") as file:
            for chunk in response.iter_content(chunk_size=block_size):
                if chunk:
                    file.write(chunk)
                    if progress_bar:
                        progress_bar.update(len(chunk))
        if progress_bar:
            progress_bar.close()

        if show_progress:
            logger.info("Download completed: %s", save_path)

    except requests.RequestException as error:
        msg = f"Failed to download file from {url}. Error: {error}"
        raise requests.RequestException(msg) from error


def download_geo_idat(probe_id, color_channel, output_dir):
    """Downloads an IDAT file for a specific probe and color channel."""
    geo_id, sentrix_id = probe_id.split("_", 1)
    file_name = f"{geo_id}_{sentrix_id}_{color_channel}.idat.gz"
    encoded_file_name = file_name.replace("_", "%5F").replace(".", "%2E")
    url = GEO_BASE_URL.format(acc=geo_id, filename=encoded_file_name)
    save_path = output_dir / file_name
    download_file(url, save_path, show_progress=False)


def download_geo_probe(probe_info):
    """Downloads both 'Grn' and 'Red' IDAT files for a single probe."""
    probe_id, output_dir = probe_info
    for color_channel in ["Grn", "Red"]:
        download_geo_idat(probe_id, color_channel, output_dir)


def download_geo_probes(output_dir, probe_ids):
    """Downloads IDAT file pairs from GEO."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with Pool() as pool:
        list(
            tqdm(
                pool.imap(
                    download_geo_probe,
                    zip(probe_ids, repeat(output_dir)),
                ),
                total=len(probe_ids),
                desc="Downloading IDAT files",
            )
        )


def unzip_and_remove_gz_files(directory, use_sentrix_id=False):
    """Function to unzip .gz files and remove the original .gz files."""
    for file_path in directory.glob("*.gz"):
        output_path = file_path.with_suffix("")
        if use_sentrix_id:
            sentrix_name = output_path.name.split("_", 1)[1]
            output_path = output_path.with_name(sentrix_name)
        with gzip.open(file_path, "rb") as fi, open(output_path, "wb") as fo:
            shutil.copyfileobj(fi, fo)
        file_path.unlink()


def setup_tutorial_files(analysis_dir, reference_dir):
    """Prepare the directory structure and files for the tutorial.

    This function sets up the necessary directory structure, processes the
    tutorial data, and downloads required IDAT files for both analysis and
    reference.

    Args:
        analysis_dir (str or Path): Path to the directory for storing analysis
            files.
        reference_dir (str or Path): Path to the directory for storing
            reference files.
    """
    control = "Control (muscle tissue)"
    tutorial_csv_path = get_resource_path("mepylome", "data/tutorial.csv.gz")

    analysis_dir = Path(analysis_dir)
    reference_dir = Path(reference_dir)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    reference_dir.mkdir(parents=True, exist_ok=True)

    tutorial_df = pd.read_csv(tutorial_csv_path)
    tutorial_df.drop(columns=["Geo_File_ID"]).to_csv(
        analysis_dir / "annotation.csv", index=False
    )
    is_control = tutorial_df["Diagnosis"] == control

    def _missing_files(dir_path, geo_ids):
        missing_files = []
        for geo_id in geo_ids:
            sentrix_id = geo_id.split("_", 1)[1]
            grn_idat_file = dir_path / f"{sentrix_id}_Grn.idat"
            red_idat_file = dir_path / f"{sentrix_id}_Red.idat"
            if not grn_idat_file.exists() or not red_idat_file.exists():
                missing_files.append(geo_id)
        return missing_files

    missing_analysis_files = _missing_files(
        analysis_dir, tutorial_df["Geo_File_ID"]
    )
    if missing_analysis_files:
        download_geo_probes(analysis_dir, missing_analysis_files)
        unzip_and_remove_gz_files(analysis_dir, use_sentrix_id=True)

    missing_reference_files = _missing_files(
        reference_dir, tutorial_df[is_control]["Geo_File_ID"]
    )
    if missing_reference_files:
        download_geo_probes(reference_dir, missing_reference_files)
        unzip_and_remove_gz_files(reference_dir, use_sentrix_id=True)


def is_file_like(obj):
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


def reset_file(filepath_or_buffer):
    """Attempts to return the open file to the beginning if it is seekable."""
    if hasattr(filepath_or_buffer, "seek"):
        filepath_or_buffer.seek(0)
