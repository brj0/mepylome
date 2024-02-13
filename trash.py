# print("__file__=", __file__); quit()
import pandas as pd
from pathlib import Path
import requests
import time
import gzip
import logging
from pathlib import Path, PurePath
import shutil
from urllib.request import urlopen
from urllib.error import URLError
import ssl
import zipfile
import io
import methylprep
from pyllumina.dtype.manifests import Manifest
from methylprep import Manifest as Manifest_old



logging.basicConfig(level=logging.INFO)
self = Manifest_old("450k")


PROBES_COLUMNS = [
    "IlmnID",
    "AddressA_ID",
    "AddressB_ID",
    "Infinium_Design_Type",
    "Color_Channel",
    "CHR",
]

def get_file_from_archive(file_or_archive, filename):
    """Retrieve a file object from a regular file or a ZIP archive."""
    if isinstance(file_or_archive, str):
        file_or_archive = Path(file_or_archive)
    if file_or_archive.suffix == ".csv":
        return open(file_or_archive, "rb")
    elif file_or_archive.suffix == ".zip":
        with zipfile.ZipFile(file_or_archive, "rb") as archive:
            file_list = archive.namelist()
            file_match = next(
                (f for f in file_list if f.endswith(filename)), None
            )
            if file_match:
                return archive.open(file_match)
            else:
                raise ValueError(
                    f"File '{filename}' not found in the ZIP archive."
                )
    else:
        raise ValueError(
            "Unsupported file type. Only '.csv' and '.zip' files are supported."
        )


def process_manifest_file(filepath, manifest_name):
    probes_file = (
        Path(MANIFEST_PROBES_DIR, manifest_name)
        .expanduser()
        .with_suffix(".csv.gz")
    )
    control_file = (
        Path(MANIFEST_CONTROL_DIR, manifest_name)
        .expanduser()
        .with_suffix(".csv.gz")
    )
    probes_pickle_file = probes_file.with_suffix('.pkl')
    control_pickle_file = control_file.with_suffix('.pkl')
    ensure_directory_exists(probes_file)
    ensure_directory_exists(control_file)
    with get_file_from_archive(filepath, manifest_name) as manifest_df:
        # Process probes
        self.seek_to_start(manifest_df)
        data_frame = pd.read_csv(
            manifest_df,
            low_memory=False,
            usecols=PROBES_COLUMNS,
        )
        n_probes = data_frame[data_frame.IlmnID.str.startswith("[")].index[0]
        data_frame = data_frame[:n_probes]
        data_frame.AddressA_ID = pd.to_numeric(data_frame.AddressA_ID).astype('Int32')
        data_frame.AddressB_ID = pd.to_numeric(data_frame.AddressB_ID).astype('Int32')
        data_frame.to_csv(probes_file, index=False)
        data_frame.to_pickle(probes_pickle_file)
        # Process controls
        self.seek_to_start(manifest_df)
        data_frame = pd.read_csv(
            manifest_df,
            header=None,
            # Skip header and probe section
            skiprows=2 + n_probes,
            usecols=range(len(CONTROL_COLUMNS)),
        )
        data_frame.columns = CONTROL_COLUMNS
        data_frame.to_csv(control_file, index=False)
        data_frame.to_pickle(control_pickle_file)


def download_default(array_type):
    """Downloads the appropriate manifest file if one does not already exist.

    Arguments:
        array_type {ArrayType} -- The type of array to process.

    Returns:
        [PurePath] -- Path to the manifest file.
    """
    probes_dir_path = Path(MANIFEST_PROBES_DIR).expanduser()
    control_dir_path = Path(MANIFEST_CONTROL_DIR).expanduser()

    filename = ARRAY_TYPE_MANIFEST_FILENAMES[array_type]
    filepath = Path(probes_dir_path).joinpath(filename)

    if Path.exists(filepath):
        return filepath

    LOGGER.info(f"Downloading manifest: {Path(filename).stem}")
    src_url = urljoin(MANIFEST_REMOTE_PATH, filename)
    download_file(filename, src_url, probes_dir_path)

    return filepath


array_type = "450k"
array_type = "epic"
array_type = "epicv2"
array_type = ArrayType(array_type)


src_url = ARRAY_URL[array_type]
filename = src_url.split("/")[-1]
dir_path = MANIFEST_DOWNLOAD_DIR

download_file(filename, src_url, dir_path)

filepath = Path(dir_path, filename).expanduser()

manifest_name = ARRAY_FILENAME[array_type]
process_manifest_file(filepath, manifest_name)


probes_file = (
    Path(MANIFEST_PROBES_DIR, manifest_name)
    .expanduser()
    .with_suffix(".csv.gz")
)
probes_pickle_file = probes_file.with_suffix('.pkl')

t = time.time()
read_data_frame = pd.read_csv(
    probes_file,
    dtype=self.get_data_types(),
)
print(time.time() - t)

t = time.time()
read_data_frame = pd.read_pickle(probes_pickle_file)
print(time.time() - t)

