"""Contains functions for setting up and downloading data for the tutorial."""

import gzip
import os
import shutil
from http import HTTPStatus
from itertools import repeat
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from mepylome.utils.files import get_resource_path

CONTROL = "Control (muscle tissue)"


def download_file(url, save_path):
    """Function to download a file."""
    import requests

    response = requests.get(url, stream=True)
    if response.status_code == HTTPStatus.OK:
        with open(save_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024):
                file.write(chunk)
    else:
        print(f"Failed to download file from {url}")


def unzip_and_remove_gz_files(directory):
    """Function to unzip .gz files and remove the original .gz files."""
    for file_path in directory.glob("*.gz"):
        output_path = file_path.with_suffix("")
        with gzip.open(file_path, "rb") as f_in, open(
            output_path, "wb"
        ) as f_out:
            shutil.copyfileobj(f_in, f_out)
        os.remove(file_path)


def download_and_process_url(data):
    """Downloads the corresponding IDAT file."""
    url, download_dir = data
    file_name = url.split("=")[-1]
    file_name = file_name.replace("%5F", "_").replace("%2E", ".")
    file_name = file_name.split("_", 1)[1]
    dest_path = download_dir / file_name
    download_file(url, dest_path)


def download_idats(download_dir, idat_grn_urls):
    """Downloads all IDAT files."""
    urls = []
    for url_grn in idat_grn_urls:
        url_red = url_grn.replace("%5FGrn%2", "%5FRed%2")
        urls.extend([url_grn, url_red])
    with Pool() as pool:
        list(
            tqdm(
                pool.imap(
                    download_and_process_url, zip(urls, repeat(download_dir))
                ),
                total=len(urls),
                desc="Downloading IDAT's",
            )
        )
    unzip_and_remove_gz_files(download_dir)


def setup_tutorial_files(analysis_dir, reference_dir):
    """Set up tutorial directory structure and download IDAT files."""
    analysis_dir = Path(analysis_dir)
    reference_dir = Path(reference_dir)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    reference_dir.mkdir(parents=True, exist_ok=True)
    TUTORIAL_CSV = get_resource_path("mepylome", "data/tutorial.csv.gz")
    tutorial_df = pd.read_csv(TUTORIAL_CSV)
    tutorial_df.drop(columns=["Url_Grn"]).to_csv(
        analysis_dir / "annotation.csv", index=False
    )
    is_control = tutorial_df["Diagnosis"] == CONTROL
    download_idats(analysis_dir, tutorial_df["Url_Grn"])
    download_idats(reference_dir, tutorial_df[is_control]["Url_Grn"])
