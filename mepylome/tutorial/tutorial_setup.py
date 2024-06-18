import gzip
import os
import shutil
from itertools import repeat
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
import pkg_resources
import requests
from tqdm import tqdm

HOME = Path.home()
DIR = HOME / "Documents" / "mepylome" / "tutorial"
DOWNLOAD_DIR = HOME / "Documents" / "mepylome" / "tutorial" / "download"
ANALSYIS_DIR = HOME / "Documents" / "mepylome" / "tutorial" / "analysis"
REFERENCE_DIR = HOME / "Documents" / "mepylome" / "tutorial" / "reference"

CONTROL = "Control (muscle tissue)"


def download_file(url, save_path):
    """Function to download a file."""
    response = requests.get(url, stream=True)
    if response.status_code == 200:
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


def setup_tutorial_files():
    DIR.mkdir(parents=True, exist_ok=True)
    ANALSYIS_DIR.mkdir(parents=True, exist_ok=True)
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    tutorial_df = pd.read_csv(
        pkg_resources.resource_filename("mepylome", "data/tutorial.csv.gz")
    )
    tutorial_df.drop(columns=["Url_Grn"]).to_csv(
        ANALSYIS_DIR / "annotation.csv", index=False
    )
    is_control = tutorial_df["Diagnosis"] == CONTROL
    download_idats(ANALSYIS_DIR, tutorial_df["Url_Grn"])
    download_idats(REFERENCE_DIR, tutorial_df[is_control]["Url_Grn"])
