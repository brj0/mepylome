"""Processes all new IDAT files and generate beta values.

The generated beta files are compatible with the 450k array.

Author: Jon Brugger
Date: 2024-03
"""

import logging
import subprocess
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from tqdm import tqdm

from mepylome import ManifestLoader, MethylData, RawData

INDEX_FILE = "/applications/reference_data/betaEPIC450Kmix_bin/index.csv"
NEW_IDAT_DIR = "/data/epidip_IDAT"
OUTPUT_DIR = "/data/epidip_CpGs_mepylome/"
BETA_BIN_FILE_SIZE = 3207696
ENDING_GRN = "_Grn.idat"
ENDING_BETAS = "_betas_filtered.bin"
LOG_FILE = "/applications/tmp/beta.logger"
ENDING_ERROR = "_error.txt"

logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)
logger.info("Processing IDAT to beta values")


if not Path(OUTPUT_DIR).exists():
    Path(OUTPUT_DIR).mkdir(parents=True)


def result_filepath(filepath, error=False):
    ending = ENDING_ERROR if error else ENDING_BETAS
    return Path(OUTPUT_DIR, filepath.name + ending)


def write_error_file(filepath, error):
    idat_files = Path(NEW_IDAT_DIR, filepath.name)
    error_file = Path(OUTPUT_DIR, filepath.name + ENDING_ERROR)
    command = f"ls -lh {idat_files}*"
    files_on_disk = subprocess.check_output(command, shell=True).decode(
        "utf-8"
    )
    error_message = (
        "During processing '"
        + filepath.name
        + "' the following error occurred:\n\n"
        + str(error)
        + "\n\nCorresponding files on disk:\n"
        + files_on_disk
        + "\n\n\nTo recalculate, delete this file."
    )
    with open(error_file, "w") as f:
        f.write(error_message)


def process_idat_file(idat_basepath):
    filepath = result_filepath(idat_basepath)
    if Path(filepath).exists():
        if Path(filepath).stat().st_size == BETA_BIN_FILE_SIZE:
            logger.info("Betas allready caluclated: " + filepath.name)
            return
        else:
            logger.info("Recalulate faulty betas: " + filepath.name)
    raw = RawData(idat_basepath)
    methyl = MethylData(raw)
    betas_450k_df = methyl.betas_for_cpgs(cpgs=cpg_index_450k, fill=0.49)
    betas_450k_np = betas_450k_df.iloc[:, 0].values
    betas_450k_np.tofile(filepath)


def process_idat_file_safe(idat_basepath):
    try:
        process_idat_file(idat_basepath)
    except Exception as e:
        write_error_file(idat_basepath, error=e)


with open(INDEX_FILE) as f:
    cpg_index_450k = np.array(f.read().splitlines())


grn_idat_files = [
    x
    for x in list(Path(NEW_IDAT_DIR).iterdir())
    if str(x).endswith(ENDING_GRN)
]

basepaths = [
    Path(str(name).replace(ENDING_GRN, "")).expanduser()
    for name in grn_idat_files
]

old_basepaths = [
    path
    for path in basepaths
    if Path(result_filepath(path)).exists()
    or Path(result_filepath(path, error=True)).exists()
]

new_basepaths = list(set(basepaths).difference(old_basepaths))


def is_valid_file(filepath):
    return filepath.exists() and filepath.stat().st_size != BETA_BIN_FILE_SIZE


faulty_basepaths = [
    path for path in basepaths if is_valid_file(result_filepath(path))
]

if len(new_basepaths + faulty_basepaths) == 0:
    logger.info("No new or faulty IDAT files to process")
    exit()

logger.info(
    f"Processing {len(new_basepaths)} new and "
    f"{len(faulty_basepaths)} incorrectly processed IDAT files."
)

# Load manifests before multiprocessing
ManifestLoader.get_manifest("450k")
ManifestLoader.get_manifest("epic")
ManifestLoader.get_manifest("epicv2")


with Pool() as pool:
    betas_450k_results = list(
        tqdm(
            pool.imap(
                process_idat_file_safe, new_basepaths + faulty_basepaths
            ),
            total=len(new_basepaths),
        )
    )


logger.info("Processing IDAT to beta values completed")
