"""Extends test_idat_extraction.py by averaging SWAN.

As SWAN is stochastic, a direct comparison between outputs of mepylome and
minfi is not straightforward. To better compare the two outputs, we apply SWAN
preprocessing multiple times and then average the results. This process
leverages the central limit theorem to obtain more stable and reliable
methylation estimates.

This script processes IDAT files located in ~/mepylome/tests using
the mepylome package. It performs SWAN preprocessing on the data and averages
the methylation and unmethylation levels obtained over multiple iterations. The
averaged data is then saved to disk for further comparison tests.

Usage:
    python mepylome_vs_minfi_swan.py
"""

from pathlib import Path

from tqdm import tqdm

from mepylome import MethylData, RawData, idat_basepaths
from mepylome.utils import ensure_directory_exists

HOME_DIR = Path.home()
TEST_DIR = Path(HOME_DIR, "mepylome", "tests")
TEST_OUTPUT_DIR = Path(HOME_DIR, "mepylome", "output_tests")
N_LOOPS = 1000

ensure_directory_exists(TEST_DIR)
ensure_directory_exists(TEST_OUTPUT_DIR)

idat_files = sorted(idat_basepaths(TEST_DIR))

for idat_file in idat_files:
    raw_data = RawData(idat_file)
    methyl_data = MethylData(raw_data, prep="swan")
    methyl_acc = methyl_data.methylated
    unmethyl_acc = methyl_data.unmethylated
    for _ in tqdm(range(N_LOOPS - 1), desc=f"{idat_file.name}"):
        methyl_data = MethylData(raw_data, prep="swan")
        methyl_acc += methyl_data.methylated
        unmethyl_acc += methyl_data.unmethylated

    methyl_mean = methyl_acc / N_LOOPS
    unmethyl_mean = unmethyl_acc / N_LOOPS

    probe = idat_file.name
    prefix_py = f"{probe}-mepylome-swan"
    methyl_path_py = Path(TEST_OUTPUT_DIR, f"{prefix_py}-methylated.csv")
    unmethyl_path_py = Path(TEST_OUTPUT_DIR, f"{prefix_py}-unmethylated.csv")

    methyl_mean.to_csv(methyl_path_py)
    unmethyl_mean.to_csv(unmethyl_path_py)
