"""Used to tests the performance of the mepylome package.

This script performs extraction of methylation information from all IDAT files
located in ~/mepylome/tests. It requires a preprocessing method as a
command-line argument (e.g., "illumina", "noob", "raw", "swan").

If '--save' is added it saves the extracted methylated and unmethylated
information to disk, enabling further comparison tests with the minfi package
using the ./test_idat_extraction.R script and validating correctness with the
./correctness.py script.

Usage:
    python test_idat_extraction.py illumina --save

    # To measure time and memory consumption
    /usr/bin/time -v python test_idat_extraction.py noob
"""

import sys
import time
from pathlib import Path

time0 = time.time()

from mepylome import MethylData, idat_basepaths
from mepylome.utils import ensure_directory_exists

time1 = time.time()


prep = None if len(sys.argv) < 2 else sys.argv[1]
preps = ["illumina", "noob", "raw", "swan"]
if prep not in preps:
    print(f"First command line argument must be in {preps}")
    print(f"Received: {prep}")
    sys.exit()


print(f"Time for importing mepylome: {time1 - time0} s")

HOME_DIR = Path.home()
TEST_DIR = Path(HOME_DIR, "mepylome", "tests")
TEST_OUTPUT_DIR = Path(HOME_DIR, "mepylome", "output_tests")

ensure_directory_exists(TEST_DIR)
ensure_directory_exists(TEST_OUTPUT_DIR)

idat_files = sorted(idat_basepaths(TEST_DIR))


time0 = time.time()

for idat_file in idat_files:
    methyl_data = MethylData(file=idat_file, prep=prep)

time1 = time.time()

N = len(idat_files)
tpc = (time1 - time0) / N

print(f"Time for analysis ({prep}): {time1 - time0} s ({N} cases)")
print(f"    Time per case: {tpc} s")

if len(sys.argv) < 3 or sys.argv[2] != "--save":
    print(
        "Exit script.\nIf you want to save output to disk for comparison "
        "with minfi, rerun this script by adding '--save'"
    )
    sys.exit()

print("\nSave results to disk.")

for idat_file in idat_files:
    probe = idat_file.name
    prefix_py = f"{probe}-mepylome-{prep}"
    prefix_r = f"{probe}-minfi-{prep}"

    methyl_path_py = Path(TEST_OUTPUT_DIR, f"{prefix_py}-methylated.csv")
    unmethyl_path_py = Path(TEST_OUTPUT_DIR, f"{prefix_py}-unmethylated.csv")

    methyl_data = MethylData(file=idat_file, prep=prep)
    methyl_data.methylated.sort_index().to_csv(methyl_path_py)
    methyl_data.unmethylated.sort_index().to_csv(unmethyl_path_py)
