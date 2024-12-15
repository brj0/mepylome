"""Performs CNV analysis for performance tests.

This script reads IDAT files from a specified subdirectory of
~/mepylome/tests/, preprocesses them using a specified method
(illumina, noob, raw, swan), and performs CNV analysis on one of them. All the
files must have the same array type.

Usage:
    python test_cnv.py <preprocessing_method> <subdir_containing_idat>
    python test_cnv.py illumina idat_epic

    # To perform profiling
    /usr/bin/time -v python test_cnv.py illumina idat_450k
"""

import sys
import time
from pathlib import Path

time0 = time.time()

from mepylome import CNV, MethylData, idat_basepaths

time1 = time.time()

HOME_DIR = Path.home()
TEST_DIR = Path(HOME_DIR, "mepylome", "tests")

prep = None if len(sys.argv) < 2 else sys.argv[1]
preps = ["illumina", "noob", "raw", "swan"]
if prep not in preps:
    print(f"First command line argument must be in {preps}")
    print(f"Received: {prep}")
    sys.exit()

subdir = None if len(sys.argv) < 3 else Path(TEST_DIR, sys.argv[2])
if subdir is None or not subdir.exists():
    print(f"Second command line argument must be a subdir of {TEST_DIR}")
    print(f"Received: {subdir}")
    sys.exit()

print(f"Time for importing mepylome: {time1 - time0}")

idat_files = sorted(idat_basepaths(subdir))

sample_file = idat_files[0]
reference_files = idat_files[1:21]

time0 = time.time()

sample_methyl = MethylData(file=sample_file)
reference_methyl = MethylData(file=reference_files)
cnv = CNV.set_all(sample_methyl, reference_methyl)

time1 = time.time()

print(f"Time for CNV analysis: {time1 - time0}")
