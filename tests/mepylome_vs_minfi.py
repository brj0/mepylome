import sys
import time
import pandas as pd
import numpy as np
from memory_profiler import memory_usage


prep = sys.argv[1]
preps = ["illumina", "swan", "noob"]
if prep not in preps:
    print(f"First command line argument must be in {preps}")
    print(f"Received: {prep}")
    quit()

memory0 = memory_usage()[0]
time0 = time.time()


from pathlib import Path

from mepylome import MethylData, idat_basepaths
from mepylome.utils import ensure_directory_exists

time1 = time.time()
memory1 = memory_usage()[0]
print(f"Import time: {time1 - time0} s")
print(f"Memory usage: {memory1 - memory0} MB\n")


HOME_DIR = Path.home()
DEFAULT_TEST_DIR = Path(HOME_DIR, "Documents", "mepylome", "tests")

ensure_directory_exists(DEFAULT_TEST_DIR)

idat_files = sorted(idat_basepaths(DEFAULT_TEST_DIR))


time1 = time.time()
for idat_file in idat_files:
    methyl_data = MethylData(file=idat_file, prep="illumina")
time2 = time.time()
memory1 = memory_usage()[0]

N = len(idat_files)
tpc = (time2 - time1) / N

print(f"Extraction time ({prep}): {time2 - time1} s")
print(f"  Time per case: {tpc} s (No. of cases: {N})\n")

print("Total:")
print(f"Time (includes benchmarking utils): {time2 - time0} s")
print(f"Memory usage: {memory1 - memory0} MB\n")

methyl_data = MethylData(file=idat_files[0], prep="illumina")

methyl_data.methylated.sort_index().to_csv(
    Path(DEFAULT_TEST_DIR, f"mepylome-{prep}-methylated.csv")
)
methyl_data.unmethylated.sort_index().to_csv(
    Path(DEFAULT_TEST_DIR, f"mepylome-{prep}-unmethylated.csv")
)

# Not in epic: 'cg08623843'
mepylome_path = Path(DEFAULT_TEST_DIR, f"mepylome-{prep}-methylated.csv")
minfi_path = Path(DEFAULT_TEST_DIR, f"minfi-{prep}-methylated.csv")

mepylome_df = pd.read_csv(mepylome_path, index_col=0)
minfi_df = pd.read_csv(minfi_path, index_col=0)

missing_cpgs = set(minfi_df.index) - set(mepylome_df.index)
extra_cpgs = set(mepylome_df.index) - set(minfi_df.index)

overlap_cpgs = mepylome_df.index.intersection(minfi_df.index)

len(missing_cpgs)
len(extra_cpgs)
len(overlap_cpgs)

diff = mepylome_df.loc[overlap_cpgs] - minfi_df.loc[overlap_cpgs]

diff.describe()

np.max(diff)
np.linalg.norm(diff)
np.mean(np.abs(diff))
np.std(np.abs(diff))
