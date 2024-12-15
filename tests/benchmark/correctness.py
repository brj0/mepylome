"""Compares minfi outputs with mepylome outputs.

This script compares the methylation data extracted by the minfi (R) and
mepylome (Python) packages. It assumes that the test_idat_extraction.py and
test_idat_extraction.R scripts have been run first to generate output files in
~/mepylome/output_tests.

Usage:
    python correctness.py
"""

from pathlib import Path

import pandas as pd

HOME_DIR = Path.home()
TEST_OUTPUT_DIR = Path(HOME_DIR, "mepylome", "output_tests")


def difference(path_py, path_r):
    """Compares mepylome output with minfi output."""
    title = f"{path_py.with_suffix('').name} vs. {path_r.with_suffix('').name}"
    print("\n\n" + title)
    print("*" * len(title))
    df_py = pd.read_csv(path_py, index_col=0)
    df_r = pd.read_csv(path_r, index_col=0)
    missing_cpgs = set(df_r.index) - set(df_py.index)
    extra_cpgs = set(df_py.index) - set(df_r.index)
    overlap_cpgs = df_py.index.intersection(df_r.index)
    diff = df_py.loc[overlap_cpgs] - df_r.loc[overlap_cpgs]
    rel_diff = diff / df_r.loc[overlap_cpgs]
    abs_max = abs(diff).max().iloc[0]
    rel_abs_max = abs(rel_diff).max().iloc[0]
    corr = df_py.iloc[:, 0].corr(df_r.iloc[:, 0])
    print(f"    No. of CpG's in minfi not in mepylome: {len(missing_cpgs)}")
    print(f"    No. of CpG's in mepylome not in minfi: {len(extra_cpgs)}")
    print(f"    No. of CpG's in mepylome and in minfi: {len(overlap_cpgs)}")
    print(f"    Correlation: {corr}")
    print(f"    RELATIVE ERROR MAX: {rel_abs_max}")
    print(f"    ERROR MAX: {abs_max}\n")
    print(f"    Difference of data frames:\n{diff}")
    print(f"    Relative difference of data frames:\n{rel_diff}")
    print(f"    Difference of data frames (summary):\n{diff.describe()}")


idat_paths = sorted(TEST_OUTPUT_DIR.iterdir())

for path in idat_paths:
    name = path.name
    split_name = name.split("-")
    package = split_name[1]
    if package != "mepylome":
        continue
    split_name[1] = "minfi"
    name_r = "-".join(split_name)
    path_r = Path(TEST_OUTPUT_DIR, name_r)
    if not path_r.exists():
        print(f"Path not found: {path_r}")
        continue
    difference(path, path_r)
