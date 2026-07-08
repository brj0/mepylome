"""Test the mepylome implementation of RFpurify.

Put test IDAT files into:
  ~/mepylome/tests/purity_tests/

The generated TSV is used as a reference to compare against the original R
RFpurify implementation.
"""

from pathlib import Path

import pandas as pd

from mepylome import MethylData, idat_basepaths

TEST_DIR = Path("~/mepylome/tests/purity_tests/").expanduser()
out_file = TEST_DIR / "rfpurify_purity_results_py.tsv"
basepaths = sorted(idat_basepaths(TEST_DIR))

rows = []

for basepath in basepaths:
    sample_id = Path(basepath).name

    print(f"Processing: {sample_id}")

    methyl = MethylData(file=basepath, prep="illumina")

    absolute = methyl.predict_purity(method="absolute").values[0]
    estimate = methyl.predict_purity(method="estimate").values[0]

    rows.append(
        {
            "sample_id": sample_id,
            "absolute": absolute,
            "estimate": estimate,
        }
    )

results = pd.DataFrame(rows)

results.to_csv(
    out_file,
    sep="\t",
    index=False,
)

print(results)
print(f"Saved: {out_file}")
