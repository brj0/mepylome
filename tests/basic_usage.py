"""Basic usage of the package."""

from pathlib import Path

import pandas as pd
import pkg_resources
import pyranges as pr

from mepylome import (
    CNV,
    Annotation,
    IdatParser,
    Manifest,
    MethylData,
    RawData,
)

# 1) IDAT files

HOME = Path.home()
DIR = HOME / "Documents" / "mepylome"

# Idat file (provide either the basename like below or a full pathname of at
# least one of the Red/Grn pairs).
sample = DIR / "203049640041_R04C01_Grn.idat"

# Reference files
reference0 = DIR / "203049640041_R03C01"
reference1 = DIR / "203049640041_R02C01_Red.idat"

# Alternatively a directory can be given. All files within will be used.
reference_dir = DIR / "cnv_neutral_450k_dir"


# 2) Parsing IDAT

# The IDAT file must contain the full path with _Red.idat/_Grn.idat ending
idat_data = IdatParser(sample)


# 3) Manifest data

manifest_450k = Manifest("450k")
manifest_epic = Manifest("epic")
manifest_epicv2 = Manifest("epicv2")


# 4) The raw methylation data

# Sample data of Grn and Red signals
raw_sample = RawData(sample)

# Reference data of Grn and Red signals
raw_reference = RawData([reference0, reference1])

# Alternative using reference directory
# raw_reference = RawData(reference_dir)


# 5) The preprocessed methylation data

# prep="illumina" is not needed here, as it is the standard.
sample_methyl = MethylData(raw_sample, prep="illumina")
reference_methyl = MethylData(raw_reference, prep="illumina")


# 6) Beta value
beta = sample_methyl.beta


# 7) Annotation object
annotation = Annotation(manifest_epic)

# 7.1) Annotation object with custom data (advanced)
# You can use custom genetic and gap data:

# Data frames

PACKAGE_DIR = Path(pkg_resources.resource_filename("mepylome", ""))
GAPS = PACKAGE_DIR / "data" / "gaps.csv.gz"
GENES = PACKAGE_DIR / "data" / "hg19_genes.tsv.gz"

# Transform to PyRanges
gap = pr.PyRanges(pd.read_csv(GAPS))
genes = pr.PyRanges(pd.read_csv(GENES, sep="\t"))

# Correction as PyRanges objects starts at 0 (not 1)
gap.Start -= 1
genes.Start -= 1

# Annotation with custom gaps and genes
annotation = Annotation(manifest_epic, gap=gap, detail=genes)


# 8) Set CNV object and apply linear regression
cnv = CNV(sample_methyl, reference_methyl)

# Get CNV for all bins
cnv.set_bins()

# Get CNV for all genes
cnv.set_detail()

# Segment Genome using binary circular segmentation algorithm
cnv.set_segments()

# For all the above steps there is a abbreviation:
cnv = CNV.set_all(sample_methyl, reference_methyl)

print(cnv)

# Visualize CNV in the browser
cnv.plot()


# 8.1) CNV object with custom annotation
cnv = CNV(sample_methyl, reference_methyl, annotation)
