"""
Basic usage of the package
"""


import pandas as pd
import pkg_resources
import pyranges as pr

from mepylome import (
    CNV,
    Annotation,
    IdatParser,
    ManifestLoader,
    MethylData,
    RawData,
)



# 0) Genetic data

# Gene data frame
GENES = pkg_resources.resource_filename("mepylome", "data/hg19_genes.tsv.gz")

# DNA gap data frame
GAP_450K = pkg_resources.resource_filename("mepylome", "data/gap_450k.csv.gz")

# DNA gaps
gap = pr.PyRanges(pd.read_csv(GAP_450K))
# PyRanges starts at 0
gap.Start -= 1

# Gene coordinates
genes = pr.PyRanges(pd.read_csv(GENES, sep="\t"))
# PyRanges starts at 0
genes.Start -= 1



# 1) IDAT files

# Idat file
sample = "/data/epidip_IDAT/6042324058_R03C02_Grn.idat"

# Reference files (either basename like below or full pathname of at least
# one of the Red/Grn pairs).
reference0 = "/data/ref_IDAT/cnvrefidat_450k/3999997083_R02C02"
reference1 = "/data/ref_IDAT/cnvrefidat_450k/5775446049_R06C01"

# Alternatively a directory can be given. All files within will be used.
reference_dir = "/data/ref_IDAT/cnvrefidat_450k"



# 2) Parsing IDAT

# The idat file must contain the full path with _Red.idat/_Grn.idat ending
idat_data = IdatParser(sample)



# 3) Manifest data

manifest = ManifestLoader.get_manifest("450k")
# manifest = ManifestLoader.get_manifest("epic")
# manifest = ManifestLoader.get_manifest("epicv2")



# 4) The raw methylation data

raw_reference = RawData([reference0, reference1])

# Alternative using reference directory
# raw_reference = RawData(reference_dir)

# Sample data must include 1 idat-pair
raw_sample = RawData(sample)



# 5) The preprocessed methylation data
ref_methyl_data = MethylData(raw_reference)
sample_methyl = MethylData(raw_sample)



# 6) Beta value
beta = sample_methyl.beta



# 7) Annotation object
annotation = Annotation(manifest, gap=gap, detail=genes)



# 8) CNV
cnv = CNV(sample_methyl, ref_methyl_data, annotation)

# Apply linear regression model
cnv.fit()

# Get CNV for all bins
cnv.set_bins()

# Get CNV for all genes
cnv.set_detail()

# Segment Genome using binary circular segmentation algorithm
cnv.set_segments()

print(cnv)
