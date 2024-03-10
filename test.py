import logging
import os
import re
import time
import warnings
from pathlib import Path
from urllib.parse import urljoin

import mepylome

# from methylprep.files.idat import IdatDataset
# from methylprep.models.probes import Channel
# from methylprep.utils.parsing import *
# import methylprep



warnings.simplefilter(action="ignore", category=FutureWarning)

from functools import reduce

import numpy as np
import pandas as pd
import pkg_resources
import pyranges as pr
from cbseg import (
    determine_cbs,
    determine_cbs_stat,
    determine_t_stat,
    segment,
    validate,
)

from mepylome import IdatParser

# TODO too long for import
from mepylome.dtypes import (
    CNV,
    Annotation,
    ArrayType,
    Channel,
    Manifest,
    ManifestLoader,
    MethylData,
    ProbeType,
    RawData,
)
from mepylome.utils import (
    download_file,
    ensure_directory_exists,
    get_file_from_archive,
    get_file_object,
    reset_file,
)

# import numexpr

LOGGER = logging.getLogger(__name__)
print("imports done")


class Timer:
    """Measures the time elapsed in milliseconds."""

    def __init__(self):
        self.time0 = time.time()

    def start(self):
        """Resets timer."""
        self.time0 = time.time()

    def stop(self, text=None):
        """Resets timer and return elapsed time."""
        delta_time = 1000 * (time.time() - self.time0)
        comment = "" if text is None else "(" + text + ")"
        print("Time passed:", delta_time, "ms", comment)
        self.time0 = time.time()
        return delta_time


timer = Timer()

filepath = "/data/ref_IDAT/cnvrefidat_450k/3999997083_R02C02_Grn.idat"
filepath = "/data/epidip_IDAT/101130760092_R05C02_Red.idat"

timer = Timer()
idat_data = mepylome.IdatParser(filepath)
timer.stop("Parsing IDAT")


file0 = "/data/ref_IDAT/cnvrefidat_450k/3999997083_R02C02_Grn.idat"
file1 = "/data/ref_IDAT/cnvrefidat_450k/5775446049_R01C02_Grn.idat"
idat_data = mepylome.IdatParser(file0)
idat_data = mepylome.IdatParser(file1)

file0 = "/data/epidip_IDAT/7970368088_R01C01_Grn.idat"
file1 = "/data/epidip_IDAT/7970368088_R01C02_Grn.idat"
file2 = "/data/ref_IDAT/cnvrefidat_450k/5775446049_R06C01_Red.idat"
file3 = "/data/ref_IDAT/cnvrefidat_450k/5775446051_R02C01"
file4 = "/data/epidip_IDAT/206171430049_R08C01"
file5 = "/data/epidip_IDAT/6042324058_R03C02"


# Set the numexpr.evaluate option to True
# pd.options.compute.use_numexpr = True

# GENES = "./data/hg19_genes.tsv.gz"
GENES = pkg_resources.resource_filename("mepylome", "data/hg19_genes.tsv.gz")
GAP_450K = pkg_resources.resource_filename("mepylome", "data/gap_450k.csv.gz")


timer.start()
refs_raw = RawData([file0, file1])
timer.stop("loading rgset ref")

timer.start()
ref_methyl = MethylData(refs_raw)
timer.stop("preproc ref")

manifest = ManifestLoader.get_manifest("450k")
# manifest = ManifestLoader.get_manifest("epic")
# manifest = ManifestLoader.get_manifest("epicv2")


timer.start()
sample_raw = RawData(file5)
timer.stop("loading rgset sample")

timer.start()
sample_methyl = MethylData(sample_raw)
timer.stop("preproc samp")

gap = pr.PyRanges(pd.read_csv(GAP_450K))
gap.Start -= 1
# gap.End -= 1

genes_df = pd.read_csv(GENES, sep="\t").rename(
    columns={
        "start": "Start",
        "end": "End",
        "name": "Gene",
        "strand": "Strand",
        "seqname": "Chromosome",
    }
)
genes_df["Strand"] = genes_df["Strand"].replace({-1: "-", 1: "+"})
genes_df.Start -= 1
genes = pr.PyRanges(genes_df)
genes = genes[["Gene"]]

timer.start()
annotation = Annotation(manifest, gap=gap, genes=genes)
timer.stop("anno")

timer.start()
cnv = CNV(sample_methyl, ref_methyl, annotation)
timer.stop("cnv")

timer.start()
cnv.fit()
timer.stop("CNV fit")

timer.start()
cnv.bin()
timer.stop("CNV bin")

timer.start()
cnv.genes()
timer.stop("CNV genes")

timer.start()
cnv.segments()
timer.stop("CNV segments")

self = cnv
sample = sample_methyl
reference = ref_methyl


filepath = "/data/epidip_CNV_data/py_cnv.zip"


def save_cnv(filepath, cnv):
    bins_df = cnv.bins.df[["Chromosome", "Start", "End", "Median"]]
    bins_df.columns = ["chrom", "start", "end", "value"]
    detail_df = cnv.genes.df[
        ["Chromosome", "Start", "End", "Gene", "Median", "N_probes"]
    ]
    detail_df.columns = ["chrom", "start", "end", "name", "value", "nprobes"]
    segments_df = cnv.segments.df[
        ["Chromosome", "Start", "End", "Mean", "Median"]
    ]
    segments_df.columns = [
        "chrom",
        "start",
        "end",
        "mean",
        "median",
    ]
    import zipfile

    dfs = [
        ("py_cnv_bins.csv", bins_df),
        ("py_cnv_detail.csv", detail_df),
        ("py_cnv_segments.csv", segments_df),
    ]
    with zipfile.ZipFile("/data/epidip_CNV_data/py_cnv.zip", "w") as zf:
        for filename, df in dfs:
            df.to_csv(filename, index=False)
            zf.write(filename)
