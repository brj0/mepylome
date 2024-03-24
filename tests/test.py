import logging
import numpy as np
import pandas as pd
import pyranges as pr
import cbseg
import logging
from sklearn.linear_model import LinearRegression
import os
import re
import time
import warnings
from pathlib import Path
from urllib.parse import urljoin
import pickle
import gzip

import mepylome

# from methylprep.files.idat import IdatDataset
# from methylprep.models.probes import Channel
# from methylprep.utils.parsing import *
# import methylprep


# warnings.simplefilter(action="ignore", category=FutureWarning)

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
    Timer,
)

# import numexpr

LOGGER = logging.getLogger(__name__)
print("imports done")


def save_cnv(cnv_zip_path, cnv):
    bins_df = cnv.bins.df[["Chromosome", "Start", "End", "Median"]]
    bins_df.columns = ["chrom", "start", "end", "value"]
    detail_df = cnv.detail.df[
        ["Chromosome", "Start", "End", "Name", "Median", "N_probes"]
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


timer = Timer()


ref_dir = "/data/ref_IDAT/cnvrefidat_450k"
smp0 = "/data/epidip_IDAT/6042324058_R03C02_Grn.idat"
smp1 = "/data/epidip_IDAT/6042324058_R04C01_Red.idat"
smp2 = "/data/epidip_IDAT/6042324058_R04C02_Red.idat"
smp3 = "/data/epidip_IDAT/6042324058_R05C01_Grn.idat"
smp4 = "/data/epidip_IDAT/6042324058_R05C02_Grn.idat"
smp5 = "/data/epidip_IDAT/6042324058_R06C01_Red.idat"
smp6 = "/data/epidip_IDAT/6042324058_R06C02_Grn.idat"
smp7 = "/data/epidip_IDAT/7970368088_R01C01_Grn.idat"
smp8 = "/data/epidip_IDAT/7970368088_R01C02_Grn.idat"
ref0 = "/data/ref_IDAT/cnvrefidat_450k/3999997083_R02C02_Grn.idat"
ref1 = "/data/ref_IDAT/cnvrefidat_450k/5775446049_R06C01_Red.idat"
ref2 = "/data/ref_IDAT/cnvrefidat_450k/5775446051_R02C01"
ref3 = "/data/ref_IDAT/cnvrefidat_450k/3999997083_R02C02_Grn.idat"
ref4 = "/data/ref_IDAT/cnvrefidat_450k/5775446049_R01C02_Grn.idat"
faulty = "/data/epidip_IDAT/10003886027_R05C02_Grn.idat"
faulty = "/data/epidip_IDAT/206486310027_R05C01_Grn.idat"

timer = Timer()
idat_data = mepylome.IdatParser(smp1)
timer.stop("Parsing IDAT")


GENES = pkg_resources.resource_filename("mepylome", "data/hg19_genes.tsv.gz")
GAP_450K = pkg_resources.resource_filename("mepylome", "data/gap_450k.csv.gz")

timer.start()
# refs_raw = RawData(ref_dir)
refs_raw = RawData([ref0, ref1])
timer.stop("RawData ref")

timer.start()
ref_methyl = MethylData(refs_raw)
timer.stop("MethylData ref")

manifest = ManifestLoader.get_manifest("450k")
# manifest = ManifestLoader.get_manifest("epic")
# manifest = ManifestLoader.get_manifest("epicv2")


timer.start()
sample_raw = RawData(smp0)
timer.stop("RawData sample")

timer.start()
sample_methyl = MethylData(sample_raw)
timer.stop("MethylData sample")

gap = pr.PyRanges(pd.read_csv(GAP_450K))
gap.Start -= 1

genes_df = pd.read_csv(GENES, sep="\t")
genes_df.Start -= 1
genes = pr.PyRanges(genes_df)
genes = genes[["Name"]]

timer.start()
annotation = Annotation(manifest, gap=gap, detail=genes)
timer.stop("Annotation")

timer.start()
cnv = CNV(sample_methyl, ref_methyl, annotation)
timer.stop("CNV")

timer.start()
cnv.fit()
timer.stop("CNV fit")

timer.start()
cnv.set_bins()
timer.stop("CNV set_bins")

timer.start()
cnv.set_detail()
timer.stop("CNV set_detail")

timer.start()
cnv._set_detail()
timer.stop("CNV set_detail")

timer.start()
cnv.set_segments()
timer.stop("CNV set_segments")

self = cnv
sample = sample_methyl
reference = ref_methyl


cnv_zip_path = "/data/epidip_CNV_data/py_cnv.zip"
# save_cnv(cnv_zip_path, cnv)


quit()

timer.start()
# r = RawData(smp7)
r = RawData([ref0, ref1])
timer.stop("1")
m = MethylData(r)
timer.stop("2")
m.illumina_control_normalization()
timer.stop("2.1")
m.illumina_bg_correction()
timer.stop("2.2")
m.preprocess_raw_cached()
timer.stop("2.5")
b = m.beta
timer.stop("3")



cn = CNV(m, ref_methyl, annotation)
timer.stop("2")
cn.fit()
timer.stop("3")

self = sample_methyl
self = ref_methyl

timer.start()
betas_450k_df = self.converted_beta(cpgs=None, fill=0.49)
timer.stop("1")

timer.start()
cn._set_bins()
# cn.set_bins()
timer.stop("4")
cn.bins

cn.set_detail()
timer.stop("5")
cn.set_segments()
timer.stop("file to csv")


filepath_gz = Path("~/Downloads/manifest.pkl.gz").expanduser()
filepath = Path("~/Downloads/manifest.pkl").expanduser()

timer.start()
with gzip.open(filepath_gz, "wb") as f:
    pickle.dump(manifest, f)

timer.stop("pickel")

timer.start()
with gzip.open(filepath_gz, "rb") as f:
    loaded_data = pickle.load(f)

timer.stop("pickel")


timer.start()
with open(filepath, "wb") as f:
    pickle.dump(manifest, f)

timer.stop("pickel")

timer.start()
with open(filepath, "rb") as f:
    loaded_data = pickle.load(f)

timer.stop("pickel")


from nanodip.config import BETA_VALUES

OUTPUT_DIR = "/data/epidip_CpGs_mepylome/"
ENDING_BETAS = "_betas_filtered.bin"

grn_idat_files = [
    x
    for x in list(Path(OUTPUT_DIR).iterdir())
    if str(x).endswith(ENDING_BETAS)
]

for x in grn_idat_files:
    # x = grn_idat_files[0]
    y = Path(BETA_VALUES, x.name)
    data_x = np.fromfile(x, dtype=np.float64)
    data_y = np.fromfile(y, dtype=np.float64)
    c = np.corrcoef(data_x, data_y)
    print(c[0, 1])




        









