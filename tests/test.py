import gzip
import logging
import inspect
import os
import pickle
import re
import time
import timeit
import warnings
from functools import reduce
from pathlib import Path
from urllib.parse import urljoin

import cbseg
import numpy as np
import pandas as pd
from functools import wraps
import pkg_resources
import pyranges as pr
import scipy.stats as stats
from cbseg import (
    determine_cbs,
    determine_cbs_stat,
    determine_t_stat,
    segment,
    validate,
)
from scipy.stats import rankdata
from sklearn.linear_model import LinearRegression

import mepylome

# TODO too long for import
from mepylome.dtypes import (
    CNV,
    Annotation,
    memoize,
    ArrayType,
    Channel,
    Manifest,
    ManifestLoader,
    IdatParser,
    MethylData,
    ProbeType,
    CNVPlot,
    RawData,
    cache,
)
from mepylome.utils import (
    download_file,
    Timer,
    ensure_directory_exists,
    get_file_from_archive,
    get_file_object,
    reset_file,
)


# from methylprep.files.idat import IdatDataset
# from methylprep.models.probes import Channel
# from methylprep.utils.parsing import *
# import methylprep


# warnings.simplefilter(action="ignore", category=FutureWarning)


# import numexpr

CNV_DATA = os.path.expanduser("/data/epidip_CNV_data/")

LOGGER = logging.getLogger(__name__)
print("imports done")

# [["Chromosome", "Start", "End", "N_probes", "Median", "Var"]]
# ["Chromosome", "Start", "End", "Name", "Median", "N_probes"]
# ["Chromosome", "Start", "End", "Mean", "Median"]


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
idat_data = IdatParser(smp1)
timer.stop("Parsing IDAT")


GENES = pkg_resources.resource_filename("mepylome", "data/hg19_genes.tsv.gz")
GAPS = pkg_resources.resource_filename("mepylome", "data/gaps.csv.gz")

timer.start()
# refs_raw = RawData(ref_dir)
refs_raw = RawData([ref0, ref1])
timer.stop("RawData ref")

timer.start()
ref_methyl = MethylData(refs_raw)
timer.stop("MethylData ref")

# quit()

manifest = Manifest("450k")
# manifest = ManifestLoader.get_manifest("epic")
# manifest = ManifestLoader.get_manifest("epicv2")


timer.start()
sample_raw = RawData(smp0)
timer.stop("RawData sample")

timer.start()
sample_methyl = MethylData(sample_raw)
timer.stop("MethylData sample")

timer.start()
betas = sample_methyl.converted_beta(cpgs=None, fill=0.49)
timer.stop("beta 1")

timer.start()
betas = sample_methyl.converted_beta(cpgs=None, fill=0.49)
timer.stop("beta 2")

gap = pr.PyRanges(pd.read_csv(GAPS))
gap.Start -= 1

timer.start()
genes_df = pd.read_csv(GENES, sep="\t")
genes_df.Start -= 1
genes = pr.PyRanges(genes_df)
genes = genes[["Name"]]
timer.stop("genes")

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
cnv.set_segments()
timer.stop("CNV set_segments")

self = cnv
sample = sample_methyl
reference = ref_methyl


file_path = "/data/epidip_CNV_data/py_cnv.zip"
file_name = "py_cnv.zip"

file_dir = "/data/epidip_CNV_data/"

timer.start()
cnv.write(Path(file_dir, "py_cnv"))
timer.stop("zip write")


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

# Time passed: 9.987592697143555 ms (Parsing IDAT)
# Time passed: 1480.562686920166 ms (RawData ref)
# Time passed: 392.32563972473145 ms (MethylData ref)
# Time passed: 7.627725601196289 ms (RawData sample)
# Time passed: 79.34379577636719 ms (MethylData sample)
# Time passed: 410.3200435638428 ms (beta 1)
# Time passed: 10.120391845703125 ms (beta 2)
# Time passed: 5241.268873214722 ms (Annotation)
# sample: Intensities smaller than 0 set to 1.
# reference: Intensities smaller than 0 set to 1.
# Time passed: 32.681941986083984 ms (CNV)
# Time passed: 462.5887870788574 ms (CNV fit)
# Time passed: 314.4857883453369 ms (CNV set_bins)
# Time passed: 329.61559295654297 ms (CNV set_detail)
# Time passed: 1489.6526336669922 ms (CNV set_detail)
# Time passed: 2024.538516998291 ms (CNV set_segments)


# self = MethylData(raw, prep="raw")


timer.start()
for _ in range(100):
    z = manifest.probe_info(ProbeType.ONE)

timer.stop("1")

timer.start()
for _ in range(100):
    z1 = manifest._probe_info([ProbeType.ONE])

timer.stop("1")

np.all(z1.values == z.values)


raw = RawData([ref0, ref1])

timer.start()
self = MethylData(raw, prep="noob")
timer.stop("*")

timer.start()
self = MethylData(sample_raw, prep="noob")
timer.stop("*")

timer.start()
self = MethylData(sample_raw, prep="swan")
timer.stop("*")

cnv.plot()

import dash
import dash_bootstrap_components as dbc

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Alert("Hello, Bootstrap!", className="m-5")

app.run_server()


smp0 = "/data/epidip_IDAT/6042324058_R03C02_Grn.idat"
filepath_or_buffer = smp0
idat_file = get_file_object(filepath_or_buffer)

timer = Timer()


timer.start()
idat_data = mepylome._IdatParser(smp0)
timer.stop("Parsing C++")

timer.start()
py_idat_data = IdatParser(smp0, intensity_only=False)
timer.stop("Parsing Python")

with np.printoptions(edgeitems=2):
    x = f"{py_idat_data.illumina_ids}"
    y = f"{py_idat_data.illumina_ids.__repr__()}"
    z = f"{repr(py_idat_data.illumina_ids)}"


class X:
    def __init__(self, a):
        self.val = a


x = X(0)


def change_x():
    # global x
    x.val = 10


print(x.val)
change_x()
print(x.val)
