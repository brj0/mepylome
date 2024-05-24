import gzip
import inspect
import logging
import os
import pickle
import re
import time
import timeit
import warnings
from functools import reduce, wraps
from pathlib import Path
from urllib.parse import urljoin

import cbseg
import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
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


from mepylome import (
    CNV,
    Annotation,
    MethylAnalysis,
    ArrayType,
    Channel,
    Chromosome,
    CNVPlot,
    IdatParser,
    Manifest,
    MethylData,
    ProbeType,
    RawData,
    ReferenceMethylData,
    cache,
    memoize,
)
from mepylome.utils import (
    get_file_object,
    reset_file,
    get_csv_file,
    ensure_directory_exists,
    download_file,
    Timer,
)

logger = logging.getLogger(__name__)

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
all_ref_dir = "/data/ref_IDAT/"
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


IDAT_DIR = "/data/epidip_IDAT"
IDAT_DIR = "/mnt/ws528695/data/epidip_IDAT"
reference_dir = "/data/ref_IDAT"
self = MethylAnalysis(analysis_dir=IDAT_DIR, reference_dir=reference_dir)
self.run_app()


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
imer.stop("2")
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
self = MethylData(raw, prep="noob")
timer.stop("*")

timer.start()
self = MethylData(sample_raw, prep="noob")
timer.stop("*")

timer.start()
self = MethylData(sample_raw, prep="swan")
timer.stop("*")


timer.start()
idat_data = mepylome._IdatParser(smp0)
timer.stop("Parsing C++")

timer.start()
py_idat_data = IdatParser(smp0, intensity_only=False)
timer.stop("Parsing Python")



m = MethylData(file=[smp0])
r = MethylData(file=[smp0, smp1, smp2, smp3, smp4])
r = MethylData(file=[smp1, smp2, smp3, smp4])
self = CNV.set_all(m, r)
self.fit()




from dash import Dash, dcc, html, Input, Output, State, callback
import time

app = Dash(__name__)

app.layout = html.Div([
    html.Div(dcc.Input(id='input-on-submit-text', type='text')),
    html.Button('Submit', id='submit-button', n_clicks=0),
    html.Div(id='container-output-text',
             children='Enter a value and press submit')
])


@callback(
    Output('container-output-text', 'children'),
    Input('submit-button', 'n_clicks'),
    State('input-on-submit-text', 'value'),
    prevent_initial_call=True,
    running=[(Output("submit-button", "disabled"), True, False)]
)
def update_output(n_clicks, value):
    time.sleep(5)
    return 'The input value was "{}" and the button has been clicked {} times'.format(
        value,
        n_clicks
    )


HOST = "localhost"
app.run(debug=True, host=HOST, use_reloader=False)
