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

timer.start()
betas = sample_methyl.converted_beta(cpgs=None, fill=0.49)
timer.stop("beta 1")

timer.start()
betas = sample_methyl.converted_beta(cpgs=None, fill=0.49)
timer.stop("beta 2")

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


timer.start()
cpg_detail = self.annotation._cpg_detail.copy()
timer.stop("1")
idx = Cache.indices(self.ratio.index, cpg_detail.IlmnID.values)
timer.stop("2")
cpg_detail["ratio"] = self.ratio.iloc[idx].ratio.values
timer.stop("3")
result = overlap.groupby("Name", dropna=False)["ratio"].agg(
    ["median", "var", "count"]
)
timer.stop("4")
df = self.annotation.detail.df.set_index("Name")
timer.stop("5")
df["Median"] = np.nan
timer.stop("6")
df["Var"] = np.nan
timer.stop("7")
df["N_probes"] = 0
timer.stop("8")
idx = Cache.indices(df.index, result.index.values)
timer.stop("9")
df.iloc[
    idx, df.columns.get_indexer(["Median", "Var", "N_probes"])
] = result.values
timer.stop("10")
df["N_probes"] = df["N_probes"].astype(int)
timer.stop("11")
df = df.reset_index()
timer.stop("12")
self.detail = pr.PyRanges(df)
timer.stop("13")


timer.start()
x = _overlap_indices(cpgs, self.methyl_ilmnid)
timer.stop("1")

timer.start()
y = Cache.overlap_indices(cpgs, self.methyl_ilmnid)
timer.stop("1")

np.all(x[0] == y[0])
np.all(x[1] == y[1])

timer.start()
x = Cache.get(cpgs.tobytes())
timer.stop("1")


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


raw = RawData([ref0, ref1])
self = MethylData(raw, prep="raw")



# BG correction
neg_controls = self.manifest.control_address("NEGATIVE")
grn_med = np.median(
    self._grn[:, np.isin(self.ids, neg_controls, assume_unique=True)], axis=1
)
red_med = np.median(
    self._red[:, np.isin(self.ids, neg_controls, assume_unique=True)], axis=1
)
bg_intensity = np.mean([grn_med, red_med], axis=0)


timer.start()
type_i = self.manifest.probe_info(ProbeType.ONE)[
    ["IlmnID", "N_CpG", "Probe_Type"]
]
type_ii = self.manifest.probe_info(ProbeType.TWO)[
    ["IlmnID", "N_CpG", "Probe_Type"]
]
all_ncpgs = pd.concat([type_i, type_ii], axis=0)
all_ncpgs = all_ncpgs.iloc[ref_methyl.methyl_index]
timer.stop("1")


locus_idx = np.sort(
    np.concatenate(
        [
            type_i.IlmnID.index,
            type_ii.IlmnID.index,
        ]
    )
)


self.methyl_ilmnid
self.methyl_index
timer.start()
all_ncpgs = self.manifest.data_frame[["Probe_Type", "N_CpG"]].iloc[
    self.methyl_index
]

subset_sizes = all_ncpgs.groupby(["Probe_Type", "N_CpG"], dropna=False).size()
subset_size = min(subset_sizes.loc[(slice(None), slice(1, 3))])

timer.start()
indices_per_type = {}
random_indices = {}
for probe_type in [ProbeType.ONE, ProbeType.TWO]:
    all_ncpgs_ = all_ncpgs[all_ncpgs.Probe_Type == probe_type]
    indices = []
    for ncpgs in range(1, 4):
        ids = all_ncpgs_.index[all_ncpgs_.N_CpG == ncpgs]
        ids_subset = np.random.permutation(len(ids))[:subset_size]
        indices.append(ids_subset)
    random_indices[probe_type] = np.sort(np.concatenate(indices))
    indices_per_type[probe_type] = all_ncpgs_.index.values


timer.stop("1")


# intensity = self.methyl[1, :]
intensity = self.methyl
self.swan = np.full((len(self.probes), len(self.methyl_index)), np.nan)

self.methyl[:, random_indices[ProbeType.ONE]]

n_probes = len(self.probes)

for probe_type in [ProbeType.ONE, ProbeType.TWO]:
    intensity[:, random_indices[probe_type]]


sorted_intensity = (
    np.sort(intensity[:, random_indices[ProbeType.ONE]], axis=1)
    + np.sort(intensity[:, random_indices[ProbeType.TWO]], axis=1)
) / 2

from scipy.stats import rankdata

normed_rank = rankdata(intensity, axis=1) / intensity.shape[1]

distribution = np.sort(normed_rank[:, random_indices[ProbeType.ONE]], axis=1)
distribution = np.sort(normed_rank[:, random_indices[ProbeType.TWO]], axis=1)


min_intensity = np.min(intensity[:, random_indices[ProbeType.ONE]], axis=1)
max_intensity = np.max(intensity[:, random_indices[ProbeType.ONE]], axis=1)

min_intensity = np.min(intensity[:, random_indices[ProbeType.TWO]], axis=1)
max_intensity = np.max(intensity[:, random_indices[ProbeType.TWO]], axis=1)

xNew = normed_rank

x = distribution[0,:]
y = intensity[0,random_indices[ProbeType.ONE]]
z = normed_rank[0,:]

len(x)
len(y)
len(z)


interp = np.interp(z, y, x)

