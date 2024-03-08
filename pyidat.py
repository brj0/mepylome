# from methylprep.files.idat import IdatDataset
# from methylprep.models.probes import Channel
# from methylprep.utils.parsing import *
# import methylprep

import logging
import os
import re
import time
from pathlib import Path
from urllib.parse import urljoin

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import numpy as np
from cbseg import (
    determine_cbs_stat,
    determine_t_stat,
    determine_cbs,
    segment,
    validate,
)
import pandas as pd
import pyranges as pr
from sklearn.linear_model import LinearRegression

from mepylome import IdatData

# TODO too long for import
from mepylome.dtypes import (
    ArrayType,
    Channel,
    Manifest,
    ManifestLoader,
    ProbeType,
)
from mepylome.utils import (
    download_file,
    ensure_directory_exists,
    get_file_from_archive,
    get_file_object,
    reset_file,
)
from functools import reduce

# import numexpr

LOGGER = logging.getLogger(__name__)
print("imports done")


# Set the numexpr.evaluate option to True
# pd.options.compute.use_numexpr = True

GENES = "./data/hg19_genes.tsv.gz"
NONE = -1


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


ENDING_RED = "_Red.idat"
ENDING_GRN = "_Grn.idat"

# TODO probes start at 1 pyranges at 0


class RawData:
    def __init__(self, basenames):
        # TODO if basenames is dir take all files in it
        # Clean up basenames
        _basenames = basenames if isinstance(basenames, list) else [basenames]
        _basenames = [
            Path(
                str(name).replace(ENDING_RED, "").replace(ENDING_GRN, "")
            ).expanduser()
            for name in _basenames
        ]
        # Remove duplicates, keep ordering
        _basenames = list(dict.fromkeys(_basenames))

        self.probes = [path.name for path in _basenames]

        grn_idat_files = [Path(str(name) + ENDING_GRN) for name in _basenames]
        red_idat_files = [Path(str(name) + ENDING_RED) for name in _basenames]

        # Check if all idat files exist
        not_found = next(
            (
                path
                for path in grn_idat_files + red_idat_files
                if not path.exists()
            ),
            None,
        )
        if not_found is not None:
            raise FileNotFoundError(
                f"The following file does not exist: {not_found}"
            )

        grn_idat = [IdatData(str(filepath)) for filepath in grn_idat_files]
        red_idat = [IdatData(str(filepath)) for filepath in red_idat_files]

        array_types = [
            ArrayType.from_probe_count(len(idat.illumina_ids))
            for idat in grn_idat + red_idat
        ]

        if len(set(array_types)) != 1:
            raise ValueError("Array types must all be the same.")

        self.array_type = array_types[0]

        self.ids = reduce(
            np.intersect1d, [idat.illumina_ids for idat in grn_idat]
        )

        self._grn = np.array(
            [
                idat.probe_means[np.isin(idat.illumina_ids, self.ids)]
                for idat in grn_idat
            ]
        )

        self._red = np.array(
            [
                idat.probe_means[np.isin(idat.illumina_ids, self.ids)]
                for idat in red_idat
            ]
        )
        self.manifest = ManifestLoader.get_manifest(self.array_type)

        self.grn = pd.DataFrame(
            self._grn.T, index=self.ids, columns=self.probes, dtype="int32"
        )
        self.red = pd.DataFrame(
            self._red.T, index=self.ids, columns=self.probes, dtype="int32"
        )
        self.methylated = None
        self.unmethylated = None

    def __repr__(self):
        title = f"RawData():"
        lines = [
            title + "\n" + "*" * len(title),
            f"array_type: {self.array_type}",
            f"manifest: {self.manifest.array_type}",
            f"probes:\n{self.probes}",
            f"ids:\n{self.ids}",
            f"_grn:\n{self._grn}",
            f"_red:\n{self._red}",
            f"grn:\n{self.grn}",
            f"red:\n{self.red}",
        ]
        return "\n\n".join(lines)


class MethylData:
    def __init__(self, raw_data, prep="illumina"):
        # TODO remove _grn and _red
        self._grn = raw_data._grn
        self._red = raw_data._red
        self.array_type = raw_data.array_type
        self.ids = raw_data.ids
        self.manifest = raw_data.manifest
        self.probes = raw_data.probes
        self.raw_data = raw_data
        if prep == "illumina":
            self.preprocess_illumina()
        else:
            self.preprocess_raw()

    def preprocess_illumina(self):
        self.illumina_control_normalization()
        self.illumina_bg_correction()
        self.preprocess_raw()

    def illumina_control_normalization(self, reference=0):
        at_controls = self.manifest.control_address(["NORM_A", "NORM_T"])
        cg_controls = self.manifest.control_address(["NORM_C", "NORM_G"])

        grn_average = np.mean(
            self._grn[:, np.isin(self.ids, cg_controls)], axis=1
        )
        red_average = np.mean(
            self._red[:, np.isin(self.ids, at_controls)], axis=1
        )

        ref = (grn_average + red_average)[reference] / 2
        grn_factor = ref / grn_average
        red_factor = ref / red_average

        self._grn = grn_factor[:, np.newaxis] * self._grn
        self._red = red_factor[:, np.newaxis] * self._red

        self.grn = pd.DataFrame(
            self._grn.T, index=self.ids, columns=self.probes, dtype="float32"
        )
        self.red = pd.DataFrame(
            self._red.T, index=self.ids, columns=self.probes, dtype="float32"
        )

    def illumina_bg_correction(self):
        neg_controls = self.manifest.control_address("NEGATIVE")

        grn_bg = np.sort(self._grn[:, np.isin(self.ids, neg_controls)])[:, 30]
        red_bg = np.sort(self._red[:, np.isin(self.ids, neg_controls)])[:, 30]

        self._grn = np.maximum(self._grn - grn_bg[:, np.newaxis], 0)
        self._red = np.maximum(self._red - red_bg[:, np.newaxis], 0)
        self.grn = pd.DataFrame(
            self._grn.T, index=self.ids, columns=self.probes, dtype="float32"
        )
        self.red = pd.DataFrame(
            self._red.T, index=self.ids, columns=self.probes, dtype="float32"
        )

    def preprocess_raw(self):
        type_i = self.manifest.probe_info(ProbeType("I"))
        type_ii = self.manifest.probe_info(ProbeType("II"))
        type_i_red = type_i[type_i.Color_Channel.values == Channel.RED.value]
        type_i_grn = type_i[type_i.Color_Channel.values == Channel.GREEN.value]
        locus_names = np.concatenate(
            [
                type_i.Name.index,
                type_ii.Name.index,
            ]
        )
        self.preprocess_raw_methylated(
            locus_names, type_i_grn, type_i_red, type_ii
        )
        self.preprocess_raw_unmethylated(
            locus_names, type_i_grn, type_i_red, type_ii
        )

    def preprocess_raw_methylated(self, locus_names, i_grn, i_red, ii):
        red = self.red
        grn = self.grn
        df = pd.DataFrame(
            np.nan,
            index=locus_names,
            columns=red.columns,
            dtype="float32",
        )
        df.loc[i_red.index] = red.loc[i_red["AddressB_ID"].values].values
        df.loc[i_grn.index] = grn.loc[i_grn["AddressB_ID"].values].values
        df.loc[ii.index] = grn.loc[ii["AddressA_ID"].values].values
        df["Name"] = self.manifest.data_frame.Name.values[locus_names]
        self.methylated = df.set_index("Name")

    def preprocess_raw_unmethylated(self, locus_names, i_grn, i_red, ii):
        red = self.red
        grn = self.grn
        df = pd.DataFrame(
            np.nan,
            index=locus_names,
            columns=red.columns,
            dtype="float32",
        )
        df.loc[i_red.index] = red.loc[i_red["AddressA_ID"].values].values
        df.loc[i_grn.index] = grn.loc[i_grn["AddressA_ID"].values].values
        df.loc[ii.index] = red.loc[ii["AddressA_ID"].values].values
        df["Name"] = self.manifest.data_frame.Name.values[locus_names]
        self.unmethylated = df.set_index("Name")

    @property
    def beta(self):
        return self.get_beta(self.methylated, self.unmethylated)

    @staticmethod
    def get_beta(
        methylated, unmethylated, offset=0, beta_threshold=0, min_zero=True
    ):
        assert offset >= 0, "offset must be non-negative"
        assert (
            0 <= beta_threshold <= 0.5
        ), "beta_threshold must be between 0 and 0.5"

        if min_zero:
            methylated = np.maximum(methylated, 0)
            unmethylated = np.maximum(unmethylated, 0)

        beta = methylated / (methylated + unmethylated + offset)

        if beta_threshold > 0:
            beta = np.minimum(
                np.maximum(beta, beta_threshold), 1 - beta_threshold
            )

        return beta

    def __repr__(self):
        title = f"RawData():"
        lines = [
            title + "\n" + "*" * len(title),
            f"array_type: {self.array_type}",
            f"manifest: {self.manifest.array_type}",
            f"probes:\n{self.probes}",
            f"_grn:\n{self._grn}",
            f"_red:\n{self._red}",
            f"grn:\n{self.grn}",
            f"red:\n{self.red}",
            f"methylated:\n{self.methylated}",
            f"unmethylated:\n{self.unmethylated}",
        ]
        if hasattr(self, "intensity"):
            lines.append(f"intensity:\n{self.intensity}")
        return "\n\n".join(lines)


class Annotation:
    def __init__(
        self,
        manifest,
        gap=None,
        genes=None,
        bin_size=50000,
        min_probes_per_bin=15,
        verbose=False,
    ):
        # TODO sort
        self.bin_size = bin_size
        self.min_probes_per_bin = min_probes_per_bin
        self.gap = gap
        self.genes = genes
        self.verbose = verbose
        self.chromsizes = pr.data.chromsizes()
        df = manifest.data_frame.copy()
        df = df[[x.startswith("cg") for x in df.Name.values]]
        df.rename(
            columns={"CHR": "Chromosome", "MAPINFO": "Start"}, inplace=True
        )
        df["End"] = df["Start"]
        df = df[
            df.Chromosome.isin(["X", "Y"] + [str(x) for x in range(1, 23)])
        ]
        df.Chromosome = "chr" + df.Chromosome
        self.manifest = pr.PyRanges(df)
        self.manifest = self.manifest.sort()
        self.probes = self.manifest.Name.values
        self.bins = self.make_bins()

    def make_bins(self):
        bins = pr.gf.tile_genome(self.chromsizes, int(5e4))
        bins = bins[bins.Chromosome != "chrM"]
        bins = bins.subtract(self.gap)
        bins = self.merge_bins(bins)
        return bins

    def merge_bins(self, bins):
        bins = bins.count_overlaps(
            self.manifest[["Name"]], overlap_col="N_probes"
        )
        bins = bins.apply(
            lambda df: self.merge_bins_in_chromosome(
                df, self.min_probes_per_bin, verbose=self.verbose
            ),
            as_pyranges=True,
        )
        return bins

    @staticmethod
    def merge_bins_in_chromosome(df, min_probes_per_bin, verbose=False):
        I_START = 0
        I_END = 1
        I_N_PROBES = 2
        INVALID = np.iinfo(np.int64).max

        matrix = df[["Start", "End", "N_probes"]].values.astype(np.int64)

        while np.any(matrix[:, I_N_PROBES] < min_probes_per_bin):
            i_min = np.argmin(matrix[:, I_N_PROBES])
            n_probes_left = INVALID
            n_probes_right = INVALID

            # Left
            if i_min > 0:
                delta_left = np.argmax(
                    matrix[i_min - 1 :: -1, I_N_PROBES] != INVALID
                )
                i_left = i_min - delta_left - 1
                if (
                    matrix[i_left, I_N_PROBES] != INVALID
                    and matrix[i_min, I_START] == matrix[i_left, I_END]
                ):
                    n_probes_left = matrix[i_left, I_N_PROBES]

            # Right
            if i_min < len(matrix) - 1:
                delta_right = np.argmax(
                    matrix[i_min + 1 :, I_N_PROBES] != INVALID
                )
                i_right = i_min + delta_right + 1
                if (
                    matrix[i_right, I_N_PROBES] != INVALID
                    and matrix[i_min, I_END] == matrix[i_right, I_START]
                ):
                    n_probes_right = matrix[i_right, I_N_PROBES]

            # Invalid
            if n_probes_left == INVALID and n_probes_right == INVALID:
                matrix[i_min, I_N_PROBES] = INVALID
                if verbose:
                    row = (
                        df.loc[i_min, ["Chromosome", "Start", "End"]]
                        .astype(str)
                        .tolist()
                    )
                    row_str = "-".join(row)
                    print(f"Could not merge {row_str}. Removed instead.")
                continue
            elif n_probes_right == INVALID or n_probes_left <= n_probes_right:
                i_merge = i_left
            else:
                i_merge = i_right
            matrix[i_merge, I_N_PROBES] += matrix[i_min, I_N_PROBES]
            matrix[i_merge, I_START] = min(
                matrix[i_merge, I_START], matrix[i_min, I_START]
            )
            matrix[i_merge, I_END] = max(
                matrix[i_merge, I_END], matrix[i_min, I_END]
            )
            matrix[i_min, I_N_PROBES] = INVALID
        df[["Start", "End", "N_probes"]] = matrix
        df = df[df.N_probes != INVALID]
        return df

    def __repr__(self):
        # TODO
        title = f"CNV():"
        lines = [
            title + "\n" + "*" * len(title),
            f"probes:\n{self.probes}",
            f"min_probes_per_bin:\n{self.min_probes_per_bin}",
            f"bin_size:\n{self.bin_size}",
            f"gap:\n{self.gap}",
            f"genes:\n{self.genes}",
            f"manifest:\n{self.manifest}",
            f"chromsizes:\n{self.chromsizes}",
            f"bins:\n{self.bins}",
        ]
        return "\n\n".join(lines)


class CNV:
    def __init__(self, sample, reference, annotation=None):
        self.sample = sample
        self.reference = reference
        self.annotation = annotation
        self.sample.intensity = self.get_itensity(self.sample)
        self.reference.intensity = self.get_itensity(self.reference)
        self.bins = annotation.bins

    def get_itensity(self, methyl_data):
        intensity = methyl_data.methylated + methyl_data.unmethylated
        prefix = "sample" if methyl_data == self.sample else "reference"
        if intensity.isna().any().any():
            print(f"{prefix}: Intensities that are NA, set to 1.")
            intensity.fillna(1, inplace=True)
        if (intensity < 1).any().any():
            print(f"{prefix}: Intensities smaller than 0 set to 1.")
            intensity[intensity < 1] = 1
        if intensity.mean(axis=0).min() < 5000:
            print(f"{prefix}: Intensities are abnormally low (< 5000).")
        if intensity.mean(axis=0).max() > 50000:
            print(f"{prefix}: Intensities are abnormally high (> 50000).")
        return intensity

    def fit(self):
        probes = self.annotation.probes
        smp_intensity = self.sample.intensity.loc[probes].values.ravel()
        idx = self.sample.intensity.loc[probes].index
        ref_intensity = self.reference.intensity.loc[
            probes,
        ].values
        correlation = np.array(
            [np.corrcoef(smp_intensity, z)[0, 1] for z in ref_intensity.T]
        )
        if any(correlation >= 0.99):
            LOGGER.warning(
                "Query sample also found in reference set. Excluded from fitting."
            )
        X = np.log2(ref_intensity)
        y = np.log2(smp_intensity)
        reg = LinearRegression().fit(X, y)
        y_pred = reg.predict(X)
        y_pred[y_pred < 0] = 0
        self.coef = reg.coef_
        self._ratio = y - y_pred
        self.ratio = pd.DataFrame({"ratio": self._ratio}, idx)
        self.noise = np.sqrt(
            np.mean((self._ratio[1:] - self._ratio[:-1]) ** 2)
        )
        self.probes = self.annotation.manifest[["Name"]]
        self.probes.ratio = self._ratio

    def bin(self):
        self.bins.bins_index = np.arange(len(self.bins.df))
        overlap = self.bins.join(self.annotation.manifest[["Name"]])
        overlap.ratio = self.ratio.loc[overlap.Name].ratio
        result = overlap.df.groupby("bins_index", dropna=False)["ratio"].agg(
            ["median", "var"]
        )
        df = self.bins.df
        df["Median"] = np.nan
        df["Var"] = np.nan
        df.loc[result.index, ["Median", "Var"]] = result.values
        self.bins = pr.PyRanges(df)

    def genes(self):
        overlap = self.annotation.genes.join(
            self.annotation.manifest[["Name"]]
        )
        overlap.ratio = self.ratio.loc[overlap.Name].ratio
        result = overlap.df.groupby("Gene", dropna=False)["ratio"].agg(
            ["median", "var", "count"]
        )
        df = self.annotation.genes.df.set_index("Gene")
        df["Median"] = np.nan
        df["Var"] = np.nan
        df["N_probes"] = 0
        df.loc[result.index, ["Median", "Var", "N_probes"]] = result.values
        df["N_probes"] = df["N_probes"].astype(int)
        df = df.reset_index()
        self.genes = pr.PyRanges(df)
        self.genes = self.genes.sort()

    @staticmethod
    def get_segments(df):
        bin_values = df["Median"].values
        chrom = df["Chromosome"].iloc[0]
        seg = segment(bin_values, shuffles=1000, p=0.001)
        seg_df = pd.DataFrame(
            [
                [chrom, df.Start.iloc[l.start], df.End.iloc[l.end - 1]]
                for l in seg
            ],
            # [[chrom, l.start, l.end] for l in seg],
            columns=["Chromosome", "Start", "End"],
        )
        return seg_df

    def segments(self):
        segments = self.bins.apply(self.get_segments)
        overlap = segments.join(self.annotation.manifest[["Name"]])
        overlap.ratio = self.ratio.loc[overlap.Name].ratio
        result = (
            overlap.df.groupby(["Chromosome", "Start", "End"], dropna=False)[
                "ratio"
            ]
            .agg(["median", "mean", "var", "count"])
            .reset_index()
            .rename(
                columns={
                    "count": "N_probes",
                    "median": "Median",
                    "mean": "Mean",
                    "var": "Var",
                }
            )
        )
        self.segments = pr.PyRanges(result)
        self.segments = self.segments.sort()

    def __repr__(self):
        # TODO
        title = f"CNV():"
        lines = [
            title + "\n" + "*" * len(title),
            f"sample:\n{self.sample.probes}",
            f"reference:\n{self.reference.probes}",
            f"annotation: {self.annotation}",
            f"_ratio: {self._ratio}",
            f"bins:\n{self.bins}",
            f"genes:\n{self.genes}",
            f"segments:\n{self.segments}",
            f"coef:\n{self.coef}",
            f"noise:\n{self.noise}",
            f"ratio:\n{self.ratio}",
        ]
        return "\n\n".join(lines)


file0 = "/data/epidip_IDAT/7970368088_R01C01_Grn.idat"
file1 = "/data/epidip_IDAT/7970368088_R01C02_Grn.idat"
file2 = "/data/ref_IDAT/cnvrefidat_450k/5775446049_R06C01_Red.idat"
file3 = "/data/ref_IDAT/cnvrefidat_450k/5775446051_R02C01"
file4 = "/data/epidip_IDAT/206171430049_R08C01"
file5 = "/data/epidip_IDAT/6042324058_R03C02"


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

gap = pr.PyRanges(pd.read_csv("./data/gap_450k.csv.gz"))
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


bins_df = cnv.bins.df[["Chromosome", "Start", "End", "Median"]]
bins_df.columns = ["chrom", "start", "end", "value"]
detail_df = cnv.genes.df[
    ["Chromosome", "Start", "End", "Gene", "Median", "N_probes"]
]
detail_df.columns = ["chrom", "start", "end", "name", "value", "nprobes"]
segments_df = cnv.segments.df[["Chromosome", "Start", "End", "Mean", "Median"]]
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
