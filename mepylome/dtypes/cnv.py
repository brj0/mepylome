import hashlib
import heapq
import io
import logging
import zipfile
from functools import lru_cache
from pathlib import Path

import cbseg
import numpy as np
import pandas as pd
import pyranges as pr

from mepylome.dtypes.arrays import ArrayType
from mepylome.dtypes.beads import MethylData, ReferenceMethylData
from mepylome.dtypes.cache import memoize
from mepylome.dtypes.chromosome import Chromosome
from mepylome.dtypes.genetic_data import GAPS, GENES
from mepylome.dtypes.manifests import MANIFEST_TMP_DIR, Manifest
from mepylome.dtypes.plots import CNVPlot
from mepylome.utils.files import ensure_directory_exists

logger = logging.getLogger(__name__)


UNSET = object()
ZIP_ENDING = "_cnv.zip"


def pd_hash(df):
    return hashlib.sha1(pd.util.hash_pandas_object(df).values).hexdigest()


@memoize
class Annotation:
    def __init__(
        self,
        manifest=None,
        array_type=None,
        gap=UNSET,
        detail=UNSET,
        bin_size=50000,
        min_probes_per_bin=15,
        verbose=False,
    ):
        if manifest is None and array_type is None:
            raise ValueError("'manifest' or 'array_type' must be given")
        self.bin_size = bin_size
        self.min_probes_per_bin = min_probes_per_bin

        self.gap = gap
        if self.gap is UNSET:
            self.gap = self.default_gaps()

        self.detail = detail
        if self.detail is UNSET:
            self.detail = self.default_detail()
        elif self.detail is not None:
            # PyRanges ranges start at 0
            self.detail.Start -= 1
            self.detail = self.detail.sort()

        self.verbose = verbose
        self.chromsizes = pr.data.chromsizes()

        if isinstance(array_type, str):
            array_type = ArrayType(array_type)

        self.array_type = array_type
        if array_type is None:
            self.array_type = manifest.array_type

        self.manifest = manifest
        if manifest is None:
            self.manifest = Manifest(array_type)

        df = self.manifest.data_frame.copy()
        df = df[[x.startswith("cg") for x in df.IlmnID.values]]
        df = df[Chromosome.is_valid_chromosome(df.Chromosome)]
        df.Chromosome = Chromosome.pd_to_string(df.Chromosome)
        # PyRanges ranges start at 0
        df.Start -= 1
        self.adjusted_manifest = pr.PyRanges(df)

        self.probes = self.adjusted_manifest.IlmnID.values
        self.bins = self.make_bins()
        self.bins.bins_index = np.arange(len(self.bins.df))
        self._cpg_bins = (
            self.bins.join(self.adjusted_manifest[["IlmnID"]])
            .df[["bins_index", "IlmnID"]]
            .set_index("bins_index")
        )
        if self.detail is None:
            self._cpg_detail = pd.DataFrame(columns=["Name", "IlmnID"])
        else:
            self._cpg_detail = self.detail.join(
                self.adjusted_manifest[["IlmnID"]]
            ).df[["Name", "IlmnID"]]

    @classmethod
    def from_array_type(
        cls,
        array_type,
        bin_size=50000,
        min_probes_per_bin=15,
        verbose=False,
    ):
        annotation = cls(
            Manifest(array_type),
            gap=Annotation.default_gaps(),
            detail=Annotation.default_detail(),
            bin_size=bin_size,
            min_probes_per_bin=min_probes_per_bin,
            verbose=verbose,
        )
        return annotation

    @staticmethod
    @lru_cache
    def default_gaps():
        gap_df = pd.read_csv(GAPS)
        # PyRanges ranges start at 0
        gap_df.Start -= 1
        gap = pr.PyRanges(gap_df)
        return gap

    @staticmethod
    @lru_cache
    def default_detail():
        genes_df = pd.read_csv(GENES, sep="\t")
        # PyRanges ranges start at 0
        genes_df.Start -= 1
        genes = pr.PyRanges(genes_df)
        return genes

    def make_bins(self):
        bins = pr.gf.tile_genome(self.chromsizes, int(5e4))
        bins = bins[bins.Chromosome != "chrM"]
        if self.gap is not None:
            bins = bins.subtract(self.gap)
        bins = self.merge_bins(bins)
        return bins

    def merge_bins(self, bins):
        bins = bins.count_overlaps(
            self.adjusted_manifest[["IlmnID"]], overlap_col="N_probes"
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
        I_START, I_END, I_N_PROBES, I_LEFT, I_RIGHT = range(5)
        INVALID = np.iinfo(np.int64).max

        # Calculate Left and Right neighbors
        df["Left"] = df.index - 1
        df["Right"] = df.index + 1

        # Need to regularly extract minimum; use min-heap
        heap = [
            (x, y)
            for x, y in zip(df.N_probes, df.index)
            if x < min_probes_per_bin
        ]
        heapq.heapify(heap)

        matrix = df[
            ["Start", "End", "N_probes", "Left", "Right"]
        ].values.astype(np.int64)

        def update_neighbors(left, mid, right):
            matrix[mid, I_N_PROBES] = INVALID
            if left > 0:
                matrix[left, I_RIGHT] = matrix[mid, I_RIGHT]
            if right < matrix.shape[0]:
                matrix[right, I_LEFT] = matrix[mid, I_LEFT]

        while heap and heap[0][0] < min_probes_per_bin:
            n_probes, i_min = heap[0]
            real_n_probes = matrix[i_min, I_N_PROBES]

            # Check if n_probes changed due to merging and needs to be updated
            if n_probes != real_n_probes:
                heapq.heapreplace(heap, (real_n_probes, i_min))
                continue

            heapq.heappop(heap)
            n_probes_left = INVALID
            n_probes_right = INVALID

            # Left
            i_left = matrix[i_min, I_LEFT]
            if (
                i_left > 0
                and matrix[i_left, I_N_PROBES] != INVALID
                and matrix[i_min, I_START] == matrix[i_left, I_END]
            ):
                n_probes_left = matrix[i_left, I_N_PROBES]

            # Right
            i_right = matrix[i_min, I_RIGHT]
            if (
                i_right < matrix.shape[0]
                and matrix[i_right, I_N_PROBES] != INVALID
                and matrix[i_min, I_END] == matrix[i_right, I_START]
            ):
                n_probes_right = matrix[i_right, I_N_PROBES]

            # Invalid
            if n_probes_left == INVALID and n_probes_right == INVALID:
                update_neighbors(i_left, i_min, i_right)
                if verbose:
                    row = (
                        df.loc[i_min, ["Chromosome", "Start", "End"]]
                        .astype(str)
                        .tolist()
                    )
                    row_str = "-".join(row)
                    print(f"Could not merge {row_str}. Removed instead.")
                continue
            if n_probes_right == INVALID or n_probes_left <= n_probes_right:
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

            update_neighbors(i_left, i_min, i_right)

        df[["Start", "End", "N_probes"]] = matrix[:, :3]
        df = df[df.N_probes != INVALID]
        return df[["Chromosome", "Start", "End", "N_probes"]]

    def __repr__(self):
        title = "Annotation():"
        lines = [
            title + "\n" + "*" * len(title),
            f"probes:\n{self.probes}",
            f"min_probes_per_bin:\n{self.min_probes_per_bin}",
            f"bin_size:\n{self.bin_size}",
            f"gap:\n{self.gap}",
            f"detail:\n{self.detail}",
            f"array_type:\n{self.array_type}",
            f"adjusted_manifest:\n{self.adjusted_manifest}",
            f"chromsizes:\n{self.chromsizes}",
            f"bins:\n{self.bins}",
        ]
        return "\n\n".join(lines)


@memoize
def cached_index(left_arr, right_arr):
    """Cached index of elements in left_array based on their presence in
    right_array.
    """
    return left_arr.get_indexer(right_arr)


def pd_loc(pd_df, pd_col):
    """Cached version of pd_df.loc[pd_col]. Speeds up computation time
    considerably.
    """
    return pd_df.iloc[cached_index(pd_df.index, pd_col.values)]


class CNV:
    def __init__(self, sample, reference, annotation=None, verbose=False):
        if len(sample.probes) != 1:
            raise ValueError("sample must contain exactly 1 probe.")
        self.sample = sample
        self.probe = self.sample.probes[0]
        if isinstance(reference, MethylData):
            self.reference = reference
        elif isinstance(reference, ReferenceMethylData):
            self.reference = reference[self.sample.array_type]
        else:
            raise ValueError(
                "'reference' must be of type 'MethylData' "
                "or 'ReferenceMethylData'"
            )
        self.verbose = verbose
        self.annotation = annotation
        if annotation is None:
            self.annotation = Annotation(array_type=sample.array_type)
        self.set_itensity(self.sample)
        self.set_itensity(self.reference)
        self.bins = self.annotation.bins
        self.probes = self.annotation.adjusted_manifest.IlmnID
        self.coef = None
        self._ratio = None
        self.ratio = None
        self.noise = None
        self.detail = None
        self.segments = None

    @classmethod
    def set_all(cls, sample, reference, annotation=None):
        cnv = cls(sample, reference, annotation)
        cnv.fit()
        cnv.set_bins()
        cnv.set_detail()
        cnv.set_segments()
        return cnv

    def set_itensity(self, methyl_data):
        if hasattr(methyl_data, "intensity"):
            return
        intensity = methyl_data.methyl + methyl_data.unmethyl
        prefix = "sample" if methyl_data == self.sample else "reference"

        # Replace NaN values with 1
        nan_indices = np.isnan(intensity)
        if np.any(nan_indices):
            intensity[nan_indices] = 1
            if self.verbose:
                print(f"{prefix}: Intensities that are NA, set to 1.")

        # Replace values less than 1 with 1
        lt_one_indices = intensity < 1
        if np.any(lt_one_indices):
            intensity[lt_one_indices] = 1
            if self.verbose:
                print(f"{prefix}: Intensities smaller than 0 set to 1.")

        # Check abnormal low and high intensities
        mean_intensity = np.mean(intensity, axis=1)
        if np.min(mean_intensity) < 5000 and self.verbose:
            print(f"{prefix}: Intensities are abnormally low (< 5000).")
        if np.max(mean_intensity) > 50000 and self.verbose:
            print(f"{prefix}: Intensities are abnormally high (> 50000).")
        methyl_data.intensity = pd.DataFrame(
            intensity.T,
            columns=methyl_data.probes,
            index=methyl_data.methyl_ilmnid,
        )

    def fit(self):
        from sklearn.linear_model import LinearRegression
        smp_intensity = pd_loc(
            self.sample.intensity, self.probes
        ).values.ravel()
        idx = pd_loc(self.sample.intensity, self.probes).index
        ref_intensity = pd_loc(self.reference.intensity, self.probes).values
        correlation = np.array(
            [np.corrcoef(smp_intensity, z)[0, 1] for z in ref_intensity.T]
        )
        if any(correlation >= 0.99):
            logger.warning(
                "Query sample found in reference set. Excluded from fitting."
            )
            ref_intensity = ref_intensity[:, correlation < 0.99]
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

    def set_bins(self):
        cpg_bins = self.annotation._cpg_bins.copy()
        cpg_bins["ratio"] = pd_loc(self.ratio, cpg_bins.IlmnID).ratio.values
        result = cpg_bins.groupby("bins_index", dropna=False)["ratio"].agg(
            ["median", "var"]
        )
        df = self.bins.df
        df["Median"] = np.nan
        df["Var"] = np.nan
        df.loc[result.index, ["Median", "Var"]] = result.values
        self.bins = pr.PyRanges(df)

    def set_detail(self):
        cpg_detail = self.annotation._cpg_detail.copy()
        cpg_detail["ratio"] = pd_loc(
            self.ratio, cpg_detail.IlmnID
        ).ratio.values
        result = cpg_detail.groupby("Name", dropna=False)["ratio"].agg(
            ["median", "var", "count"]
        )
        df = self.annotation.detail.df.set_index("Name")
        df["Median"] = np.nan
        df["Var"] = np.nan
        df["N_probes"] = 0
        idx = cached_index(df.index, result.index.values)
        df.iloc[idx, df.columns.get_indexer(["Median", "Var", "N_probes"])] = (
            result.values
        )
        df["N_probes"] = df["N_probes"].astype(int)
        df = df.reset_index()
        self.detail = pr.PyRanges(df)

    @staticmethod
    def get_segments(df):
        bin_values = df["Median"].values
        chrom = df["Chromosome"].iloc[0]
        seg = cbseg.segment(bin_values, shuffles=1000, p=0.0001)
        seg_df = pd.DataFrame(
            [
                [chrom, df.Start.iloc[s.start], df.End.iloc[s.end - 1]]
                for s in seg
            ],
            columns=["Chromosome", "Start", "End"],
        )
        return seg_df

    def set_segments(self):
        segments = self.bins.apply(self.get_segments)
        overlap = segments.join(self.annotation.adjusted_manifest[["IlmnID"]])
        overlap.ratio = self.ratio.loc[overlap.IlmnID].ratio
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

    def write(self, path, data="all"):
        default = {"all", "bins", "detail", "segments", "metadata"}
        if not isinstance(data, list):
            data = [data]
        data = set(data)
        if "all" in data:
            data = default
        invalid = data - default
        if invalid:
            raise ValueError(
                f"Invalid file(s) specified: {invalid}. "
                f"Valid options are: {default}"
            )
        dfs_to_write = []
        if "bins" in data:
            dfs_to_write.append(("bins.csv", self.bins.df))
        if "detail" in data:
            dfs_to_write.append(("detail.csv", self.detail.df))
        if "segments" in data:
            dfs_to_write.append(("segments.csv", self.segments.df))
        if "metadata" in data:
            metadata_df = pd.DataFrame(
                {"Array_type": [str(self.annotation.array_type)]},
            )
            dfs_to_write.append(("metadata.csv", metadata_df))
        base_path = Path(path).expanduser()
        if base_path.suffix == ".zip":
            base_path = base_path.with_suffix("")
        file_name = base_path.name
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as zf:
            for df_name, df in dfs_to_write:
                csv_bytes = df.to_csv(index=False).encode("utf-8")
                zf.writestr(f"{file_name}_{df_name}", csv_bytes)
        buffer.seek(0)
        with open(base_path.with_suffix(".zip"), "wb") as f:
            f.write(buffer.read())

    def plot(self):
        cnv_dir = Path(MANIFEST_TMP_DIR, "cnv")
        ensure_directory_exists(cnv_dir)
        cnv_file = self.probe + ZIP_ENDING
        cnv_path = Path(cnv_dir, cnv_file)
        self.write(cnv_path)
        cnv_plot = CNVPlot(cnv_dir, cnv_file)
        cnv_plot.run_app()

    def __repr__(self):
        title = "CNV():"
        lines = [
            title + "\n" + "*" * len(title),
            f"sample:\n{self.sample.probes}",
            f"reference:\n{self.reference.probes}",
            f"_ratio: {self._ratio}",
            f"bins:\n{self.bins}",
            f"detail:\n{self.detail}",
            f"segments:\n{self.segments}",
            f"coef:\n{self.coef}",
            f"noise:\n{self.noise}",
            f"ratio:\n{self.ratio}",
        ]
        return "\n\n".join(lines)
