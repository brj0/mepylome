import heapq
import logging

import cbseg
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import pyranges as pr
from sklearn.linear_model import LinearRegression

from mepylome.utils import Timer, ensure_directory_exists
from mepylome.dtypes import Manifest, cache, MANIFEST_TMP_DIR, CNVPlot

cnv_timer = Timer()

LOGGER = logging.getLogger(__name__)
ZIP_ENDING = "_cnv.zip"


class Annotation:
    def __init__(
        self,
        manifest,
        gap=None,
        detail=None,
        bin_size=50000,
        min_probes_per_bin=15,
        verbose=False,
    ):
        # TODO sort
        self.bin_size = bin_size
        self.min_probes_per_bin = min_probes_per_bin
        self.gap = gap
        self.detail = detail
        self.detail = self.detail.sort()
        self.verbose = verbose
        self.chromsizes = pr.data.chromsizes()
        self.array_type = manifest.array_type
        df = manifest.data_frame.copy()
        df = df[[x.startswith("cg") for x in df.IlmnID.values]]
        df = df[
            df.Chromosome.isin(["X", "Y"] + [str(x) for x in range(1, 23)])
        ]
        df.Chromosome = "chr" + df.Chromosome.astype(str)
        self.manifest = pr.PyRanges(df)
        # self.manifest = self.manifest.sort()
        self.probes = self.manifest.IlmnID.values
        self.bins = self.make_bins()
        self.bins.bins_index = np.arange(len(self.bins.df))
        self._cpg_bins = (
            self.bins.join(self.manifest[["IlmnID"]])
            .df[["bins_index", "IlmnID"]]
            .set_index("bins_index")
        )
        self._cpg_detail = self.detail.join(self.manifest[["IlmnID"]]).df[
            ["Name", "IlmnID"]
        ]

    def make_bins(self):
        bins = pr.gf.tile_genome(self.chromsizes, int(5e4))
        bins = bins[bins.Chromosome != "chrM"]
        bins = bins.subtract(self.gap)
        bins = self.merge_bins(bins)
        return bins

    def merge_bins(self, bins):
        bins = bins.count_overlaps(
            self.manifest[["IlmnID"]], overlap_col="N_probes"
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

            update_neighbors(i_left, i_min, i_right)

        df[["Start", "End", "N_probes"]] = matrix[:, :3]
        df = df[df.N_probes != INVALID]
        return df[["Chromosome", "Start", "End", "N_probes"]]

    @staticmethod
    def _merge_bins_in_chromosome(df, min_probes_per_bin, verbose=False):
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
            f"detail:\n{self.detail}",
            f"array_type:\n{self.array_type}",
            f"manifest:\n{self.manifest}",
            f"chromsizes:\n{self.chromsizes}",
            f"bins:\n{self.bins}",
        ]
        return "\n\n".join(lines)


@cache
def indices(left_arr, right_arr):
    return left_arr.get_indexer(right_arr)


class CNV:
    def __init__(self, sample, reference, annotation=None):
        if len(sample.probes) != 1:
            raise ValueError("sample must contain exactly 1 probe.")
        self.sample = sample
        self.probe = self.sample.probes[0]
        self.reference = reference
        self.annotation = annotation
        self.set_itensity(self.sample)
        self.set_itensity(self.reference)
        self.bins = annotation.bins
        self.probes = self.annotation.manifest.IlmnID
        self.coef = None
        self._ratio = None
        self.ratio = None
        self.noise = None
        self.detail = None
        self.segments = None

    def set_itensity(self, methyl_data):
        if hasattr(methyl_data, "intensity"):
            return
        intensity = methyl_data.methyl + methyl_data.unmethyl
        prefix = "sample" if methyl_data == self.sample else "reference"

        # Replace NaN values with 1
        nan_indices = np.isnan(intensity)
        if np.any(nan_indices):
            print(f"{prefix}: Intensities that are NA, set to 1.")
            intensity[nan_indices] = 1

        # Replace values less than 1 with 1
        lt_one_indices = intensity < 1
        if np.any(lt_one_indices):
            print(f"{prefix}: Intensities smaller than 0 set to 1.")
            intensity[lt_one_indices] = 1

        # Check abnormal low and high intensities
        mean_intensity = np.mean(intensity, axis=1)
        if np.min(mean_intensity) < 5000:
            print(f"{prefix}: Intensities are abnormally low (< 5000).")
        if np.max(mean_intensity) > 50000:
            print(f"{prefix}: Intensities are abnormally high (> 50000).")
        methyl_data.intensity = pd.DataFrame(
            intensity.T,
            columns=methyl_data.probes,
            index=methyl_data.methyl_ilmnid,
        )

    def fit(self):
        probes = self.probes.values
        smp_prob_idx = indices(self.sample.intensity.index, probes)
        smp_intensity = self.sample.intensity.iloc[smp_prob_idx].values.ravel()
        idx = self.sample.intensity.iloc[smp_prob_idx].index
        ref_prob_idx = indices(self.reference.intensity.index, probes)
        ref_intensity = self.reference.intensity.iloc[ref_prob_idx,].values
        # smp_intensity = self.sample.intensity.loc[probes].values.ravel()
        # idx = self.sample.intensity.loc[probes].index
        # ref_intensity = self.reference.intensity.loc[
        # probes,
        # ].values
        correlation = np.array(
            [np.corrcoef(smp_intensity, z)[0, 1] for z in ref_intensity.T]
        )
        if any(correlation >= 0.99):
            # TODO exclude
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

    def set_bins(self):
        cpg_bins = self.annotation._cpg_bins.copy()
        idx = indices(self.ratio.index, cpg_bins.IlmnID.values)
        cpg_bins["ratio"] = self.ratio.iloc[idx].ratio.values
        # cpg_bins["ratio"] = self.ratio.loc[cpg_bins.IlmnID].ratio.values
        result = cpg_bins.groupby("bins_index", dropna=False)["ratio"].agg(
            ["median", "var"]
        )
        df = self.bins.df
        df["Median"] = np.nan
        df["Var"] = np.nan
        df.loc[result.index, ["Median", "Var"]] = result.values
        self.bins = pr.PyRanges(df)

    def _set_bins(self):
        self.bins.bins_index = np.arange(len(self.bins.df))
        overlap = self.bins.join(self.annotation.manifest[["IlmnID"]])
        overlap.ratio = self.ratio.loc[overlap.IlmnID].ratio
        result = overlap.df.groupby("bins_index", dropna=False)["ratio"].agg(
            ["median", "var"]
        )
        df = self.bins.df
        df["Median"] = np.nan
        df["Var"] = np.nan
        df.loc[result.index, ["Median", "Var"]] = result.values
        self.bins = pr.PyRanges(df)

    def set_detail(self):
        cpg_detail = self.annotation._cpg_detail.copy()
        idx = indices(self.ratio.index, cpg_detail.IlmnID.values)
        cpg_detail["ratio"] = self.ratio.iloc[idx].ratio.values
        result = cpg_detail.groupby("Name", dropna=False)["ratio"].agg(
            ["median", "var", "count"]
        )
        df = self.annotation.detail.df.set_index("Name")
        df["Median"] = np.nan
        df["Var"] = np.nan
        df["N_probes"] = 0
        idx = indices(df.index, result.index.values)
        df.iloc[
            idx, df.columns.get_indexer(["Median", "Var", "N_probes"])
        ] = result.values
        df["N_probes"] = df["N_probes"].astype(int)
        df = df.reset_index()
        self.detail = pr.PyRanges(df)

    def _set_detail(self):
        overlap = self.annotation.detail.join(
            self.annotation.manifest[["IlmnID"]]
        )
        overlap.ratio = self.ratio.loc[overlap.IlmnID].ratio
        result = overlap.df.groupby("Name", dropna=False)["ratio"].agg(
            ["median", "var", "count"]
        )
        df = self.annotation.detail.df.set_index("Name")
        df["Median"] = np.nan
        df["Var"] = np.nan
        df["N_probes"] = 0
        df.loc[result.index, ["Median", "Var", "N_probes"]] = result.values
        df["N_probes"] = df["N_probes"].astype(int)
        df = df.reset_index()
        self.detail = pr.PyRanges(df)
        self.detail = self.detail.sort()

    @staticmethod
    def get_segments(df):
        bin_values = df["Median"].values
        chrom = df["Chromosome"].iloc[0]
        seg = cbseg.segment(bin_values, shuffles=1000, p=0.001)
        seg_df = pd.DataFrame(
            [
                [chrom, df.Start.iloc[l.start], df.End.iloc[l.end - 1]]
                for l in seg
            ],
            columns=["Chromosome", "Start", "End"],
        )
        return seg_df

    def set_segments(self):
        segments = self.bins.apply(self.get_segments)
        overlap = segments.join(self.annotation.manifest[["IlmnID"]])
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

    def write(self, file_dir, file_name=None, files="all"):
        default = {"all", "bins", "detail", "segments", "metadata"}
        if not isinstance(files, list):
            files = [files]
        files = set(files)
        if "all" in files:
            files = default
        invalid = files - default
        if invalid:
            raise ValueError(
                f"Invalid file(s) specified: {invalid}. "
                f"Valid options are: {default}"
            )
        dfs_to_write = []
        if "bins" in files:
            dfs_to_write.append(("bins.csv", self.bins.df))
        if "detail" in files:
            dfs_to_write.append(("detail.csv", self.detail.df))
        if "segments" in files:
            dfs_to_write.append(("segments.csv", self.segments.df))
        if "metadata" in files:
            metadata_df = pd.DataFrame(
                {"Array_type": [str(self.annotation.array_type)]},
            )
            dfs_to_write.append(("metadata.csv", metadata_df))
        file_dir = Path(file_dir).expanduser()
        if file_name is None:
            file_name = Path(file_dir).joinpath(
                self.probe + ZIP_ENDING
            )
        base_path = Path(file_dir).joinpath(file_name)
        if base_path.suffix == ".zip":
            base_path = base_path.with_suffix("")
        with zipfile.ZipFile(base_path.with_suffix(".zip"), "w") as zf:
            for filename, df in dfs_to_write:
                csv_path = Path(str(base_path) + "_" + filename)
                df.to_csv(csv_path, index=False)
                zf.write(csv_path, arcname=csv_path.name)
                csv_path.unlink()

    def plot(self):
        cnv_dir = Path(MANIFEST_TMP_DIR, "cnv")
        ensure_directory_exists(cnv_dir)
        cnv_file = self.probe + ZIP_ENDING
        self.write(cnv_dir, cnv_file)
        CNVPlot(cnv_dir, cnv_file)


    def __repr__(self):
        # TODO
        title = f"CNV():"
        lines = [
            title + "\n" + "*" * len(title),
            f"sample:\n{self.sample.probes}",
            f"reference:\n{self.reference.probes}",
            # f"annotation: {self.annotation}",
            f"_ratio: {self._ratio}",
            f"bins:\n{self.bins}",
            f"detail:\n{self.detail}",
            f"segments:\n{self.segments}",
            f"coef:\n{self.coef}",
            f"noise:\n{self.noise}",
            f"ratio:\n{self.ratio}",
        ]
        return "\n\n".join(lines)
