"""Provides CNV analysis functionality including segmentation and plotting.

This module provides classes and functions for copy number variation (CNV)
analysis.
"""

import heapq
import io
import logging
import zipfile
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pyranges as pr

from mepylome.dtypes.arrays import ArrayType
from mepylome.dtypes.beads import MethylData, ReferenceMethylData
from mepylome.dtypes.cache import cache_key, memoize
from mepylome.dtypes.chromosome import Chromosome
from mepylome.dtypes.manifests import Manifest
from mepylome.dtypes.plots import CNVPlot
from mepylome.utils.files import ensure_directory_exists, get_resource_path
from mepylome.utils.varia import CONFIG, MEPYLOME_TMP_DIR

logger = logging.getLogger(__name__)

UNSET = object()
ZIP_ENDING = CONFIG["suffixes"]["cnv_zip"]
PACKAGE_DIR = get_resource_path("mepylome")

# Data copied from conumee
GAPS = PACKAGE_DIR / CONFIG["paths"]["gaps"]

# HG19 Gene data downloaded from:
# https://grch37.ensembl.org/biomart/martview
GENES = PACKAGE_DIR / CONFIG["paths"]["genes"]


def _get_cgsegment():
    try:
        import ruptures

        def function(bin_values):
            algo = ruptures.KernelCPD("linear").fit(bin_values)
            segments = algo.predict(pen=1)
            return [
                [start, end] for start, end in zip([0] + segments, segments)
            ]

        return function
    except Exception:
        pass
    try:
        import linear_segment

        def function(bin_values):
            segments = linear_segment.segment(
                bin_values, shuffles=1000, cutoff=0.3
            )
            return [[s.start, s.end] for s in segments]

        return function
    except Exception:
        pass
    try:
        import cbseg

        def function(bin_values):
            segments = cbseg.segment(bin_values, shuffles=1000, p=0.0001)
            return [[s.start, s.end] for s in segments]

        return function
    except Exception:
        logger.warning(
            "**Warning**: Segmentation won't be calculated due to missing "
            "'linear_segment', 'cbseg' or 'ruptures' package. See "
            "documentation"
        )
        return None


class Annotation:
    """Genomic annotations for CNV such as as binning and gene locations.

    Args:
        manifest (Manifest, optional): The manifest containing annotation
            details. Can be determined from array_type.
        array_type (str, optional): The type of array used for annotation.
            Can be determined from manifest.
        gap (pyranges.PyRanges): The genomic gaps. If unset default values
            will be used.
        detail (pyranges.PyRanges, optional): Detailed annotation (usually
            genes).
        bin_size (int, optional): The base-pair size of annotation bins.
            Defaults to 50000.
        min_probes_per_bin (int, optional): The minimum number of probes
            per bin. Defaults to 15.

    Attributes:
        manifest (Manifest): The manifest to use.
        array_type (str): The array type of the manifest.
        probes (list): The Illumina ID's of the manifest after adjusting the
            manifest to relevant genomic ranges.
        gap (pyranges.PyRanges): The genomic gaps except for the CNV analysis.
        detail (pyranges.PyRanges): Detailed annotation information (usually
            genes).
        bin_size (int): The base-pair size of the bins.
        min_probes_per_bin (int): The minimum number of probes per bin.
        chromsizes (dict): Dictionary containing chromosome sizes.
    """

    _cache = {}

    def __new__(
        cls,
        manifest=None,
        array_type=None,
        gap=UNSET,
        detail=UNSET,
        bin_size=50000,
        min_probes_per_bin=15,
    ):
        key = cache_key(
            manifest,
            array_type,
            gap,
            detail,
            bin_size,
            min_probes_per_bin,
        )
        if key in cls._cache:
            return cls._cache[key]

        instance = super().__new__(cls)

        # Cache the instance
        cls._cache[key] = instance
        return instance

    def __init__(
        self,
        manifest=None,
        array_type=None,
        gap=UNSET,
        detail=UNSET,
        bin_size=50000,
        min_probes_per_bin=15,
    ):
        # Don't need to initialize if instance is cached.
        if hasattr(self, "_cached"):
            return
        self._cached = True

        logger.info("Constructing annotation...")
        if manifest is None and array_type is None:
            msg = "'manifest' or 'array_type' must be given"
            raise ValueError(msg)
        self.bin_size = bin_size
        self.min_probes_per_bin = min_probes_per_bin

        self.gap = gap
        if self.gap is UNSET:
            self.gap = self.default_gaps()

        self.detail = detail
        if self.detail is UNSET:
            self.detail = self.default_genes()
        elif self.detail is not None:
            # PyRanges ranges start at 0
            self.detail.Start -= 1
            self.detail = self.detail.sort()

        self.chromsizes = pr.data.chromsizes()

        if array_type is None:
            self.array_type = manifest.array_type
        else:
            self.array_type = ArrayType(array_type)

        self.manifest = manifest
        if manifest is None:
            self.manifest = Manifest(self.array_type)

        manifest_df = self.manifest.data_frame.copy()
        manifest_df = manifest_df[
            [x.startswith("cg") for x in manifest_df.IlmnID.values]
        ]
        manifest_df = manifest_df[
            Chromosome.is_valid_chromosome(manifest_df.Chromosome)
        ]
        manifest_df.Chromosome = Chromosome.pd_to_string(
            manifest_df.Chromosome
        )
        # PyRanges ranges start at 0
        manifest_df.Start -= 1
        self.adjusted_manifest = pr.PyRanges(manifest_df)

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
        logger.info("Constructing annotation done")

    @staticmethod
    @lru_cache
    def default_gaps():
        """Default genomic gaps.

        Details:
            The default value of conumee2.
        """
        gap_df = pd.read_csv(GAPS)
        # PyRanges ranges start at 0
        gap_df.Start -= 1
        return pr.PyRanges(gap_df)

    @staticmethod
    @lru_cache
    def default_genes():
        """Default PyRanges object including gene names with coordinates.

        Details:
            Data downloaded from: https://grch37.ensembl.org/biomart/martview
        """
        genes_df = pd.read_csv(GENES, sep="\t")
        # PyRanges ranges start at 0
        genes_df.Start -= 1
        return pr.PyRanges(genes_df)

    def make_bins(self):
        """Creates equidistant bins and then removes genomic gaps."""
        bins = pr.gf.tile_genome(self.chromsizes, int(self.bin_size))
        bins = bins[bins.Chromosome != "chrM"]
        if self.gap is not None:
            bins = bins.subtract(self.gap)
        return self.merge_bins(bins)

    def merge_bins(self, bins):
        """Merges adjacent bins until all contain a minimum of probes."""
        bins = bins.count_overlaps(
            self.adjusted_manifest[["IlmnID"]], overlap_col="N_probes"
        )
        return bins.apply(
            lambda df: self.merge_bins_in_chromosome(
                df, self.min_probes_per_bin
            ),
            as_pyranges=True,
        )

    @staticmethod
    def merge_bins_in_chromosome(bin_df, min_probes_per_bin):
        """Merges adjacent bins until all contain a minimum of probes.

        Args:
            bin_df (DataFrame): DataFrame containing bin information for a
                single chromosome.
            min_probes_per_bin (int): Minimum number of probes per bin required
                for merging.

        Returns:
            DataFrame: Merged bins in the chromosome.
        """
        I_START, I_END, I_N_PROBES, I_LEFT, I_RIGHT = range(5)
        INVALID = np.iinfo(np.int64).max

        # Calculate Left and Right neighbors
        bin_df["Left"] = bin_df.index - 1
        bin_df["Right"] = bin_df.index + 1

        # Need to regularly extract minimum; use min-heap
        heap = [
            (x, y)
            for x, y in zip(bin_df.N_probes, bin_df.index)
            if x < min_probes_per_bin
        ]
        heapq.heapify(heap)

        matrix = bin_df[
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
                if logger.isEnabledFor(logging.DEBUG):
                    row = (
                        bin_df.loc[i_min, ["Chromosome", "Start", "End"]]
                        .astype(str)
                        .tolist()
                    )
                    row_str = "-".join(row)
                    logger.debug(
                        "Could not merge %s. Removed instead.", row_str
                    )
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

        bin_df[["Start", "End", "N_probes"]] = matrix[:, :3]
        bin_df = bin_df[bin_df.N_probes != INVALID]
        return bin_df[["Chromosome", "Start", "End", "N_probes"]]

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
    """Cached indices to improve speed of pandas loc/iloc operations.

    Return the cached indices of elements in left_array based on their presence
    in right_array.
    """
    return left_arr.get_indexer(right_arr)


def _pd_loc(pd_df, pd_col):
    """Cached version of pd_df.loc[pd_col] to speed up computation."""
    return pd_df.iloc[cached_index(pd_df.index, pd_col.values)]


class CNV:
    """Class for Copy Number Variation (CNV) analysis.

    Attributes:
        sample (MethylData): MethylData object representing the sample.
        reference (MethylData): MethylData object representing the CNV-
            neutral references.
        annotation (Annotation): Annotation object containing genomic
            annotation information.
        bins (PyRanges): PyRanges object representing genomic bins.
        probes (Index): Index of probe IDs.
        coef: Coefficient of linear regression.
        _ratio: Difference between observed sample intensity and expected
            intensity calculated by linear regression from references.
        ratio: The values from _ratio as DataFrame with Illumina ID's as
            indices.
        noise: Noise level. A quality measure for the sample bead.
        detail: Detailed information (usually Genes).
        segments: Segments calculated by circular binary segmentation.

    Args:
        sample (MethylData): MethylData object representing the sample.
        reference (MethylData or ReferenceMethylData): MethylData object
            representing the reference, or ReferenceMethylData object for
            multiple references.
        annotation (Annotation, optional): Annotation object containing
            genomic annotation information. Defaults to annotation
            associated with the sample array type.

    Examples:
        >>> sample = MethylData(file="path/to/idat/file")
        >>> reference = MethylData(file="path/to/idat/reference/dir")
        >>> cnv = CNV(sample, reference)
        >>> cnv.set_bins()
        >>> cnv.set_detail()
        >>> cnv.set_segments()
        >>> cnv.plot()

    Raises:
        ValueError: If sample does not contain exactly 1 probe, or if
            reference is not of type MethylData or ReferenceMethylData.

    Reference:
        Daenekas, B., Pérez, E., Boniolo, F., Stefan, S., Benfatto, S., Sill,
        M., Sturm, D., Jones, D. T. W., Capper, D., Zapatka, M., & Hovestadt,
        V. (2024). Conumee 2.0: enhanced copy-number variation analysis from
        DNA methylation arrays for humans and mice. In J. Kelso (Ed.),
        Bioinformatics (Vol. 40, Issue 2). Oxford University Press (OUP).
        https://doi.org/10.1093/bioinformatics/btae029
    """

    def __init__(self, sample, reference, annotation=None):
        if len(sample.probes) != 1:
            msg = "sample must contain exactly 1 probe."
            raise ValueError(msg)
        self.sample = sample
        self.probe = self.sample.probes[0]
        if isinstance(reference, MethylData):
            self.reference = reference
        elif isinstance(reference, ReferenceMethylData):
            self.reference = reference[self.sample.array_type]
        else:
            msg = (
                "'reference' must be of type 'MethylData' "
                "or 'ReferenceMethylData'"
            )
            raise TypeError(msg)
        self.annotation = annotation
        if annotation is None:
            self.annotation = Annotation(
                array_type=sample.array_type,
            )
        if not (
            self.sample.array_type
            == self.annotation.array_type
            == self.reference.array_type
        ):
            msg = (
                f"Array type mismatch: sample ({self.sample.array_type}), "
                f"reference ({self.reference.array_type}), "
                f"annotation ({self.annotation.array_type}).\n"
                "All must be the same."
            )
            raise ValueError(msg)
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

        self.fit()

    @classmethod
    def set_all(cls, sample, reference, annotation=None, *, do_seg=True):
        """Create a CNV object and perform CNV analysis.

        Args:
            sample (MethylData): MethylData object representing the sample.
            reference (MethylData or ReferenceMethylData): MethylData object
                representing the reference, or ReferenceMethylData object for
                multiple references.
            annotation (Annotation, optional): Annotation object containing
                genomic annotation information. Defaults to annotation
                associated with the sample array type.
            do_seg (bool, optional): Indicates whether to perform
                segmentation, which can be computationally intensive. Defaults
                to True.

        Returns:
            CNV: CNV object with fitted data and optionally segmented.

        Examples:
            >>> cnv = CNV.set_all(sample, reference, do_seg=do_seg)
            >>> # Note: This command is equivalent to:
            >>> cnv = CNV(sample, reference)
            >>> cnv.set_bins()
            >>> cnv.set_detail()
            >>> if do_seg:
            >>>     cnv.set_segments()

        """
        cnv = cls(
            sample=sample,
            reference=reference,
            annotation=annotation,
        )
        cnv.set_bins()
        cnv.set_detail()
        if do_seg:
            cnv.set_segments()
        return cnv

    def set_itensity(self, methyl_data):
        """Calculates intensity values from methylation data."""
        if hasattr(methyl_data, "intensity"):
            return
        logger.debug("%s Setting intensity...", self.probe)
        intensity = methyl_data.methyl + methyl_data.unmethyl
        prefix = (
            f"{self.probe}"
            if methyl_data == self.sample
            else f"{self.reference.probes[0]},..."
        )

        # Replace NaN values with 1
        nan_indices = np.isnan(intensity)
        if np.any(nan_indices):
            intensity[nan_indices] = 1
            logger.debug("%s: Intensities that are NA set to 1", prefix)

        # Replace values less than 1 with 1
        lt_one_indices = intensity < 1
        if np.any(lt_one_indices):
            intensity[lt_one_indices] = 1
            logger.debug("%s: Intensities < 0 set to 1", prefix)

        # Check abnormal low and high intensities
        mean_intensity = np.mean(intensity, axis=1)
        if np.min(mean_intensity) < 5000:
            logger.info("%s: Intensities are abnormally low (< 5000)", prefix)
        if np.max(mean_intensity) > 50000:
            logger.info(
                "%s: Intensities are abnormally high (> 50000)", prefix
            )
        methyl_data.intensity = pd.DataFrame(
            intensity.T,
            columns=methyl_data.probes,
            index=methyl_data.methyl_ilmnid,
        )

    def fit(self):
        """Fits linear regression model to calculate CNV at every CpG site.

        This method fits a linear regression model to the intensity data of the
        sample and reference and calculates the CNV at every CpG site.
        """
        logger.info("%s Performing fit...", self.probe)
        from sklearn.linear_model import LinearRegression

        smp_intensity = _pd_loc(
            self.sample.intensity, self.probes
        ).values.ravel()
        idx = _pd_loc(self.sample.intensity, self.probes).index
        ref_intensity = _pd_loc(self.reference.intensity, self.probes).values
        correlation = np.array(
            [np.corrcoef(smp_intensity, z)[0, 1] for z in ref_intensity.T]
        )
        if any(correlation >= 0.99):
            logger.info(
                "%s Sample found in reference set. Excluded from fitting.",
                self.probe,
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
        """Calculates CNV within each bin based on the results of 'fit'.

        This method calculates copy number variation (CNV) within each bin
        by taking the median of the ratios obtained from the linear regression
        model fit in the 'fit' method.
        """
        logger.info("%s Setting bins...", self.probe)
        cpg_bins = self.annotation._cpg_bins.copy()
        cpg_bins["ratio"] = _pd_loc(self.ratio, cpg_bins.IlmnID).ratio.values
        result = cpg_bins.groupby("bins_index", dropna=False)["ratio"].agg(
            ["median", "var"]
        )
        bins_df = self.bins.df
        bins_df["Median"] = np.nan
        bins_df["Var"] = np.nan
        bins_df.loc[result.index, ["Median", "Var"]] = result.values
        self.bins = pr.PyRanges(bins_df)

    def set_detail(self):
        """Calculates CNV for the detail object based on the results of 'fit'.

        This method calculates copy number variation (CNV) for the detail
        object (usually genes) by aggregating the ratios obtained from the
        linear regression model fit in the 'fit' method for each genomic region
        specified in the detail object. The result includes the median ratio,
        variance, and count of probes within each region.
        """
        logger.info("%s Setting detail...", self.probe)
        cpg_detail = self.annotation._cpg_detail.copy()
        cpg_detail["ratio"] = _pd_loc(
            self.ratio, cpg_detail.IlmnID
        ).ratio.values
        result = cpg_detail.groupby("Name", dropna=False)["ratio"].agg(
            ["median", "var", "count"]
        )
        detail_df = self.annotation.detail.df.set_index("Name")
        detail_df["Median"] = np.nan
        detail_df["Var"] = np.nan
        detail_df["N_probes"] = 0
        idx = cached_index(detail_df.index, result.index.values)
        detail_df.iloc[
            idx, detail_df.columns.get_indexer(["Median", "Var", "N_probes"])
        ] = result.values
        detail_df["N_probes"] = detail_df["N_probes"].astype(int)
        detail_df = detail_df.reset_index()
        self.detail = pr.PyRanges(detail_df)

    @staticmethod
    def _get_segments(df):
        """Performs circular binary segmentation to identify CNV segments.

        This method applies the circular binary segmentation (CBS) algorithm
        to identify copy number variation (CNV) segments based on the median
        values of genomic bins. The CBS algorithm is a time-intensive
        operation that segments genomic regions based on change-points in
        intensity values.

        Args:
            df (DataFrame): DataFrame containing the median values of
                genomic bins.

        Returns:
            DataFrame: DataFrame containing CNV segments with columns for
                chromosome, start position, and end position.

        """
        cbsegment = _get_cgsegment()
        bin_values = df["Median"].values
        chrom = df["Chromosome"].iloc[0]
        seg = cbsegment(bin_values)
        return pd.DataFrame(
            [[chrom, df.Start.iloc[s[0]], df.End.iloc[s[1] - 1]] for s in seg],
            columns=["Chromosome", "Start", "End"],
        )

    def set_segments(self):
        """Sets CNV segments based on circular binary segmentation.

        This method applies the circular binary segmentation (CBS) algorithm
        to identify copy number variation (CNV) segments in the dataset.
        It calculates the CNV segments for each chromosome and stores them
        in the 'segments' attribute of the object.
        """
        if _get_cgsegment() is None:
            return
        logger.info("%s Setting segments...", self.probe)
        segments = self.bins.apply(self._get_segments)
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
        """Writes CNV data to disk as a zip file.

        This method writes the CNV data to disk as a zip file containing
        CSV files. It allows specifying which data to include in the zip
        file, such as bins, detail, segments, and metadata.

        Args:
            path (str): The path to save the zip file.
            data (str or list of str, optional): Specifies which data to
                include in the zip file. Valid options are "all", "bins",
                "detail", "segments", and "metadata". Defaults to "all".

        Raises:
            ValueError: If an invalid data option is specified.
        """
        logger.info("%s Write data to disk...", self.probe)
        default = {"all", "bins", "detail", "segments", "metadata"}
        if not isinstance(data, list):
            data = [data]
        data = set(data)
        if "all" in data:
            data = default
        invalid = data - default
        if invalid:
            msg = (
                f"Invalid file(s) specified: {invalid}. "
                f"Valid options are: {default}"
            )
            raise ValueError(msg)
        dfs_to_write = []
        if "bins" in data and self.bins is not None:
            dfs_to_write.append(("bins.csv", self.bins.df))
        if "detail" in data and self.detail is not None:
            dfs_to_write.append(("detail.csv", self.detail.df))
        if "segments" in data and self.segments is not None:
            dfs_to_write.append(("segments.csv", self.segments.df))
        if "metadata" in data:
            metadata_df = pd.DataFrame(
                {
                    "Array_type": [str(self.annotation.array_type)],
                    "Noise": [self.noise],
                },
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
        with base_path.with_suffix(".zip").open("wb") as f:
            f.write(buffer.read())

    def plot(self):
        """Generates and displays a plot of the CNV data."""
        logger.info("%s Plotting...", self.probe)
        cnv_dir = Path(MEPYLOME_TMP_DIR, "cnv_zips")
        ensure_directory_exists(cnv_dir)
        cnv_file = self.probe + ZIP_ENDING
        cnv_path = Path(cnv_dir, cnv_file)
        if "Median" not in self.bins.columns:
            self.set_bins()
        if self.detail is None:
            self.set_detail()
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
