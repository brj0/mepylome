"""Contains classes and function for processing Illumina methylation arrays.

It includes methods for extracting methylation information, various
preprocessing techniques, normalization, and data handling.
"""

import collections
import os
import pickle
from functools import reduce
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from mepylome.dtypes.arrays import ArrayType
from mepylome.dtypes.cache import cache_key, input_args_id, memoize
from mepylome.dtypes.idat import IdatParser
from mepylome.dtypes.manifests import Manifest
from mepylome.dtypes.probes import Channel, ProbeType
from mepylome.utils.varia import MEPYLOME_TMP_DIR, normexp_get_xs

ENDING_GRN = "_Grn.idat"
ENDING_RED = "_Red.idat"
ENDING_GZ = ".gz"
ENDING_SUFFIXES = ("_Grn.idat", "_Red.idat", "_Grn.idat.gz", "_Red.idat.gz")

NEUTRAL_BETA = 0.5


def is_valid_idat_basepath(basepath):
    """Checks if the given basepath(s) point to valid IDAT files."""
    if not isinstance(basepath, list):
        basepath = [basepath]
    basepath = [str(x) for x in basepath]
    return all(
        (
            os.path.exists(x + ENDING_GRN)
            or os.path.exists(x + ENDING_GRN + ENDING_GZ)
        )
        and (
            os.path.exists(x + ENDING_RED)
            or os.path.exists(x + ENDING_RED + ENDING_GZ)
        )
        for x in basepath
    )


def idat_basepaths(files, only_valid=False):
    """Returns unique basepaths from IDAT files or directory.

    This function processes a list of IDAT files or a directory containing IDAT
    files and returns their basepaths by removing the file endings. The
    function ensures that there are no duplicate basepaths in the returned list
    and maintains the order of the files as they appear in the input.

    Args:
        files (path or list): A file or directory path or a list of file paths.
        only_valid (bool): If True, only returns basepaths that point to valid
            IDAT file pairs. Defaults is 'False'.

    Returns:
        list: A list of unique basepaths corresponding to the provided IDAT
            files. If a directory is provided, all IDAT files are recursively
            considered.

    Example:
        >>> idat_basepaths("/path/to/dir")
        [PosixPath('/path/to/dir/file1'), PosixPath('/path/to/dir/file2')]

        >>> idat_basepaths(["/path1/file1_Grn.idat", "/path2/file2_Red.idat"])
        [PosixPath('/path1/file1'), PosixPath('/path2/file2')]

        >>> idat_basepaths("/path/to/idat/file_Grn.idat.gz")
        [PosixPath('/path/to/idat/file')]
    """

    def get_idat_files(file_or_dir):
        path = os.path.expanduser(file_or_dir)
        # If path is dir take all files in it
        if os.path.isdir(path):
            for dirpath, _, filenames in os.walk(path, followlinks=True):
                for filename in filenames:
                    if filename.endswith(ENDING_SUFFIXES):
                        yield os.path.join(dirpath, filename)
        else:
            yield path

    def strip_suffix(file_path):
        for suffix in ENDING_SUFFIXES:
            if file_path.endswith(suffix):
                return file_path[: -len(suffix)]
        return file_path

    if not isinstance(files, list):
        files = [files]
    _files = [
        strip_suffix(idat_file)
        for file_or_dir in files
        for idat_file in get_idat_files(file_or_dir)
    ]
    # Remove duplicates, keep ordering
    unique_basepaths_dict = dict.fromkeys(_files)
    if only_valid:
        return [
            Path(base)
            for base in unique_basepaths_dict
            if is_valid_idat_basepath(base)
        ]
    return [Path(base) for base in unique_basepaths_dict]


def idat_paths_from_basenames(basenames):
    """Returns paths to green and red IDAT files.

    Args:
        basenames (list): List of basepaths for IDAT files.

    Returns:
        tuple: Paths to green and red IDAT files.

    Raises:
        FileNotFoundError: If any IDAT file is not found.
    """
    grn_idat_files = np.array(
        [Path(str(name) + ENDING_GRN) for name in basenames]
    )
    red_idat_files = np.array(
        [Path(str(name) + ENDING_RED) for name in basenames]
    )

    def check_and_fix(files):
        not_existing = [i for i, path in enumerate(files) if not path.exists()]
        files[not_existing] = [
            x.parent / (x.name + ENDING_GZ) for x in files[not_existing]
        ]
        return next((x for x in files[not_existing] if not x.exists()), None)

    not_found = check_and_fix(grn_idat_files)
    not_found = (
        check_and_fix(red_idat_files) if not_found is None else not_found
    )
    if not_found is not None:
        idat_file = str(not_found).replace(ENDING_GZ, "")
        msg = f"IDAT file not found: {idat_file}."
        raise FileNotFoundError(msg)
    return grn_idat_files, red_idat_files


class RawData:
    """Represents raw intensity data extracted from IDAT files.

    This class initializes with a list of basepaths to IDAT files and parses
    them to extract raw intensity data from the green and red channels.

    Args:
        basenames (list): List of basepaths to IDAT files.
        manifest (Manifest, optional): The manifest associated with the array.
            If not provided, it will be determined from the probe count.

    Attributes:
        array_type (str): Type of Illumina array.
        probes (list): List of probe names corresponding to the IDAT files.
        ids (array): Array of probe IDs.
        _grn (array): Array of raw intensity values from the green channel.
        _red (array): Array of raw intensity values from the red channel.

    Example:
        >>> idat_basepath0 = directory_path / "200925700125_R07C01"
        >>> idat_basepath1 = directory_path / "200925700133_R02C01_Grn.idat"
        >>> raw_data = RawData(idat_basepath0)
        >>> raw_data = RawData([idat_basepath0, idat_basepath1])
    """

    def __init__(self, basenames, *, manifest=None):
        self.manifest = manifest
        _basenames = idat_basepaths(basenames)

        self.probes = [path.name.replace(ENDING_GZ, "") for path in _basenames]

        grn_idat_files, red_idat_files = idat_paths_from_basenames(_basenames)

        grn_idat = [
            IdatParser(str(filepath), intensity_only=True)
            for filepath in grn_idat_files
        ]
        red_idat = [
            IdatParser(str(filepath), intensity_only=True)
            for filepath in red_idat_files
        ]

        array_types = [
            ArrayType.from_probe_count(len(idat.illumina_ids))
            for idat in grn_idat + red_idat
        ]

        if len(set(array_types)) != 1:
            msg = "Array types must all be the same."
            raise ValueError(msg)

        self.array_type = array_types[0]

        all_illumina_ids = [idat.illumina_ids for idat in grn_idat + red_idat]

        if all(
            np.array_equal(all_illumina_ids[0], arr)
            for arr in all_illumina_ids
        ):
            self.ids = all_illumina_ids[0]
            self._grn = np.array([idat.probe_means for idat in grn_idat])
            self._red = np.array([idat.probe_means for idat in red_idat])
        else:
            self.ids = reduce(
                np.intersect1d, [idat.illumina_ids for idat in grn_idat]
            )
            self._grn = np.array(
                [
                    idat.probe_means[
                        np.isin(
                            idat.illumina_ids, self.ids, assume_unique=True
                        )
                    ]
                    for idat in grn_idat
                ]
            )
            self._red = np.array(
                [
                    idat.probe_means[
                        np.isin(
                            idat.illumina_ids, self.ids, assume_unique=True
                        )
                    ]
                    for idat in red_idat
                ]
            )
        if self.manifest is None:
            self.manifest = Manifest(self.array_type)

        self._grn_df = None
        self._red_df = None
        self.methylated = None
        self.unmethylated = None

    @property
    def grn(self):
        """DataFrame: Green channel raw intensity indexed by probe IDs."""
        if self._grn_df is None:
            self._grn_df = pd.DataFrame(
                self._grn.T, index=self.ids, columns=self.probes, dtype="int32"
            )
        return self._grn_df

    @property
    def red(self):
        """DataFrame: Red channel raw intensity indexed by probe IDs."""
        if self._red_df is None:
            self._red_df = pd.DataFrame(
                self._red.T, index=self.ids, columns=self.probes, dtype="int32"
            )
        return self._red_df

    def __repr__(self):
        title = "RawData():"
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


@memoize
def _overlap_indices(left_arr, right_arr):
    """Compute the indices of overlapping elements between two arrays.

    This function finds the common elements (indices) between two input arrays
    and returns their positions in both arrays. It uses pandas Index objects
    and memoization for performance improvement.

    Example:
        >>> left_arr = ['a', 'b', 'c', 'd']
        >>> right_arr = ['b', 'c', 'e']
        >>> left_idx, right_idx = _overlap_indices(left_arr, right_arr)
        >>> print(left_idx)
        [1 2]
        >>> print(right_idx)
        [0 1]
    """
    if not isinstance(left_arr, pd.Index):
        left_arr = pd.Index(left_arr)
    if not isinstance(right_arr, pd.Index):
        right_arr = pd.Index(right_arr)
    common_indices = left_arr.intersection(right_arr)
    left_index = left_arr.get_indexer(common_indices)
    right_index = right_arr.get_indexer(common_indices)
    return left_index, right_index


class MethylData:
    """Represents methylated and unmethylated intensity data from RawData.

    This class provides methods for preprocessing Illumina methylation data and
    computing beta values from methylated and unmethylated intensities.

    Args:
        data (RawData): RawData object containing raw intensity data.
        file (str): Path to file or dir or list of paths containing raw
            intensity data.
        prep (str): Preprocessing method. Options: "illumina", "swan",
            "noob".
        seed (int, optional): Seed value used for random number generation in
            the SWAN preprocessing method. Default is None.

    Note:
        If 'data' is not provided, it will attempt to create a RawData object
        using the specified 'file'.

    Raises:
        ValueError: If neither 'data' nor 'file' is provided.
        ValueError: If 'data' is provided but is not of type 'RawData'.

    Examples:
        >>> methyl_data = MethylData(raw_data)
        >>> methyl_data = MethylData(file=file_path, prep="swan")
    """

    def __init__(self, data=None, file=None, prep="illumina", seed=None):
        if data is None and file is None:
            msg = "'data' or 'file' must be given."
            raise ValueError(msg)
        if data is None:
            data = RawData(file)
        elif not isinstance(data, RawData):
            msg = "'data' is not of type 'RawData'."
            raise ValueError(msg)
        self.seed = seed
        self._grn = data._grn
        self._red = data._red
        self.array_type = data.array_type
        self.ids = data.ids
        self.manifest = data.manifest
        self.probes = data.probes
        self.data = data
        self._grn_df = None
        self._red_df = None
        self._methylated_df = None
        self._unmethylated_df = None
        if prep == "illumina":
            self.preprocess_illumina()
        elif prep == "swan":
            self.preprocess_swan()
        elif prep == "noob":
            self.preprocess_noob()
        elif prep == "raw":
            self.preprocess_raw()
        else:
            msg = f"invalid 'prep' value {prep}"
            raise ValueError(msg)

    @property
    def grn(self):
        """DataFrame: Normalized green intensity by probe ID."""
        if self._grn_df is None:
            self._grn_df = pd.DataFrame(
                self._grn.T,
                index=self.ids,
                columns=self.probes,
                dtype="float32",
            )
        return self._grn_df

    @property
    def red(self):
        """DataFrame: Normalized red intensity by probe ID."""
        if self._red_df is None:
            self._red_df = pd.DataFrame(
                self._red.T,
                index=self.ids,
                columns=self.probes,
                dtype="float32",
            )
        return self._red_df

    @property
    def methylated(self):
        """DataFrame: Methylated intensity values indexed by IlmnID."""
        if self._methylated_df is None:
            self._methylated_df = pd.DataFrame(
                self.methyl.T,
                index=self.methyl_ilmnid,
                columns=self.probes,
                dtype="float32",
            )
            self._methylated_df.index.name = "IlmnID"
        return self._methylated_df

    @property
    def unmethylated(self):
        """DataFrame: Unmethylated intensity values indexed by IlmnID."""
        if self._unmethylated_df is None:
            self._unmethylated_df = pd.DataFrame(
                self.unmethyl.T,
                index=self.methyl_ilmnid,
                columns=self.probes,
                dtype="float32",
            )
            self._unmethylated_df.index.name = "IlmnID"
        return self._unmethylated_df

    def preprocess_illumina(self):
        """Performs preprocessing usings Illuminas method.

        This function implements preprocessing for Illumina methylation
        microarrays as used in Genome Studio, the standard software provided by
        Illumina.

        Details:
            This implementation is adapted from 'minfi'.
        """
        self._methylated_df = None
        self._unmethylated_df = None
        ci = MethylData._cached_indices(self.manifest, self.ids, "illumina")
        self._illumina_control_normalization(ci=ci)
        self._illumina_bg_correction(ci)
        self._preprocess_raw(ci)

    def _illumina_control_normalization(self, ci, reference=0):
        """Performs normalization using control probes."""
        grn_average = np.mean(
            self._grn[:, ci["ids_cg_cont"]],
            axis=1,
        )
        red_average = np.mean(
            self._red[:, ci["ids_at_cont"]],
            axis=1,
        )

        ref = (grn_average + red_average)[reference] / 2
        grn_factor = ref / grn_average
        red_factor = ref / red_average

        self._grn = grn_factor[:, np.newaxis] * self._grn
        self._red = red_factor[:, np.newaxis] * self._red

    def _illumina_bg_correction(self, ci):
        """Performs background normalization using negative control probes."""
        if len(ci["ids_ng_cont"]) <= 30:
            return

        grn_bg = np.partition(self._grn[:, ci["ids_ng_cont"]], 30)[:, 30]
        red_bg = np.partition(self._red[:, ci["ids_ng_cont"]], 30)[:, 30]

        # Subtract and threshold at zero, using in-place operations
        np.subtract(self._grn, grn_bg[:, np.newaxis], out=self._grn)
        np.maximum(self._grn, 0, out=self._grn)

        # Subtract and threshold at zero, using in-place operations
        np.subtract(self._red, red_bg[:, np.newaxis], out=self._red)
        np.maximum(self._red, 0, out=self._red)

    def _preprocess_raw_uncached(self):
        """Calculates methylated/unmethylated arrays without preprocessing.

        Converts the Red/Green channel for an Illumina methylation array
        into methylation signal, without using any normalization.

        Note:
            Uncached, slower version of ``preprocess_raw``.
        """
        type_1 = self.manifest.probe_info(ProbeType.ONE)
        type_2 = self.manifest.probe_info(ProbeType.TWO)
        type_1_red = type_1[type_1.Color_Channel.values == Channel.RED.value]
        type_1_grn = type_1[type_1.Color_Channel.values == Channel.GRN.value]
        man_idx_np = np.sort(
            np.concatenate(
                [
                    type_1.IlmnID.index,
                    type_2.IlmnID.index,
                ]
            )
        )
        self._preprocess_raw_methylated(
            man_idx_np, type_1_grn, type_1_red, type_2
        )
        self._preprocess_raw_unmethylated(
            man_idx_np, type_1_grn, type_1_red, type_2
        )

    def _preprocess_raw_methylated(self, man_idx_np, t1_grn, t1_red, t2):
        """Calculates methylated data frame without preprocessing."""
        red = self.red
        grn = self.grn
        result = pd.DataFrame(
            np.nan,
            index=man_idx_np,
            columns=red.columns,
            dtype="float32",
        )
        result.loc[t1_red.index] = red.loc[t1_red["AddressB_ID"].values].values
        result.loc[t1_grn.index] = grn.loc[t1_grn["AddressB_ID"].values].values
        result.loc[t2.index] = grn.loc[t2["AddressA_ID"].values].values
        result["IlmnID"] = self.manifest.data_frame.IlmnID.values[man_idx_np]
        self._methylated_df = result.set_index("IlmnID")

    def _preprocess_raw_unmethylated(self, man_idx_np, t1_grn, t1_red, t2):
        """Calculates unmethylated data frame without preprocessing."""
        red = self.red
        grn = self.grn
        result = pd.DataFrame(
            np.nan,
            index=man_idx_np,
            columns=red.columns,
            dtype="float32",
        )
        result.loc[t1_red.index] = red.loc[t1_red["AddressA_ID"].values].values
        result.loc[t1_grn.index] = grn.loc[t1_grn["AddressA_ID"].values].values
        result.loc[t2.index] = red.loc[t2["AddressA_ID"].values].values
        result["IlmnID"] = self.manifest.data_frame.IlmnID.values[man_idx_np]
        self._unmethylated_df = result.set_index("IlmnID")

    @memoize
    def _cached_indices(manifest, illumina_ids, prep):
        """Cache the indices required for data processing.

        Args:
            manifest (Manifest): Manifest object.
            illumina_ids (array): Array of Illumina IDs.
            prep (str): Preprocessing method. Options: "illumina", "noob",
                "swan", "raw".

        Returns:
            dict: Cached indices including probe indices, Illumina IDs indices,
                and probe type indices.
        """
        type_1 = manifest.probe_info(ProbeType.ONE)
        type_2 = manifest.probe_info(ProbeType.TWO)
        type_1_red = type_1[type_1.Color_Channel.values == Channel.RED.value]
        type_1_grn = type_1[type_1.Color_Channel.values == Channel.GRN.value]
        idx = pd.Index(
            np.sort(
                np.concatenate(
                    [
                        type_1.IlmnID.index,
                        type_2.IlmnID.index,
                    ]
                )
            )
        )
        ilmnid = manifest.data_frame.IlmnID.values[idx.values]
        ids = pd.Index(illumina_ids)
        ci = {"idx": idx.values, "ilmnid": ilmnid}
        ci["ids_1_red_a"] = ids.get_indexer(type_1_red["AddressA_ID"])
        ci["ids_1_red_b"] = ids.get_indexer(type_1_red["AddressB_ID"])
        ci["ids_1_grn_a"] = ids.get_indexer(type_1_grn["AddressA_ID"])
        ci["ids_1_grn_b"] = ids.get_indexer(type_1_grn["AddressB_ID"])
        ci["ids_2_____a"] = ids.get_indexer(type_2["AddressA_ID"])
        ci["idx_1_red__"] = idx.get_indexer(type_1_red.index)
        ci["idx_1_grn__"] = idx.get_indexer(type_1_grn.index)
        ci["idx_2______"] = idx.get_indexer(type_2.index)

        if prep == "illumina":
            at_controls = manifest.control_address(["NORM_A", "NORM_T"])
            ng_controls = manifest.control_address("NEGATIVE")
            cg_controls = manifest.control_address(["NORM_C", "NORM_G"])

            def valid_ids(indices):
                return indices[indices != -1]

            ci["ids_at_cont"] = valid_ids(ids.get_indexer(at_controls))
            ci["ids_cg_cont"] = valid_ids(ids.get_indexer(cg_controls))
            ci["ids_ng_cont"] = valid_ids(ids.get_indexer(ng_controls))

        if prep == "swan":
            ng_controls = manifest.control_address("NEGATIVE")

            def valid_ids(indices):
                return indices[indices != -1]

            ci["ids_ng_cont"] = valid_ids(ids.get_indexer(ng_controls))

        if prep == "noob":
            control_probes = manifest.control_data_frame
            control_probes = control_probes[
                control_probes.Address_ID.isin(ids)
            ].reset_index(drop=True)
            ci["idx_cg"] = control_probes[
                control_probes.Control_Type.isin(["NORM_C", "NORM_G"])
            ].index.values
            ci["idx_at"] = control_probes[
                control_probes.Control_Type.isin(["NORM_A", "NORM_T"])
            ].index.values
            ci["ids_ctr"] = ids.get_indexer(control_probes["Address_ID"])

        return ci

    def preprocess_raw(self):
        """Calculates methylated/unmethylated arrays without preprocessing.

        Converts the Red/Green channel for an Illumina methylation array
        into methylation signal, without using any normalization.
        """
        ci = MethylData._cached_indices(self.manifest, self.ids, "raw")
        self._preprocess_raw(ci)

    def _preprocess_raw_old(self, ci):
        """Same as _preprocess_raw just slower and cleaner."""
        self.methyl = np.full((len(self.probes), len(ci["idx"])), np.nan)
        self.methyl[:, ci["idx_1_red__"]] = self._red[:, ci["ids_1_red_b"]]
        self.methyl[:, ci["idx_1_grn__"]] = self._grn[:, ci["ids_1_grn_b"]]
        self.methyl[:, ci["idx_2______"]] = self._grn[:, ci["ids_2_____a"]]
        self.unmethyl = np.full((len(self.probes), len(ci["idx"])), np.nan)
        self.unmethyl[:, ci["idx_1_red__"]] = self._red[:, ci["ids_1_red_a"]]
        self.unmethyl[:, ci["idx_1_grn__"]] = self._grn[:, ci["ids_1_grn_a"]]
        self.unmethyl[:, ci["idx_2______"]] = self._red[:, ci["ids_2_____a"]]
        self.methyl_index = ci["idx"]
        self.methyl_ilmnid = ci["ilmnid"]

    def _preprocess_raw(self, ci):
        """Internal preprocess logic."""
        methyl_shape = (len(self.probes), len(ci["idx"]))
        self.methyl = np.full(methyl_shape, np.nan)
        self.unmethyl = np.full(methyl_shape, np.nan)
        self.methyl[:, ci["idx_1_red__"]] = np.take(
            self._red, ci["ids_1_red_b"], axis=1
        )
        self.methyl[:, ci["idx_1_grn__"]] = np.take(
            self._grn, ci["ids_1_grn_b"], axis=1
        )
        self.methyl[:, ci["idx_2______"]] = np.take(
            self._grn, ci["ids_2_____a"], axis=1
        )
        self.unmethyl[:, ci["idx_1_red__"]] = np.take(
            self._red, ci["ids_1_red_a"], axis=1
        )
        self.unmethyl[:, ci["idx_1_grn__"]] = np.take(
            self._grn, ci["ids_1_grn_a"], axis=1
        )
        self.unmethyl[:, ci["idx_2______"]] = np.take(
            self._red, ci["ids_2_____a"], axis=1
        )
        self.methyl_index = ci["idx"]
        self.methyl_ilmnid = ci["ilmnid"]

    def _swan_bg_intensity(self, ci):
        """Intensity background normalization used for SWAN preprocessing."""
        grn_med = np.median(
            self._grn[:, ci["ids_ng_cont"]],
            axis=1,
        )
        red_med = np.median(
            self._red[:, ci["ids_ng_cont"]],
            axis=1,
        )
        return np.mean([grn_med, red_med], axis=0)

    @staticmethod
    def _swan_indices(manifest, methyl_index, seed=None):
        rng = np.random.default_rng(seed)
        all_ncpgs = (
            manifest.data_frame[["Probe_Type", "N_CpG"]]
            .loc[methyl_index]
            .reset_index(drop=True)
        )
        subset_sizes = all_ncpgs.groupby(
            ["Probe_Type", "N_CpG"], dropna=False
        ).size()
        subset_size = min(
            subset_sizes.get((probe_type, n_cpg), 0)
            for probe_type in [ProbeType.ONE, ProbeType.TWO]
            for n_cpg in [1, 2, 3]
        )
        all_indices = {}
        random_subset_indices = {}
        for probe_type in [ProbeType.ONE, ProbeType.TWO]:
            all_ncpts_type = all_ncpgs[all_ncpgs.Probe_Type == probe_type]
            all_indices[probe_type] = all_ncpts_type.index.values
            all_ncpts_type = all_ncpts_type.reset_index(drop=True)
            indices = []
            for ncpgs in range(1, 4):
                ids = all_ncpts_type.index[all_ncpts_type.N_CpG == ncpgs]
                ids_subset = rng.permutation(ids)[:subset_size]
                indices.append(ids_subset)
            random_subset_indices[probe_type] = np.sort(
                np.concatenate(indices)
            )
        return all_indices, random_subset_indices

    def preprocess_swan(self):
        """Subset-quantile Within Array Normalization (SWAN).

        Details:
            The SWAN method has two parts. First, an average quantile
            distribution is created using a subset of probes defined to be
            biologically similar based on the number of CpGs underlying the
            probe body. This is achieved by randomly selecting N Infinium I and
            II probes that have 1, 2 and 3 underlying CpGs, where N is the
            minimum number of probes in the 6 sets of Infinium I and II probes
            with 1, 2 or 3 probe body CpGs. This results in a pool of 3N
            Infinium I and 3N Infinium II probes. The subset for each probe
            type is then sorted by increasing intensity.  The value of each of
            the 3N pairs of observations is subsequently assigned to be the
            mean intensity of the two probe types for that row or “quantile”.
            This is the standard quantile procedure. The intensities of the
            remaining probes are then separately adjusted for each probe type
            using linear interpolation between the subset probes.


            Implementation adapted from 'minfi'

        Note:
            SWAN uses a random subset of probes for between array
            normalization. To achieve reproducible results, set the seed.

        References:
            J Maksimovic, L Gordon and A Oshlack (2012). SWAN: Subset quantile
            Within-Array Normalization for Illumina Infinium
            HumanMethylation450 BeadChips. Genome Biology 13, R44.
        """
        self._methylated_df = None
        self._unmethylated_df = None
        ci = MethylData._cached_indices(self.manifest, self.ids, "swan")
        self._preprocess_raw(ci)
        bg_intensity = self._swan_bg_intensity(ci)
        all_indices, random_subset_indices = MethylData._swan_indices(
            self.manifest, self.methyl_index, self.seed
        )
        self.methyl = MethylData._preprocess_swan_main(
            self.methyl, bg_intensity, all_indices, random_subset_indices
        )
        self.unmethyl = MethylData._preprocess_swan_main(
            self.unmethyl, bg_intensity, all_indices, random_subset_indices
        )

    @staticmethod
    def _preprocess_swan_main(
        intensity, bg_intensity, all_indices, random_subset_indices
    ):
        """Main function for preprocess_swan."""
        from scipy.stats import rankdata

        random_subset_one = all_indices[ProbeType.ONE][
            random_subset_indices[ProbeType.ONE]
        ]
        random_subset_two = all_indices[ProbeType.TWO][
            random_subset_indices[ProbeType.TWO]
        ]
        sorted_subset_intensity = (
            np.sort(intensity[:, random_subset_one], axis=1)
            + np.sort(intensity[:, random_subset_two], axis=1)
        ) / 2
        swan = np.full(intensity.shape, np.nan)
        for i in range(len(intensity)):
            for probe_type in [ProbeType.ONE, ProbeType.TWO]:
                curr_intensity = intensity[i, all_indices[probe_type]]
                x = rankdata(curr_intensity) / len(curr_intensity)
                xp = np.sort(x[random_subset_indices[probe_type]])
                fp = sorted_subset_intensity[i, :]
                interp = np.interp(x=x, xp=xp, fp=fp)
                intensity_min = np.min(
                    curr_intensity[random_subset_indices[probe_type]]
                )
                intensity_max = np.max(
                    curr_intensity[random_subset_indices[probe_type]]
                )
                interp[x > np.max(xp)] += (
                    curr_intensity[x > np.max(xp)] - intensity_max
                )
                interp[x < np.min(xp)] += (
                    curr_intensity[x < np.min(xp)] - intensity_min
                )
                interp[interp <= 0] = bg_intensity[i]
                swan[i, all_indices[probe_type]] = interp
        return swan

    def preprocess_noob(self, offset=15, dye_method="single"):
        """The Noob preprocessing method.

        Description:
            Noob (normal-exponential out-of-band) is a background correction
            method with dye-bias normalization.

        Args:
            offset (float): An offset for the normexp background correction.
            dye_method (str): How should dye bias correction be done: "single"
                for single sample approach, or "reference" for a reference
                array.

        References:
            TJ Triche, DJ Weisenberger, D Van Den Berg, PW Laird and KD
            Siegmund _Low-level processing of Illumina Infinium DNA
            Methylation BeadArrays.  Nucleic Acids Res (2013) 41, e90.
            doi:10.1093/nar/gkt090.
        """
        self._methylated_df = None
        self._unmethylated_df = None

        ci = MethylData._cached_indices(self.manifest, self.ids, "noob")

        self._preprocess_raw(ci)

        grn_oob = np.concatenate(
            [self._grn[:, ci["ids_1_red_a"]], self._grn[:, ci["ids_1_red_b"]]],
            axis=1,
        )
        red_oob = np.concatenate(
            [self._red[:, ci["ids_1_grn_a"]], self._red[:, ci["ids_1_grn_b"]]],
            axis=1,
        )

        methyl = self.methyl
        unmethyl = self.unmethyl
        methyl[methyl <= 0] = 1
        unmethyl[unmethyl <= 0] = 1

        grn_m = methyl[:, ci["idx_1_grn__"]]
        grn_u = unmethyl[:, ci["idx_1_grn__"]]
        grn_2 = methyl[:, ci["idx_2______"]]

        xf_grn = np.concatenate([grn_m, grn_u, grn_2], axis=1)
        xs_grn = normexp_get_xs(xf_grn, controls=grn_oob, offset=offset)

        cumsum = np.cumsum([0, grn_m.shape[1], grn_u.shape[1], grn_2.shape[1]])
        slice_grn_m = slice(cumsum[0], cumsum[1])
        slice_grn_u = slice(cumsum[1], cumsum[2])
        slice_grn_2 = slice(cumsum[2], cumsum[3])

        red_m = methyl[:, ci["idx_1_red__"]]
        red_u = unmethyl[:, ci["idx_1_red__"]]
        red_2 = unmethyl[:, ci["idx_2______"]]

        xf_red = np.concatenate([red_m, red_u, red_2], axis=1)
        xs_red = normexp_get_xs(xf_red, controls=red_oob, offset=offset)

        cumsum = np.cumsum([0, red_m.shape[1], red_u.shape[1], red_2.shape[1]])
        slice_red_m = slice(cumsum[0], cumsum[1])
        slice_red_u = slice(cumsum[1], cumsum[2])
        slice_red_2 = slice(cumsum[2], cumsum[3])

        methyl[:, ci["idx_1_grn__"]] = xs_grn["xs"][:, slice_grn_m]
        unmethyl[:, ci["idx_1_grn__"]] = xs_grn["xs"][:, slice_grn_u]

        methyl[:, ci["idx_1_red__"]] = xs_red["xs"][:, slice_red_m]
        unmethyl[:, ci["idx_1_red__"]] = xs_red["xs"][:, slice_red_u]

        methyl[:, ci["idx_2______"]] = xs_grn["xs"][:, slice_grn_2]
        unmethyl[:, ci["idx_2______"]] = xs_red["xs"][:, slice_red_2]

        # Dye correction

        grn_control = self._grn[:, ci["ids_ctr"]]
        red_control = self._red[:, ci["ids_ctr"]]

        xcs_grn = normexp_get_xs(
            grn_control, param=xs_grn["param"], offset=offset
        )
        xcs_red = normexp_get_xs(
            red_control, param=xs_red["param"], offset=offset
        )

        grn_avg = np.mean(xcs_grn["xs"][:, ci["idx_cg"]], axis=1)
        red_avg = np.mean(xcs_red["xs"][:, ci["idx_at"]], axis=1)

        red_grn_ratio = red_avg / grn_avg

        if dye_method == "single":
            red_factor = 1 / red_grn_ratio
            grn_factor = np.array([1, 1])
        elif dye_method == "reference":
            ref_idx = np.argmin(np.abs(red_grn_ratio - 1))
            ref = (grn_avg + red_avg)[ref_idx] / 2
            if np.isnan(ref):
                msg = "'ref_idx' refers to an array that is not present"
                raise ValueError(msg)
            grn_factor = ref / grn_avg
            red_factor = ref / red_avg
        else:
            msg = "dye_method must be 'single' or 'reference'"
            raise ValueError(msg)

        red_factor = red_factor.reshape(-1, 1)
        methyl[:, ci["idx_1_red__"]] *= red_factor
        unmethyl[:, ci["idx_1_red__"]] *= red_factor
        unmethyl[:, ci["idx_2______"]] *= red_factor

        if dye_method == "reference":
            grn_factor = grn_factor.reshape(-1, 1)
            methyl[:, ci["idx_1_grn__"]] *= grn_factor
            unmethyl[:, ci["idx_1_grn__"]] *= grn_factor
            methyl[:, ci["idx_2______"]] *= grn_factor

        self.methyl = methyl
        self.unmethyl = unmethyl

    @property
    def betas(self):
        """Returns beta values."""
        betas = self._get_beta(self.methyl, self.unmethyl)
        return pd.DataFrame(
            betas.T, columns=self.probes, index=self.methyl_ilmnid
        )

    def betas_at(self, cpgs=None, fill=NEUTRAL_BETA):
        """Calculates beta values for specified CpG sites.

        Args:
            cpgs (array-like): Array of CpG IDs.
            fill (float): Value to fill for CpGs not found in the used
                manifest or equal to NaN.

        Returns:
            pandas.DataFrame: DataFrame containing beta values for specified
                CpGs.

        Nore:
            If 'cpgs' is None, all CpGs from the used manifest are considered.
        """
        if cpgs is None:
            cpgs = self.manifest.methylation_probes
        betas = self._get_beta(self.methyl, self.unmethyl)
        converted = np.full((len(self.probes), len(cpgs)), fill)
        left_idx, right_idx = _overlap_indices(cpgs, self.methyl_ilmnid)
        converted[:, left_idx] = betas[:, right_idx]
        converted[np.isnan(converted)] = fill
        return pd.DataFrame(converted.T, columns=self.probes, index=cpgs)

    @staticmethod
    def _get_beta(
        methylated, unmethylated, offset=0, beta_threshold=0, *, min_zero=True
    ):
        if offset < 0:
            msg = "'offset' must be non-negative"
            raise ValueError(msg)

        if not (0 <= beta_threshold <= 0.5):
            msg = "'beta_threshold' must be between 0 and 0.5"
            raise ValueError(msg)

        if min_zero:
            methylated = np.maximum(methylated, 0)
            unmethylated = np.maximum(unmethylated, 0)

        # Ignore division by zero
        with np.errstate(divide="ignore", invalid="ignore"):
            betas = methylated / (methylated + unmethylated + offset)

        if beta_threshold > 0:
            betas = np.minimum(
                np.maximum(betas, beta_threshold), 1 - beta_threshold
            )

        return betas

    def __repr__(self):
        title = "MethylData():"
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


class ReferenceMethylData:
    """Stores and manages reference cases for different array types.

    This class categorizes and processes reference IDAT files to create
    MethylData objects for different array types. It is intended for CNV
    neutral reference cases used in CNV calculation.

    Args:
        file (list): List of file paths to IDAT files or directory containing
            IDAT files.
        prep (str): Preprocessing method. Options: "illumina", "swan", "noob".

    Attributes:
        _methyl_data (dict): Internal dictionary to cache MethylData objects
            for each array type.

    Raises:
        ValueError: If no reference files are found for the specified array
            type.

    Examples:
        >>> # 'directory' contains 450k, EPIC and EPICv2 idat files
        >>> reference = ReferenceMethylData(file=directory, prep="illumina")
        >>> sample_450k = MethylData(file=idat_file_450k)
        >>> sample_epic = MethylData(file=idat_file_epic)
        >>> sample_epicv2 = MethylData(file=idat_file_epicv2)
        >>> # reference can be used for all types
        >>> cnv_450k = CNV(sample_450k, reference)
        >>> cnv_epic = CNV(sample_epic, reference)
        >>> cnv_epicv2 = CNV(sample_epicv2, reference)
    """

    _cache = {}

    def __new__(cls, file, prep="illumina", save_to_disk=False):
        key = cache_key(file, prep)
        if key in cls._cache:
            return cls._cache[key]

        instance = super().__new__(cls)

        # Cache the instance
        cls._cache[key] = instance
        return instance

    def __getnewargs__(self):
        # Necessary for pickle
        return self.file, self.prep, self.save_to_disk

    def __init__(self, file, prep="illumina", save_to_disk=False):
        # Don't need to initialize if instance is cached.
        if hasattr(self, "_cached"):
            return
        self._cached = True

        self.file = file
        self.prep = prep
        self.save_to_disk = save_to_disk
        idat_files = idat_basepaths(self.file)

        # Load data from disk
        filepath = ReferenceMethylData.pickle_filename(self.prep, idat_files)
        if self.save_to_disk and filepath.exists():
            with filepath.open("rb") as f:
                saved_instance = pickle.load(f)
                self.__dict__.update(saved_instance.__dict__)
                return

        reference_files = collections.defaultdict(list)
        self._methyl_data = {}
        for idat_file in tqdm(
            idat_files, desc="Categorizing reference IDAT files"
        ):
            array_type = ArrayType.from_idat(idat_file)
            reference_files[array_type].append(idat_file)
        for array_type, file_list in tqdm(
            reference_files.items(), desc="Processing reference IDAT files"
        ):
            raw_data = RawData(file_list)
            self._methyl_data[array_type] = MethylData(
                raw_data, prep=self.prep
            )

        # Save to disk
        if self.save_to_disk:
            with filepath.open("wb") as f:
                pickle.dump(self, f)

    @staticmethod
    def pickle_filename(prep, idat_files):
        return MEPYLOME_TMP_DIR / input_args_id(
            "Ref", prep, sorted(str(x) for x in idat_files)
        )

    def __getitem__(self, array_type):
        if array_type not in self._methyl_data:
            msg = (
                f"No copy number neutral reference files found for "
                f"array type '{array_type.value}'."
            )
            raise ValueError(msg)
        return self._methyl_data[array_type]
