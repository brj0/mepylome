from functools import reduce
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from mepylome import IdatParser
from mepylome.dtypes import ArrayType, Channel, ManifestLoader, ProbeType
from mepylome.utils import Timer

ENDING_RED = "_Red.idat"
ENDING_GRN = "_Grn.idat"

# TODO probes start at 1 pyranges at 0

LOGGER = logging.getLogger(__name__)

beads_timer = Timer()

def idat_basepaths(files):
    # If basenames is dir take all files in it
    if isinstance(files, list):
        _files = files
    elif Path(files).is_dir():
        _files = list(Path(files).iterdir())
    else:
        _files = [files]
    # Remove file endings
    _files = [
        Path(
            str(name).replace(ENDING_RED, "").replace(ENDING_GRN, "")
        ).expanduser()
        for name in _files
    ]
    # Remove duplicates, keep ordering
    _files = list(dict.fromkeys(_files))
    return _files

class RawData:
    def __init__(self, basenames):
        _basenames = idat_basepaths(basenames)

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

        grn_idat = [IdatParser(str(filepath)) for filepath in grn_idat_files]
        red_idat = [IdatParser(str(filepath)) for filepath in red_idat_files]

        array_types = [
            ArrayType.from_probe_count(len(idat.illumina_ids))
            for idat in grn_idat + red_idat
        ]

        if len(set(array_types)) != 1:
            raise ValueError("Array types must all be the same.")

        self.array_type = array_types[0]

        all_illumina_ids = [idat.illumina_ids for idat in grn_idat + red_idat]

        if all([np.array_equal(all_illumina_ids[0], arr) for arr in all_illumina_ids]):
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
                        np.isin(idat.illumina_ids, self.ids, assume_unique=True)
                    ]
                    for idat in grn_idat
                ]
            )
            self._red = np.array(
                [
                    idat.probe_means[
                        np.isin(idat.illumina_ids, self.ids, assume_unique=True)
                    ]
                    for idat in red_idat
                ]
            )
        self.manifest = ManifestLoader.get_manifest(self.array_type)

        self._grn_df = None
        self._red_df = None
        self.methylated = None
        self.unmethylated = None

    @property
    def grn(self):
        if self._grn_df is None:
            self._grn_df = pd.DataFrame(
                self._grn.T, index=self.ids, columns=self.probes, dtype="int32"
            )
        return self._grn_df

    @property
    def red(self):
        if self._red_df is None:
            self._red_df = pd.DataFrame(
                self._red.T, index=self.ids, columns=self.probes, dtype="int32"
            )
        return self._red_df

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


class Cache:
    cache = {}
    @classmethod
    def overlap_indices(self, left_arr, right_arr):
        key = Cache.fast_hash(left_arr, right_arr)
        # key = hash(left_arr.data.tobytes() + right_arr.data.tobytes())
        if key in Cache.cache:
            return Cache.cache[key]
        if not isinstance(left_arr, pd.Index):
            left_arr = pd.Index(left_arr)
        if not isinstance(right_arr, pd.Index):
            right_arr = pd.Index(right_arr)
        common_indices = left_arr.intersection(right_arr)
        left_index = left_arr.get_indexer(common_indices)
        right_index = right_arr.get_indexer(common_indices)
        Cache.cache[key] = left_index, right_index
        return left_index, right_index
    def fast_hash(left_arr, right_arr):
        N = len(left_arr)
        M = len(right_arr)
        L = 57
        idx_left = [x * N // L for x in range(L)] + [-1]
        idx_right = [x * M // L for x in range(L)] + [-1]
        key = (
            tuple(left_arr[idx_left]),
            tuple(right_arr[idx_right]),
            N,
            M,
        )
        return key


class MethylData:
    def __init__(self, data=None, file=None, prep="illumina"):
        if data is None and file is None:
            raise ValueError("'data' or 'file' must be given.")
        if data is None:
            data = RawData(file)
        # TODO remove _grn and _red
        self._grn = data._grn
        self._red = data._red
        self.array_type = data.array_type
        self.ids = data.ids
        self.manifest = data.manifest
        self.probes = data.probes
        self.data = data
        self._grn_df = None
        self._red_df = None
        if prep == "illumina":
            self.preprocess_illumina()
        else:
            self.preprocess_raw_cached()

    @property
    def grn(self):
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
        if self._red_df is None:
            self._red_df = pd.DataFrame(
                self._red.T,
                index=self.ids,
                columns=self.probes,
                dtype="float32",
            )
        return self._red_df

    def preprocess_illumina(self):
        self.illumina_control_normalization()
        self.illumina_bg_correction()
        self.preprocess_raw_cached()

    def illumina_control_normalization(self, reference=0):
        at_controls = self.manifest.control_address(["NORM_A", "NORM_T"])
        cg_controls = self.manifest.control_address(["NORM_C", "NORM_G"])

        grn_average = np.mean(
            self._grn[:, np.isin(self.ids, cg_controls, assume_unique=True)],
            axis=1,
        )
        red_average = np.mean(
            self._red[:, np.isin(self.ids, at_controls, assume_unique=True)],
            axis=1,
        )

        ref = (grn_average + red_average)[reference] / 2
        grn_factor = ref / grn_average
        red_factor = ref / red_average

        self._grn = grn_factor[:, np.newaxis] * self._grn
        self._red = red_factor[:, np.newaxis] * self._red

    def illumina_bg_correction(self):
        neg_controls = self.manifest.control_address("NEGATIVE")

        grn_bg = np.sort(
            self._grn[:, np.isin(self.ids, neg_controls, assume_unique=True)]
        )[:, 30]
        red_bg = np.sort(
            self._red[:, np.isin(self.ids, neg_controls, assume_unique=True)]
        )[:, 30]

        self._grn = np.maximum(self._grn - grn_bg[:, np.newaxis], 0)
        self._red = np.maximum(self._red - red_bg[:, np.newaxis], 0)

    def preprocess_raw(self):
        type_i = self.manifest.probe_info(ProbeType.ONE)
        type_ii = self.manifest.probe_info(ProbeType.TWO)
        type_i_red = type_i[type_i.Color_Channel.values == Channel.RED.value]
        type_i_grn = type_i[type_i.Color_Channel.values == Channel.GREEN.value]
        locus_idx = np.sort(
            np.concatenate(
                [
                    type_i.IlmnID.index,
                    type_ii.IlmnID.index,
                ]
            )
        )
        self.preprocess_raw_methylated(
            locus_idx, type_i_grn, type_i_red, type_ii
        )
        self.preprocess_raw_unmethylated(
            locus_idx, type_i_grn, type_i_red, type_ii
        )

    def preprocess_raw_methylated(self, locus_idx, i_grn, i_red, ii):
        red = self.red
        grn = self.grn
        df = pd.DataFrame(
            np.nan,
            index=locus_idx,
            columns=red.columns,
            dtype="float32",
        )
        df.loc[i_red.index] = red.loc[i_red["AddressB_ID"].values].values
        df.loc[i_grn.index] = grn.loc[i_grn["AddressB_ID"].values].values
        df.loc[ii.index] = grn.loc[ii["AddressA_ID"].values].values
        df["IlmnID"] = self.manifest.data_frame.IlmnID.values[locus_idx]
        self.methylated = df.set_index("IlmnID")

    def preprocess_raw_unmethylated(self, locus_idx, i_grn, i_red, ii):
        red = self.red
        grn = self.grn
        df = pd.DataFrame(
            np.nan,
            index=locus_idx,
            columns=red.columns,
            dtype="float32",
        )
        df.loc[i_red.index] = red.loc[i_red["AddressA_ID"].values].values
        df.loc[i_grn.index] = grn.loc[i_grn["AddressA_ID"].values].values
        df.loc[ii.index] = red.loc[ii["AddressA_ID"].values].values
        df["IlmnID"] = self.manifest.data_frame.IlmnID.values[locus_idx]
        self.unmethylated = df.set_index("IlmnID")

    def preprocess_raw_cached(self):
        key = (self.manifest.array_type, self.ids.tobytes())
        if key in Cache.cache:
            (
                lnm_idx,
                ids__i_red_a,
                ids__i_red_b,
                ids__i_grn_a,
                ids__i_grn_b,
                ids_ii_____a,
                lnm__i_red__,
                lnm__i_grn__,
                lnm_ii______,
            ) = Cache.cache[key]
        else:
            type_i = self.manifest.probe_info(ProbeType.ONE)
            type_ii = self.manifest.probe_info(ProbeType.TWO)
            type_i_red = type_i[
                type_i.Color_Channel.values == Channel.RED.value
            ]
            type_i_grn = type_i[
                type_i.Color_Channel.values == Channel.GREEN.value
            ]
            locus_idx = np.sort(
                np.concatenate(
                    [
                        type_i.IlmnID.index,
                        type_ii.IlmnID.index,
                    ]
                )
            )
            ids_idx = pd.Index(self.ids)
            lnm_idx = pd.Index(locus_idx)
            ids__i_red_a = ids_idx.get_indexer(type_i_red["AddressA_ID"])
            ids__i_red_b = ids_idx.get_indexer(type_i_red["AddressB_ID"])
            ids__i_grn_a = ids_idx.get_indexer(type_i_grn["AddressA_ID"])
            ids__i_grn_b = ids_idx.get_indexer(type_i_grn["AddressB_ID"])
            ids_ii_____a = ids_idx.get_indexer(type_ii["AddressA_ID"])
            lnm__i_red__ = lnm_idx.get_indexer(type_i_red.index)
            lnm__i_grn__ = lnm_idx.get_indexer(type_i_grn.index)
            lnm_ii______ = lnm_idx.get_indexer(type_ii.index)
            Cache.cache[key] = (
                lnm_idx,
                ids__i_red_a,
                ids__i_red_b,
                ids__i_grn_a,
                ids__i_grn_b,
                ids_ii_____a,
                lnm__i_red__,
                lnm__i_grn__,
                lnm_ii______,
            )
        self.methyl = np.full((len(self.probes), len(lnm_idx)), np.nan)
        self.methyl[:, lnm__i_red__] = self._red[:, ids__i_red_b]
        self.methyl[:, lnm__i_grn__] = self._grn[:, ids__i_grn_b]
        self.methyl[:, lnm_ii______] = self._grn[:, ids_ii_____a]
        self.unmethyl = np.full((len(self.probes), len(lnm_idx)), np.nan)
        self.unmethyl[:, lnm__i_red__] = self._red[:, ids__i_red_a]
        self.unmethyl[:, lnm__i_grn__] = self._grn[:, ids__i_grn_a]
        self.unmethyl[:, lnm_ii______] = self._red[:, ids_ii_____a]
        self.methyl_index = lnm_idx
        self.methyl_ilmnid = self.manifest.data_frame.IlmnID.values[lnm_idx]
        self.methylated = pd.DataFrame(
            self.methyl.T,
            index=self.methyl_ilmnid,
            columns=self.probes,
            dtype="float32",
        )
        self.methylated.index.name = "IlmnID"
        self.unmethylated = pd.DataFrame(
            self.unmethyl.T,
            index=self.methyl_ilmnid,
            columns=self.probes,
            dtype="float32",
        )
        self.unmethylated.index.name = "IlmnID"

    @property
    def beta(self):
        beta = self.get_beta(self.methyl, self.unmethyl)
        return pd.DataFrame(
            beta.T, columns=self.probes, index=self.methyl_ilmnid
        )

    def converted_beta(self, cpgs=None, fill=0.49):
        if cpgs is None:
            cpgs = self.manifest.methylation_probes
        beta = self.get_beta(self.methyl, self.unmethyl)
        converted = np.full((len(self.probes), len(cpgs)), fill)
        left_idx, right_idx = Cache.overlap_indices(cpgs, self.methyl_ilmnid)
        converted[:, left_idx] = beta[:, right_idx]
        converted[np.isnan(converted)] = fill
        return pd.DataFrame(converted.T, columns=self.probes, index=cpgs)

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
        # Ignore division by zero
        with np.errstate(divide="ignore", invalid="ignore"):
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
