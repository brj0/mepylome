from functools import reduce
from pathlib import Path

import numpy as np
import pandas as pd

from mepylome import IdatParser
from mepylome.dtypes import ArrayType, Channel, ManifestLoader, ProbeType

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

        grn_idat = [IdatParser(str(filepath)) for filepath in grn_idat_files]
        red_idat = [IdatParser(str(filepath)) for filepath in red_idat_files]

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
