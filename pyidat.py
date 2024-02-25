# from methylprep.files.idat import IdatDataset
# from methylprep.models.probes import Channel
# from methylprep.utils.parsing import *
# import methylprep

import logging
import os
import re
import time
import pyranges as pr
from pathlib import Path
from urllib.parse import urljoin

import numpy as np
import pandas as pd

from pyllumina import IdatData

# TODO too long for import
from pyllumina.dtypes import (
    ArrayType,
    Channel,
    Manifest,
    ManifestLoader,
    ProbeType,
)
from pyllumina.utils import (
    download_file,
    ensure_directory_exists,
    get_file_from_archive,
    get_file_object,
    reset_file,
)

print("imports done")

NONE = -1

from functools import reduce


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



class RawData:
    def __init__(self, basenames):
        # Clean up basenames
        _basenames = basenames if isinstance(basenames, list) else [basenames]
        _basenames = [
            Path(
                str(name).replace(ENDING_RED, "").replace(ENDING_GRN, "")
            ).expanduser()
            for name in _basenames
        ]
        # Remove duplicates keep ordering
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
            NONE,
            index=locus_names,
            columns=red.columns,
            dtype="float32",
        )
        df.loc[i_red.index] = red.loc[i_red["AddressB_ID"].values].values
        df.loc[i_grn.index] = grn.loc[i_grn["AddressB_ID"].values].values
        df.loc[ii.index] = grn.loc[ii["AddressA_ID"].values].values
        df["Name"] = self.manifest.data_frame.Name.values[locus_names]
        self.methylated = df

    def preprocess_raw_unmethylated(self, locus_names, i_grn, i_red, ii):
        red = self.red
        grn = self.grn
        df = pd.DataFrame(
            NONE,
            index=locus_names,
            columns=red.columns,
            dtype="float32",
        )
        df.loc[i_red.index] = red.loc[i_red["AddressA_ID"].values].values
        df.loc[i_grn.index] = grn.loc[i_grn["AddressA_ID"].values].values
        df.loc[ii.index] = red.loc[ii["AddressA_ID"].values].values
        df["Name"] = self.manifest.data_frame.Name.values[locus_names]
        self.unmethylated = df

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
        return "\n\n".join(lines)


file0 = "/data/epidip_IDAT/7970368088_R01C01_Grn.idat"
file1 = "/data/epidip_IDAT/7970368088_R01C02_Grn.idat"
file2 = "/data/ref_IDAT/450k/5775446049_R06C01_Red.idat"
file3 = "/data/ref_IDAT/450k/5775446051_R02C01"


timer.start()
raw_data = RawData([file0, file1])
timer.stop("loading rgset")

timer.start()
m_data = MethylData(raw_data)
timer.stop("preproc")

manifest = ManifestLoader.get_manifest("450k")
# manifest = ManifestLoader.get_manifest("epic")
# manifest = ManifestLoader.get_manifest("epicv2")


timer.start()
sample_r_data = RawData(file3)
timer.stop("loading rgset")

timer.start()
sample_m_data = MethylData(sample_r_data)
timer.stop("preproc")
self = sample_m_data

class Annotation:
    def __init__(self, manifest):
        self.probes = manifest.data_frame.Name

anno = Annotation(manifest)

df = manifest.data_frame
df.columns = [
    "IlmnID",
    "Name",
    "AddressA_ID",
    "AddressB_ID",
    "Infinium_Design_Type",
    "Color_Channel",
    "Chromosome",
    "Start",
    "probe_type",
]
df["End"] = df["Start"]
df = df[df.Chromosome.isin(["X", "Y"] + [str(x) for x in range(1, 23)])]
df = df[df["Name"].str.startswith("cg")]

pran = pr.PyRanges(df)
chromsizes = pr.data.chromsizes()
bins = pr.gf.tile_genome(chromsizes, int(5e4))

probes = df.Name.unique()


# #' @rdname CNV.fit
# setMethod("CNV.fit", signature(query = "CNV.data", ref = "CNV.data", anno = "CNV.anno"),
# function(query, ref, anno, intercept = TRUE) {
# if (ncol(query@intensity) == 0)
# stop("query intensities unavailable, run CNV.load")
# if (ncol(ref@intensity) == 0)
# stop("reference set intensities unavailable, run CNV.load")

# if (ncol(query@intensity) != 1)
# message("using multiple query samples")
# if (ncol(ref@intensity) == 1)
# warning("reference set contains only a single sample. use more samples for better results.")

# p <- unique(names(anno@probes))  # ordered by location
# if (!all(is.element(p, rownames(query@intensity))))
# stop("query intensities not given for all probes.")
# if (!all(is.element(p, rownames(ref@intensity))))
# stop("reference set intensities not given for all probes.")

# object <- new("CNV.analysis")
# object@date <- date()
# object@fit$args <- list(intercept = intercept)

# object@anno <- anno

# object@fit$coef <- data.frame(matrix(ncol = 0, nrow = ncol(ref@intensity)))
# object@fit$ratio <- data.frame(matrix(ncol = 0, nrow = length(p)))
# for (i in 1:ncol(query@intensity)) {

# message(paste(colnames(query@intensity)[i]), " (",round(i/ncol(query@intensity)*100, digits = 3), "%", ")", sep = "")
# r <- cor(query@intensity[p, ], ref@intensity[p, ])[i, ] < 0.99
# if (any(!r)) message("query sample seems to also be in the reference set. not used for fit.")
# if (intercept) {
# ref.fit <- lm(y ~ ., data = data.frame(y = log2(query@intensity[p,i]), X = log2(ref@intensity[p, r])))
# } else {
# ref.fit <- lm(y ~ . - 1, data = data.frame(y = log2(query@intensity[p,i]), X = log2(ref@intensity[p, r])))
# }
# object@fit$coef <- cbind(object@fit$coef,as.numeric(ref.fit$coefficients[-1]))

# ref.predict <- predict(ref.fit)
# ref.predict[ref.predict < 0] <- 0

# object@fit$ratio <- cbind(object@fit$ratio, log2(query@intensity[p,i]) - ref.predict[p])
# }


# colnames(object@fit$coef) <- colnames(query@intensity)
# rownames(object@fit$coef) <- colnames(ref@intensity)
# colnames(object@fit$ratio) <- colnames(query@intensity)
# rownames(object@fit$ratio) <- p

# object@fit$noise <- as.numeric()
# for (i in 1:ncol(query@intensity)) {
# object@fit$noise <- c(object@fit$noise, sqrt(mean((object@fit$ratio[-1,i] - object@fit$ratio[-nrow(object@fit$ratio),i])^2,na.rm = TRUE)))
# }

# names(object@fit$noise) <- colnames(query@intensity)
# return(object)
# })
