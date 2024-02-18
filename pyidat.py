from enum import IntEnum, unique
from functools import reduce
# from methylprep.files.idat import IdatDataset
# from methylprep.models.probes import Channel
# from methylprep.utils.parsing import *
# import methylprep
from pathlib import Path
from pyllumina import IdatData
from pyllumina.dtypes import ArrayType, Channel, ProbeType, Manifest, ManifestLoader
from urllib.parse import urljoin
import logging
import numpy as np
import os
import pandas as pd
import pickle
import re
import time

from pyllumina.utils import (
    download_file,
    get_file_object,
    get_file_from_archive,
    reset_file,
    ensure_directory_exists,
)


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


class RGSet:
    def __init__(
        self,
        basenames,
    ):
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

        grn_idat = [IdatData(filepath) for filepath in grn_idat_files]
        red_idat = [IdatData(filepath) for filepath in red_idat_files]

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

        self.grn_mean = np.array(
            [
                idat.probe_means[np.isin(idat.illumina_ids, self.ids)]
                for idat in grn_idat
            ]
        )

        self.red_mean = np.array(
            [
                idat.probe_means[np.isin(idat.illumina_ids, self.ids)]
                for idat in red_idat
            ]
        )
        self.manifest = ManifestLoader.get_manifest(self.array_type)

    def __repr__(self):
        grn_result = pd.DataFrame({"ids":self.ids})
        grn_result[self.probes] = pd.DataFrame(self.grn_mean.T)
        red_result = pd.DataFrame({"ids":self.ids})
        red_result[self.probes] = pd.DataFrame(self.grn_mean.T)
        title = f"RGSet():"
        lines = [
            title + "\n" + "*" * len(title),
            f"array_type: {self.array_type}",
            f"probes:\n{self.probes}",
            f"ids:\n{self.ids}",
            f"grn:\n{self.grn_mean}",
            f"red:\n{self.red_mean}",
            f"grn:\n{grn_result}",
            f"red:\n{red_result}",
        ]
        return "\n\n".join(lines)



file0 = "/data/ref_IDAT/450k/3999997083_R02C02_Grn.idat"
file1 = "/data/ref_IDAT/450k/5775446049_R01C02_Grn.idat"

basenames = os.path.expanduser("/data/epidip_IDAT/10003885067_R02C02_Grn.idat")
basenames = ["~/path/to/file_Red.idat", "path/to/another_file_Grn.idat"]
basenames = [
    Path("path/to/file_Red.idat"),
    Path("path/to/another_file_Grn.idat"),
]
basenames = [file0, file1]

quit()

timer.start()
rg_set = RGSet(basenames)
timer.stop()

self = rg_set

manifest = ManifestLoader.get_manifest(ArrayType('450k'))
manifest = ManifestLoader.get_manifest(ArrayType('epic'))
manifest = ManifestLoader.get_manifest(ArrayType('epicv2'))


# class MyClass:
    # def __init__(self, array_type):
        # self.manifest = ManifestLoader.get_manifest(array_type)

    # def do_something_with_manifest(self):
        # # Use self.manifest here
        # pass

# # Usage
# my_instance = MyClass('450k')
# my_instance.do_something_with_manifest()

