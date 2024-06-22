"""Module for handling Illumina array manifest files.

This module contains a single class ``Manifest`` for reading and processing
Illumina array manifest files, which contain information about probes and their
characteristics.
"""

import logging
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyranges as pr

from mepylome.dtypes.arrays import ArrayType
from mepylome.dtypes.chromosome import Chromosome
from mepylome.dtypes.cache import cache_key
from mepylome.dtypes.probes import Channel, InfiniumDesignType, ProbeType
from mepylome.utils.files import (
    download_file,
    ensure_directory_exists,
    get_csv_file,
    reset_file,
)

__all__ = ["Manifest"]


logger = logging.getLogger(__name__)


MANIFEST_DIR = Path.home() / ".mepylome" / "manifest_files"
MEPYLOME_TMP_DIR = Path(tempfile.gettempdir()) / "mepylome"

ENDING_CONTROL_PROBES = "_control-probes"
NONE = -1

MANIFEST_URL = {
    ArrayType.ILLUMINA_450K: (
        "https://webdata.illumina.com/downloads/productfiles/humanmethylation450/humanmethylation450_15017482_v1-2.csv"
    ),
    ArrayType.ILLUMINA_EPIC: (
        "https://webdata.illumina.com/downloads/productfiles/methylationEPIC/infinium-methylationepic-v-1-0-b5-manifest-file-csv.zip"
    ),
    ArrayType.ILLUMINA_EPIC_V2: (
        "https://support.illumina.com.cn/content/dam/illumina-support/documents/downloads/productfiles/methylationepic/InfiniumMethylationEPICv2.0ProductFiles(ZIPFormat).zip"
    ),
    ArrayType.ILLUMINA_MOUSE: (
        "https://support.illumina.com/content/dam/illumina-support/documents/downloads/productfiles/mouse-methylation/infinium-mouse-methylation-manifest-file-csv.zip"
    ),
}

REMOTE_FILENAME = {
    ArrayType.ILLUMINA_450K: "humanmethylation450_15017482_v1-2.csv",
    ArrayType.ILLUMINA_EPIC: "infinium-methylationepic-v-1-0-b5-manifest-file.csv",
    ArrayType.ILLUMINA_EPIC_V2: "EPIC-8v2-0_A1.csv",
    ArrayType.ILLUMINA_MOUSE: "MouseMethylation-12v1-0_A2.csv",
}

LOCAL_FILENAME = {
    ArrayType.ILLUMINA_450K: "manifest-450k.csv.gz",
    ArrayType.ILLUMINA_EPIC: "manifest-epic.csv.gz",
    ArrayType.ILLUMINA_EPIC_V2: "manifest-epicv2.csv.gz",
    ArrayType.ILLUMINA_MOUSE: "manifest-mouse.csv.gz",
}


PROBES_COLUMNS = [
    "IlmnID",
    "Name",
    "AddressA_ID",
    "AddressB_ID",
    "Infinium_Design_Type",
    "Color_Channel",
    "CHR",
    "MAPINFO",
    "AlleleA_ProbeSeq",
    "AlleleB_ProbeSeq",
]


CONTROL_COLUMNS = (
    "Address_ID",
    "Control_Type",
    "Color",
    "Extended_Type",
)


class Manifest:
    """Provides an object interface to an Illumina array manifest file.

    This class provides functionality for reading and processing Illumina array
    manifest files. If the manifest is used for the first time, the data is
    automatically downloaded, transformed, and saved locally, which may take
    some time.

    Args:
        array_type (str or ArrayType): The type of array to process. Use either
            ArrayType (ArrayType.ILLUMINA_450K, ArrayType.ILLUMINA_EPIC,
            ArrayType.ILLUMINA_EPIC_V2) or corresponding string ('450k',
            'epic', 'epicv2')

        filepath (str or Path): A pre-existing manifest filepath (default:
            None)

        verbose: If True, enables verbose output. (default: True)

    Examples:
        >>> # To initialize a manifest object for Illumina 450k array:
        >>> manifest = Manifest("450k")
        >>> manifest

        >>> # To initialize a manifest object for Illumina EPIC array
        >>> manifest = Manifest(ArrayType.ILLUMINA_EPIC)
        >>> type_1 = manifest.probe_info(ProbeType.ONE)

        >>> # To load all manifests when first used:
        >>> Manifest.load()
    """

    _cache = {}

    def __new__(
        cls,
        array_type,
        filepath=None,
        verbose=True,
    ):
        key = cache_key(array_type, filepath)
        if key in cls._cache:
            return cls._cache[key]

        instance = super().__new__(cls)

        # Cache the instance
        cls._cache[key] = instance
        return instance

    def __getnewargs__(self):
        # Necessary for pickle
        return self.array_type, self.filepath, self.verbose

    def __init__(
        self,
        array_type,
        filepath=None,
        verbose=True,
    ):
        if hasattr(self, "_cached"):
            return
        self._cached = True
        if isinstance(array_type, str):
            array_type = ArrayType(array_type)

        self.array_type = array_type
        self.filepath = filepath
        self.verbose = verbose

        if self.filepath is None:
            (
                self.filepath,
                control_filepath,
            ) = self._get_processed_manifest(array_type)
        else:
            control_filepath = self._get_control_path(self.filepath)

        self.__data_frame = self._read_probes(self.filepath)
        self.__control_data_frame = self._read_control_probes(control_filepath)
        self.__snp_data_frame = self._read_snp_probes()
        self.__methyl_probes = None

    @property
    def data_frame(self):
        """Pandas data frame of all manifest probes."""
        return self.__data_frame

    @property
    def control_data_frame(self):
        """Pandas data frame of all manifest control probes."""
        return self.__control_data_frame

    @property
    def snp_data_frame(self):
        """SNP probes from the manifest data frame."""
        return self.__snp_data_frame

    @property
    def methylation_probes(self):
        """All type I and II probes."""
        if self.__methyl_probes is None:
            type_1 = self.probe_info(ProbeType.ONE)
            type_2 = self.probe_info(ProbeType.TWO)
            idx = np.sort(
                np.concatenate(
                    [
                        type_1.IlmnID.index,
                        type_2.IlmnID.index,
                    ]
                )
            )
            self.__methyl_probes = self.__data_frame.iloc[idx]["IlmnID"].values
        return self.__methyl_probes

    def control_address(self, control_type=None):
        """Returns address IDs of all control probes of the specified type."""
        if control_type is None:
            return self.__control_data_frame.Address_ID
        # Ensure control_type is a list-like object
        if not isinstance(control_type, (list, tuple)):
            control_type = [control_type]
        # Use isin() with the list-like object
        return self.__control_data_frame[
            self.__control_data_frame.Control_Type.isin(control_type)
        ].Address_ID

    @staticmethod
    def load(array_types=None):
        """Loads specified manifests into memory.

        Args:
            array_types (list or ArrayType, optional): List of array types or
                a single array type to load. Defaults to all available types.

        Examples:
            >>> # Load all manifests:
            >>> Manifest.load()

            >>> # Load specific manifests:
            >>> Manifest.load([ArrayType.ILLUMINA_450K, ArrayType.ILLUMINA_EPIC])
            >>> Manifest.load("epicv2")
        """
        if array_types is None:
            array_types = [
                ArrayType.ILLUMINA_450K,
                ArrayType.ILLUMINA_EPIC,
                ArrayType.ILLUMINA_EPIC_V2,
            ]
        if not isinstance(array_types, list):
            array_types = [array_types]
        for array_type in array_types:
            _ = Manifest(array_type)

    @staticmethod
    def _get_processed_manifest(array_type):
        """Downloads the appropriate manifest file if necessary.

        Args:
            array_type: The manifest array type.

        Returns:
            tuple of paths: Local paths to the manifest file and its control
                file.
        """
        local_filename = LOCAL_FILENAME[array_type]
        local_filepath = Path(MANIFEST_DIR, local_filename).expanduser()
        control_filepath = Manifest._get_control_path(local_filepath)

        if local_filepath.exists() and control_filepath.exists():
            return local_filepath, control_filepath

        download_dir = Path(MEPYLOME_TMP_DIR).expanduser()

        source_url = MANIFEST_URL[array_type]
        source_filename = Path(source_url).name
        logger.info("Downloading manifest: %s", source_filename)
        download_file(source_url, Path(download_dir, source_filename))

        downloaded_filepath = Path(download_dir, source_filename).expanduser()
        # Remove the .gz suffix
        remote_filename = REMOTE_FILENAME[array_type]
        Manifest._process_manifest(
            filepath=downloaded_filepath,
            manifest_name=remote_filename,
            dest_probes=local_filepath,
            dest_control=control_filepath,
        )

        # Remove downloaded files
        # shutil.rmtree(MEPYLOME_TMP_DIR)

        return local_filepath, control_filepath

    @staticmethod
    def _get_control_path(probes_path):
        """Converts probes path to the control probes path."""
        split_filename = probes_path.name.split(".")
        split_filename[0] += ENDING_CONTROL_PROBES
        return Path(probes_path.parent, ".".join(split_filename))

    @staticmethod
    def _process_manifest(filepath, manifest_name, dest_probes, dest_control):
        """Processes the manifest file and saves it locally to disk.

        Args:
            filepath (Path): Path to the downloaded manifest archive.
            manifest_name (str): Name of the manifest file inside the archive.
            dest_probes (Path): Local destination path to save the processed
                probes.
            dest_control (Path): Local destination path to save the processed
                control probes.
        """
        ensure_directory_exists(dest_probes)
        ensure_directory_exists(dest_control)
        with get_csv_file(filepath, manifest_name) as manifest_file:
            # Process probes
            Manifest._seek_to_start(manifest_file)
            probes_df = pd.read_csv(
                manifest_file,
                low_memory=False,
                usecols=PROBES_COLUMNS,
            )
            n_probes = probes_df[probes_df.IlmnID.str.startswith("[")].index[0]
            probes_df = probes_df[:n_probes]
            probes_df = Manifest._process_probes(probes_df)
            probes_df.to_csv(dest_probes, index=False)
            # Process controls
            Manifest._seek_to_start(manifest_file)
            controls_df = pd.read_csv(
                manifest_file,
                header=None,
                # Skip metadata and probe section
                skiprows=(3 + n_probes),
                usecols=range(len(CONTROL_COLUMNS)),
            )
            controls_df.columns = CONTROL_COLUMNS
            controls_df["Address_ID"] = controls_df["Address_ID"].astype(
                "int32"
            )
            controls_df.to_csv(dest_control, index=False)

    @staticmethod
    def _process_probes(data_frame):
        """Transforms manifest probes to a more efficient internal format."""
        data_frame = data_frame.rename(
            columns={
                "MAPINFO": "Start",
                "CHR": "Chromosome",
            },
        )
        data_frame["Chromosome"] = Chromosome.pd_from_string(
            data_frame["Chromosome"]
        )
        # IlmnID and Name are different in EPICv2
        data_frame.IlmnID = data_frame.Name
        data_frame = data_frame.drop(columns="Name")
        channel_to_int = {"Grn": Channel.GRN, "Red": Channel.RED}
        data_frame["Color_Channel"] = data_frame["Color_Channel"].replace(
            channel_to_int
        )
        data_frame["Infinium_Design_Type"] = data_frame[
            "Infinium_Design_Type"
        ].replace({"I": InfiniumDesignType.I, "II": InfiniumDesignType.II})
        data_frame["TypeI_N_CpG"] = np.maximum(
            0, data_frame["AlleleB_ProbeSeq"].fillna("").str.count("CG") - 1
        )
        # R Stands for CG in AlleleA_ProbeSeq
        data_frame["TypeII_N_CpG"] = data_frame["AlleleA_ProbeSeq"].str.count(
            "R"
        )
        data_frame["N_CpG"] = NONE
        data_frame.loc[
            data_frame["Infinium_Design_Type"] == InfiniumDesignType.I,
            "N_CpG",
        ] = data_frame.loc[
            data_frame["Infinium_Design_Type"] == InfiniumDesignType.I,
            "TypeI_N_CpG",
        ]
        data_frame.loc[
            data_frame["Infinium_Design_Type"] == InfiniumDesignType.II,
            "N_CpG",
        ] = data_frame.loc[
            data_frame["Infinium_Design_Type"] == InfiniumDesignType.II,
            "TypeII_N_CpG",
        ]

        # Use int32 to improve performance of indexing
        int_cols = [
            "AddressA_ID",
            "AddressB_ID",
            "Infinium_Design_Type",
            "Start",
            "Color_Channel",
            "TypeI_N_CpG",
            "TypeII_N_CpG",
        ]
        data_frame[int_cols] = (
            data_frame[int_cols].fillna(NONE).astype("int32")
        )
        data_frame["End"] = data_frame["Start"]
        data_frame = data_frame.drop(
            columns=["AlleleA_ProbeSeq", "AlleleB_ProbeSeq"]
        )

        def get_probe_type(name, infinium_type):
            """Determines the probe type (I, II, SnpI, SnpII or Control)."""
            probe_type = ProbeType.from_manifest_values(name, infinium_type)
            return probe_type.value

        vectorized_get_type = np.vectorize(get_probe_type)
        data_frame["Probe_Type"] = vectorized_get_type(
            data_frame.IlmnID.values,
            data_frame["Infinium_Design_Type"].values,
        )
        # TODO drop Infinium_Design_Type
        probes_ranges = pr.PyRanges(data_frame).sort()
        return probes_ranges.df

    @staticmethod
    def _seek_to_start(manifest_file):
        """Move the manifest file pointer to the start of the data section.

        Details:
            The function searches for the first occurrence of the left-most
            column named "IlmnID".
        """
        reset_file(manifest_file)

        current_pos = manifest_file.tell()
        header_line = manifest_file.readline()

        while not header_line.startswith(b"IlmnID"):
            current_pos = manifest_file.tell()
            if not header_line:
                msg = (
                    "The first (left-most) column in your manifest must "
                    "contain 'IlmnID'. This defines the header row."
                )
                raise EOFError(msg)
            header_line = manifest_file.readline()

        if current_pos == 0:
            manifest_file.seek(current_pos)
        else:
            manifest_file.seek(current_pos - 1)

    def _read_probes(self, probes_file):
        """Reads and returns probes from local file {probes_file}."""
        if self.verbose:
            logger.info(
                "Reading manifest file: %s", Path(probes_file.name).stem
            )

        data_frame = pd.read_csv(
            probes_file,
            dtype=self._get_data_types(),
        )
        # TODO epicv2 has duplicated ID's for example c='cg22367159'
        data_frame = data_frame.drop_duplicates(
            subset=["IlmnID"], keep="first"
        )
        return data_frame.reset_index(drop=True)

    def _read_control_probes(self, control_file):
        """Reads and returns control probes from local file {control_file}."""
        data_frame = pd.read_csv(
            control_file,
            dtype=self._get_data_types(),
        )
        # Use int32 instead of int64 to improve performance of indexing
        data_frame["Address_ID"] = (
            data_frame["Address_ID"].astype("int32").copy()
        )
        return data_frame

    def _read_snp_probes(self):
        """Extracts SNP probes from the manifest data frame."""
        snp_df = self.data_frame.copy()
        return snp_df[snp_df.IlmnID.str.match("rs", na=False)]

    def _get_data_types(self):
        """Returns data types for the manifest columns."""
        return {
            "IlmnID": np.dtype(object),
            "AddressA_ID": np.int32,
            "AddressB_ID": np.int32,
            "Infinium_Design_Type": np.int8,
            "Color_Channel": np.int8,
            "Chromosome": np.int8,
            "Start": np.int32,
            "TypeI_N_CpG": np.int8,
            "TypeII_N_CpG": np.int8,
            "N_CpG": np.int8,
            "End": np.int32,
            "Probe_Type": np.int8,
        }

    def probe_info(self, probe_type, channel=None):
        """Retrieves information about probes of a specified type and channel.

        Args:
            probe_type (ProbeType): The type of probe (I, II, SnpI, SnpII,
                Control).
            channel (Channel, optional): The color channel (RED or GRN).
                Defaults to None.

        Returns:
            DataFrame: DataFrame containing information about the specified
                probes.

        Raises:
            ValueError: If probe_type is not a valid ProbeType or if channel is
            not a valid Channel.
        """
        if not isinstance(probe_type, ProbeType):
            msg = "probe_type is not a valid ProbeType"
            raise TypeError(msg)

        if channel and not isinstance(channel, Channel):
            msg = "channel not a valid Channel"
            raise TypeError(msg)

        data_frame = self.data_frame
        probe_type_mask = data_frame["Probe_Type"].values == probe_type.value

        if channel is None:
            return data_frame[probe_type_mask]

        channel_mask = data_frame["Color_Channel"].values == channel.value
        return data_frame[probe_type_mask & channel_mask]

    def __repr__(self):
        title = f"Manifest({self.array_type}):"
        lines = [
            title + "\n" + "*" * len(title),
            f"data_frame:\n{self.data_frame}",
            f"control_data_frame:\n{self.control_data_frame}",
            f"snp_data_frame:\n{self.snp_data_frame}",
        ]
        return "\n\n".join(lines)
