"""Module for handling Illumina array manifest files.

This module contains a single class ``Manifest`` for reading and processing
Illumina array manifest files, which contain information about probes and their
characteristics.
"""

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pyranges as pr

from mepylome.dtypes.arrays import ArrayType
from mepylome.dtypes.cache import cache_key, input_args_id
from mepylome.dtypes.chromosome import Chromosome
from mepylome.dtypes.probes import Channel, InfiniumDesignType, ProbeType
from mepylome.utils.files import (
    download_file,
    ensure_directory_exists,
    get_csv_file,
    reset_file,
)
from mepylome.utils.varia import CONFIG, MEPYLOME_TMP_DIR

logger = logging.getLogger(__name__)

__all__ = ["Manifest"]


MANIFEST_DIR = Path.home() / ".mepylome" / "manifest_files_v0"
DOWNLOAD_DIR = MEPYLOME_TMP_DIR / "manifests"

ENDING_CONTROL_PROBES = CONFIG["suffixes"]["manifest_control_probes"]
NONE = -1

MANIFEST_URL = {
    ArrayType(type_): url for type_, url in CONFIG["urls"]["manifest"].items()
}
PROCESSED_MANIFEST_URL = (
    CONFIG["urls"]["processed_manifests"] % MANIFEST_DIR.name
)
REMOTE_FILENAME = {
    ArrayType(type_): url for type_, url in CONFIG["files"]["remote"].items()
}
LOCAL_FILENAME = {
    ArrayType(type_): url for type_, url in CONFIG["files"]["local"].items()
}


PROBES_COLUMNS = [
    "IlmnID",
    "Name",
    "AddressA_ID",
    "AddressB_ID",
    "Infinium_Design_Type",
    "Color_Channel",
    "CHR",
    "Chr",
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
    manifest files. A manifest can be loaded automatically based on the array
    type or provided as a raw manifest file. On first use, the necessary data
    is automatically downloaded if needed, transformed, and saved locally,
    which might take some time. The processed manifest is then saved locally
    and loaded in its processed form on subsequent uses. During a running
    session, all loaded manifests are cached in memory.

    Args:
        array_type (str or ArrayType): The type of array to process. Use either
            ArrayType (ArrayType.ILLUMINA_450K, ArrayType.ILLUMINA_EPIC,
            ArrayType.ILLUMINA_EPIC_V2) or corresponding string ('450k',
            'epic', 'epicv2', 'msa48')
        proc_path (str or Path): The path to the local processed manifest file
            (default: None).
        raw_path (str or Path, optional): Path to the raw manifest file.
            Default is None.
        download_proc (bool, optional): If True and there is no locally saved
            processed manifest file, attempts to download the processed
            manifest file instead of the raw one. Defaults to True.

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
        array_type=None,
        raw_path=None,
        proc_path=None,
        download_proc=True,
    ):
        key = cache_key(array_type, raw_path, proc_path)
        if key in cls._cache:
            return cls._cache[key]

        instance = super().__new__(cls)

        # Cache the instance
        cls._cache[key] = instance
        return instance

    def __getnewargs__(self):
        # Necessary for pickle
        return (
            self.array_type,
            self.raw_path,
            self.proc_path,
            self.download_proc,
        )

    def __init__(
        self,
        array_type=None,
        raw_path=None,
        proc_path=None,
        download_proc=True,
    ):
        if hasattr(self, "_cached"):
            return
        self._cached = True

        def to_path(x):
            return x if x is None else Path(x)

        self.array_type = ArrayType(array_type) if array_type else None
        self.raw_path = to_path(raw_path)
        self.proc_path = to_path(proc_path)
        self.download_proc = download_proc

        # Load cached data from disk
        pickle_hash = input_args_id(
            "manifest", self.array_type, self.raw_path, self.proc_path
        )
        self._pickle_path = MEPYLOME_TMP_DIR / f"{pickle_hash}.pkl"
        if self._pickle_path.exists():
            with self._pickle_path.open("rb") as file:
                saved_instance = pickle.load(file)
                self.__dict__.update(saved_instance.__dict__)
                return

        if self.array_type == ArrayType.UNKNOWN:
            self._data_frame = pd.DataFrame()
            self._control_data_frame = pd.DataFrame()
            self._snp_data_frame = pd.DataFrame()
            self._methyl_probes = None
            return

        # Set processed manifest path
        if self.proc_path is None:
            if self.array_type is not None:
                self.proc_path = MANIFEST_DIR / LOCAL_FILENAME[self.array_type]
            elif self.raw_path is not None:
                if self.raw_path.suffix == ".zip":
                    gz_filename = "proc_" + self.raw_path.stem + ".gz"
                else:
                    gz_filename = "proc_" + self.raw_path.name + ".gz"
                self.proc_path = DOWNLOAD_DIR / gz_filename
            else:
                msg = "Provide either array_type or proc_path or raw_path"
                raise ValueError(msg)

        self.ctrl_path = Manifest._get_control_path(self.proc_path)

        # Create processed manifest files if they do not exist
        if not (self.proc_path.exists() and self.ctrl_path.exists()):
            # If array type is given, download the files
            if self.array_type is not None:
                if (
                    not self.download_proc
                    or not self._download_processed_manifest()
                ):
                    self._download_manifest()
                    csv_filename = REMOTE_FILENAME[self.array_type]
                    self._process_manifest(csv_filename=csv_filename)
            # Else process from raw_path
            elif self.raw_path is not None:
                if self.raw_path.suffix == ".zip":
                    csv_filename = self.raw_path.stem
                else:
                    csv_filename = self.raw_path.name
                self._process_manifest(csv_filename=csv_filename)
            else:
                msg = "Provide either array_type or proc_path or raw_path"
                raise ValueError(msg)

        self._data_frame = self._read_probes(self.proc_path)
        self._control_data_frame = self._read_control_probes(self.ctrl_path)
        self._snp_data_frame = self._read_snp_probes()
        self._methyl_probes = None

        # Save to disk
        with self._pickle_path.open("wb") as file:
            pickle.dump(self, file)

    @property
    def data_frame(self):
        """Pandas data frame of all manifest probes."""
        return self._data_frame

    @property
    def control_data_frame(self):
        """Pandas data frame of all manifest control probes."""
        return self._control_data_frame

    @property
    def snp_data_frame(self):
        """SNP probes from the manifest data frame."""
        return self._snp_data_frame

    @property
    def methylation_probes(self):
        """All type I and II probes."""
        if self._methyl_probes is None:
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
            self._methyl_probes = self._data_frame.iloc[idx]["IlmnID"].values
        return self._methyl_probes

    def control_address(self, control_type=None):
        """Returns address IDs of all control probes of the specified type."""
        if control_type is None:
            return self._control_data_frame.Address_ID
        # Ensure control_type is a list-like object
        if not isinstance(control_type, (list, tuple)):
            control_type = [control_type]
        # Use isin() with the list-like object
        return self._control_data_frame[
            self._control_data_frame.Control_Type.isin(control_type)
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
            >>> Manifest.load(
            >>>     [ArrayType.ILLUMINA_450K, ArrayType.ILLUMINA_EPIC]
            >>> )
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

    def _download_manifest(
        self,
    ):
        """Downloads the appropriate manifest file.

        This method downloads the manifest file based on the `array_type`
        specified during the initialization of the Manifest object. It stores
        the downloaded file in the path of the `raw_path` attribute.

        Raises:
            KeyError: If the array type is not found in MANIFEST_URL.
        """
        source_url = MANIFEST_URL[self.array_type]
        source_filename = Path(source_url).name

        logger.info("Downloading manifest: %s", source_filename)

        self.raw_path = DOWNLOAD_DIR / source_filename
        download_file(source_url, self.raw_path)

    def _download_processed_manifest(self):
        """Download processed manifest file and return true if successful."""
        logger.info("Downloading processed %s manifest", self.array_type)
        try:
            for path in [self.proc_path, self.ctrl_path]:
                ensure_directory_exists(path.parent)
                url = PROCESSED_MANIFEST_URL + path.name
                download_file(url, path)
            return True
        except Exception:
            return False

    @staticmethod
    def _get_control_path(probes_path):
        """Converts probes path to the control probes path."""
        split_filename = probes_path.name.split(".")
        split_filename[0] += ENDING_CONTROL_PROBES
        return Path(probes_path.parent, ".".join(split_filename))

    def _process_manifest(self, csv_filename=None):
        """Process the manifest file and save it locally to disk.

        This method processes the raw manifest file by extracting the necessary
        probe and control probe information, and then saves these processed
        details to local files with pathnames `probes_path` and `ctrl_path`.

        Args:
            csv_filename (str, optional): Name of the manifest file inside the
                archive. If not provided, it defaults to the name of the
                raw_path file.
        """
        logger.info("Process raw manifest %s", self.raw_path)
        if csv_filename is None:
            csv_filename = self.raw_path.name
        ensure_directory_exists(self.proc_path.parent)
        ensure_directory_exists(self.ctrl_path.parent)
        with get_csv_file(self.raw_path, csv_filename) as manifest_file:
            # Process probes
            Manifest._seek_to_start(manifest_file)
            available_columns = pd.read_csv(manifest_file, nrows=0).columns
            Manifest._seek_to_start(manifest_file)
            valid_columns = [
                col for col in PROBES_COLUMNS if col in available_columns
            ]
            probes_df = pd.read_csv(
                manifest_file,
                low_memory=False,
                usecols=valid_columns,
            )
            n_probes = probes_df[probes_df.IlmnID.str.startswith("[")].index[0]
            probes_df = probes_df[:n_probes]
            probes_df = Manifest._process_probes(probes_df)
            probes_df.to_csv(self.proc_path, index=False)
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
            if (
                pd.to_numeric(controls_df["Address_ID"], errors="coerce")
                .notna()
                .all()
            ):
                controls_df["Address_ID"] = controls_df["Address_ID"].astype(
                    "int32"
                )
            controls_df.to_csv(self.ctrl_path, index=False)

    @staticmethod
    def _process_probes(data_frame):
        """Transforms manifest probes to a more efficient internal format."""
        rename_map = {
            "MAPINFO": "Start",
            "CHR": "Chromosome",
            "Chr": "Chromosome",
        }
        data_frame = data_frame.rename(columns=rename_map)
        data_frame["Chromosome"] = Chromosome.pd_from_string(
            data_frame["Chromosome"]
        )
        # IlmnID and Name are different in EPICv2
        data_frame["IlmnID"] = data_frame["Name"]
        data_frame = data_frame.drop(columns=["Name"])
        channel_to_int = {"Grn": Channel.GRN, "Red": Channel.RED}
        design_type_map = {
            "I": InfiniumDesignType.I,
            "II": InfiniumDesignType.II,
        }
        with pd.option_context("future.no_silent_downcasting", True):
            if "Color_Channel" in data_frame.columns:
                data_frame["Color_Channel"] = (
                    data_frame["Color_Channel"]
                    .replace(channel_to_int)
                    .infer_objects()
                )

            if "Infinium_Design_Type" in data_frame.columns:
                data_frame["Infinium_Design_Type"] = (
                    data_frame["Infinium_Design_Type"]
                    .replace(design_type_map)
                    .infer_objects()
                )

        data_frame["TypeI_N_CpG"] = 0
        data_frame["TypeII_N_CpG"] = 0

        if "AlleleB_ProbeSeq" in data_frame.columns:
            data_frame["TypeI_N_CpG"] = np.maximum(
                0,
                data_frame["AlleleB_ProbeSeq"].fillna("").str.count("CG") - 1,
            )

        if "AlleleA_ProbeSeq" in data_frame.columns:
            # R Stands for CG in AlleleA_ProbeSeq
            data_frame["TypeII_N_CpG"] = (
                data_frame["AlleleA_ProbeSeq"].fillna("").str.count("R")
            )

        data_frame["N_CpG"] = NONE
        if "Infinium_Design_Type" in data_frame.columns:
            is_type_I = (
                data_frame["Infinium_Design_Type"] == InfiniumDesignType.I
            )
            is_type_II = (
                data_frame["Infinium_Design_Type"] == InfiniumDesignType.II
            )

            data_frame.loc[is_type_I, "N_CpG"] = data_frame.loc[
                is_type_I, "TypeI_N_CpG"
            ]
            data_frame.loc[is_type_II, "N_CpG"] = data_frame.loc[
                is_type_II, "TypeII_N_CpG"
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
        for col in int_cols:
            if col in data_frame.columns:
                data_frame[col] = data_frame[col].fillna(NONE).astype("int32")

        if "Start" in data_frame.columns:
            data_frame["End"] = data_frame["Start"]

        drop_cols = ["AlleleA_ProbeSeq", "AlleleB_ProbeSeq"]
        existing_drop_cols = [
            col for col in drop_cols if col in data_frame.columns
        ]
        data_frame = data_frame.drop(columns=existing_drop_cols)

        def get_probe_type(name, infinium_type):
            """Determines the probe type (I, II, SnpI, SnpII or Control)."""
            probe_type = ProbeType.from_manifest_values(name, infinium_type)
            return probe_type.value

        if (
            "IlmnID" in data_frame.columns
            and "Infinium_Design_Type" in data_frame.columns
        ):
            vectorized_get_type = np.vectorize(get_probe_type)
            data_frame["Probe_Type"] = vectorized_get_type(
                data_frame["IlmnID"].values,
                data_frame["Infinium_Design_Type"].values,
            )

        # TODO: drop Infinium_Design_Type
        if not {"Chromosome", "Start", "End"}.issubset(data_frame.columns):
            return data_frame

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
        """Reads and returns probes from local file `probes_file`."""
        data_frame = pd.read_csv(
            probes_file,
            dtype=self._get_data_types(),
        )
        # TODO: epicv2 has duplicated ID's for example c='cg22367159'
        data_frame = data_frame.drop_duplicates(
            subset=["IlmnID"], keep="first"
        )
        return data_frame.reset_index(drop=True)

    def _read_control_probes(self, control_file):
        """Reads and returns control probes from local file `control_file`."""
        data_frame = pd.read_csv(
            control_file,
            dtype=self._get_data_types(),
        )
        # Use int32 instead of int64 to improve performance of indexing
        if (
            pd.to_numeric(data_frame["Address_ID"], errors="coerce")
            .notna()
            .all()
        ):
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
