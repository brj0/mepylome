import logging
import pickle
from pathlib import Path
from urllib.parse import urljoin

import numpy as np
import pandas as pd
import pyranges as pr

from mepylome.dtypes import (
    ArrayType,
    Channel,
    Chromosome,
    InfiniumDesignType,
    ProbeType,
    memoize,
)
from mepylome.utils import (
    download_file,
    ensure_directory_exists,
    get_file_from_archive,
    get_file_object,
    reset_file,
)

__all__ = ["Manifest"]


NONE = -1
LOGGER = logging.getLogger(__name__)

MANIFEST_DIR = f"~/.mepylome/manifest_files"
MANIFEST_TMP_DIR = f"/tmp/mepylome"
ENDING_CONTROL_PROBES = "_control-probes"


REMOTE_FILENAME = {
    ArrayType.ILLUMINA_450K: "humanmethylation450_15017482_v1-2.csv.gz",
    ArrayType.ILLUMINA_EPIC: "infinium-methylationepic-v-1-0-b5-manifest-file.csv.gz",
    ArrayType.ILLUMINA_EPIC_V2: "EPIC-8v2-0_A1.csv.gz",
    ArrayType.ILLUMINA_MOUSE: "MouseMethylation-12v1-0_A2.csv.gz",
}

SOURCE_URL = {
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

def get_key(array_type, filepath):
    return (str(array_type), str(filepath))


class _Manifest:
    """Provides an object interface to an Illumina array manifest file.

    Arguments:
        array_type {ArrayType} -- The type of array to process.
        values are styled like ArrayType.ILLUMINA_27K, ArrayType.ILLUMINA_EPIC
        or ArrayType('epic'), ArrayType('mouse')

    Keyword Arguments:
        filepath {file-like} -- a pre-existing manifest filepath
        (default: {None})

    Raises:
        ValueError: The sample sheet is not formatted properly or a sample
        cannot be found.
    """
    # _cache = {}

    # def __new__(
        # cls,
        # array_type,
        # filepath=None,
        # verbose=True,
    # ):
        # cache_key = get_key(array_type, filepath)
        # if cache_key in cls._cache:
            # return cls._cache[cache_key]
        # instance = super().__new__(cls)
        # # Cache the instance
        # cls._cache[cache_key] = instance
        # print("---MANIFEST CACHED", cache_key)
        # return instance

    # def __getnewargs__(self):
        # """ Necessary for pickle"""
        # return self.array_type, self.filepath, self.verbose

    def __init__(
        self,
        array_type,
        filepath=None,
        verbose=True,
    ):
        # if hasattr(self, 'cache_key'):
            # return
        # self.cache_key = get_key(array_type, filepath)
        if isinstance(array_type, str):
            array_type = ArrayType(array_type)

        self.array_type = array_type
        self.filepath = filepath
        self.verbose = verbose

        if self.filepath is None:
            (
                self.filepath,
                control_filepath,
            ) = self.get_processed_manifest(array_type)
        else:
            control_filepath = self.get_control_path(self.filepath)

        self.__data_frame = self.read_probes(self.filepath)
        self.__control_data_frame = self.read_control_probes(control_filepath)
        self.__snp_data_frame = self.read_snp_probes()
        self.__methyl_probes = self.get_methyl_probes()

    def get_methyl_probes(self):
        type_1 = self.probe_info(ProbeType.ONE)
        type_2 = self.probe_info(ProbeType.TWO)
        locus_names = np.sort(
            np.concatenate(
                [
                    type_1.IlmnID.index,
                    type_2.IlmnID.index,
                ]
            )
        )
        return self.__data_frame.iloc[locus_names]["IlmnID"].values

    def __repr__(self):
        title = f"Manifest({self.array_type}):"
        lines = [
            title + "\n" + "*" * len(title),
            f"data_frame:\n{self.data_frame}",
            f"control_data_frame:\n{self.control_data_frame}",
            f"snp_data_frame:\n{self.snp_data_frame}",
        ]
        return "\n\n".join(lines)

    # def __getstate__(self):
        # state = self.__dict__.copy()
        # return state

    # def __setstate__(self, state):
        # self.__dict__.update(state)

    @staticmethod
    def load(
        array_types=[
            ArrayType.ILLUMINA_450K,
            ArrayType.ILLUMINA_EPIC,
            ArrayType.ILLUMINA_EPIC_V2,
        ]
    ):
        for array_type in array_types:
            _ = Manifest(array_type)

    @property
    def data_frame(self):
        return self.__data_frame

    @property
    def control_data_frame(self):
        return self.__control_data_frame

    def control_address(self, control_type=None):
        if control_type is None:
            return self.__control_data_frame.Address_ID
        # Ensure control_type is a list-like object
        if not isinstance(control_type, (list, tuple)):
            control_type = [control_type]
        # Use isin() with the list-like object
        return self.__control_data_frame[
            self.__control_data_frame.Control_Type.isin(control_type)
        ].Address_ID

    @property
    def snp_data_frame(self):
        return self.__snp_data_frame

    @property
    def methylation_probes(self):
        return self.__methyl_probes

    @staticmethod
    def get_processed_manifest(array_type):
        """Downloads the appropriate manifest file if one does not already exist.

        Args:
            array_type (ArrayType): The type of array to process.

        Returns:
            tuple of PurePath: Paths to the manifest file and its control file.
        """
        local_filename = LOCAL_FILENAME[array_type]
        local_filepath = Path(MANIFEST_DIR, local_filename).expanduser()
        control_filepath = Manifest.get_control_path(local_filepath)

        if local_filepath.exists() and control_filepath.exists():
            return local_filepath, control_filepath

        download_dir = Path(MANIFEST_TMP_DIR).expanduser()

        source_url = SOURCE_URL[array_type]
        source_filename = Path(source_url).name
        LOGGER.info(f"Downloading manifest: {source_filename}")
        download_file(source_filename, source_url, download_dir)

        downloaded_filepath = Path(download_dir, source_filename).expanduser()
        # Remove the .gz suffix
        remote_filename = REMOTE_FILENAME[array_type].replace(".gz", "")
        Manifest.process_manifest(
            filepath=downloaded_filepath,
            manifest_name=remote_filename,
            dest_probes=local_filepath,
            dest_control=control_filepath,
        )

        # Remove downloaded files
        # TODO uncomment this
        # shutil.rmtree(MANIFEST_TMP_DIR)

        return local_filepath, control_filepath

    @staticmethod
    def get_control_path(probes_path):
        split_filename = probes_path.name.split(".")
        split_filename[0] += ENDING_CONTROL_PROBES
        return Path(probes_path.parent, ".".join(split_filename))

    @staticmethod
    def process_probes(data_frame):
        # TODO Chromosomes not consistent
        data_frame.rename(
            columns={
                "MAPINFO": "Start",
                "CHR": "Chromosome",
            },
            inplace=True,
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
            # TODO drop Infinium_Design_Type
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
        data_frame.drop(
            columns=["AlleleA_ProbeSeq", "AlleleB_ProbeSeq"], inplace=True
        )

        def get_probe_type(name, infinium_type):
            """returns one of (I, II, SnpI, SnpII, Control)
            .from_manifest_values() returns probe type using either the
            Infinium_Design_Type (I or II) or the name
            (starts with 'rs' == SnpI) and 'Control' is none of the above.
            """
            probe_type = ProbeType.from_manifest_values(name, infinium_type)
            return probe_type.value

        vectorized_get_type = np.vectorize(get_probe_type)
        data_frame["Probe_Type"] = vectorized_get_type(
            data_frame.IlmnID.values,
            data_frame["Infinium_Design_Type"].values,
        )
        probes_ranges = pr.PyRanges(data_frame).sort()
        return probes_ranges.df

    @staticmethod
    def process_manifest(filepath, manifest_name, dest_probes, dest_control):
        ensure_directory_exists(dest_probes)
        ensure_directory_exists(dest_control)
        with get_file_from_archive(filepath, manifest_name) as manifest_file:
            # Process probes
            Manifest.seek_to_start(manifest_file)
            probes_df = pd.read_csv(
                manifest_file,
                low_memory=False,
                usecols=PROBES_COLUMNS,
            )
            n_probes = probes_df[probes_df.IlmnID.str.startswith("[")].index[0]
            probes_df = probes_df[:n_probes]
            probes_df = Manifest.process_probes(probes_df)
            probes_df.to_csv(dest_probes, index=False)
            # Process controls
            Manifest.seek_to_start(manifest_file)
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
    def seek_to_start(manifest_file):
        """find the start of the data part of the manifest. first left-most
        column must be "IlmnID" to be found.
        """
        reset_file(manifest_file)

        current_pos = manifest_file.tell()
        header_line = manifest_file.readline()

        while not header_line.startswith(b"IlmnID"):
            current_pos = manifest_file.tell()
            if not header_line:  # EOF
                raise EOFError(
                    "The first (left-most) column in your manifest must "
                    "contain 'IlmnID'. This defines the header row."
                )
            header_line = manifest_file.readline()

        if current_pos == 0:
            manifest_file.seek(current_pos)
        else:
            manifest_file.seek(current_pos - 1)

    def read_probes(self, probes_file):
        # TODO better do this before saving gz
        if self.verbose:
            LOGGER.info(
                f"Reading manifest file: {Path(probes_file.name).stem}"
            )

        data_frame = pd.read_csv(
            probes_file,
            dtype=self.get_data_types(),
        )
        # TODO epicv2 has duplicated ID's for example c='cg22367159'
        data_frame.drop_duplicates(
            subset=["IlmnID"], keep="first", inplace=True
        )
        # data_frame.index.name = "_IlmnID"
        data_frame.reset_index(inplace=True, drop=True)
        return data_frame

    def read_control_probes(self, control_file):
        """Unlike other probes, control probes have no IlmnID because they're
        not locus-specific.  they also use arbitrary columns, ignoring the
        header at start of manifest file.
        """
        data_frame = pd.read_csv(
            control_file,
            dtype=self.get_data_types(),
            # index_col=0,
        )
        # Use int32 instead of int64 to improve performance of indexing
        data_frame["Address_ID"] = (
            data_frame["Address_ID"].astype("int32").copy()
        )
        return data_frame

    def read_snp_probes(self):
        """Unlike cpg and control probes, these rs probes are NOT sequential in
        all arrays.
        """
        snp_df = self.data_frame.copy()
        # TODO use int
        snp_df = snp_df[snp_df.IlmnID.str.match("rs", na=False)]
        return snp_df

    def get_data_types(self):
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
            "Probe_Type": np.int8
        }

    def probe_info(self, probe_type, channel=None):
        """used by infer_channel_switch. Given a probe type (I, II, SnpI,
        SnpII, Control) and a channel (Channel.RED | Channel.GRN), this will
        return info needed to map probes to their names (e.g. cg0031313 or
        rs00542420), which are NOT in the idat files.
        """
        if not isinstance(probe_type, ProbeType):
            raise Exception("probe_type is not a valid ProbeType")

        if channel and not isinstance(channel, Channel):
            raise Exception("channel not a valid Channel")

        data_frame = self.data_frame
        probe_type_mask = data_frame["Probe_Type"].values == probe_type.value

        if channel is None:
            return data_frame[probe_type_mask]

        channel_mask = data_frame["Color_Channel"].values == channel.value
        return data_frame[probe_type_mask & channel_mask]

    def _probe_info(self, probe_types, channel=None):
        """Retrieve probe information based on probe types and channel."""
        data_frame = self.data_frame
        if isinstance(probe_types, ProbeType):
            probe_type_mask = (
                data_frame["Probe_Type"].values == probe_types.value
            )

        else:
            if not isinstance(probe_types, list):
                raise ValueError(
                    "probe_types must be a ProbeType or a list of ProbeType"
                )
            if not all(
                isinstance(probe_type, ProbeType) for probe_type in probe_types
            ):
                raise ValueError(f"{probe_type} is not a valid ProbeType")

            probe_type_values = np.array(
                [probe_type.value for probe_type in probe_types]
            )
            probe_type_mask = np.isin(
                data_frame["Probe_Type"].values, probe_type_values
            )

        if channel and not isinstance(channel, Channel):
            raise ValueError("channel not a valid Channel")

        if channel is None:
            return data_frame[probe_type_mask]

        channel_mask = data_frame["Color_Channel"].values == channel.value
        return data_frame[probe_type_mask & channel_mask]


# When using decorator, pickle wont work
Manifest = memoize(_Manifest)
