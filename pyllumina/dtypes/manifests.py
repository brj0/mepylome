# Lib
import logging
from pathlib import Path
from urllib.parse import urljoin
import numpy as np
import pandas as pd
import pickle

# App
from pyllumina.dtypes import ArrayType, Channel, ProbeType
from pyllumina.utils import (
    download_file,
    get_file_object,
    get_file_from_archive,
    reset_file,
    ensure_directory_exists,
)


__all__ = ["Manifest", "ManifestLoader"]


NONE = -1
LOGGER = logging.getLogger(__name__)

MANIFEST_DIR = f"~/.pyllumina/manifest_files"
MANIFEST_DOWNLOAD_DIR = f"/tmp/pyllumina"
ENDING_CONTROL_PROBES = "_control-probes"
# MANIFEST_BUCKET_NAME = "array-manifest-files"

# MANIFEST_REMOTE_PATH = f"https://s3.amazonaws.com/{MANIFEST_BUCKET_NAME}/"

ARRAY_FILENAME = {
    # "27k": "humanmethylation450_15017482_v1-2.csv",
    "450k": "humanmethylation450_15017482_v1-2.csv.gz",
    "epic": "infinium-methylationepic-v-1-0-b5-manifest-file.csv.gz",
    # "epic+": "CombinedManifestEPIC_manifest_CoreColumns_v2.csv.gz",
    "epicv2": "EPIC-8v2-0_A1.csv.gz",
    "mouse": "MouseMethylation-12v1-0_A2.csv.gz",
}

ARRAY_URL = {
    # "27k": "",
    ArrayType.ILLUMINA_450K: "https://webdata.illumina.com/downloads/productfiles/humanmethylation450/humanmethylation450_15017482_v1-2.csv",
    ArrayType.ILLUMINA_EPIC: "https://webdata.illumina.com/downloads/productfiles/methylationEPIC/infinium-methylationepic-v-1-0-b5-manifest-file-csv.zip",
    # "epic+": "CombinedManifestEPIC_manifest_CoreColumns_v2.csv.gz",
    ArrayType.ILLUMINA_MOUSE: "https://support.illumina.com/content/dam/illumina-support/documents/downloads/productfiles/mouse-methylation/infinium-mouse-methylation-manifest-file-csv.zip",
    ArrayType.ILLUMINA_EPIC_V2: "https://support.illumina.com.cn/content/dam/illumina-support/documents/downloads/productfiles/methylationepic/InfiniumMethylationEPICv2.0ProductFiles(ZIPFormat).zip",
}

ARRAY_TYPE_MANIFEST_FILENAMES = {
    # ArrayType.ILLUMINA_27K: ARRAY_FILENAME["27k"],
    ArrayType.ILLUMINA_450K: ARRAY_FILENAME["450k"],
    ArrayType.ILLUMINA_EPIC: ARRAY_FILENAME["epic"],
    ArrayType.ILLUMINA_EPIC_V2: ARRAY_FILENAME["epicv2"],
    # ArrayType.ILLUMINA_EPIC_PLUS: ARRAY_FILENAME["epic+"],
    ArrayType.ILLUMINA_MOUSE: ARRAY_FILENAME["mouse"],
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
]

MANIFEST_COLUMNS = (
    "IlmnID",
    "AddressA_ID",
    "AddressB_ID",
    "Infinium_Design_Type",
    "Color_Channel",
    "Genome_Build",
    "CHR",
    "MAPINFO",
    "Strand",
    "OLD_Genome_Build",
    "OLD_CHR",
    "OLD_MAPINFO",
    "OLD_Strand",
)

MOUSE_MANIFEST_COLUMNS = (
    "IlmnID",
    "AddressA_ID",
    "AddressB_ID",
    "Infinium_Design_Type",
    "Color_Channel",
    # replaces Probe_Type in v1.4.6+ with tons of design meta data. only
    # 'Random' and 'Multi' matter in code.
    #'Probe_Type', # pre v1.4.6, needed to identify mouse-specific probes (mu)
    # | and control probe sub_types
    "design",
    "Genome_Build",
    "CHR",
    "MAPINFO",
    "Strand",
    "OLD_Genome_Build",
    "OLD_CHR",
    "OLD_MAPINFO",
    "OLD_Strand",
)

CONTROL_COLUMNS = (
    "Address_ID",
    "Control_Type",
    "Color",
    "Extended_Type",
    # control probes don't have 'IlmnID' values set -- these probes are not
    # locii specific
    # these column names don't appear in manifest. they are added when
    # importing the control section of rows
)


class Manifest:
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


    def __init__(
        self,
        array_type,
        filepath=None,
        verbose=True,
    ):
        array_str_to_class = dict(
            zip(
                list(ARRAY_FILENAME.keys()),
                list(ARRAY_TYPE_MANIFEST_FILENAMES.keys()),
            )
        )
        if array_type in array_str_to_class:
            array_type = array_str_to_class[array_type]
        self.array_type = array_type
        self.verbose = verbose

        if filepath is None:
            (
                filepath,
                control_filepath,
            ) = self.get_processed_manifest(array_type)
        else:
            control_filepath = self.get_control_path(filepath)

        self.__data_frame = self.read_probes(filepath)
        self.__control_data_frame = self.read_control_probes(control_filepath)
        self.__snp_data_frame = self.read_snp_probes()
        if self.array_type == ArrayType.ILLUMINA_MOUSE:
            self.__mouse_data_frame = self.read_mouse_probes()
        else:
            self.__mouse_data_frame = pd.DataFrame()

    def __repr__(self):
        title = f"Manifest({self.array_type}):"
        lines = [
            title + "\n" + "*" * len(title),
            f"data_frame:\n{self.data_frame}",
            f"control_data_frame:\n{self.control_data_frame}",
            f"snp_data_frame:\n{self.snp_data_frame}",
            f"mouse_data_frame:\n{self.mouse_data_frame}",
        ]
        return "\n\n".join(lines)

    @property
    def columns(self):
        if self.array_type == ArrayType.ILLUMINA_MOUSE:
            return MOUSE_MANIFEST_COLUMNS
        else:
            return MANIFEST_COLUMNS

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
    def mouse_data_frame(self):
        return self.__mouse_data_frame

    @staticmethod
    def download_default(array_type, on_lambda=False):
        """Downloads the appropriate manifest file if one does not already exist.

        Arguments:
            array_type {ArrayType} -- The type of array to process.

        Returns:
            [PurePath] -- Path to the manifest file.
        """
        dir_path = Path(MANIFEST_DIR).expanduser()
        if on_lambda:
            dir_path = Path(MANIFEST_DIR_PATH_LAMBDA).expanduser()
        filename = ARRAY_TYPE_MANIFEST_FILENAMES[array_type]
        filepath = Path(dir_path).joinpath(filename)

        if Path.exists(filepath):
            return filepath

        LOGGER.info(f"Downloading manifest: {Path(filename).stem}")
        # src_url = urljoin(MANIFEST_REMOTE_PATH, filename)
        # download_file(filename, src_url, dir_path)

        return filepath

    @staticmethod
    def get_processed_manifest(array_type):
        """Downloads the appropriate manifest file if one does not already exist.

        Args:
            array_type (ArrayType): The type of array to process.

        Returns:
            tuple of PurePath: Paths to the manifest file and its control file.
        """
        manifest_filename = ARRAY_TYPE_MANIFEST_FILENAMES[array_type]
        manifest_filepath = Path(MANIFEST_DIR, manifest_filename).expanduser()
        control_filepath = Manifest.get_control_path(manifest_filepath)

        if manifest_filepath.exists() and control_filepath.exists():
            return manifest_filepath, control_filepath

        download_dir = Path(MANIFEST_DOWNLOAD_DIR).expanduser()

        source_url = ARRAY_URL[array_type]
        source_filename = Path(source_url).name
        LOGGER.info(f"Downloading manifest: {source_filename}")
        download_file(source_filename, source_url, download_dir)

        downloaded_filepath = Path(download_dir, source_filename).expanduser()
        # Remove the .gz suffix
        manifest_name = manifest_filepath.with_suffix("").name
        Manifest.process_manifest(
            filepath=downloaded_filepath,
            manifest_name=manifest_name,
            dest_probes=manifest_filepath,
            dest_control=control_filepath,
        )

        # Remove downloaded files
        # TODO uncomment this
        # shutil.rmtree(MANIFEST_DOWNLOAD_DIR)

        return manifest_filepath, control_filepath

    @staticmethod
    def get_control_path(probes_path):
        split_filename = probes_path.name.split(".")
        split_filename[0] += ENDING_CONTROL_PROBES
        return Path(probes_path.parent, ".".join(split_filename))

    @staticmethod
    def process_manifest(filepath, manifest_name, dest_probes, dest_control):
        ensure_directory_exists(dest_probes)
        ensure_directory_exists(dest_control)
        with get_file_from_archive(filepath, manifest_name) as manifest_file:
            # Process probes
            Manifest.seek_to_start(manifest_file)
            data_frame = pd.read_csv(
                manifest_file,
                low_memory=False,
                usecols=PROBES_COLUMNS,
            )
            n_probes = data_frame[data_frame.IlmnID.str.startswith("[")].index[
                0
            ]
            data_frame_probes = data_frame[:n_probes].copy()
            data_frame_probes[["AddressA_ID", "AddressB_ID", "MAPINFO"]] = (
                data_frame_probes[["AddressA_ID", "AddressB_ID", "MAPINFO"]]
                .apply(pd.to_numeric)
                .astype("Int32")
            )
            data_frame_probes.to_csv(dest_probes, index=False)
            # Process controls
            Manifest.seek_to_start(manifest_file)
            data_frame_controls = pd.read_csv(
                manifest_file,
                header=None,
                # Skip metadata and probe section
                skiprows=(3 + n_probes),
                usecols=range(len(CONTROL_COLUMNS)),
            )
            data_frame_controls.columns = CONTROL_COLUMNS
            data_frame_controls["Address_ID"] = data_frame_controls[
                "Address_ID"
            ].astype("int32")
            data_frame_controls.to_csv(dest_control, index=False)

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
        if self.verbose:
            LOGGER.info(
                f"Reading manifest file: {Path(probes_file.name).stem}"
            )

        data_frame = pd.read_csv(
            probes_file,
            dtype=self.get_data_types(),
            # index_col="IlmnID",
        )

        # Use int32 instead of Int32 to improve performance of indexing
        data_frame["AddressA_ID"] = (
            data_frame["AddressA_ID"].fillna(NONE).astype("int32").copy()
        )
        data_frame["AddressB_ID"] = (
            data_frame["AddressB_ID"].fillna(NONE).astype("int32").copy()
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
        data_frame["probe_type"] = vectorized_get_type(
            data_frame.IlmnID.values,
            data_frame["Infinium_Design_Type"].values,
        )
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
        data_frame["Address_ID"] = data_frame["Address_ID"].astype("int32").copy()
        return data_frame

    def read_snp_probes(self):
        """Unlike cpg and control probes, these rs probes are NOT sequential in
        all arrays.
        """
        # since these are not sequential, loading everything and filtering by
        # IlmnID.
        snp_df = self.data_frame.copy()
        # 'O' type columns won't match in SigSet, so forcing float64 here.
        # Also, float32 won't cover all probe IDs; must be float64.
        # snp_df = snp_df[snp_df.index.str.match("rs", na=False)].astype(
        # {"AddressA_ID": "float64", "AddressB_ID": "float64"}
        # )
        # TODO use int
        snp_df = snp_df[snp_df.IlmnID.str.match("rs", na=False)]
        return snp_df

    def read_mouse_probes(self):
        """ILLUMINA_MOUSE contains unique probes whose names begin with 'mu'
        and 'rp' for 'murine' and 'repeat', respectively. This creates a
        dataframe of these probes, which are not processed like normal cg/ch
        probes.
        """
        # low_memory=Fase is required because control probes create mixed-types
        # in columns.
        # --- pre v1.4.6: mouse_df = mouse_df[(mouse_df['Probe_Type'] == 'rp')
        # | (mouse_df['IlmnID'].str.startswith('uk', na=False)) |
        # (mouse_df['Probe_Type'] == 'mu')]
        # --- pre v1.4.6: 'mu' probes start with 'cg' instead and have 'mu' in
        # Probe_Type column
        mouse_df = self.dataframe.copy()
        mouse_df = mouse_df[
            (mouse_df["design"] == "Multi") | (mouse_df["design"] == "Random")
        ]
        return mouse_df

    def get_data_types(self):
        data_types = {key: str for key in self.columns}
        data_types["AddressA_ID"] = "Int32"
        data_types["AddressB_ID"] = "Int32"
        return data_types

    def probe_info(self, probe_type, channel=None):
        """used by infer_channel_switch. Given a probe type (I, II, SnpI,
        SnpII, Control) and a channel (Channel.RED | Channel.GREEN), this will
        return info needed to map probes to their names (e.g. cg0031313 or
        rs00542420), which are NOT in the idat files.
        """
        if not isinstance(probe_type, ProbeType):
            raise Exception("probe_type is not a valid ProbeType")

        if channel and not isinstance(channel, Channel):
            raise Exception("channel not a valid Channel")

        data_frame = self.data_frame
        probe_type_mask = data_frame["probe_type"].values == probe_type.value

        if not channel:
            return data_frame[probe_type_mask]

        channel_mask = data_frame["Color_Channel"].values == channel.value
        return data_frame[probe_type_mask & channel_mask]

    def _probe_info(self, probe_types, channel=None):
        """Retrieve probe information based on probe types and channel."""
        if isinstance(probe_types, ProbeType):
            probe_types = [probe_types]

        if not isinstance(probe_types, list):
            raise ValueError(
                "probe_types must be a ProbeType or a list of ProbeType"
            )

        if not all(
            isinstance(probe_type, ProbeType) for probe_type in probe_types
        ):
            raise ValueError(f"{probe_type} is not a valid ProbeType")

        if channel and not isinstance(channel, Channel):
            raise ValueError("channel not a valid Channel")

        data_frame = self.data_frame
        probe_type_mask = data_frame["probe_type"].isin(
            [probe_type.value for probe_type in probe_types]
        )

        if not channel:
            return data_frame[probe_type_mask]

        channel_mask = data_frame["Color_Channel"].values == channel.value
        return data_frame[probe_type_mask & channel_mask]


class ManifestLoader:
    _manifests = {}

    @classmethod
    def get_manifest(cls, array_type):
        array_str_to_class = dict(
            zip(
                list(ARRAY_FILENAME.keys()),
                list(ARRAY_TYPE_MANIFEST_FILENAMES.keys()),
            )
        )
        if array_type in array_str_to_class:
            array_type = array_str_to_class[array_type]
        if array_type not in cls._manifests:
            manifest = Manifest(array_type)
            cls._manifests[array_type] = manifest
        return cls._manifests[array_type]
