"""Contains a IDAT file parser."""

import gzip
import io
import os
from enum import IntEnum, unique

import numpy as np

from mepylome.utils.files import get_file_object

__all__ = ["IdatParser"]

DEFAULT_IDAT_VERSION = 3
DEFAULT_IDAT_FILE_ID = "IDAT"


def read_byte(infile):
    return int.from_bytes(infile.read(1), byteorder="little", signed=False)


def read_short(infile):
    return int.from_bytes(infile.read(2), byteorder="little", signed=False)


def read_int(infile):
    return int.from_bytes(infile.read(4), byteorder="little", signed=True)


def read_long(infile):
    return int.from_bytes(infile.read(8), byteorder="little", signed=True)


def read_char(infile, num_bytes):
    return infile.read(num_bytes).decode("utf-8")


def read_string(infile):
    num_bytes = read_byte(infile)
    num_chars = num_bytes % 128
    shift = 0
    while num_bytes // 128 == 1:
        num_bytes = read_byte(infile)
        shift += 7
        offset = (num_bytes % 128) * (2**shift)
        num_chars += offset
    return read_char(infile, num_chars)


def read_array(infile, dtype, n):
    dtype = np.dtype(dtype)
    total_size = dtype.itemsize * n
    alldata = infile.read(total_size)

    if len(alldata) != total_size:
        msg = "End of file reached before number of results parsed"
        raise EOFError(msg)

    return np.frombuffer(alldata, dtype)


@unique
class IdatSectionCode(IntEnum):
    """Section codes used in Illumina idat files."""

    ILLUMINA_ID = 102
    STD_DEV = 103
    MEAN = 104
    NUM_BEADS = 107  # how many replicate measurements for each probe
    MID_BLOCK = 200
    RUN_INFO = 300
    RED_GREEN = 400
    MOSTLY_NULL = 401  # manifest
    BARCODE = 402
    CHIP_TYPE = 403  # format
    MOSTLY_A = 404  # label
    UNKNOWN_1 = 405  # opa
    UNKNOWN_2 = 406  # sampleid
    UNKNOWN_3 = 407  # descr
    UNKNOWN_4 = 408  # plate
    UNKNOWN_5 = 409  # well
    UNKNOWN_6 = 410
    UNKNOWN_7 = 510  # unknown
    NUM_SNPS_READ = 1000


def _get_file_size(file_like):
    """Get the size of a file-like object."""
    # Check if the file-like object has a fileno method
    if isinstance(file_like, (io.BufferedReader, gzip.GzipFile)):
        return os.fstat(file_like.fileno()).st_size

    if isinstance(file_like, io.BytesIO):
        current_pos = file_like.tell()
        file_like.seek(0, io.SEEK_END)
        size = file_like.tell()
        file_like.seek(current_pos)
        return size

    msg = "Cannot determine file size. Unknown file format"
    raise ValueError(msg)


class IdatParser:
    """Reads and parses an IDAT file.

    Stores all extracted values from the IDAT file as attributes.

    Args:
        file (str or file-like object): Path to the IDAT file or
            a file-like object. Can also be a gzipped IDAT file.
        intensity_only (bool, optional): Whether to read only intensity values,
            which makes parsing faster. Defaults to False.

    Examples:
        >>> filepath = "/path/to/idat/file_Grn.idat"
        >>> idat_data = IdatParser(filepath)
        >>> ids = idat_data.illumina_ids
        >>> print(idat_data)
    """

    def __init__(
        self,
        file,
        *,
        intensity_only=False,
        array_type_only=False,
    ):
        """Reads and parses the IDAT file."""
        self.intensity_only = intensity_only
        self.array_type_only = array_type_only
        self._file = file

        with get_file_object(file) as idat_file:
            self.file_size = _get_file_size(idat_file)
            self._parse_header(idat_file)
            self._parse_body(idat_file)

    def _parse_header(self, idat_file):
        file_type = read_char(idat_file, len(DEFAULT_IDAT_FILE_ID))
        # Assert file is indeed IDAT format
        if file_type != DEFAULT_IDAT_FILE_ID:
            msg = (
                f"Parser could not open file {self._file} as its not a valid "
                "IDAT file."
            )
            raise ValueError(msg)

        idat_version = read_long(idat_file)

        # Assert correct IDAT file version
        if idat_version != DEFAULT_IDAT_VERSION:
            msg = (
                f"Parser could not open file {self._file} as its not a "
                "version 3 IDAT file."
            )
            raise ValueError(msg)

        self.num_fields = read_int(idat_file)

        self.offsets = {
            read_short(idat_file): read_long(idat_file)
            for _ in range(self.num_fields)
        }

    def _parse_body(self, idat_file):
        def seek_to_section(section_code):
            idat_file.seek(self.offsets[section_code.value])

        seek_to_section(IdatSectionCode.NUM_SNPS_READ)
        self.n_snps_read = read_int(idat_file)

        if self.array_type_only:
            return

        seek_to_section(IdatSectionCode.ILLUMINA_ID)
        self.illumina_ids = read_array(idat_file, "<i4", self.n_snps_read)

        seek_to_section(IdatSectionCode.MEAN)
        self.probe_means = read_array(idat_file, "<u2", self.n_snps_read)

        if self.intensity_only:
            return

        seek_to_section(IdatSectionCode.STD_DEV)
        self.std_dev = read_array(idat_file, "<u2", self.n_snps_read)

        seek_to_section(IdatSectionCode.NUM_BEADS)
        self.n_beads = read_array(idat_file, "<u1", self.n_snps_read)

        seek_to_section(IdatSectionCode.MID_BLOCK)
        n_mid_block = read_int(idat_file)
        self.mid_block = read_array(idat_file, "<i4", n_mid_block)

        seek_to_section(IdatSectionCode.RUN_INFO)
        runinfo_entry_count = read_int(idat_file)

        self.run_info = [None] * runinfo_entry_count
        for i in range(runinfo_entry_count):
            self.run_info[i] = [
                read_string(idat_file),  # run_time
                read_string(idat_file),  # block_type
                read_string(idat_file),  # block_pars
                read_string(idat_file),  # block_code
                read_string(idat_file),  # code_version
            ]

        seek_to_section(IdatSectionCode.RED_GREEN)
        self.red_green = read_int(idat_file)

        seek_to_section(IdatSectionCode.MOSTLY_NULL)
        self.mostly_null = read_string(idat_file)

        seek_to_section(IdatSectionCode.BARCODE)
        self.barcode = read_string(idat_file)

        seek_to_section(IdatSectionCode.CHIP_TYPE)
        self.chip_type = read_string(idat_file)

        seek_to_section(IdatSectionCode.MOSTLY_A)
        self.mostly_a = read_string(idat_file)

        seek_to_section(IdatSectionCode.UNKNOWN_1)
        self.unknown_1 = read_string(idat_file)

        seek_to_section(IdatSectionCode.UNKNOWN_2)
        self.unknown_2 = read_string(idat_file)

        seek_to_section(IdatSectionCode.UNKNOWN_3)
        self.unknown_3 = read_string(idat_file)

        seek_to_section(IdatSectionCode.UNKNOWN_4)
        self.unknown_4 = read_string(idat_file)

        seek_to_section(IdatSectionCode.UNKNOWN_5)
        self.unknown_5 = read_string(idat_file)

        seek_to_section(IdatSectionCode.UNKNOWN_6)
        self.unknown_6 = read_string(idat_file)

        seek_to_section(IdatSectionCode.UNKNOWN_7)
        self.unknown_7 = read_string(idat_file)

    def __repr__(self):
        with np.printoptions(edgeitems=2):
            result = (
                f"IdatParser(\n"
                f"    file_size: {self.file_size}\n"
                f"    num_fields: {self.num_fields}\n"
                f"    n_snps_read: {self.n_snps_read}\n"
                f"    illumina_ids: {self.illumina_ids!r}\n"
                f"    probe_means: {self.probe_means!r}\n"
            )

            if self.intensity_only:
                return result + ")"

            return result + (
                f"    std_dev: {self.std_dev!r}\n"
                f"    n_beads: {self.n_beads!r}\n"
                f"    mid_block: {self.mid_block!r}\n"
                f"    red_green: {self.red_green}\n"
                f"    mostly_null: {self.mostly_null}\n"
                f"    barcode: {self.barcode}\n"
                f"    chip_type: {self.chip_type}\n"
                f"    mostly_a: {self.mostly_a}\n"
                f"    unknown_1: {self.unknown_1}\n"
                f"    unknown_2: {self.unknown_2}\n"
                f"    unknown_3: {self.unknown_3}\n"
                f"    unknown_4: {self.unknown_4}\n"
                f"    unknown_5: {self.unknown_5}\n"
                f"    unknown_6: {self.unknown_6}\n"
                f"    unknown_7: {self.unknown_7}\n"
                ")"
            )
