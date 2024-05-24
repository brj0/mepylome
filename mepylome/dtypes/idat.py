import logging
import os
from enum import IntEnum, unique

import numpy as np

from mepylome.utils.files import get_file_object

LOGGER = logging.getLogger(__name__)

__all__ = ["IdatParser"]


def bytes_to_int(input_bytes, signed=False):
    return int.from_bytes(input_bytes, byteorder="little", signed=signed)


def read_byte(infile):
    return bytes_to_int(infile.read(1), signed=False)


def read_short(infile):
    return bytes_to_int(infile.read(2), signed=False)


def read_int(infile):
    return bytes_to_int(infile.read(4), signed=True)


def read_long(infile):
    return bytes_to_int(infile.read(8), signed=True)


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
    # np.readfile is not able to read from gzopene-d file
    alldata = infile.read(dtype.itemsize * n)
    if len(alldata) != dtype.itemsize * n:
        raise EOFError("End of file reached before number of results parsed")
    readdata = np.frombuffer(alldata, dtype, n)
    if readdata.size != n:
        raise EOFError("End of file reached before number of results parsed")
    return readdata


LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.WARNING)


DEFAULT_IDAT_VERSION = 3
DEFAULT_IDAT_FILE_ID = "IDAT"


@unique
class IdatSectionCode(IntEnum):
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


class IdatParser:
    def __init__(
        self,
        filepath_or_buffer,
        intensity_only=False,
    ):
        """Reads and parses the IDAT file."""
        self.intensity_only = intensity_only
        with get_file_object(filepath_or_buffer) as idat_file:
            self.file_size = os.fstat(idat_file.fileno()).st_size

            # Parse IDAT header

            file_type = read_char(idat_file, len(DEFAULT_IDAT_FILE_ID))
            # Assert file is indeed IDAT format
            if file_type != DEFAULT_IDAT_FILE_ID:
                raise ValueError(
                    "Parser could not open file as its not a valid IDAT file."
                )

            idat_version = read_long(idat_file)

            # Assert correct IDAT file version
            if idat_version != DEFAULT_IDAT_VERSION:
                raise ValueError(
                    "Parser could not open file as its not a version 3 "
                    "IDAT file."
                )

            self.num_fields = read_int(idat_file)

            self.offsets = {}
            for _idx in range(self.num_fields):
                key = read_short(idat_file)
                self.offsets[key] = read_long(idat_file)

            # Parse IDAT body

            def seek_to_section(section_code):
                idat_file.seek(self.offsets[section_code.value])

            seek_to_section(IdatSectionCode.NUM_SNPS_READ)
            self.n_snps_read = read_int(idat_file)

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

            self.run_info = []
            for _ in range(runinfo_entry_count):
                run_time = read_string(idat_file)
                block_type = read_string(idat_file)
                block_pars = read_string(idat_file)
                block_code = read_string(idat_file)
                code_version = read_string(idat_file)
                self.run_info.append(
                    (
                        run_time,
                        block_type,
                        block_pars,
                        block_code,
                        code_version,
                    )
                )

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
                f"    illumina_ids: {repr(self.illumina_ids)}\n"
                f"    probe_means: {repr(self.probe_means)}\n"
            )

            if self.intensity_only:
                return result + ")"

            return result + (
                f"    std_dev: {repr(self.std_dev)}\n"
                f"    n_beads: {repr(self.n_beads)}\n"
                f"    mid_block: {repr(self.mid_block)}\n"
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
