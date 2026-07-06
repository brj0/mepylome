"""Contains a IDAT file parser."""

import gzip
import io
import os
import struct
from collections.abc import Generator
from contextlib import contextmanager
from enum import IntEnum, unique
from pathlib import Path
from typing import BinaryIO, Literal, cast

import numpy as np

__all__ = ["IdatParser"]

DEFAULT_IDAT_VERSION = 3
DEFAULT_IDAT_FILE_ID = "IDAT"

_D_INT32 = np.dtype("<i4")
_D_UINT16 = np.dtype("<u2")
_D_UINT8 = np.dtype("<u1")


_read_byte = struct.Struct("<B").unpack
_read_short = struct.Struct("<H").unpack
_read_int = struct.Struct("<i").unpack
_read_long = struct.Struct("<q").unpack


def read_byte(f: BinaryIO) -> int:
    return _read_byte(f.read(1))[0]


def read_short(f: BinaryIO) -> int:
    return _read_short(f.read(2))[0]


def read_int(f: BinaryIO) -> int:
    return _read_int(f.read(4))[0]


def read_long(f: BinaryIO) -> int:
    return _read_long(f.read(8))[0]


def read_char(f: BinaryIO, num_bytes: int) -> str:
    return f.read(num_bytes).decode("utf-8")


def read_string(f: BinaryIO) -> str:
    num_bytes = read_byte(f)
    num_chars = num_bytes & 0x7F
    shift = 0
    while num_bytes & 0x80:
        num_bytes = read_byte(f)
        shift += 7
        num_chars += (num_bytes & 0x7F) << shift
    return read_char(f, num_chars)


def read_array(
    f: BinaryIO,
    dtype: np.dtype,
    n: int,
) -> np.ndarray:
    total_size = dtype.itemsize * n
    alldata = f.read(total_size)

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


def _get_file_size(file_like: BinaryIO) -> int:
    """Get the size of a file-like object."""
    # Check if the file-like object has a fileno method
    if isinstance(file_like, io.BufferedReader | gzip.GzipFile):
        return os.fstat(file_like.fileno()).st_size

    if isinstance(file_like, io.BytesIO):
        current_pos = file_like.tell()
        file_like.seek(0, io.SEEK_END)
        size = file_like.tell()
        file_like.seek(current_pos)
        return size

    msg = "Cannot determine file size. Unknown file format"
    raise ValueError(msg)


@contextmanager
def get_file_object(
    file: str | Path | BinaryIO,
) -> Generator[BinaryIO, None, None]:
    """Returns a file-like object for reading an IDAT file.

    Supports plain and gzipped IDAT files.

    Args:
        file: Path to the file (str or Path) or an already open file-like
            object.

    Yields:
        A binary file-like object.
    """
    if isinstance(file, io.BufferedIOBase | io.BytesIO):
        # Already a file-like object
        yield file
    else:
        if not isinstance(file, str | Path):
            raise TypeError(f"Expected file-like object, got {type(file)}")
        path = Path(file)
        f = gzip.open(path, "rb") if path.suffix == ".gz" else path.open("rb")
        try:
            yield cast(BinaryIO, f)
        finally:
            f.close()


class IdatParser:
    """Reads and parses an IDAT file.

    Stores all extracted values from the IDAT file as attributes.

    Args:
        file (str or file-like object): Path to the IDAT file or
            a file-like object. Can also be a gzipped IDAT file.
        mode (str): Controls parsing depth. Options are 'full' (reads the
            entire file), 'intensity' (reads only intensity values), or
            'array_type' (reads only structural metadata). Defaults to 'full'.

    Examples:
        >>> filepath = "/path/to/idat/file_Grn.idat"
        >>> idat_data = IdatParser(filepath)
        >>> ids = idat_data.illumina_ids
        >>> print(idat_data)
    """

    def __init__(
        self,
        file: str | Path | BinaryIO,
        *,
        mode: Literal["full", "intensity", "array_type"] = "full",
    ) -> None:
        """Reads and parses the IDAT file."""
        self.mode = mode
        self._file = file

        with get_file_object(file) as idat_file:
            self.file_size = _get_file_size(idat_file)
            self._parse_header(idat_file)
            self._parse_body(idat_file)

    def _parse_header(self, idat_file: BinaryIO) -> None:
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

    def _parse_body(self, idat_file: BinaryIO) -> None:
        seek = idat_file.seek

        seek(self.offsets[IdatSectionCode.NUM_SNPS_READ])
        self.n_snps_read = read_int(idat_file)

        if self.mode == "array_type":
            return

        seek(self.offsets[IdatSectionCode.ILLUMINA_ID])
        self.illumina_ids = read_array(idat_file, _D_INT32, self.n_snps_read)

        seek(self.offsets[IdatSectionCode.MEAN])
        self.probe_means = read_array(idat_file, _D_UINT16, self.n_snps_read)

        if self.mode == "intensity":
            return

        seek(self.offsets[IdatSectionCode.STD_DEV])
        self.std_dev = read_array(idat_file, _D_UINT16, self.n_snps_read)

        seek(self.offsets[IdatSectionCode.NUM_BEADS])
        self.n_beads = read_array(idat_file, _D_UINT8, self.n_snps_read)

        seek(self.offsets[IdatSectionCode.MID_BLOCK])
        n_mid_block = read_int(idat_file)
        self.mid_block = read_array(idat_file, _D_INT32, n_mid_block)

        seek(self.offsets[IdatSectionCode.RUN_INFO])
        runinfo_entry_count = read_int(idat_file)

        self.run_info = [
            [
                read_string(idat_file),  # run_time
                read_string(idat_file),  # block_type
                read_string(idat_file),  # block_pars
                read_string(idat_file),  # block_code
                read_string(idat_file),  # code_version
            ]
            for _ in range(runinfo_entry_count)
        ]

        seek(self.offsets[IdatSectionCode.RED_GREEN])
        self.red_green = read_int(idat_file)

        seek(self.offsets[IdatSectionCode.MOSTLY_NULL])
        self.mostly_null = read_string(idat_file)

        seek(self.offsets[IdatSectionCode.BARCODE])
        self.barcode = read_string(idat_file)

        seek(self.offsets[IdatSectionCode.CHIP_TYPE])
        self.chip_type = read_string(idat_file)

        seek(self.offsets[IdatSectionCode.MOSTLY_A])
        self.mostly_a = read_string(idat_file)

        seek(self.offsets[IdatSectionCode.UNKNOWN_1])
        self.unknown_1 = read_string(idat_file)

        seek(self.offsets[IdatSectionCode.UNKNOWN_2])
        self.unknown_2 = read_string(idat_file)

        seek(self.offsets[IdatSectionCode.UNKNOWN_3])
        self.unknown_3 = read_string(idat_file)

        seek(self.offsets[IdatSectionCode.UNKNOWN_4])
        self.unknown_4 = read_string(idat_file)

        seek(self.offsets[IdatSectionCode.UNKNOWN_5])
        self.unknown_5 = read_string(idat_file)

        seek(self.offsets[IdatSectionCode.UNKNOWN_6])
        self.unknown_6 = read_string(idat_file)

        seek(self.offsets[IdatSectionCode.UNKNOWN_7])
        self.unknown_7 = read_string(idat_file)

    def __repr__(self) -> str:
        with np.printoptions(edgeitems=2):
            result = (
                f"IdatParser(\n"
                f"    file_size: {self.file_size}\n"
                f"    num_fields: {self.num_fields}\n"
                f"    n_snps_read: {self.n_snps_read}\n"
                f"    illumina_ids: {self.illumina_ids!r}\n"
                f"    probe_means: {self.probe_means!r}\n"
            )

            if self.mode == "intensity":
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
