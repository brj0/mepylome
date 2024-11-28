"""Writes idat files to disk or to buffer."""

import io

import numpy as np

from mepylome.dtypes.idat import (
    DEFAULT_IDAT_FILE_ID,
    DEFAULT_IDAT_VERSION,
    IdatSectionCode,
)


def write_byte(outfile, value):
    outfile.write(value.to_bytes(1, byteorder="little", signed=False))


def write_short(outfile, value):
    outfile.write(value.to_bytes(2, byteorder="little", signed=False))


def write_int(outfile, value):
    outfile.write(value.to_bytes(4, byteorder="little", signed=True))


def write_long(outfile, value):
    outfile.write(value.to_bytes(8, byteorder="little", signed=True))


def write_char(outfile, value):
    outfile.write(value.encode("utf-8"))


def write_string(outfile, value):
    encoded_value = value.encode("utf-8")
    num_chars = len(encoded_value)
    num_bytes = num_chars
    while num_chars > 127:
        num_bytes = (num_chars & 0x7F) | 0x80
        num_chars >>= 7
        write_byte(outfile, num_bytes)
    write_byte(outfile, num_chars)
    outfile.write(encoded_value)


def write_array(outfile, array):
    outfile.write(array.tobytes())


class IdatWriter:
    """Writes data to an IDAT file with dummy values for missing data."""

    def __init__(self, file=None, data=None):
        """Initializes and writes data to the IDAT file or buffer."""
        self.file = file
        self.data = data
        self.buffer = None
        if self.file:
            with open(self.file, "wb") as outfile:
                self._write_header(outfile)
                self._write_body(outfile)
        else:
            self.buffer = io.BytesIO()
            self._write_header(self.buffer)
            self._write_body(self.buffer)
            self.buffer.seek(0)  # Rewind buffer to the beginning

    def _write_header(self, outfile):
        # Write IDAT header
        write_char(outfile, DEFAULT_IDAT_FILE_ID)
        write_long(outfile, DEFAULT_IDAT_VERSION)
        num_fields = len(IdatSectionCode)
        # Write number of fields
        write_int(outfile, num_fields)
        self._offsets_offset = outfile.tell()
        for _ in IdatSectionCode:
            write_short(outfile, 0)
            write_long(outfile, 0)

    def _get_default(self, n_snps_read):
        return {
            "n_snps_read": n_snps_read,
            "illumina_ids": np.arange(0, n_snps_read, dtype="<i4"),
            "probe_means": np.arange(10, 10 + n_snps_read, dtype="<u2"),
            "std_dev": np.arange(20, 20 + n_snps_read, dtype="<u2"),
            "n_beads": np.arange(30, 30 + n_snps_read, dtype="<u1"),
            "mid_block": np.arange(40, 40 + n_snps_read, dtype="<i4"),
            "run_info": [
                [
                    "run_time_0",
                    "block_type_0",
                    "block_pars_0",
                    "block_code_0",
                    "code_version_0",
                ],
                [
                    "run_time_1",
                    "block_type_1",
                    "block_pars_1",
                    "block_code_1",
                    "code_version_1",
                ],
            ],
            "red_green": 0,
            "mostly_null": "mostly_null",
            "barcode": "barcode",
            "chip_type": "chip_type",
            "mostly_a": "mostly_a",
            "unknown_1": "unknown_1",
            "unknown_2": "unknown_2",
            "unknown_3": "unknown_3",
            "unknown_4": "unknown_4",
            "unknown_5": "unknown_5",
            "unknown_6": "unknown_6",
            "unknown_7": "unknown_7",
        }

    def _write_body(self, outfile):
        # Ensure data is available or use defaults
        illumina_ids = self.data.get("illumina_ids", None)
        n_snps_read = 10 if illumina_ids is None else len(illumina_ids)
        default = self._get_default(n_snps_read)

        self.data = {
            **self.data,
            **{k: v for k, v in default.items() if k not in self.data},
        }

        def check_type(key, dtype):
            if self.data[key].dtype != dtype:
                expected = np.dtype(dtype)
                msg = f"Invalid type: {key} must be of type {expected}."
                raise TypeError(msg)

        check_type("illumina_ids", "<i4")
        check_type("probe_means", "<u2")
        check_type("std_dev", "<u2")
        check_type("n_beads", "<u1")
        check_type("mid_block", "<i4")

        def get_data(key):
            return self.data[key]

        offsets = {}

        # Write NUM_SNPS_READ
        offsets[IdatSectionCode.NUM_SNPS_READ] = outfile.tell()
        write_int(outfile, n_snps_read)

        # Write ILLUMINA_ID
        offsets[IdatSectionCode.ILLUMINA_ID] = outfile.tell()
        write_array(outfile, get_data("illumina_ids"))

        # Write MEAN
        offsets[IdatSectionCode.MEAN] = outfile.tell()
        write_array(outfile, get_data("probe_means"))

        # Write STD_DEV
        offsets[IdatSectionCode.STD_DEV] = outfile.tell()
        write_array(outfile, get_data("std_dev"))

        # Write NUM_BEADS
        offsets[IdatSectionCode.NUM_BEADS] = outfile.tell()
        write_array(outfile, get_data("n_beads"))

        # Write MID_BLOCK
        offsets[IdatSectionCode.MID_BLOCK] = outfile.tell()
        write_int(outfile, len(get_data("mid_block")))
        write_array(outfile, get_data("mid_block"))

        # Write RUN_INFO
        offsets[IdatSectionCode.RUN_INFO] = outfile.tell()
        run_info = get_data("run_info")
        write_int(outfile, len(run_info))
        for entry in run_info:
            for field in entry:
                write_string(outfile, field)

        # Write RED_GREEN
        offsets[IdatSectionCode.RED_GREEN] = outfile.tell()
        write_int(outfile, get_data("red_green"))

        # Write MOSTLY_NULL
        offsets[IdatSectionCode.MOSTLY_NULL] = outfile.tell()
        write_string(outfile, get_data("mostly_null"))

        # Write BARCODE
        offsets[IdatSectionCode.BARCODE] = outfile.tell()
        write_string(outfile, get_data("barcode"))

        # Write CHIP_TYPE
        offsets[IdatSectionCode.CHIP_TYPE] = outfile.tell()
        write_string(outfile, get_data("chip_type"))

        # Write MOSTLY_A
        offsets[IdatSectionCode.MOSTLY_A] = outfile.tell()
        write_string(outfile, get_data("mostly_a"))

        # Write UNKNOWN_1
        offsets[IdatSectionCode.UNKNOWN_1] = outfile.tell()
        write_string(outfile, get_data("unknown_1"))

        # Write UNKNOWN_2
        offsets[IdatSectionCode.UNKNOWN_2] = outfile.tell()
        write_string(outfile, get_data("unknown_2"))

        # Write UNKNOWN_3
        offsets[IdatSectionCode.UNKNOWN_3] = outfile.tell()
        write_string(outfile, get_data("unknown_3"))

        # Write UNKNOWN_4
        offsets[IdatSectionCode.UNKNOWN_4] = outfile.tell()
        write_string(outfile, get_data("unknown_4"))

        # Write UNKNOWN_5
        offsets[IdatSectionCode.UNKNOWN_5] = outfile.tell()
        write_string(outfile, get_data("unknown_5"))

        # Write UNKNOWN_6
        offsets[IdatSectionCode.UNKNOWN_6] = outfile.tell()
        write_string(outfile, get_data("unknown_6"))

        # Write UNKNOWN_7
        offsets[IdatSectionCode.UNKNOWN_7] = outfile.tell()
        write_string(outfile, get_data("unknown_7"))

        # Write offsets
        outfile.seek(self._offsets_offset)
        for code in IdatSectionCode:
            write_short(outfile, code.value)
            write_long(outfile, offsets[code])


def write_idat(data=None, file=None):
    writer = IdatWriter(file=file, data=data)
    if file is None:
        return writer.buffer
