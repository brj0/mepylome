"""Pytest for IDAT parser."""

import gzip
import io
import struct
from pathlib import Path
from typing import Any

import numpy as np
import numpy.testing as npt
import pytest

from mepylome.dtypes.idat import (
    IdatParser,
    _get_file_size,
    get_file_object,
    read_array,
    read_string,
)
from mepylome.tests.helpers import TempIdatFile

TEST_DATA_DICT: dict[str, dict[str, Any]] = {
    "EmptyData": {},
    "Ids_Short": {"illumina_ids": np.array([1, 10, 1000, 10000], dtype="<i4")},
    "Ids_Large_ZeroStart": {"illumina_ids": np.arange(0, 10000, dtype="<i4")},
    "Ids_Large_Offset": {"illumina_ids": np.arange(100, 10000, dtype="<i4")},
    "FullData_Comprehensive": {
        "n_snps_read": 20,
        "illumina_ids": np.arange(1000, 1000 + 20, dtype="<i4"),
        "probe_means": np.arange(100, 100 + 20, dtype="<u2"),
        "std_dev": np.arange(2000, 2000 + 20, dtype="<u2"),
        "n_beads": np.arange(3, 3 + 20, dtype="<u1"),
        "mid_block": np.arange(400000, 400000 + 20, dtype="<i4"),
        "run_info": [
            ["time_0", "type_0", "pars_0", "code_0", "vers_0"],
            ["time_1", "type_1", "pars_1", "code_1", "vers_1"],
        ],
        "red_green": 1,
        "mostly_null": "asdf",
        "barcode": "ghij",
        "chip_type": "klmn",
        "mostly_a": "opqr",
        "unknown_1": "stuv",
        "unknown_2": "wxxz",
        "unknown_3": "1234",
        "unknown_4": "5678",
        "unknown_5": "90.,",
        "unknown_6": "<>-:",
        "unknown_7": "!=|#",
    },
}


def _check_idat_parser(
    dirpath: Path,
    test_data: dict[str, Any],
    gzipped: bool,
) -> None:
    """Check the data consistency between writer and parser."""
    tmp_idat = TempIdatFile(dirpath, test_data, gzipped)
    idat_file = IdatParser(tmp_idat.path)

    for key in tmp_idat.data:
        expected_data = tmp_idat.data[key]
        parsed_data = getattr(idat_file, key)

        if isinstance(expected_data, np.ndarray):
            npt.assert_array_equal(
                expected_data,
                parsed_data,
                err_msg=(
                    f"Arrays for key '{key}' do not match (gzipped={gzipped})."
                ),
            )
        else:
            assert expected_data == parsed_data, (
                f"Values for key '{key}' do not match (gzipped={gzipped}). "
                f"Expected: {expected_data}, Got: {parsed_data}"
            )


@pytest.mark.parametrize(
    "gzipped",
    [False, True],
    ids=["unzipped", "gzipped"],
)
@pytest.mark.parametrize(
    "test_data",
    list(TEST_DATA_DICT.values()),
    ids=list(TEST_DATA_DICT.keys()),
)
def test_idat_parser(
    test_data: dict[str, Any], gzipped: bool, tmp_path: Path
) -> None:
    """Tests IDAT parser."""
    _check_idat_parser(tmp_path, test_data, gzipped)


# ---------------------------------------------------------------------------
# Helpers – minimal valid IDAT binary builder
# ---------------------------------------------------------------------------


def _pack_string(s: str) -> bytes:
    """Encode a string with a single-byte length prefix."""
    b = s.encode("utf-8")
    return struct.pack("B", len(b)) + b


def _build_idat(
    chip_type: str = "EPIC",
    barcode: str = "SentrixBC",
    n_snps: int = 3,
) -> bytes:
    """Build a minimal but fully valid IDAT v3 binary in memory."""
    body = io.BytesIO()
    offsets: dict[int, int] = {}

    # NUM_SNPS_READ (1000)
    offsets[1000] = body.tell()
    body.write(struct.pack("<i", n_snps))

    # ILLUMINA_ID (102)
    offsets[102] = body.tell()
    body.write(np.array([1001, 1002, 1003], dtype="<i4").tobytes())

    # MEAN (104)
    offsets[104] = body.tell()
    body.write(np.array([100, 200, 300], dtype="<u2").tobytes())

    # STD_DEV (103)
    offsets[103] = body.tell()
    body.write(np.array([10, 20, 30], dtype="<u2").tobytes())

    # NUM_BEADS (107)
    offsets[107] = body.tell()
    body.write(np.array([5, 6, 7], dtype="<u1").tobytes())

    # MID_BLOCK (200)
    offsets[200] = body.tell()
    body.write(struct.pack("<i", 2))
    body.write(np.array([9, 8], dtype="<i4").tobytes())

    # RUN_INFO (300) – 1 entry × 5 strings
    offsets[300] = body.tell()
    body.write(struct.pack("<i", 1))
    for s in ["2024-01-01", "decode", "pars", "code", "1.0"]:
        body.write(_pack_string(s))

    # RED_GREEN (400)
    offsets[400] = body.tell()
    body.write(struct.pack("<i", 0))

    # String sections
    for code, val in [
        (401, ""),
        (402, barcode),
        (403, chip_type),
        (404, "AAAA"),
        (405, ""),
        (406, ""),
        (407, ""),
        (408, ""),
        (409, ""),
        (410, ""),
        (510, ""),
    ]:
        offsets[code] = body.tell()
        body.write(_pack_string(val))

    # Header: "IDAT" + version (long) + num_fields + (short+long) per field
    header = io.BytesIO()
    header.write(b"IDAT")
    header.write(struct.pack("<q", 3))
    header.write(struct.pack("<i", len(offsets)))
    header_size = 4 + 8 + 4 + len(offsets) * 10
    for code, offset in offsets.items():
        header.write(struct.pack("<H", code))
        header.write(struct.pack("<q", offset + header_size))

    return header.getvalue() + body.getvalue()


# ---------------------------------------------------------------------------
# read_string – multi-byte length prefix (lines 45-48)
# ---------------------------------------------------------------------------


def test_read_string_multi_byte_length() -> None:
    """read_string handles strings whose length requires >1 continuation byte.

    A length >=128 is encoded as two bytes where the first byte has bit 7 set.
    """
    # Encode length 200 as two bytes: 0x80 | (200 & 0x7F) = 0xC8, 0x01
    payload = b"x" * 200
    first_byte = (200 & 0x7F) | 0x80  # = 0xC8  (sets the "more bytes" flag)
    second_byte = 200 >> 7  # = 1
    encoded = bytes([first_byte, second_byte]) + payload
    buf = io.BytesIO(encoded)
    assert read_string(buf) == "x" * 200


# ---------------------------------------------------------------------------
# read_array – premature EOF (lines 62-63)
# ---------------------------------------------------------------------------


def test_read_array_eof_raises() -> None:
    """Raises EOFError when the file ends before all data is read."""
    # Ask for 4 int32s (16 bytes) but supply only 4 bytes
    buf = io.BytesIO(b"\x00" * 4)
    with pytest.raises(EOFError, match="End of file"):
        read_array(buf, "<i4", 4)


# ---------------------------------------------------------------------------
# _get_file_size – BytesIO branch (lines 99-104) and unknown type (106-107)
# ---------------------------------------------------------------------------


def test_get_file_size_bytesio() -> None:
    """_get_file_size returns the correct size for a BytesIO object."""
    data = b"hello world"
    buf = io.BytesIO(data)
    buf.seek(5)  # advance position to confirm it is restored
    assert _get_file_size(buf) == len(data)
    assert buf.tell() == 5  # position preserved


def test_get_file_size_unknown_type_raises() -> None:
    """_get_file_size raises ValueError for unsupported file-like objects."""

    class _Fake:
        pass

    with pytest.raises(ValueError, match="Unknown file format"):
        _get_file_size(_Fake())  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# get_file_object – already-open BytesIO (line 127) and bad type (line 130)
# ---------------------------------------------------------------------------


def test_get_file_object_with_bytesio() -> None:
    """get_file_object yields the BytesIO as-is without closing it."""
    buf = io.BytesIO(b"data")
    with get_file_object(buf) as f:
        assert f is buf
    # The BytesIO must still be readable after the context manager exits
    buf.seek(0)
    assert buf.read() == b"data"


def test_get_file_object_bad_type_raises() -> None:
    """Faises TypeError for anything that is not a path or file-like object."""
    with (
        pytest.raises(TypeError, match="Expected file-like object"),
        get_file_object(42),  # type: ignore[arg-type]
    ):
        pass


# ---------------------------------------------------------------------------
# IdatParser – invalid file ID / wrong version (lines 178-182, 188-192)
# ---------------------------------------------------------------------------


def _make_idat_header(
    file_id: bytes = b"IDAT",
    version: int = 3,
    num_fields: int = 0,
) -> bytes:
    """Return a bare header with no section offsets."""
    return file_id + struct.pack("<q", version) + struct.pack("<i", num_fields)


def test_idatparser_invalid_file_id_raises() -> None:
    """IdatParser raises when the file does not start with 'IDAT'."""
    bad = _make_idat_header(file_id=b"XXXX")
    with pytest.raises(ValueError, match="not a valid IDAT file"):
        IdatParser(io.BytesIO(bad))


def test_idatparser_wrong_version_raises() -> None:
    """IdatParser raises ValueError for IDAT files with version != 3."""
    bad = _make_idat_header(version=2)
    with pytest.raises(ValueError, match="version 3 IDAT file"):
        IdatParser(io.BytesIO(bad))


# ---------------------------------------------------------------------------
# IdatParser – full parse from BytesIO (covers file_size via BytesIO branch)
# ---------------------------------------------------------------------------


@pytest.fixture
def idat_bytes() -> bytes:
    return _build_idat()


@pytest.fixture
def idat_parser(idat_bytes: bytes) -> IdatParser:
    return IdatParser(io.BytesIO(idat_bytes))


def test_idatparser_parses_n_snps(idat_parser: IdatParser) -> None:
    assert idat_parser.n_snps_read == 3


def test_idatparser_parses_illumina_ids(idat_parser: IdatParser) -> None:
    np.testing.assert_array_equal(idat_parser.illumina_ids, [1001, 1002, 1003])


def test_idatparser_parses_probe_means(idat_parser: IdatParser) -> None:
    np.testing.assert_array_equal(idat_parser.probe_means, [100, 200, 300])


def test_idatparser_parses_chip_type(idat_parser: IdatParser) -> None:
    assert idat_parser.chip_type == "EPIC"


def test_idatparser_parses_barcode(idat_parser: IdatParser) -> None:
    assert idat_parser.barcode == "SentrixBC"


def test_idatparser_file_size_nonzero(idat_parser: IdatParser) -> None:
    assert idat_parser.file_size > 0


# ---------------------------------------------------------------------------
# IdatParser – intensity_only=True (lines 188-192 early return in _parse_body)
# ---------------------------------------------------------------------------


def test_idatparser_intensity_only_skips_extra_fields(
    idat_bytes: bytes,
) -> None:
    """intensity_only=True parses ids and means but not std_dev/beads/etc."""
    parser = IdatParser(io.BytesIO(idat_bytes), intensity_only=True)
    np.testing.assert_array_equal(parser.probe_means, [100, 200, 300])
    assert not hasattr(parser, "std_dev")
    assert not hasattr(parser, "n_beads")
    assert not hasattr(parser, "chip_type")


# ---------------------------------------------------------------------------
# IdatParser – array_type_only=True
# ---------------------------------------------------------------------------


def test_idatparser_array_type_only_reads_n_snps_only(
    idat_bytes: bytes,
) -> None:
    """array_type_only=True reads only n_snps_read and returns early."""
    parser = IdatParser(io.BytesIO(idat_bytes), array_type_only=True)
    assert parser.n_snps_read == 3
    assert not hasattr(parser, "illumina_ids")
    assert not hasattr(parser, "probe_means")


# ---------------------------------------------------------------------------
# IdatParser – plain file and gzipped file on disk
# ---------------------------------------------------------------------------


def test_idatparser_reads_plain_file(
    tmp_path: Path, idat_bytes: bytes
) -> None:
    """IdatParser accepts a plain path string."""
    idat_file = tmp_path / "sample_Grn.idat"
    idat_file.write_bytes(idat_bytes)
    parser = IdatParser(str(idat_file))
    assert parser.chip_type == "EPIC"


def test_idatparser_reads_gzipped_file(
    tmp_path: Path, idat_bytes: bytes
) -> None:
    """IdatParser transparently decompresses .idat.gz files."""
    gz_file = tmp_path / "sample_Grn.idat.gz"
    with gzip.open(gz_file, "wb") as f:
        f.write(idat_bytes)
    parser = IdatParser(gz_file)
    assert parser.chip_type == "EPIC"
    assert parser.n_snps_read == 3


# ---------------------------------------------------------------------------
# IdatParser.__repr__ – full repr (lines 280-293) and intensity_only branch
# ---------------------------------------------------------------------------


def test_repr_full(idat_parser: IdatParser) -> None:
    """__repr__ returns a multi-line string containing all field names."""
    r = repr(idat_parser)
    for field in (
        "file_size",
        "num_fields",
        "n_snps_read",
        "illumina_ids",
        "probe_means",
        "std_dev",
        "n_beads",
        "mid_block",
        "red_green",
        "barcode",
        "chip_type",
    ):
        assert field in r, f"Expected '{field}' in repr"


def test_repr_intensity_only(idat_bytes: bytes) -> None:
    """With intensity_only=True returns the short form ending in ')'."""
    parser = IdatParser(io.BytesIO(idat_bytes), intensity_only=True)
    r = repr(parser)
    assert r.endswith(")")
    assert "std_dev" not in r
    assert "probe_means" in r
