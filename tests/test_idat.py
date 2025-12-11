"""Pytest for IDAT parser."""

from typing import Any

import numpy as np
import numpy.testing as npt
import pytest

from mepylome.dtypes.idat import IdatParser
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
    test_data: dict[str, Any],
    gzipped: bool,
) -> None:
    """Check the data consistency between writer and parser."""
    tmp_idat = TempIdatFile(test_data, gzipped)
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
def test_idat_parser(test_data: dict[str, Any], gzipped: bool) -> None:
    """Tests IDAT parser."""
    _check_idat_parser(test_data, gzipped)
