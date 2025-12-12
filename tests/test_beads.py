"""Pytest for IDAT preprocessing."""

from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest

from mepylome.dtypes import ArrayType, Manifest, MethylData, PrepType, RawData
from mepylome.tests.helpers import TempIdatFilePair, TempManifest

ENDING_GRN = "_Grn.idat"
ENDING_RED = "_Red.idat"
ENDING_GZ = ".gz"
ENDING_SUFFIXES = (
    ENDING_GRN,
    ENDING_RED,
    ENDING_GRN + ENDING_GZ,
    ENDING_RED + ENDING_GZ,
)

from mepylome.dtypes.beads import (
    idat_basepaths,
    idat_paths_from_basenames,
    is_valid_idat_basepath,
)


def create_idat_pair(base: Path, gz: bool = False) -> None:
    """Create a valid Grn/Red file pair."""
    if gz:
        (base.parent / (base.name + ENDING_GRN + ENDING_GZ)).touch()
        (base.parent / (base.name + ENDING_RED + ENDING_GZ)).touch()
    else:
        (base.parent / (base.name + ENDING_GRN)).touch()
        (base.parent / (base.name + ENDING_RED)).touch()


# ----------------------------------------------------------------------------
# Tests for is_valid_idat_basepath
# ----------------------------------------------------------------------------


def test_is_valid_basepath_true(tmp_path: Path) -> None:
    base = tmp_path / "sample"
    create_idat_pair(base, gz=False)
    assert is_valid_idat_basepath(base) is True


def test_is_valid_basepath_true_gz(tmp_path: Path) -> None:
    base = tmp_path / "sample"
    create_idat_pair(base, gz=True)
    assert is_valid_idat_basepath(base) is True


def test_is_valid_multiple_paths(tmp_path: Path) -> None:
    base1 = tmp_path / "a"
    base2 = tmp_path / "b"
    create_idat_pair(base1, gz=True)
    create_idat_pair(base2, gz=False)
    assert is_valid_idat_basepath([base1, base2]) is True


def test_is_valid_basepath_false(tmp_path: Path) -> None:
    base = tmp_path / "sample"
    (tmp_path / ("sample" + ENDING_GRN)).touch()
    assert is_valid_idat_basepath(base) is False


# ----------------------------------------------------------------------------
# Tests for idat_basepaths
# ----------------------------------------------------------------------------


def test_idat_basepaths_from_directory(tmp_path: Path) -> None:
    base1 = tmp_path / "x1"
    base2 = tmp_path / "x2"
    create_idat_pair(base1)
    create_idat_pair(base2, gz=True)
    result = idat_basepaths(tmp_path)
    assert base1 in result
    assert base2 in result
    assert len(result) == 2


def test_idat_basepaths_strip_suffix(tmp_path: Path) -> None:
    base = tmp_path / "sample"
    create_idat_pair(base)
    files = [
        tmp_path / ("sample" + ENDING_GRN),
        tmp_path / ("sample" + ENDING_RED),
    ]
    result = idat_basepaths(files)
    assert result == [base]


def test_idat_basepaths_only_valid(tmp_path: Path) -> None:
    base_valid = tmp_path / "valid"
    base_invalid = tmp_path / "invalid"
    create_idat_pair(base_valid)
    (tmp_path / ("invalid" + ENDING_GRN)).touch()
    result = idat_basepaths([base_valid, base_invalid], only_valid=True)
    assert result == [base_valid]


def test_idat_basepaths_deduplicates(tmp_path: Path) -> None:
    base = tmp_path / "sample"
    create_idat_pair(base)
    files = [
        tmp_path / ("sample" + ENDING_GRN),
        tmp_path / ("sample" + ENDING_RED),
    ]
    result = idat_basepaths(files)
    assert result == [base]


# -------------------------------------------------------------------------
# Tests for idat_paths_from_basenames
# -------------------------------------------------------------------------


def test_idat_paths_from_basenames_success(tmp_path: Path) -> None:
    base = tmp_path / "test"
    create_idat_pair(base)
    grn, red = idat_paths_from_basenames([base])
    assert grn[0] == base.with_name(base.name + ENDING_GRN)
    assert red[0] == base.with_name(base.name + ENDING_RED)


def test_idat_paths_from_basenames_gz(tmp_path: Path) -> None:
    base = tmp_path / "test"
    create_idat_pair(base, gz=True)
    grn, red = idat_paths_from_basenames([base])
    assert grn[0].suffix == ".gz"
    assert red[0].suffix == ".gz"


def test_idat_paths_from_basenames_missing_file_raises(tmp_path: Path) -> None:
    base = tmp_path / "missing"
    (tmp_path / ("missing" + ENDING_GRN)).touch()
    with pytest.raises(FileNotFoundError):
        idat_paths_from_basenames([base])


def test_idat_paths_multiple(tmp_path: Path) -> None:
    base1 = tmp_path / "a"
    base2 = tmp_path / "b"
    create_idat_pair(base1)
    create_idat_pair(base2)
    grn, red = idat_paths_from_basenames([base1, base2])
    assert len(grn) == 2
    assert len(red) == 2


# -------------------------------------------------------------------------
# Tests for MethylData
# -------------------------------------------------------------------------


def _test_raw_data(
    dirpath: Path,
    n_cpgs: int,
    manifest: Manifest,
    n_probes: int,
) -> None:
    id_list = sorted(
        (
            set(manifest.data_frame.AddressA_ID)
            | set(manifest.data_frame.AddressB_ID)
            | set(manifest.control_data_frame.Address_ID)
        )
        - {-1}
    )
    id_list = id_list[:n_cpgs]
    ids = np.full(n_cpgs, -1, dtype="<i4")
    ids[: len(id_list)] = np.array(id_list, dtype="<i4")
    start_value = id_list[-1] + 1
    ids[len(id_list) :] = np.arange(
        start_value,
        start_value + (n_cpgs - len(id_list)),
        dtype="<i4",
    )

    # Create and test two sets of data
    idat_pairs = []
    for index in range(n_probes):
        mean_grn = ((ids + index) % 23).astype(dtype="<u2")
        mean_red = ((ids + index) % 21).astype(dtype="<u2")
        test_grn = {
            "n_snps_read": n_cpgs,
            "illumina_ids": ids,
            "probe_means": mean_grn,
        }
        test_red = {
            "n_snps_read": n_cpgs,
            "illumina_ids": ids,
            "probe_means": mean_red,
        }
        idat_pairs.append(
            TempIdatFilePair(
                dirpath=dirpath, data_grn=test_grn, data_red=test_red
            )
        )

    if manifest.array_type == ArrayType.UNKNOWN:
        raw_data = RawData(
            [file.basepath for file in idat_pairs], manifest=manifest
        )
    else:
        raw_data = RawData([file.basepath for file in idat_pairs])

    assert str(raw_data) is not None
    assert str(raw_data.manifest) is not None

    npt.assert_array_equal(raw_data.ids, ids, err_msg="Mismatch in IDs array")
    npt.assert_array_equal(
        raw_data._grn,
        np.array([pair.data_grn["probe_means"] for pair in idat_pairs]),
        err_msg="Mismatch in Green channel array",
    )
    npt.assert_array_equal(
        raw_data._red,
        np.array([pair.data_red["probe_means"] for pair in idat_pairs]),
        err_msg="Mismatch in Red channel array",
    )
    assert raw_data.probes == [pair.basepath.name for pair in idat_pairs], (
        "Mismatch in probe name"
    )
    assert raw_data.array_type == manifest.array_type, "Mismatch in array type"

    if manifest.array_type == ArrayType.UNKNOWN:
        _test_methyl_data_raw(raw_data)
        _test_methyl_data_illumina(raw_data)
        _test_methyl_data_noob(raw_data)
        _test_methyl_data_swan(raw_data)
    else:
        prep_values: list[PrepType] = ["raw", "illumina", "swan", "noob"]

        for prep in prep_values:
            assert MethylData(raw_data, prep=prep) is not None


def _test_methyl_data_raw(raw_data: RawData) -> None:
    """Test MethylData with raw preparation."""
    methyl_data = MethylData(raw_data, prep="raw")

    expected_methyl = np.array(
        [
            [18, 1, 14, 0, 9, 17, 6, 14, 9, 20, 13, 17, 8, 13, 6, 19],
            [19, 2, 15, 1, 10, 18, 7, 15, 10, 21, 14, 18, 9, 14, 7, 20],
        ]
    )
    npt.assert_array_equal(methyl_data.methyl, expected_methyl)

    expected_unmethyl = np.array(
        [
            [0, 4, 21, 3, 17, 0, 13, 13, 9, 14, 15, 7, 1, 3, 16, 12],
            [1, 5, 22, 4, 18, 1, 14, 14, 10, 15, 16, 8, 2, 4, 17, 13],
        ]
    )
    npt.assert_array_equal(methyl_data.unmethyl, expected_unmethyl)


def _test_methyl_data_illumina(raw_data: RawData) -> None:
    """Test MethylData with illumina preparation."""
    methyl_data = MethylData(raw_data, prep="illumina")

    expected_methyl = np.array(
        [
            [1.46428571, 12.90740741, 10.62962963, 19.03571429],
            [2.61702128, 25.15909091, 20.96590909, 18.31914894],
        ]
    )
    npt.assert_almost_equal(
        methyl_data.methyl[:, [1, 5, 7, 13]], expected_methyl
    )

    expected_unmethyl = np.array(
        [
            [3.03703704, 0.0, 9.87037037, 2.27777778],
            [6.98863636, 1.39772727, 19.56818182, 5.59090909],
        ]
    )
    npt.assert_almost_equal(
        methyl_data.unmethyl[:, [1, 5, 7, 13]], expected_unmethyl
    )


def _test_methyl_data_noob(raw_data: RawData) -> None:
    """Test MethylData with noob preparation."""
    methyl_data = MethylData(raw_data, prep="noob")

    expected_methyl = np.array(
        [
            [17.68105245, 19.39793999, 18.57913213, 19.78231226],
            [17.95140526, 21.91645282, 20.99133582, 20.4679614],
        ]
    )
    npt.assert_almost_equal(
        methyl_data.methyl[:, [1, 5, 7, 13]], expected_methyl
    )

    expected_unmethyl = np.array(
        [
            [16.73134405, 16.36054317, 18.3387984, 16.60061064],
            [18.90364196, 18.36008626, 20.7197986, 18.75593491],
        ]
    )
    npt.assert_almost_equal(
        methyl_data.unmethyl[:, [1, 5, 7, 13]], expected_unmethyl
    )


def _test_methyl_data_swan(raw_data: RawData) -> None:
    """Test MethylData with swan preparation."""
    methyl_data = MethylData(raw_data, prep="swan", seed=1234)

    expected_methyl = np.array(
        [[5.0, 18.5, 13.5, 13.5], [6.0, 19.5, 14.5, 14.5]]
    )
    npt.assert_almost_equal(
        methyl_data.methyl[:, [1, 5, 7, 13]], expected_methyl
    )

    expected_unmethyl = np.array(
        [[8.42857143, 10.0, 13.0, 5.0], [9.42857143, 11.0, 14.0, 6.0]]
    )
    npt.assert_almost_equal(
        methyl_data.unmethyl[:, [1, 5, 7, 13]], expected_unmethyl
    )


def test_raw_data(tmp_path: Path) -> None:
    """Main test entry for IDAT processing."""
    tmp_manifest = TempManifest(dirpath=tmp_path)
    manifest = Manifest(raw_path=tmp_manifest.path)
    manifest.array_type = ArrayType.UNKNOWN

    _test_raw_data(tmp_path, 54, manifest, 2)
    # _test_raw_data(tmp_path, 622500, Manifest("450k"), 1)
    # _test_raw_data(tmp_path, 622500, Manifest("450k"), 4)
    # _test_raw_data(tmp_path, 1051000, Manifest("epic"), 3)
    # _test_raw_data(tmp_path, 1104000, Manifest("epicv2"), 2)

    # Clean up
    manifest.proc_path.unlink()
    manifest.ctrl_path.unlink()
    manifest._pickle_path.unlink()
