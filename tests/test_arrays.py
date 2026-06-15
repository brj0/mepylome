"""Pytest for ArrayType."""

from pathlib import Path

import pytest

from mepylome.dtypes.arrays import ArrayType, _find_valid_path
from mepylome.tests.helpers import TempIdatFile

# ---------------------------------------------------------------------------
# Tests for ArrayType enum values and __str__
# ---------------------------------------------------------------------------


def test_array_type_str_values() -> None:
    """Each ArrayType str matches its value."""
    assert str(ArrayType.ILLUMINA_450K) == "450k"
    assert str(ArrayType.ILLUMINA_EPIC) == "epic"
    assert str(ArrayType.ILLUMINA_EPIC_V2) == "epicv2"
    assert str(ArrayType.ILLUMINA_27K) == "27k"
    assert str(ArrayType.ILLUMINA_MOUSE) == "mouse"
    assert str(ArrayType.ILLUMINA_MSA48) == "msa48"
    assert str(ArrayType.HORVATH_MAMMAL_40) == "mammal40"
    assert str(ArrayType.UNKNOWN) == "unknown"


# ---------------------------------------------------------------------------
# Tests for ArrayType.from_probe_count
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "probe_count, expected",
    [
        # 450k boundaries
        (622000, ArrayType.ILLUMINA_450K),
        (622500, ArrayType.ILLUMINA_450K),
        (623000, ArrayType.ILLUMINA_450K),
        # EPIC (current range)
        (1050000, ArrayType.ILLUMINA_EPIC),
        (1051500, ArrayType.ILLUMINA_EPIC),
        (1053000, ArrayType.ILLUMINA_EPIC),
        # EPIC (old range)
        (1032000, ArrayType.ILLUMINA_EPIC),
        (1032500, ArrayType.ILLUMINA_EPIC),
        (1033000, ArrayType.ILLUMINA_EPIC),
        # EPICv2
        (1100000, ArrayType.ILLUMINA_EPIC_V2),
        (1104000, ArrayType.ILLUMINA_EPIC_V2),
        (1108000, ArrayType.ILLUMINA_EPIC_V2),
        # MSA48
        (384400, ArrayType.ILLUMINA_MSA48),
        (384500, ArrayType.ILLUMINA_MSA48),
        (384600, ArrayType.ILLUMINA_MSA48),
        # 27k
        (55200, ArrayType.ILLUMINA_27K),
        (55300, ArrayType.ILLUMINA_27K),
        (55400, ArrayType.ILLUMINA_27K),
        # Mouse
        (315000, ArrayType.ILLUMINA_MOUSE),
        (340000, ArrayType.ILLUMINA_MOUSE),
        (362000, ArrayType.ILLUMINA_MOUSE),
        # Mammal40
        (41000, ArrayType.HORVATH_MAMMAL_40),
        (41050, ArrayType.HORVATH_MAMMAL_40),
        (41100, ArrayType.HORVATH_MAMMAL_40),
        # UNKNOWN – values outside any known range
        (0, ArrayType.UNKNOWN),
        (100, ArrayType.UNKNOWN),
        (999999, ArrayType.UNKNOWN),
        (2000000, ArrayType.UNKNOWN),
    ],
    ids=lambda x: str(x),
)
def test_from_probe_count(probe_count: int, expected: ArrayType) -> None:
    assert ArrayType.from_probe_count(probe_count) == expected


# ---------------------------------------------------------------------------
# Tests for _find_valid_path
# ---------------------------------------------------------------------------


def test_find_valid_path_direct(tmp_path: Path) -> None:
    """Returns path if the file itself exists."""
    p = tmp_path / "file.idat"
    p.touch()
    assert _find_valid_path(p) == p


def test_find_valid_path_grn(tmp_path: Path) -> None:
    """Falls back to _Grn.idat when base does not exist."""
    base = tmp_path / "sample"
    grn = tmp_path / "sample_Grn.idat"
    grn.touch()
    assert _find_valid_path(base) == grn


def test_find_valid_path_grn_gz(tmp_path: Path) -> None:
    """Falls back to _Grn.idat.gz when nothing else exists."""
    base = tmp_path / "sample"
    grn_gz = tmp_path / "sample_Grn.idat.gz"
    grn_gz.touch()
    assert _find_valid_path(base) == grn_gz


def test_find_valid_path_missing_raises(tmp_path: Path) -> None:
    """Raises ValueError when no valid file can be found."""
    with pytest.raises(ValueError, match="No valid file found"):
        _find_valid_path(tmp_path / "nonexistent")


# ---------------------------------------------------------------------------
# Tests for ArrayType.from_idat
# ---------------------------------------------------------------------------


def test_from_idat_direct_path(tmp_path: Path) -> None:
    """from_idat infers UNKNOWN for a small synthetic IDAT."""
    import numpy as np

    idat = TempIdatFile(
        tmp_path,
        {
            "n_snps_read": 10,
            "illumina_ids": np.arange(10, dtype="<i4"),
            "probe_means": np.zeros(10, dtype="<u2"),
        },
    )
    result = ArrayType.from_idat(idat.path)
    assert result == ArrayType.UNKNOWN


def test_from_idat_grn_suffix(tmp_path: Path) -> None:
    """from_idat resolves _Grn.idat when given the basepath."""
    import numpy as np

    n = 10
    idat = TempIdatFile(
        tmp_path,
        {
            "n_snps_read": n,
            "illumina_ids": np.arange(n, dtype="<i4"),
            "probe_means": np.zeros(n, dtype="<u2"),
        },
    )
    # Rename to look like a _Grn.idat file so from_idat can find it via base
    grn_path = idat.path.with_name(idat.path.stem + "_Grn.idat")
    idat.path.rename(grn_path)

    base = grn_path.with_name(grn_path.name.replace("_Grn.idat", ""))
    result = ArrayType.from_idat(base)
    assert result == ArrayType.UNKNOWN
