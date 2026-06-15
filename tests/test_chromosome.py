"""Pytest for Chromosome enum."""

import pandas as pd
import pytest

from mepylome.dtypes.chromosome import Chromosome

# ---------------------------------------------------------------------------
# Basic enum membership and values
# ---------------------------------------------------------------------------


def test_chromosome_values() -> None:
    assert Chromosome.CHR1 == 1
    assert Chromosome.CHR22 == 22
    assert Chromosome.CHRX == 23
    assert Chromosome.CHRY == 24
    assert Chromosome.CHRM == 25
    assert Chromosome.INVALID == -1
    assert Chromosome.CHR0 == 0


# ---------------------------------------------------------------------------
# is_valid_chromosome
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "chrom, expected",
    [
        (Chromosome.CHR1, True),
        (Chromosome.CHR22, True),
        (Chromosome.CHRX, True),
        (Chromosome.CHRY, True),
        (Chromosome.CHR0, False),  # 0 is below CHR1
        (Chromosome.CHRM, False),  # mitochondrial is above CHRY
        (Chromosome.INVALID, False),
    ],
)
def test_is_valid_chromosome(chrom: Chromosome, expected: bool) -> None:
    assert Chromosome.is_valid_chromosome(chrom) == expected


# ---------------------------------------------------------------------------
# pd_from_string – numeric strings ("1" .. "22")
# ---------------------------------------------------------------------------


def test_pd_from_string_numeric() -> None:
    col = pd.Series(["1", "2", "22"])
    result = Chromosome.pd_from_string(col)
    assert list(result) == [1, 2, 22]


# ---------------------------------------------------------------------------
# pd_from_string – "chrN" prefix
# ---------------------------------------------------------------------------


def test_pd_from_string_chr_prefix() -> None:
    col = pd.Series(["chr1", "chr10", "chr22"])
    result = Chromosome.pd_from_string(col)
    assert list(result) == [1, 10, 22]


# ---------------------------------------------------------------------------
# pd_from_string – sex and mito chromosomes (all forms)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "label, expected_int",
    [
        ("X", Chromosome.CHRX),
        ("x", Chromosome.CHRX),
        ("chrX", Chromosome.CHRX),
        ("chrx", Chromosome.CHRX),
        ("Y", Chromosome.CHRY),
        ("y", Chromosome.CHRY),
        ("chrY", Chromosome.CHRY),
        ("chry", Chromosome.CHRY),
        ("M", Chromosome.CHRM),
        ("m", Chromosome.CHRM),
        ("chrM", Chromosome.CHRM),
        ("chrm", Chromosome.CHRM),
    ],
)
def test_pd_from_string_sex_mito(label: str, expected_int: int) -> None:
    col = pd.Series([label])
    result = Chromosome.pd_from_string(col)
    assert result.iloc[0] == int(expected_int)


# ---------------------------------------------------------------------------
# pd_from_string – unknown/invalid labels map to INVALID
# ---------------------------------------------------------------------------


def test_pd_from_string_invalid() -> None:
    col = pd.Series(["foo", "chr99", "MT", ""])
    result = Chromosome.pd_from_string(col)
    assert all(v == int(Chromosome.INVALID) for v in result)


# ---------------------------------------------------------------------------
# pd_to_string – autosomal
# ---------------------------------------------------------------------------


def test_pd_to_string_autosomal() -> None:
    col = pd.Series([int(Chromosome.CHR1), int(Chromosome.CHR22)])
    result = Chromosome.pd_to_string(col)
    assert list(result) == ["chr1", "chr22"]


# ---------------------------------------------------------------------------
# pd_to_string – sex and mito
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "chrom, expected_str",
    [
        (Chromosome.CHRX, "chrX"),
        (Chromosome.CHRY, "chrY"),
        (Chromosome.CHRM, "chrM"),
    ],
)
def test_pd_to_string_sex_mito(chrom: Chromosome, expected_str: str) -> None:
    col = pd.Series([int(chrom)])
    result = Chromosome.pd_to_string(col)
    assert result.iloc[0] == expected_str


# ---------------------------------------------------------------------------
# pd_to_string – unmapped values become "NaN"
# ---------------------------------------------------------------------------


def test_pd_to_string_unmapped() -> None:
    # INVALID (-1) and out-of-range integers (99) are not in the chrom_map
    # and should fall back to "NaN".  CHR0 IS mapped to "chr0", so excluded.
    col = pd.Series([int(Chromosome.INVALID), 99])
    result = Chromosome.pd_to_string(col)
    assert all(v == "NaN" for v in result)


def test_pd_to_string_chr0_mapped() -> None:
    """CHR0 (value 0) is present in the map and renders as 'chr0'."""
    col = pd.Series([int(Chromosome.CHR0)])
    result = Chromosome.pd_to_string(col)
    assert result.iloc[0] == "chr0"


# ---------------------------------------------------------------------------
# Round-trip: from_string → to_string for all autosomal chromosomes
# ---------------------------------------------------------------------------


def test_round_trip_autosomal() -> None:
    labels = [f"chr{i}" for i in range(1, 23)]
    col_in = pd.Series(labels)
    ints = Chromosome.pd_from_string(col_in)
    col_out = Chromosome.pd_to_string(ints)
    assert list(col_out) == labels


# ---------------------------------------------------------------------------
# Round-trip: from_string → to_string for sex chromosomes
# ---------------------------------------------------------------------------


def test_round_trip_sex() -> None:
    labels = ["chrX", "chrY"]
    col_in = pd.Series(labels)
    ints = Chromosome.pd_from_string(col_in)
    col_out = Chromosome.pd_to_string(ints)
    assert list(col_out) == labels


# ---------------------------------------------------------------------------
# Mixed series
# ---------------------------------------------------------------------------


def test_pd_from_string_mixed() -> None:
    col = pd.Series(["1", "chr2", "X", "chrY", "invalid"])
    result = Chromosome.pd_from_string(col)
    expected = [
        int(Chromosome.CHR1),
        int(Chromosome.CHR2),
        int(Chromosome.CHRX),
        int(Chromosome.CHRY),
        int(Chromosome.INVALID),
    ]
    assert list(result) == expected
