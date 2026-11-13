"""Tests for SNP beta values (genotype fingerprint) from RawData."""

import uuid
from collections.abc import Callable, Generator
from pathlib import Path

import numpy as np
import pytest

from mepylome.dtypes import ArrayType, Manifest, MethylData, RawData
from mepylome.dtypes.beads import _compute_snp_betas, _snp_indices
from mepylome.tests.helpers import TEST_MANIFEST_CSV, TempIdatFilePair

OFFSET = 100

# SNP probes of the test manifest (see mepylome.tests.helpers):
# rs10796216: Type I, Red,   A=14622465, B=41635319
# rs715359:   Type I, Green, A=18796328, B=48710462
RS_RED_A, RS_RED_B = 14622465, 41635319
RS_GRN_A, RS_GRN_B = 18796328, 48710462

# The test manifest has no Type II SNP probe, so one is added.
RS_T2_A = 12345678
TYPE_2_SNP_ROW = b"rs_type2,rs_type2,12345678,ACGTACGTAC,,,II,,,,\n"
MANIFEST_WITH_TYPE_2_SNP = TEST_MANIFEST_CSV.replace(
    b"[Controls]", TYPE_2_SNP_ROW + b"[Controls]", 1
)
MANIFEST_WITHOUT_SNPS = b"\n".join(
    line
    for line in TEST_MANIFEST_CSV.split(b"\n")
    if not line.startswith(b"rs")
)


@pytest.fixture(autouse=True)
def clear_snp_cache() -> Generator[None, None, None]:
    """Avoid stale cache entries between tests."""
    _snp_indices._cache.clear()
    yield
    _snp_indices._cache.clear()


@pytest.fixture()
def make_manifest(
    tmp_path: Path,
) -> Generator[Callable[[bytes], Manifest], None, None]:
    """Creates a Manifest from CSV content and cleans up its cache files."""
    created: list[Manifest] = []

    def _make(csv: bytes) -> Manifest:
        path = tmp_path / f"tmp_manifest_{uuid.uuid4()}.csv"
        path.write_bytes(csv)
        manifest = Manifest(raw_path=path)
        manifest.array_type = ArrayType.UNKNOWN
        created.append(manifest)
        return manifest

    yield _make
    for manifest in created:
        manifest.proc_path.unlink(missing_ok=True)
        manifest.ctrl_path.unlink(missing_ok=True)
        manifest._pickle_path.unlink(missing_ok=True)


@pytest.fixture()
def tmp_manifest(
    make_manifest: Callable[[bytes], Manifest],
) -> Manifest:
    return make_manifest(TEST_MANIFEST_CSV)


# ---------------------------------------------------------------------------
# _compute_snp_betas (covers Type I red/green and Type II SNP probes)
# ---------------------------------------------------------------------------


def _channel(
    addresses: np.ndarray, values: dict[int, int], dtype: str
) -> np.ndarray:
    arr = np.zeros((1, len(addresses)), dtype=dtype)
    for address, value in values.items():
        arr[0, np.flatnonzero(addresses == address)[0]] = value
    return arr


SNP_ADDRESSES = np.array(
    [RS_RED_A, RS_RED_B, RS_GRN_A, RS_GRN_B, RS_T2_A], dtype="int32"
)


def test_snp_betas_all_probe_types(
    make_manifest: Callable[[bytes], Manifest],
) -> None:
    manifest = make_manifest(MANIFEST_WITH_TYPE_2_SNP)
    # Type I uses A (unmethylated) and B (methylated) of its own channel.
    # Type II: green is methylated, red is unmethylated.
    green = _channel(
        SNP_ADDRESSES,
        {RS_RED_A: 7, RS_RED_B: 7, RS_GRN_A: 50, RS_GRN_B: 850, RS_T2_A: 300},
        "<u2",
    )
    red = _channel(
        SNP_ADDRESSES,
        {RS_RED_A: 100, RS_RED_B: 900, RS_GRN_A: 7, RS_GRN_B: 7, RS_T2_A: 100},
        "<u2",
    )

    result = _compute_snp_betas(
        green, red, manifest, SNP_ADDRESSES, ["sample"], OFFSET
    )

    # Only SNP probes, no cg probes.
    assert set(result.index) == {"rs10796216", "rs715359", "rs_type2"}
    assert result.index.name == "IlmnID"
    assert list(result.columns) == ["sample"]
    assert result.loc["rs10796216", "sample"] == pytest.approx(
        900 / (900 + 100 + OFFSET)
    )
    assert result.loc["rs715359", "sample"] == pytest.approx(
        850 / (850 + 50 + OFFSET)
    )
    assert result.loc["rs_type2", "sample"] == pytest.approx(
        300 / (300 + 100 + OFFSET)
    )


def test_snp_betas_do_not_overflow_for_uint16_intensities(
    make_manifest: Callable[[bytes], Manifest],
) -> None:
    manifest = make_manifest(MANIFEST_WITH_TYPE_2_SNP)
    green = _channel(SNP_ADDRESSES, {RS_T2_A: 40000}, "<u2")
    red = _channel(SNP_ADDRESSES, {RS_T2_A: 40000}, "<u2")

    result = _compute_snp_betas(
        green, red, manifest, SNP_ADDRESSES, ["sample"], OFFSET
    )

    # M + U = 80000 would wrap around in uint16 arithmetic.
    assert result.loc["rs_type2", "sample"] == pytest.approx(
        40000 / (80000 + OFFSET)
    )


def test_snp_betas_custom_offset(
    make_manifest: Callable[[bytes], Manifest],
) -> None:
    manifest = make_manifest(MANIFEST_WITH_TYPE_2_SNP)
    green = _channel(SNP_ADDRESSES, {RS_T2_A: 300}, "<u2")
    red = _channel(SNP_ADDRESSES, {RS_T2_A: 100}, "<u2")

    result = _compute_snp_betas(
        green, red, manifest, SNP_ADDRESSES, ["s"], offset=0
    )

    assert result.loc["rs_type2", "s"] == pytest.approx(0.75)


def test_snp_betas_without_snp_probes_is_empty(
    make_manifest: Callable[[bytes], Manifest],
) -> None:
    manifest = make_manifest(MANIFEST_WITHOUT_SNPS)
    addresses = np.array([10, 11], dtype="int32")
    zeros = np.zeros((2, 2), dtype="<u2")

    result = _compute_snp_betas(
        zeros, zeros, manifest, addresses, ["a", "b"], OFFSET
    )

    assert result.shape == (0, 2)
    assert list(result.columns) == ["a", "b"]


# ---------------------------------------------------------------------------
# RawData with synthetic IDAT files
# ---------------------------------------------------------------------------


def _build_raw_data(
    dirpath: Path,
    manifest: Manifest,
    samples: list[tuple[dict[int, int], dict[int, int]]],
) -> tuple[RawData, list]:
    """Synthetic IDATs; each sample is (green, red) intensity overrides."""
    ids = np.array(
        sorted(
            (
                set(manifest.data_frame.AddressA_ID)
                | set(manifest.data_frame.AddressB_ID)
                | set(manifest.control_data_frame.Address_ID)
            )
            - {-1}
        ),
        dtype="<i4",
    )
    pairs = []
    for green_values, red_values in samples:
        green = (ids % 23).astype("<u2")
        red = (ids % 21).astype("<u2")
        for address, value in green_values.items():
            green[ids == address] = value
        for address, value in red_values.items():
            red[ids == address] = value
        pairs.append(
            TempIdatFilePair(
                dirpath=dirpath,
                data_grn={
                    "n_snps_read": len(ids),
                    "illumina_ids": ids,
                    "probe_means": green,
                },
                data_red={
                    "n_snps_read": len(ids),
                    "illumina_ids": ids,
                    "probe_means": red,
                },
            )
        )
    raw = RawData([p.basepath for p in pairs], manifest=manifest)
    return raw, pairs


SAMPLES = [
    # Heterozygous at the red SNP, homozygous (low beta) at the green SNP.
    (
        {RS_GRN_A: 850, RS_GRN_B: 50},
        {RS_RED_A: 500, RS_RED_B: 500},
    ),
    # Homozygous (high beta) at the red SNP, heterozygous at the green SNP.
    (
        {RS_GRN_A: 450, RS_GRN_B: 450},
        {RS_RED_A: 50, RS_RED_B: 850},
    ),
]


def test_raw_data_snp_betas(tmp_path: Path, tmp_manifest: Manifest) -> None:
    raw, _pairs = _build_raw_data(tmp_path, tmp_manifest, SAMPLES)

    result = raw.snp_betas()

    assert set(result.index) == {"rs10796216", "rs715359"}
    assert list(result.columns) == raw.sample_ids
    red = result.loc["rs10796216"].values
    grn = result.loc["rs715359"].values
    np.testing.assert_allclose(
        red, [500 / (1000 + OFFSET), 850 / (900 + OFFSET)]
    )
    np.testing.assert_allclose(
        grn, [50 / (900 + OFFSET), 450 / (900 + OFFSET)]
    )


def test_raw_data_snp_betas_offset(
    tmp_path: Path, tmp_manifest: Manifest
) -> None:
    raw, _pairs = _build_raw_data(tmp_path, tmp_manifest, SAMPLES)

    result = raw.snp_betas(offset=0)

    np.testing.assert_allclose(
        result.loc["rs10796216"].values, [0.5, 850 / 900]
    )


def test_raw_data_snp_betas_does_not_change_raw_data(
    tmp_path: Path, tmp_manifest: Manifest
) -> None:
    raw, _pairs = _build_raw_data(tmp_path, tmp_manifest, SAMPLES)
    green_before = raw.green.copy()
    red_before = raw.red.copy()

    raw.snp_betas()

    np.testing.assert_array_equal(raw.green, green_before)
    np.testing.assert_array_equal(raw.red, red_before)


def test_snp_probes_not_part_of_methylation_betas(
    tmp_path: Path, tmp_manifest: Manifest
) -> None:
    raw, _pairs = _build_raw_data(tmp_path, tmp_manifest, SAMPLES)

    betas = MethylData(raw, prep="raw").betas

    assert not any(str(i).startswith("rs") for i in betas.index)
