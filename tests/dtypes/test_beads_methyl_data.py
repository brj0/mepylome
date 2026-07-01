"""Extended pytest for MethylData – beta/M-value computation and DataFrames."""

from collections.abc import Generator
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.testing as npt
import pytest

from mepylome.dtypes import ArrayType, Manifest, MethylData, RawData
from mepylome.tests.helpers import TempIdatFilePair, TempManifest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_raw_data(
    dirpath: Path,
    manifest: Manifest,
    n_cpgs: int = 54,
    n_probes: int = 2,
) -> tuple[RawData, list]:
    """Create a RawData object from synthetic IDAT pairs using manifest IDs."""
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
    if len(id_list) < n_cpgs:
        start = id_list[-1] + 1
        ids[len(id_list) :] = np.arange(
            start, start + (n_cpgs - len(id_list)), dtype="<i4"
        )

    pairs = []
    for idx in range(n_probes):
        mean_grn = ((ids + idx) % 23).astype("<u2")
        mean_red = ((ids + idx) % 21).astype("<u2")
        pair = TempIdatFilePair(
            dirpath=dirpath,
            data_grn={
                "n_snps_read": n_cpgs,
                "illumina_ids": ids,
                "probe_means": mean_grn,
            },
            data_red={
                "n_snps_read": n_cpgs,
                "illumina_ids": ids,
                "probe_means": mean_red,
            },
        )
        pairs.append(pair)

    raw = RawData([p.basepath for p in pairs], manifest=manifest)
    return raw, pairs


@pytest.fixture()
def tmp_manifest(tmp_path: Path) -> Generator[Manifest, None, None]:
    tm = TempManifest(dirpath=tmp_path)
    m = Manifest(raw_path=tm.path)
    m.array_type = ArrayType.UNKNOWN
    yield m
    m.proc_path.unlink(missing_ok=True)
    m.ctrl_path.unlink(missing_ok=True)
    m._pickle_path.unlink(missing_ok=True)


@pytest.fixture()
def raw_data(tmp_path: Path, tmp_manifest: Manifest) -> RawData:
    rd, _ = _build_raw_data(tmp_path, tmp_manifest, n_cpgs=54, n_probes=2)
    return rd


# ---------------------------------------------------------------------------
# MethylData constructor errors
# ---------------------------------------------------------------------------


def test_methyl_data_no_args_raises() -> None:
    with pytest.raises(ValueError, match="Exactly one"):
        MethylData()


def test_methyl_data_both_args_raises(
    raw_data: RawData, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="Exactly one"):
        MethylData(data=raw_data, file=tmp_path)


def test_methyl_data_wrong_data_type_raises() -> None:
    with pytest.raises(ValueError, match="not of type"):
        MethylData(data="not_raw_data")  # type: ignore[arg-type]


def test_methyl_data_invalid_prep_raises(raw_data: RawData) -> None:
    with pytest.raises(ValueError):
        MethylData(raw_data, prep="invalid_prep")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# _get_beta – unit tests
# ---------------------------------------------------------------------------


def test_get_beta_basic() -> None:
    m = np.array([[100.0, 200.0]])
    u = np.array([[100.0, 0.0]])
    result = MethylData._get_beta(m, u)
    # beta = m / (m + u + 100)
    expected = np.array([[100 / 300, 200 / 300]])
    npt.assert_almost_equal(result, expected)


def test_get_beta_negative_offset_raises() -> None:
    with pytest.raises(ValueError, match="offset"):
        MethylData._get_beta(np.ones((1, 1)), np.ones((1, 1)), offset=-1)


def test_get_beta_bad_threshold_raises() -> None:
    with pytest.raises(ValueError, match="beta_threshold"):
        MethylData._get_beta(
            np.ones((1, 1)), np.ones((1, 1)), beta_threshold=0.6
        )


def test_get_beta_clamps_negatives_when_min_zero() -> None:
    m = np.array([[-50.0]])
    u = np.array([[0.0]])
    result = MethylData._get_beta(m, u, min_zero=True)
    # After clamping: m=0, beta = 0 / (0 + 0 + 100) = 0
    assert result[0, 0] == 0.0


def test_get_beta_threshold_applied() -> None:
    m = np.array([[1000.0]])
    u = np.array([[0.0]])
    result = MethylData._get_beta(m, u, beta_threshold=0.1)
    # Raw beta ≈ 1000/1100 ≈ 0.909, clamped to 1-0.1 = 0.9
    assert result[0, 0] <= 0.9 + 1e-6


def test_get_beta_zero_denominator() -> None:
    """When m=0, u=0, result should not raise (division handled gracefully)."""
    m = np.array([[0.0]])
    u = np.array([[0.0]])
    result = MethylData._get_beta(m, u, offset=0)
    assert (
        np.isnan(result[0, 0]) or np.isinf(result[0, 0]) or result[0, 0] == 0.0
    )


# ---------------------------------------------------------------------------
# _get_m_value – unit tests
# ---------------------------------------------------------------------------


def test_get_m_value_basic() -> None:
    m = np.array([[8.0]])
    u = np.array([[0.0]])
    # mval = log2((8+1)/(0+1)) = log2(9)
    result = MethylData._get_m_value(m, u, offset=1.0)
    npt.assert_almost_equal(result, np.log2(9))


def test_get_m_value_negative_offset_raises() -> None:
    with pytest.raises(ValueError, match="offset"):
        MethylData._get_m_value(np.ones((1, 1)), np.ones((1, 1)), offset=-0.1)


def test_get_m_value_negative_clamped() -> None:
    m = np.array([[-10.0]])
    u = np.array([[0.0]])
    result = MethylData._get_m_value(m, u, offset=1.0, min_zero=True)
    # After clamp: log2((0+1)/(0+1)) = 0
    npt.assert_almost_equal(result, 0.0)


def test_get_m_value_symmetric() -> None:
    """Equal methylated and unmethylated → M-value of 0."""
    m = np.array([[50.0]])
    u = np.array([[50.0]])
    result = MethylData._get_m_value(m, u, offset=0.0)
    npt.assert_almost_equal(result, 0.0)


# ---------------------------------------------------------------------------
# MethylData.betas property
# ---------------------------------------------------------------------------


def test_betas_shape_and_type(tmp_path: Path, tmp_manifest: Manifest) -> None:
    raw, _ = _build_raw_data(tmp_path, tmp_manifest, n_cpgs=54, n_probes=2)
    md = MethylData(raw, prep="raw")
    betas = md.betas
    assert betas.shape[1] == len(raw.sample_ids)
    assert (betas.values >= 0).all()
    assert (betas.values <= 1).all()


def test_betas_index_matches_methyl_ilmnid(
    tmp_path: Path, tmp_manifest: Manifest
) -> None:
    raw, _ = _build_raw_data(tmp_path, tmp_manifest)
    md = MethylData(raw, prep="raw")
    assert list(md.betas.index) == list(md.probe_ids)


def test_betas_columns_match_sample_ids(
    tmp_path: Path, tmp_manifest: Manifest
) -> None:
    raw, _ = _build_raw_data(tmp_path, tmp_manifest)
    md = MethylData(raw, prep="raw")
    assert list(md.betas.columns) == raw.sample_ids


# ---------------------------------------------------------------------------
# MethylData.betas_at
# ---------------------------------------------------------------------------


def test_betas_at_subset(tmp_path: Path, tmp_manifest: Manifest) -> None:
    raw, _ = _build_raw_data(tmp_path, tmp_manifest)
    md = MethylData(raw, prep="raw")
    subset = md.probe_ids[:3]
    betas_sub = md.betas_at(subset)
    assert list(betas_sub.index) == list(subset)
    assert betas_sub.shape[1] == len(raw.sample_ids)


def test_betas_at_missing_cpg_filled(
    tmp_path: Path, tmp_manifest: Manifest
) -> None:
    raw, _ = _build_raw_data(tmp_path, tmp_manifest)
    md = MethylData(raw, prep="raw")
    fake_cpg = np.array(["nonexistent_cg"])
    betas_fill = md.betas_at(fake_cpg, fill=0.42)
    assert betas_fill.shape == (1, len(raw.sample_ids))
    npt.assert_array_almost_equal(betas_fill.values, 0.42)


def test_betas_at_none_uses_manifest_probes(
    tmp_path: Path, tmp_manifest: Manifest
) -> None:
    raw, _ = _build_raw_data(tmp_path, tmp_manifest)
    md = MethylData(raw, prep="raw")
    betas_all = md.betas_at(None)
    assert betas_all.shape[0] == len(tmp_manifest.methylation_probes)


# ---------------------------------------------------------------------------
# MethylData.mvalues property
# ---------------------------------------------------------------------------


def test_mvalues_shape(tmp_path: Path, tmp_manifest: Manifest) -> None:
    raw, _ = _build_raw_data(tmp_path, tmp_manifest)
    md = MethylData(raw, prep="raw")
    mv = md.mvalues
    assert mv.shape == md.betas.shape


def test_mvalues_index_matches_ilmnid(
    tmp_path: Path, tmp_manifest: Manifest
) -> None:
    raw, _ = _build_raw_data(tmp_path, tmp_manifest)
    md = MethylData(raw, prep="raw")
    assert list(md.mvalues.index) == list(md.probe_ids)


# ---------------------------------------------------------------------------
# MethylData.mvalues_at
# ---------------------------------------------------------------------------


def test_mvalues_at_missing_filled(
    tmp_path: Path, tmp_manifest: Manifest
) -> None:
    raw, _ = _build_raw_data(tmp_path, tmp_manifest)
    md = MethylData(raw, prep="raw")
    fake = np.array(["ghost_cg"])
    mv = md.mvalues_at(fake, fill=-99.0)
    npt.assert_array_almost_equal(mv.values, -99.0)


# ---------------------------------------------------------------------------
# RawData grn/red DataFrames
# ---------------------------------------------------------------------------


def test_raw_data_grn_df_shape(tmp_path: Path, tmp_manifest: Manifest) -> None:
    raw, pairs = _build_raw_data(tmp_path, tmp_manifest, n_cpgs=54, n_probes=3)
    grn_df = raw.green_df
    assert grn_df.shape == (len(raw.bead_addresses), 3)
    assert list(grn_df.columns) == raw.sample_ids
    assert list(grn_df.index) == list(raw.bead_addresses)


def test_raw_data_red_df_shape(tmp_path: Path, tmp_manifest: Manifest) -> None:
    raw, _ = _build_raw_data(tmp_path, tmp_manifest, n_cpgs=54, n_probes=2)
    red_df = raw.red_df
    assert red_df.shape == (len(raw.bead_addresses), 2)


def test_raw_data_grn_values_match_array(
    tmp_path: Path, tmp_manifest: Manifest
) -> None:
    raw, _ = _build_raw_data(tmp_path, tmp_manifest)
    npt.assert_array_equal(raw.green_df.values, raw.green.T)


def test_raw_data_red_values_match_array(
    tmp_path: Path, tmp_manifest: Manifest
) -> None:
    raw, _ = _build_raw_data(tmp_path, tmp_manifest)
    npt.assert_array_equal(raw.red_df.values, raw.red.T)


# ---------------------------------------------------------------------------
# MethylData grn/red DataFrames
# ---------------------------------------------------------------------------


def test_methyl_data_grn_df(tmp_path: Path, tmp_manifest: Manifest) -> None:
    raw, _ = _build_raw_data(tmp_path, tmp_manifest)
    md = MethylData(raw, prep="raw")
    assert md.green_df.shape == (len(raw.bead_addresses), len(raw.sample_ids))


def test_methyl_data_red_df(tmp_path: Path, tmp_manifest: Manifest) -> None:
    raw, _ = _build_raw_data(tmp_path, tmp_manifest)
    md = MethylData(raw, prep="raw")
    assert md.red_df.shape == (len(raw.bead_addresses), len(raw.sample_ids))


# ---------------------------------------------------------------------------
# MethylData.methylated_df / unmethylated_df DataFrames
# ---------------------------------------------------------------------------


def test_methyl_data_methylated_df(
    tmp_path: Path, tmp_manifest: Manifest
) -> None:
    raw, _ = _build_raw_data(tmp_path, tmp_manifest)
    md = MethylData(raw, prep="raw")
    df = md.methylated_df
    assert df.shape[1] == len(raw.sample_ids)
    assert df.index.name == "IlmnID"


def test_methyl_data_unmethylated_df(
    tmp_path: Path, tmp_manifest: Manifest
) -> None:
    raw, _ = _build_raw_data(tmp_path, tmp_manifest)
    md = MethylData(raw, prep="raw")
    df = md.unmethylated_df
    assert df.shape == md.methylated_df.shape
    assert df.index.name == "IlmnID"


# ---------------------------------------------------------------------------
# MethylData.intensity_df
# ---------------------------------------------------------------------------


def test_methyl_data_intensity(tmp_path: Path, tmp_manifest: Manifest) -> None:
    raw, _ = _build_raw_data(tmp_path, tmp_manifest)
    md = MethylData(raw, prep="raw")
    intensity_df = md.intensity_df
    # intensity = methylated + unmethylated, all non-negative in our data
    assert (intensity_df.values >= 0).all()
    assert intensity_df.shape == md.methylated_df.shape


# ---------------------------------------------------------------------------
# MethylData all prep types return valid objects
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("prep", ["raw", "illumina", "noob", "swan"])
def test_all_prep_types(
    prep: Literal["raw", "illumina", "noob", "swan"],
    tmp_path: Path,
    tmp_manifest: Manifest,
) -> None:
    raw, _ = _build_raw_data(tmp_path, tmp_manifest)
    md = MethylData(raw, prep=prep, seed=42)
    assert md is not None
    assert str(md) is not None
    assert md.betas.shape[1] == len(raw.sample_ids)


# ---------------------------------------------------------------------------
# RawData raises on mismatched array types
# ---------------------------------------------------------------------------


def test_raw_data_mixed_array_types_raises(
    tmp_path: Path, tmp_manifest: Manifest
) -> None:
    """Creating RawData from IDAT files with different probe counts raises.

    We need two counts that map to *different* non-UNKNOWN ArrayTypes.
    450k range: 622000–623000; EPICv2 range: 1100000–1108000.
    """
    import numpy as np

    def _make_pair(n: int) -> "TempIdatFilePair":
        ids = np.arange(n, dtype="<i4")
        return TempIdatFilePair(
            dirpath=tmp_path,
            data_grn={
                "n_snps_read": n,
                "illumina_ids": ids,
                "probe_means": np.zeros(n, dtype="<u2"),
            },
            data_red={
                "n_snps_read": n,
                "illumina_ids": ids,
                "probe_means": np.zeros(n, dtype="<u2"),
            },
        )

    pair_a = _make_pair(622500)  # → ILLUMINA_450K
    pair_b = _make_pair(1104000)  # → ILLUMINA_EPIC_V2

    with pytest.raises(ValueError, match="Array types must all be the same"):
        RawData([pair_a.basepath, pair_b.basepath], manifest=tmp_manifest)
