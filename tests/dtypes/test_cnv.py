"""Pytest for CNV-related utilities."""

import io
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pyranges1 as pr
import pytest

from mepylome.dtypes.arrays import ArrayType
from mepylome.dtypes.beads import MethylData
from mepylome.dtypes.cnv import (
    CNV,
    Annotation,
    _get_cgsegment,
    _linear_regression,
)

N_PROBES: int = 200
N_REFS: int = 4
N_BINS: int = 20
N_GENES: int = 5


def test_get_cgsegment_returns_callable_or_none() -> None:
    """Backend detection returns a callable or None."""
    seg = _get_cgsegment()

    assert seg is None or callable(seg)


@pytest.fixture
def probe_ids() -> np.ndarray:
    return np.array([f"cg{i:08d}" for i in range(N_PROBES)])


@pytest.fixture
def sample_mock(probe_ids: np.ndarray) -> MagicMock:
    rng = np.random.default_rng(0)
    mock = MagicMock(spec=MethylData)
    mock.sample_ids = ["SAMPLE_001"]
    mock.array_type = ArrayType.ILLUMINA_450K
    mock.intensity = rng.uniform(500, 5000, (1, N_PROBES))
    mock.probe_ids = pd.Index(probe_ids)
    return mock


@pytest.fixture
def ref_mock(probe_ids: np.ndarray) -> MagicMock:
    rng = np.random.default_rng(1)
    mock = MagicMock(spec=MethylData)
    mock.sample_ids = [f"REF_{i:03d}" for i in range(N_REFS)]
    mock.array_type = ArrayType.ILLUMINA_450K
    log_X: np.ndarray = np.log2(rng.uniform(500, 5000, (N_PROBES, N_REFS)))
    mock.log_intensity_fit = np.hstack([log_X, np.ones((N_PROBES, 1))])
    mock.probe_ids = pd.Index(probe_ids)
    return mock


@pytest.fixture
def annotation_mock(probe_ids: np.ndarray) -> MagicMock:
    ann = MagicMock()
    ann.array_type = ArrayType.ILLUMINA_450K

    adj_man = MagicMock()
    adj_man.IlmnID = pd.Index(probe_ids)
    ann._adjusted_manifest = adj_man

    starts: np.ndarray = np.arange(N_BINS) * 50_000
    ends: np.ndarray = starts + 50_000
    ann.bins = pr.PyRanges(
        pd.DataFrame(
            {"Chromosome": ["chr1"] * N_BINS, "Start": starts, "End": ends}
        )
    )

    bins_index: np.ndarray = np.tile(
        np.arange(N_BINS), N_PROBES // N_BINS + 1
    )[:N_PROBES]
    ann._cpg_bins = pd.DataFrame(
        {"bins_index": bins_index, "IlmnID": probe_ids}
    )

    gene_starts: np.ndarray = np.arange(N_GENES) * 200_000
    gene_ends: np.ndarray = gene_starts + 150_000
    gene_names: list[str] = [f"GENE{i}" for i in range(N_GENES)]
    ann.detail = pr.PyRanges(
        pd.DataFrame(
            {
                "Chromosome": ["chr1"] * N_GENES,
                "Start": gene_starts,
                "End": gene_ends,
                "Name": gene_names,
            }
        )
    )
    gene_name_col: np.ndarray = np.repeat(gene_names, N_PROBES // N_GENES)[
        :N_PROBES
    ]
    ann._cpg_detail = pd.DataFrame(
        {"Name": gene_name_col, "IlmnID": probe_ids}
    )

    return ann


@pytest.fixture
def fitted_cnv(
    sample_mock: MagicMock,
    ref_mock: MagicMock,
    annotation_mock: MagicMock,
) -> CNV:
    """CNV instance after fit() but before set_bins / set_detail."""
    obj: CNV = object.__new__(CNV)
    obj.sample = sample_mock
    obj.sample_id = sample_mock.sample_ids[0]
    obj.reference = ref_mock
    obj.annotation = annotation_mock
    obj.bins = annotation_mock.bins
    obj.probes = annotation_mock._adjusted_manifest.IlmnID
    obj.detail = None
    obj.segments = None
    obj._idx_cached = None
    obj.fit()
    return obj


@pytest.fixture
def full_cnv(fitted_cnv: CNV) -> CNV:
    """CNV instance after fit, set_bins, and set_detail."""
    fitted_cnv.set_bins()
    fitted_cnv.set_detail()
    return fitted_cnv


def _bin_df(
    starts: list[int],
    ends: list[int],
    n_probes: list[int],
) -> pd.DataFrame:
    return pd.DataFrame({"Start": starts, "End": ends, "N_probes": n_probes})


# ---------------------------------------------------------------------------
# _linear_regression
# ---------------------------------------------------------------------------


def test_linear_regression_perfect_fit() -> None:
    rng = np.random.default_rng(0)
    n: int = 50
    x: np.ndarray = rng.standard_normal(n)
    y: np.ndarray = 3.0 * x + 5.0
    X: np.ndarray = np.column_stack([x, np.ones(n)])
    y_pred, coef = _linear_regression(X, y)
    assert coef.shape == (1,)
    np.testing.assert_allclose(coef[0], 3.0, atol=1e-8)
    np.testing.assert_allclose(y_pred, np.maximum(y, 0), atol=1e-8)


def test_linear_regression_clipped_at_zero() -> None:
    n: int = 10
    X: np.ndarray = np.column_stack([np.ones(n), np.ones(n)])
    y: np.ndarray = -np.ones(n)
    y_pred, _ = _linear_regression(X, y)
    assert np.all(y_pred >= 0)


def test_linear_regression_multiple_predictors() -> None:
    rng = np.random.default_rng(42)
    n, m = 100, 3
    X_raw: np.ndarray = rng.standard_normal((n, m))
    true_coef: np.ndarray = np.array([1.0, -2.0, 0.5])
    y: np.ndarray = X_raw @ true_coef + 0.1
    X: np.ndarray = np.column_stack([X_raw, np.ones(n)])
    y_pred, coef = _linear_regression(X, y)
    assert coef.shape == (m,)
    np.testing.assert_allclose(coef, true_coef, atol=1e-6)


def test_linear_regression_returns_two_elements() -> None:
    n: int = 20
    X: np.ndarray = np.column_stack([np.arange(n, dtype=float), np.ones(n)])
    y: np.ndarray = np.arange(n, dtype=float)
    result: tuple[np.ndarray, np.ndarray] = _linear_regression(X, y)
    assert isinstance(result, tuple) and len(result) == 2


# ---------------------------------------------------------------------------
# Annotation.merge_bins_in_chromosome
# ---------------------------------------------------------------------------


def test_annotation_requires_manifest_or_array_type() -> None:
    """Either manifest or array_type must be supplied."""
    with pytest.raises(
        ValueError,
        match="Either array_type or manifest must be provided",
    ):
        Annotation(manifest=None, array_type=None)


def test_merge_bins_no_merge_needed() -> None:
    df: pd.DataFrame = _bin_df([0, 100, 200], [100, 200, 300], [20, 25, 30])
    result: pd.DataFrame = Annotation.merge_bins_in_chromosome(
        df.copy(), min_probes_per_bin=15
    )
    assert len(result) == 3


def test_merge_bins_small_bin_merged() -> None:
    df: pd.DataFrame = _bin_df(
        [0, 100, 200, 300], [100, 200, 300, 400], [20, 5, 20, 20]
    )
    result: pd.DataFrame = Annotation.merge_bins_in_chromosome(
        df.copy(), min_probes_per_bin=15
    )
    assert len(result) == 3
    assert result["N_probes"].sum() == 65


def test_merge_bins_all_small_merged() -> None:
    df: pd.DataFrame = _bin_df([0, 100, 200], [100, 200, 300], [5, 5, 5])
    result: pd.DataFrame = Annotation.merge_bins_in_chromosome(
        df.copy(), min_probes_per_bin=15
    )
    assert result["N_probes"].sum() == 15


def test_merge_bins_output_columns() -> None:
    df: pd.DataFrame = _bin_df([0, 100], [100, 200], [20, 20])
    result: pd.DataFrame = Annotation.merge_bins_in_chromosome(
        df.copy(), min_probes_per_bin=5
    )
    assert {"Start", "End", "N_probes"}.issubset(result.columns)


def test_merge_bins_empty_dataframe() -> None:
    df: pd.DataFrame = _bin_df([], [], [])
    result: pd.DataFrame = Annotation.merge_bins_in_chromosome(
        df.copy(), min_probes_per_bin=15
    )
    assert len(result) == 0


def test_merge_bins_single_bin() -> None:
    df: pd.DataFrame = _bin_df([0], [100], [5])
    result: pd.DataFrame = Annotation.merge_bins_in_chromosome(
        df.copy(), min_probes_per_bin=15
    )
    assert len(result) <= 1


# ---------------------------------------------------------------------------
# CNV init validation
# ---------------------------------------------------------------------------


def test_cnv_init_raises_on_multi_sample(
    ref_mock: MagicMock,
    annotation_mock: MagicMock,
) -> None:
    bad_sample: MagicMock = MagicMock(spec=MethylData)
    bad_sample.sample_ids = ["S1", "S2"]
    obj: CNV = object.__new__(CNV)
    with pytest.raises(ValueError, match="exactly 1"):
        CNV.__init__(obj, bad_sample, ref_mock, annotation_mock)


def test_cnv_init_raises_on_invalid_reference_type(
    sample_mock: MagicMock,
    annotation_mock: MagicMock,
) -> None:
    obj: CNV = object.__new__(CNV)
    with pytest.raises(TypeError, match="MethylData"):
        CNV.__init__(
            obj,
            sample_mock,
            "not_a_reference",  # type: ignore[arg-type]
            annotation_mock,
        )


def test_cnv_init_raises_on_array_type_mismatch(
    sample_mock: MagicMock,
    ref_mock: MagicMock,
    annotation_mock: MagicMock,
) -> None:
    ref_mock.array_type = ArrayType.ILLUMINA_EPIC
    obj: CNV = object.__new__(CNV)
    with pytest.raises(ValueError, match="Array type mismatch"):
        CNV.__init__(obj, sample_mock, ref_mock, annotation_mock)


# ---------------------------------------------------------------------------
# CNV.fit
# ---------------------------------------------------------------------------


def test_fit_sets_ratio(fitted_cnv: CNV) -> None:
    assert hasattr(fitted_cnv, "_ratio")
    assert len(fitted_cnv._ratio) == N_PROBES


def test_fit_ratio_dataframe_shape(fitted_cnv: CNV) -> None:
    assert fitted_cnv.ratio.shape == (N_PROBES, 1)
    assert "ratio" in fitted_cnv.ratio.columns


def test_fit_ratio_is_finite(fitted_cnv: CNV) -> None:
    assert np.all(np.isfinite(fitted_cnv._ratio))


def test_fit_noise_non_negative(fitted_cnv: CNV) -> None:
    assert fitted_cnv.noise >= 0


def test_fit_coef_shape(fitted_cnv: CNV) -> None:
    assert fitted_cnv.coef.shape == (N_REFS,)


# ---------------------------------------------------------------------------
# CNV.set_bins
# ---------------------------------------------------------------------------


def test_set_bins_adds_median_column(fitted_cnv: CNV) -> None:
    fitted_cnv.set_bins()
    assert "Median" in fitted_cnv.bins.columns


def test_set_bins_adds_var_column(fitted_cnv: CNV) -> None:
    fitted_cnv.set_bins()
    assert "Var" in fitted_cnv.bins.columns


def test_set_bins_median_is_float(fitted_cnv: CNV) -> None:
    fitted_cnv.set_bins()
    medians: pd.Series = pd.DataFrame(fitted_cnv.bins)["Median"]
    assert pd.api.types.is_float_dtype(medians)


# ---------------------------------------------------------------------------
# CNV.set_detail
# ---------------------------------------------------------------------------


def test_set_detail_populates_attribute(fitted_cnv: CNV) -> None:
    assert fitted_cnv.detail is None
    fitted_cnv.set_detail()
    assert fitted_cnv.detail is not None


def test_set_detail_has_median_column(fitted_cnv: CNV) -> None:
    fitted_cnv.set_detail()
    assert "Median" in fitted_cnv.detail.columns  # type: ignore[union-attr]


def test_set_detail_has_n_probes_column(fitted_cnv: CNV) -> None:
    fitted_cnv.set_detail()
    assert "N_probes" in fitted_cnv.detail.columns  # type: ignore[union-attr]


def test_set_detail_n_probes_non_negative(fitted_cnv: CNV) -> None:
    fitted_cnv.set_detail()
    assert (pd.DataFrame(fitted_cnv.detail)["N_probes"] >= 0).all()


# ---------------------------------------------------------------------------
# CNV.write
# ---------------------------------------------------------------------------


def test_write_creates_zip(full_cnv: CNV, tmp_path: Path) -> None:
    full_cnv.write(tmp_path / "sample")
    assert (tmp_path / "sample.zip").exists()


def test_write_bins_csv_present(full_cnv: CNV, tmp_path: Path) -> None:
    full_cnv.write(tmp_path / "sample", data="bins")
    with zipfile.ZipFile(tmp_path / "sample.zip") as zf:
        assert any("bins" in n for n in zf.namelist())


def test_write_metadata_csv_present(full_cnv: CNV, tmp_path: Path) -> None:
    full_cnv.write(tmp_path / "sample", data="metadata")
    with zipfile.ZipFile(tmp_path / "sample.zip") as zf:
        assert any("metadata" in n for n in zf.namelist())


def test_write_metadata_has_noise(full_cnv: CNV, tmp_path: Path) -> None:
    full_cnv.write(tmp_path / "sample", data="metadata")
    with zipfile.ZipFile(tmp_path / "sample.zip") as zf:
        csv_name: str = next(n for n in zf.namelist() if "metadata" in n)
        df: pd.DataFrame = pd.read_csv(io.BytesIO(zf.read(csv_name)))
    assert "Noise" in df.columns
    assert float(df["Noise"].iloc[0]) >= 0


def test_write_bins_roundtrip(full_cnv: CNV, tmp_path: Path) -> None:
    full_cnv.write(tmp_path / "sample", data="bins")
    with zipfile.ZipFile(tmp_path / "sample.zip") as zf:
        csv_name: str = next(n for n in zf.namelist() if "bins" in n)
        df: pd.DataFrame = pd.read_csv(io.BytesIO(zf.read(csv_name)))
    assert "Median" in df.columns
    assert len(df) > 0


def test_write_all_includes_bins_detail_metadata(
    full_cnv: CNV, tmp_path: Path
) -> None:
    full_cnv.write(tmp_path / "sample", data="all")
    with zipfile.ZipFile(tmp_path / "sample.zip") as zf:
        joined: str = " ".join(zf.namelist())
    assert "bins" in joined
    assert "detail" in joined
    assert "metadata" in joined


def test_write_raises_on_invalid_data_key(
    full_cnv: CNV, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="Invalid file"):
        full_cnv.write(tmp_path / "sample", data="bogus_key")


def test_write_accepts_zip_suffix_in_path(
    full_cnv: CNV, tmp_path: Path
) -> None:
    full_cnv.write(tmp_path / "sample.zip", data="metadata")
    assert (tmp_path / "sample.zip").exists()


# ---------------------------------------------------------------------------
# CNV.__repr__
# ---------------------------------------------------------------------------


def test_repr_is_string(fitted_cnv: CNV) -> None:
    assert isinstance(repr(fitted_cnv), str)


def test_repr_contains_cnv(fitted_cnv: CNV) -> None:
    assert "CNV" in repr(fitted_cnv)


def test_repr_contains_noise(fitted_cnv: CNV) -> None:
    assert "noise" in repr(fitted_cnv).lower()


# ---------------------------------------------------------------------------
# CNV.set_all
# ---------------------------------------------------------------------------


def test_set_all_returns_cnv_instance(
    sample_mock: MagicMock,
    ref_mock: MagicMock,
    annotation_mock: MagicMock,
) -> None:
    with patch.object(CNV, "set_segments"):
        cnv: CNV = CNV.set_all(
            sample_mock, ref_mock, annotation=annotation_mock, do_seg=False
        )
    assert isinstance(cnv, CNV)


def test_set_all_do_seg_false_skips_segmentation(
    sample_mock: MagicMock,
    ref_mock: MagicMock,
    annotation_mock: MagicMock,
) -> None:
    with patch.object(CNV, "set_segments") as mock_seg:
        cnv: CNV = CNV.set_all(
            sample_mock, ref_mock, annotation=annotation_mock, do_seg=False
        )
    mock_seg.assert_not_called()
    assert cnv.segments is None


def test_set_all_do_seg_true_calls_set_segments(
    sample_mock: MagicMock,
    ref_mock: MagicMock,
    annotation_mock: MagicMock,
) -> None:
    with patch.object(CNV, "set_segments") as mock_seg:
        CNV.set_all(
            sample_mock, ref_mock, annotation=annotation_mock, do_seg=True
        )
    mock_seg.assert_called_once()


def test_set_all_bins_have_median(
    sample_mock: MagicMock,
    ref_mock: MagicMock,
    annotation_mock: MagicMock,
) -> None:
    with patch.object(CNV, "set_segments"):
        cnv: CNV = CNV.set_all(
            sample_mock, ref_mock, annotation=annotation_mock, do_seg=False
        )
    assert "Median" in cnv.bins.columns


def test_set_all_detail_is_populated(
    sample_mock: MagicMock,
    ref_mock: MagicMock,
    annotation_mock: MagicMock,
) -> None:
    with patch.object(CNV, "set_segments"):
        cnv: CNV = CNV.set_all(
            sample_mock, ref_mock, annotation=annotation_mock, do_seg=False
        )
    assert cnv.detail is not None


# ---------------------------------------------------------------------------
# Annotation singleton cache (skipped when data files are absent)
# ---------------------------------------------------------------------------


def test_annotation_same_args_returns_same_instance() -> None:
    pytest.importorskip("pyranges1")
    try:
        a1: Annotation = Annotation(array_type="450k")
        a2: Annotation = Annotation(array_type="450k")
    except Exception:
        pytest.skip("Annotation data files not available")
    assert a1 is a2


def test_annotation_different_bin_size_gives_different_instance() -> None:
    pytest.importorskip("pyranges1")
    try:
        a1: Annotation = Annotation(array_type="450k", bin_size=50_000)
        a2: Annotation = Annotation(array_type="450k", bin_size=100_000)
    except Exception:
        pytest.skip("Annotation data files not available")
    assert a1 is not a2
