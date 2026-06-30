"""Pytest for module-level helpers in mepylome.analysis.core."""

from __future__ import annotations

import shutil
import sys
import types
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from sklearn.model_selection import StratifiedKFold

from mepylome.analysis import core
from mepylome.analysis.core import (
    DualOutput,
    MethylAnalysis,
    extract_sub_dataframe,
    get_balanced_indices,
    get_cpgs_from_file,
)

AnalysisFactory = Callable[..., MethylAnalysis]


# ---------------------------------------------------------------------------
# Fakes / test doubles
# ---------------------------------------------------------------------------


class FakeIdatHandler:
    """Minimal stand-in for ``mepylome.analysis.utils.IdatHandler``.

    Exposes just the attributes/methods ``MethylAnalysis`` touches, so
    tests don't need real IDAT files or annotation spreadsheets.
    """

    analysis_dir: Path
    annotation: Path
    overlap: bool
    test_dir: Path
    analysis_ids: list[str] | None
    test_ids: list[str] | None
    paths: list[Path]
    ids: list[str]
    id_to_path: dict[str, Path]
    id_to_basename: dict[str, str]
    samples_annotated: pd.DataFrame
    selected_columns: list[str]

    def __init__(
        self,
        *,
        analysis_dir: Path,
        annotation: Path,
        overlap: bool,
        test_dir: Path,
        analysis_ids: list[str] | None,
        test_ids: list[str] | None,
    ) -> None:
        self.analysis_dir = analysis_dir
        self.annotation = annotation
        self.overlap = overlap
        self.test_dir = test_dir
        self.analysis_ids = analysis_ids
        # IMPORTANT: store exactly as given (do not coerce None -> []).
        # MethylAnalysis.idat_handler compares this against
        # `self.test_ids`, which itself defaults to None. Coercing here
        # would make every property access look "changed" and defeat
        # caching.
        self.test_ids = test_ids

        self.paths = []
        self.ids = []
        self.id_to_path = {}
        self.id_to_basename = {}
        self.samples_annotated = pd.DataFrame()
        self.selected_columns = []

    def init_parameters(self) -> dict[str, Any]:
        return {
            "analysis_dir": self.analysis_dir,
            "annotation": self.annotation,
            "overlap": self.overlap,
            "test_dir": self.test_dir,
            "analysis_ids": self.analysis_ids,
            "test_ids": self.test_ids,
        }

    def features(
        self, columns: Sequence[str] | str | None = None
    ) -> pd.Series:
        return pd.Series(dtype=object)

    def __len__(self) -> int:
        return len(self.ids)

    def __bool__(self) -> bool:
        # MethylAnalysis.set_betas() does `if not self.idat_handler: ...`
        return True


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def patch_core_dependencies(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Iterator[None]:
    """Isolate MethylAnalysis from filesystem/IO-heavy collaborators.

    Applied to every test in this module so that constructing a
    ``MethylAnalysis`` never touches real IDAT files, annotation lookup,
    or the real (non-deterministic) hashing helper.
    """
    monkeypatch.setattr(
        core, "make_log_file", lambda name: tmp_path / f"{name}.log"
    )
    monkeypatch.setattr(
        core,
        "guess_annotation_file",
        lambda analysis_dir: core.INVALID_PATH,
    )
    # Pretend the optional segmentation backend (cbseg/ruptures/...) is
    # unavailable, matching a typical minimal test environment.
    monkeypatch.setattr(core, "_get_cgsegment", lambda: None)
    monkeypatch.setattr(
        core,
        "input_args_id",
        lambda *args, extra_hash=None, **kwargs: (
            "fixedhash-" + "-".join(str(a) for a in args)
        ),
    )
    monkeypatch.setattr(core, "IdatHandler", FakeIdatHandler)
    yield


@pytest.fixture
def make_analysis(tmp_path: Path) -> AnalysisFactory:
    """Factory for creating a lightweight MethylAnalysis instance."""

    def _make(**overrides: Any) -> MethylAnalysis:
        analysis_dir = tmp_path / "analysis"
        output_dir = tmp_path / "output"
        kwargs: dict[str, Any] = {
            "analysis_dir": analysis_dir,
            "output_dir": output_dir,
            # Explicit empty CpG list avoids the "auto" array-detection
            # codepath, which is covered by its own dedicated tests.
            "cpgs": [],
        }
        kwargs.update(overrides)
        return core.MethylAnalysis(**kwargs)

    return _make


# ---------------------------------------------------------------------------
# Module-level helper functions
# ---------------------------------------------------------------------------


def test_extract_sub_dataframe_full_overlap() -> None:
    """All requested columns exist in the source frame."""
    df = pd.DataFrame(
        {"a": [1.0, 2.0], "b": [3.0, 4.0], "c": [5.0, 6.0]},
        index=["s1", "s2"],
    )
    result = extract_sub_dataframe(df, np.array(["c", "a"]))
    assert list(result.columns) == ["c", "a"]
    assert list(result.index) == ["s1", "s2"]
    assert result.loc["s1", "a"] == 1.0
    assert result.loc["s1", "c"] == 5.0


def test_extract_sub_dataframe_partial_overlap_uses_fill() -> None:
    """Columns missing from the source frame are filled with `fill`."""
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]}, index=["s1", "s2"])
    result = extract_sub_dataframe(df, np.array(["a", "z"]), fill=0.42)
    assert result.loc["s1", "a"] == 1.0
    assert result.loc["s2", "a"] == 2.0
    assert result.loc["s1", "z"] == 0.42
    assert result.loc["s2", "z"] == 0.42


def test_extract_sub_dataframe_no_overlap_all_filled() -> None:
    """When no columns overlap, the whole result equals `fill`."""
    df = pd.DataFrame({"a": [1.0, 2.0]}, index=["s1", "s2"])
    result = extract_sub_dataframe(df, np.array(["x", "y"]), fill=-1.0)
    assert (result.values == -1.0).all()
    assert list(result.columns) == ["x", "y"]


def test_extract_sub_dataframe_default_fill() -> None:
    """Default fill value of 0.49 is used when not specified."""
    df = pd.DataFrame({"a": [1.0]}, index=["s1"])
    result = extract_sub_dataframe(df, np.array(["missing"]))
    assert result.loc["s1", "missing"] == 0.49


def test_extract_sub_dataframe_preserves_row_count() -> None:
    """Result has the same number of rows as the input, regardless of cols."""
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0]}, index=["s1", "s2", "s3"])
    result = extract_sub_dataframe(df, np.array(["a", "b", "c"]))
    assert len(result) == 3


def test_get_balanced_indices_equal_classes() -> None:
    """When classes are already balanced, all indices are returned."""
    labels = ["a", "b", "a", "b"]
    result = get_balanced_indices(labels, seed=0)
    assert sorted(result.tolist()) == [0, 1, 2, 3]


def test_get_balanced_indices_downsamples_majority_class() -> None:
    """Majority class is downsampled to match the minority class size."""
    labels = ["a", "a", "a", "b", "b"]
    result = get_balanced_indices(labels, seed=0)
    # Minority class "b" has 2 samples, so result has 2 + 2 = 4 indices.
    assert len(result) == 4
    selected_labels = [labels[i] for i in result]
    assert selected_labels.count("a") == 2
    assert selected_labels.count("b") == 2


def test_get_balanced_indices_is_sorted() -> None:
    """Returned indices are sorted in ascending order."""
    labels = ["a", "b", "a", "b", "a"]
    result = get_balanced_indices(labels, seed=1)
    assert list(result) == sorted(result.tolist())


def test_get_balanced_indices_reproducible_with_seed() -> None:
    """Same seed produces the same balanced selection."""
    labels = ["a", "a", "a", "a", "b", "b"]
    result1 = get_balanced_indices(labels, seed=42)
    result2 = get_balanced_indices(labels, seed=42)
    assert list(result1) == list(result2)


def test_get_balanced_indices_raises_on_singleton_class() -> None:
    """A class with only one member raises a ValueError."""
    labels = ["a", "a", "b"]
    with pytest.raises(ValueError, match="Only 1 sample"):
        get_balanced_indices(labels)


def test_get_balanced_indices_raises_on_empty_class() -> None:
    """A class with zero effective members raises."""
    labels = ["a", "b"]
    with pytest.raises(ValueError, match="Only 1 sample"):
        get_balanced_indices(labels)


def test_get_cpgs_from_file_non_path_input_returns_none() -> None:
    """Non str/Path input (e.g. a list) returns None."""
    assert get_cpgs_from_file(["cg00000001", "cg00000002"]) is None
    assert get_cpgs_from_file(None) is None
    assert get_cpgs_from_file(123) is None


def test_get_cpgs_from_file_nonexistent_string_path_returns_none() -> None:
    """A string path that does not exist on disk returns None."""
    assert get_cpgs_from_file("/no/such/file/at/all.csv") is None


def test_get_cpgs_from_file_reads_csv(tmp_path: Path) -> None:
    """CpGs are read from a simple single-column CSV file."""
    csv_path = tmp_path / "cpgs.csv"
    csv_path.write_text("cg00000001\ncg00000002\ncg00000003\n")
    result = get_cpgs_from_file(csv_path)
    assert result is not None
    assert sorted(result.tolist()) == [
        "cg00000001",
        "cg00000002",
        "cg00000003",
    ]


def test_get_cpgs_from_file_drops_na_values(tmp_path: Path) -> None:
    """Empty/NaN cells in the file are dropped from the result."""
    csv_path = tmp_path / "cpgs_with_gaps.csv"
    csv_path.write_text("cg00000001\n\ncg00000003\n")
    result = get_cpgs_from_file(csv_path)
    assert result is not None
    assert "cg00000001" in result
    assert "cg00000003" in result
    assert len(result) == 2


def test_get_cpgs_from_file_accepts_str_object(tmp_path: Path) -> None:
    """A pathlib.Path object (not just Path) is also accepted."""
    csv_path = tmp_path / "cpgs.csv"
    csv_path.write_text("cg00000001\n")
    result = get_cpgs_from_file(str(csv_path))
    assert result is not None
    assert list(result) == ["cg00000001"]


def test_get_cpgs_from_file_unsupported_format_raises(tmp_path: Path) -> None:
    """An unsupported file extension raises a wrapped ValueError."""
    bad_path = tmp_path / "cpgs.unsupported_ext"
    bad_path.write_text("cg00000001\n")
    with pytest.raises(ValueError, match="Failed to read CpGs"):
        get_cpgs_from_file(bad_path)


def test_dual_output_writes_to_file_and_restores_stdout(
    tmp_path: Path,
) -> None:
    """Text printed inside the context manager is written to the log file."""
    import sys

    log_path = tmp_path / "log.txt"
    original_stdout = sys.stdout

    with DualOutput(log_path):
        print("hello dual output")

    assert sys.stdout is original_stdout
    assert "hello dual output" in log_path.read_text()


def test_dual_output_appends_across_multiple_uses(tmp_path: Path) -> None:
    """DualOutput opens the file in append mode, preserving prior content."""
    log_path = tmp_path / "log.txt"

    with DualOutput(log_path):
        print("first")

    with DualOutput(log_path):
        print("second")

    content = log_path.read_text()
    assert "first" in content
    assert "second" in content


def test_dual_output_write_method_writes_to_both_targets(
    tmp_path: Path,
) -> None:
    """Calling .write() directly forwards text to terminal and log."""
    log_path = tmp_path / "log.txt"
    dual = DualOutput(log_path)
    try:
        dual.write("manual write\n")
        dual.flush()
    finally:
        dual.close()
    assert "manual write" in log_path.read_text()


# ---------------------------------------------------------------------------
# MethylAnalysis.__init__ validation and basic construction
# ---------------------------------------------------------------------------


def test_init_raises_for_invalid_cpg_selection(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cpg_selection"):
        core.MethylAnalysis(
            analysis_dir=tmp_path / "analysis",
            output_dir=tmp_path / "output",
            cpg_selection="invalid_choice",
        )


def test_init_resolves_string_paths_to_path_objects(tmp_path: Path) -> None:
    analysis_dir = tmp_path / "analysis"
    output_dir = tmp_path / "output"

    analysis = core.MethylAnalysis(
        analysis_dir=str(analysis_dir),
        output_dir=str(output_dir),
        cpgs=[],
    )

    assert analysis.analysis_dir == analysis_dir
    assert analysis.output_dir == output_dir
    assert output_dir.exists()


def test_init_disables_segmentation_when_dependency_missing(
    make_analysis: AnalysisFactory,
) -> None:
    # patch_core_dependencies forces _get_cgsegment() -> None
    analysis = make_analysis(do_seg=True)
    assert analysis.do_seg is False


def test_init_respects_do_seg_when_dependency_available(
    make_analysis: AnalysisFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(core, "_get_cgsegment", lambda: object())
    analysis = make_analysis(do_seg=True)
    assert analysis.do_seg is True


def test_init_handles_missing_annotation_gracefully(
    make_analysis: AnalysisFactory, tmp_path: Path
) -> None:
    bogus_annotation = tmp_path / "missing_annotation.csv"
    analysis = make_analysis(annotation=bogus_annotation)
    assert not analysis.annotation.exists()


def test_repr_contains_class_name_and_key_attribute(
    make_analysis: AnalysisFactory,
) -> None:
    analysis = make_analysis()
    text = repr(analysis)
    assert "MethylAnalysis" in text
    assert "analysis_dir" in text


# ---------------------------------------------------------------------------
# cpgs property / _get_cpgs
# ---------------------------------------------------------------------------


def test_cpgs_property_sorted_and_excludes_blacklist(
    make_analysis: AnalysisFactory,
) -> None:
    analysis = make_analysis(
        cpgs=["cpg3", "cpg1", "cpg2"], cpg_blacklist=["cpg2"]
    )
    assert list(analysis.cpgs) == ["cpg1", "cpg3"]


def test_cpgs_setter_updates_internal_array(
    make_analysis: AnalysisFactory,
) -> None:
    analysis = make_analysis(cpgs=["cpg1"])
    analysis.cpgs = ["cpgA", "cpgB"]
    assert sorted(analysis.cpgs) == ["cpgA", "cpgB"]


def test_get_cpgs_accepts_set_and_array_inputs(
    make_analysis: AnalysisFactory,
) -> None:
    analysis = make_analysis(cpgs=[])
    from_set = analysis._get_cpgs({"cpg2", "cpg1"})
    from_array = analysis._get_cpgs(np.array(["cpg2", "cpg1"]))
    assert list(from_set) == ["cpg1", "cpg2"]
    assert list(from_array) == ["cpg1", "cpg2"]


def test_get_cpgs_auto_with_no_idat_files_returns_empty_array(
    make_analysis: AnalysisFactory,
) -> None:
    analysis = make_analysis()
    result = analysis._get_cpgs("auto")
    assert isinstance(result, np.ndarray)
    assert result.size == 0


def test_get_cpgs_invalid_array_type_string_raises(
    make_analysis: AnalysisFactory,
) -> None:
    analysis = make_analysis()
    with pytest.raises(ValueError, match="must be one of the following"):
        analysis._get_cpgs("not_a_real_array_type")


# ---------------------------------------------------------------------------
# classifiers property / _get_classifiers
# ---------------------------------------------------------------------------


def test_classifiers_defaults_to_empty_list(
    make_analysis: AnalysisFactory,
) -> None:
    analysis = make_analysis()
    assert analysis.classifiers == []


def test_classifiers_setter_wraps_bare_model_with_defaults(
    make_analysis: AnalysisFactory,
) -> None:
    analysis = make_analysis()
    model: object = object()
    analysis.classifiers = model

    result = analysis.classifiers
    assert len(result) == 1
    assert result[0]["model"] is model
    assert result[0]["name"] == "Custom_Classifier_0"
    assert isinstance(result[0]["cv"], StratifiedKFold)


def test_classifiers_dict_without_model_raises(
    make_analysis: AnalysisFactory,
) -> None:
    analysis = make_analysis()
    analysis.classifiers = [{"name": "no model here"}]
    with pytest.raises(ValueError, match="must have a 'model'"):
        _ = analysis.classifiers


def test_classifiers_converts_int_cv_to_stratified_kfold(
    make_analysis: AnalysisFactory,
) -> None:
    analysis = make_analysis()
    analysis.classifiers = [{"model": object(), "cv": 3}]
    cv = analysis.classifiers[0]["cv"]
    assert isinstance(cv, StratifiedKFold)
    assert cv.get_n_splits() == 3


def test_get_classifiers_handles_mixed_list_of_strings_and_dicts(
    make_analysis: AnalysisFactory,
) -> None:
    analysis = make_analysis()
    clf_dict: dict[str, Any] = {
        "model": "kbest-rf",
        "name": "Custom RF",
        "cv": 10,
    }

    result = analysis._get_classifiers(["plain-model-string", clf_dict])

    assert result[0]["model"] == "plain-model-string"
    assert result[0]["name"] == "Custom_Classifier_0"
    assert result[1]["model"] == "kbest-rf"
    assert result[1]["name"] == "Custom RF"
    assert result[1]["cv"].get_n_splits() == 10


def test_get_classifiers_returns_empty_list_for_none(
    make_analysis: AnalysisFactory,
) -> None:
    analysis = make_analysis()
    assert analysis._get_classifiers(None) == []


# ---------------------------------------------------------------------------
# _get_umap_parms (static helper)
# ---------------------------------------------------------------------------


def test_get_umap_parms_uses_defaults_when_none_given() -> None:
    parms = core.MethylAnalysis._get_umap_parms(None)
    assert parms == {
        "metric": "manhattan",
        "min_dist": 0.1,
        "n_neighbors": 15,
        "verbose": True,
    }


def test_get_umap_parms_overrides_and_extends_defaults() -> None:
    parms = core.MethylAnalysis._get_umap_parms(
        {"n_neighbors": 5, "extra_param": "x"}
    )
    assert parms["n_neighbors"] == 5
    assert parms["metric"] == "manhattan"
    assert parms["extra_param"] == "x"


# ---------------------------------------------------------------------------
# idat_handler property caching
# ---------------------------------------------------------------------------


def test_idat_handler_is_cached_when_params_unchanged(
    make_analysis: AnalysisFactory,
) -> None:
    analysis = make_analysis()
    handler1 = analysis.idat_handler
    handler2 = analysis.idat_handler
    assert handler1 is handler2


def test_idat_handler_reinitialized_when_analysis_dir_changes(
    make_analysis: AnalysisFactory, tmp_path: Path
) -> None:
    analysis = make_analysis()
    handler1 = analysis.idat_handler

    new_dir = tmp_path / "other_analysis"
    new_dir.mkdir()
    analysis.analysis_dir = new_dir

    handler2 = analysis.idat_handler
    assert handler1 is not handler2


# ---------------------------------------------------------------------------
# Hashing / caching helpers
# ---------------------------------------------------------------------------


def test_get_cpgs_hash_is_computed_once_and_cached(
    make_analysis: AnalysisFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    analysis = make_analysis(cpgs=["a", "b"])
    calls: list[tuple[Any, ...]] = []

    def fake_input_args_id(*args: Any, **kwargs: Any) -> str:
        calls.append(args)
        return "hash-value"

    # __init__ already populated `_internal_cpgs_hash` using the
    # autouse-patched `input_args_id`. Reset it so this test's
    # call-counting fake actually gets exercised.
    analysis._internal_cpgs_hash = None
    monkeypatch.setattr(core, "input_args_id", fake_input_args_id)

    first = analysis._get_cpgs_hash()
    second = analysis._get_cpgs_hash()

    assert first == second == "hash-value"
    assert len(calls) == 1


def test_get_test_files_hash_empty_string_when_dir_missing(
    make_analysis: AnalysisFactory,
) -> None:
    analysis = make_analysis()
    shutil.rmtree(analysis.test_dir, ignore_errors=True)
    assert analysis._get_test_files_hash() == ""


def test_get_test_files_hash_nonempty_when_files_present(
    make_analysis: AnalysisFactory,
) -> None:
    analysis = make_analysis()
    (analysis.test_dir / "sample.idat").write_text("data")
    result = analysis._get_test_files_hash()
    assert result != ""


def test_get_vars_or_hashes_contains_expected_keys(
    make_analysis: AnalysisFactory,
) -> None:
    analysis = make_analysis()
    current = analysis._get_vars_or_hashes()
    assert set(current) == {
        "analysis_dir",
        "prep",
        "n_cpgs",
        "cpg_selection",
        "cpgs",
        "test_files",
        "analysis_ids",
        "test_ids",
    }


# ---------------------------------------------------------------------------
# Test directory handling
# ---------------------------------------------------------------------------


def test_set_test_dir_creates_default_dir_under_output_dir(
    make_analysis: AnalysisFactory,
) -> None:
    analysis = make_analysis()
    assert analysis.test_dir.exists()
    assert str(analysis.test_dir).startswith(str(analysis.output_dir))


def test_set_test_dir_uses_explicitly_provided_directory(
    make_analysis: AnalysisFactory, tmp_path: Path
) -> None:
    custom_test_dir = tmp_path / "my_test_dir"
    analysis = make_analysis(test_dir=custom_test_dir)
    assert analysis.test_dir == custom_test_dir.expanduser()
    assert analysis.test_dir.exists()


# ---------------------------------------------------------------------------
# compute_umap
# ---------------------------------------------------------------------------


def test_compute_umap_raises_without_betas_or_feature_matrix(
    make_analysis: AnalysisFactory,
) -> None:
    analysis = make_analysis()
    analysis.betas_sel = None
    analysis.feature_matrix = None
    with pytest.raises(AttributeError, match="betas_sel"):
        analysis.compute_umap()


def test_compute_umap_raises_on_row_count_mismatch(
    make_analysis: AnalysisFactory,
) -> None:
    analysis = make_analysis()
    analysis.betas_sel = None
    analysis.idat_handler.ids = ["s1", "s2"]  # type: ignore[misc]
    analysis.feature_matrix = np.zeros((3, 5))
    with pytest.raises(ValueError, match="Dimension mismatch"):
        analysis.compute_umap()


def test_compute_umap_raises_for_unknown_sample_in_feature_matrix_index(
    make_analysis: AnalysisFactory,
) -> None:
    analysis = make_analysis()
    analysis.betas_sel = None
    analysis.idat_handler.ids = ["s1", "s2"]  # type: ignore[misc]
    analysis.feature_matrix = pd.DataFrame(
        {"f1": [0.1, 0.2]}, index=["s1", "unknown_sample"]
    )
    with pytest.raises(ValueError, match="Invalid sample IDs"):
        analysis.compute_umap()


# ---------------------------------------------------------------------------
# make_umap_plot
# ---------------------------------------------------------------------------


def test_make_umap_plot_raises_when_umap_df_not_set(
    make_analysis: AnalysisFactory,
) -> None:
    analysis = make_analysis()
    analysis.umap_df = None
    with pytest.raises(AttributeError, match="umap_df"):
        analysis.make_umap_plot()


def test_make_umap_plot_raises_keyerror_for_missing_feature_ids(
    make_analysis: AnalysisFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    analysis = make_analysis()
    analysis.umap_df = pd.DataFrame(
        {"Umap_x": [0.0, 1.0], "Umap_y": [0.0, 1.0]},
        index=["s1", "s2"],
    )
    monkeypatch.setattr(
        analysis.idat_handler,
        "features",
        lambda *a, **k: pd.Series(["A"], index=["s1"]),
    )
    with pytest.raises(KeyError, match="Missing"):
        analysis.make_umap_plot()


# ---------------------------------------------------------------------------
# _get_coordinates
# ---------------------------------------------------------------------------


def test_get_coordinates_raises_when_umap_df_not_set(
    make_analysis: AnalysisFactory,
) -> None:
    analysis = make_analysis()
    analysis.umap_df = None
    with pytest.raises(ValueError, match="umap_df"):
        analysis._get_coordinates("sample1")


def test_get_coordinates_returns_xy_for_known_sample(
    make_analysis: AnalysisFactory,
) -> None:
    analysis = make_analysis()
    analysis.umap_df = pd.DataFrame(
        {"Umap_x": [1.5, 2.5], "Umap_y": [3.5, 4.5]},
        index=["s1", "s2"],
    )
    coords = analysis._get_coordinates("s2")
    assert coords["Umap_x"] == pytest.approx(2.5)
    assert coords["Umap_y"] == pytest.approx(4.5)


# ---------------------------------------------------------------------------
# make_cnv_plot
# ---------------------------------------------------------------------------


def test_make_cnv_plot_raises_for_invalid_sample(
    make_analysis: AnalysisFactory,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    ref_dir = tmp_path / "ref"
    ref_dir.mkdir(parents=True, exist_ok=True)
    analysis = make_analysis(reference_dir=ref_dir)
    analysis.idat_handler.id_to_path["sample1"] = tmp_path / "sample1"

    monkeypatch.setattr(core, "is_valid_idat_basepath", lambda path: False)

    with pytest.raises(FileNotFoundError, match="not found"):
        analysis.make_cnv_plot("sample1")


def test_make_cnv_plot_raises_when_reference_dir_missing(
    make_analysis: AnalysisFactory,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    analysis = make_analysis(reference_dir=tmp_path / "missing_ref")
    analysis.idat_handler.id_to_path["sample1"] = tmp_path / "sample1"

    monkeypatch.setattr(core, "is_valid_idat_basepath", lambda path: True)

    with pytest.raises(FileNotFoundError, match="Reference dir"):
        analysis.make_cnv_plot("sample1")


# ---------------------------------------------------------------------------
# cn_summary
# ---------------------------------------------------------------------------


def test_cn_summary_raises_when_segmentation_disabled(
    make_analysis: AnalysisFactory,
) -> None:
    analysis = make_analysis()
    assert analysis.do_seg is False  # forced by patched _get_cgsegment
    with pytest.raises(ValueError, match="do_seg"):
        analysis.cn_summary(["sample1"])


# ---------------------------------------------------------------------------
# classify
# ---------------------------------------------------------------------------


def test_classify_raises_when_neither_ids_nor_values_given(
    make_analysis: AnalysisFactory,
) -> None:
    analysis = make_analysis()
    with pytest.raises(ValueError, match="exactly one"):
        analysis.classify(ids=None, values=None, clf_list=object())


def test_classify_raises_when_both_ids_and_values_given(
    make_analysis: AnalysisFactory,
) -> None:
    analysis = make_analysis()
    with pytest.raises(ValueError, match="exactly one"):
        analysis.classify(ids=["s1"], values=pd.DataFrame(), clf_list=object())


# ---------------------------------------------------------------------------
# Additional coverage: _resolve_path invalid type
# ---------------------------------------------------------------------------


def test_init_raises_for_invalid_path_type(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="_resolve_path"):
        core.MethylAnalysis(
            analysis_dir=12345,  # type: ignore[arg-type]
            output_dir=tmp_path / "output",
            cpgs=[],
        )


# ---------------------------------------------------------------------------
# idat_handler: restoring previously selected columns
# ---------------------------------------------------------------------------


def test_idat_handler_restores_previously_selected_columns(
    make_analysis: AnalysisFactory,
) -> None:
    analysis = make_analysis()
    handler = analysis.idat_handler
    handler.selected_columns = ["group"]
    handler.samples_annotated = pd.DataFrame(columns=["group"])
    handler_again = analysis.idat_handler
    assert handler_again.selected_columns == ["group"]


# ---------------------------------------------------------------------------
# _get_cpgs: file-based and manifest-intersection branches
# ---------------------------------------------------------------------------


def test_get_cpgs_reads_from_file_path(
    make_analysis: AnalysisFactory,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cpg_file = tmp_path / "cpgs.csv"
    cpg_file.write_text("cpg1\ncpg2\n")
    monkeypatch.setattr(
        core,
        "read_dataframe",
        lambda path, header=None: pd.DataFrame(["cpg1", "cpg2"]),
    )
    analysis = make_analysis(cpgs=[])
    result = analysis._get_cpgs(cpg_file)
    assert sorted(result) == ["cpg1", "cpg2"]


def test_get_cpgs_loads_manifest_intersection_for_array_type_string(
    make_analysis: AnalysisFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeManifest:
        _probes = {
            "450k": {"cpgA", "cpgB", "cpgC"},
            "epic": {"cpgB", "cpgC", "cpgD"},
        }

        def __init__(self, array_type: str) -> None:
            self.array_type = array_type

        @property
        def methylation_probes(self) -> set[str]:
            return self._probes.get(str(self.array_type), set())

    monkeypatch.setattr(core, "Manifest", FakeManifest)
    analysis = make_analysis(cpgs=[])
    result = analysis._get_cpgs("450k+epic")
    assert sorted(result) == ["cpgB", "cpgC"]


# ---------------------------------------------------------------------------
# _update_paths branches
# ---------------------------------------------------------------------------


def test_update_paths_returns_early_when_output_dir_missing(
    make_analysis: AnalysisFactory,
) -> None:
    analysis = make_analysis()
    cnv_dir_before = analysis.cnv_dir
    shutil.rmtree(analysis.output_dir, ignore_errors=True)
    analysis._update_paths()
    assert analysis.cnv_dir == cnv_dir_before


def test_update_paths_recomputes_cpgs_when_analysis_dir_changes(
    make_analysis: AnalysisFactory, tmp_path: Path
) -> None:
    analysis = make_analysis(cpgs=["x"])
    new_dir = tmp_path / "another"
    new_dir.mkdir()
    analysis.analysis_dir = new_dir
    analysis._update_paths()
    assert analysis.cpgs.size == 0


def test_update_paths_uses_feature_matrix_for_clf_hash(
    make_analysis: AnalysisFactory,
) -> None:
    analysis = make_analysis(feature_matrix=pd.DataFrame({"f": [1, 2]}))
    assert analysis.clf_dir is not None


def test_update_paths_uses_n_cpgs_for_clf_hash_when_full_betas_disabled(
    make_analysis: AnalysisFactory,
) -> None:
    analysis = make_analysis(load_full_betas=False)
    assert analysis.clf_dir is not None


def test_update_paths_resets_betas_when_prep_changes(
    make_analysis: AnalysisFactory,
) -> None:
    analysis = make_analysis()
    analysis.betas_sel = pd.DataFrame({"a": [1]})
    analysis.betas_all = pd.DataFrame({"a": [1]})
    analysis.prep = "noob"
    analysis._update_paths()
    assert analysis.betas_sel is None
    assert analysis.betas_all is None


# ---------------------------------------------------------------------------
# make_umap orchestration
# ---------------------------------------------------------------------------


def test_make_umap_calls_set_betas_compute_and_plot_in_order(
    make_analysis: AnalysisFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    analysis = make_analysis()
    call_order: list[str] = []
    monkeypatch.setattr(
        analysis, "set_betas", lambda: call_order.append("set_betas")
    )
    monkeypatch.setattr(
        analysis, "compute_umap", lambda: call_order.append("compute_umap")
    )
    monkeypatch.setattr(
        analysis,
        "make_umap_plot",
        lambda: call_order.append("make_umap_plot"),
    )
    analysis.make_umap()
    assert call_order == ["set_betas", "compute_umap", "make_umap_plot"]


# ---------------------------------------------------------------------------
# compute_umap success paths (fake UMAP/cuML injected via sys.modules)
# ---------------------------------------------------------------------------


class _FakeUMAP:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    def fit_transform(self, matrix: Any) -> np.ndarray:
        return np.zeros((len(matrix), 2))


def test_compute_umap_warns_and_completes_on_chosen_index_mismatch(
    make_analysis: AnalysisFactory,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_umap_module = types.ModuleType("umap")
    fake_umap_module.UMAP = _FakeUMAP  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "umap", fake_umap_module)
    monkeypatch.setattr(core, "LOG_FILE", tmp_path / "umap.log")

    analysis = make_analysis()
    analysis.idat_handler.ids = ["s1", "s2", "s3"]  # type: ignore[misc]
    analysis.betas_sel = pd.DataFrame({"cpg1": [0.1, 0.2]}, index=["s1", "s2"])
    analysis.feature_matrix = np.zeros((2, 5))
    analysis.use_gpu = False

    analysis.umap_plot_path = tmp_path / "umap.csv"
    analysis.compute_umap()

    assert analysis.umap_df is not None
    assert list(analysis.umap_df.index) == ["s1", "s2"]
    assert analysis.umap_plot_path.exists()


def test_compute_umap_uses_gpu_backend_when_enabled(
    make_analysis: AnalysisFactory,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_cupy = types.ModuleType("cupy")
    fake_cupy.asarray = lambda x: np.asarray(x)  # type: ignore[attr-defined]
    fake_cupy.asnumpy = lambda x: np.asarray(x)  # type: ignore[attr-defined]
    fake_cuml_manifold = types.ModuleType("cuml.manifold")
    fake_cuml_manifold.UMAP = _FakeUMAP  # type: ignore[attr-defined]
    fake_cuml = types.ModuleType("cuml")
    fake_cuml.manifold = fake_cuml_manifold  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "cupy", fake_cupy)
    monkeypatch.setitem(sys.modules, "cuml", fake_cuml)
    monkeypatch.setitem(sys.modules, "cuml.manifold", fake_cuml_manifold)
    monkeypatch.setattr(core, "LOG_FILE", tmp_path / "umap_gpu.log")

    analysis = make_analysis()
    analysis.idat_handler.ids = ["s1", "s2"]  # type: ignore[misc]
    analysis.betas_sel = pd.DataFrame({"cpg1": [0.1, 0.2]}, index=["s1", "s2"])
    analysis.feature_matrix = None
    analysis.use_gpu = True

    analysis.umap_plot_path = tmp_path / "umap.csv"
    analysis.compute_umap()

    assert analysis.umap_df is not None
    assert list(analysis.umap_df.index) == ["s1", "s2"]


# ---------------------------------------------------------------------------
# make_umap_plot success path + _umap_plot_highlight + _retrieve_zoom
# ---------------------------------------------------------------------------


def test_make_umap_plot_builds_plot_for_valid_data(
    make_analysis: AnalysisFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    analysis = make_analysis()
    analysis.umap_df = pd.DataFrame(
        {"Umap_x": [0.0, 1.0], "Umap_y": [1.0, 0.0]}, index=["s1", "s2"]
    )
    monkeypatch.setattr(
        analysis.idat_handler,
        "features",
        lambda *a, **k: pd.Series(["A", "B"], index=["s1", "s2"]),
    )
    analysis.make_umap_plot()
    assert analysis.ids == ["s1", "s2"]
    assert analysis.umap_plot is not None
    assert analysis.raw_umap_plot is not None
    assert analysis.raw_umap_plot.to_dict() == analysis.umap_plot.to_dict()
    assert analysis.dropdown_id == analysis.ids_to_highlight


def test_umap_plot_highlight_annotates_highlighted_and_cnv_ids(
    make_analysis: AnalysisFactory,
) -> None:
    analysis = make_analysis()
    analysis.umap_df = pd.DataFrame(
        {"Umap_x": [0.0, 1.0, 2.0], "Umap_y": [1.0, 0.0, 2.0]},
        index=["s1", "s2", "s3"],
    )
    analysis.raw_umap_plot = go.Figure()
    analysis.ids_to_highlight = ["s1"]

    analysis._umap_plot_highlight(cnv_id="s2")

    assert analysis.cnv_id == "s2"
    assert analysis.dropdown_id == ["s1"]
    assert len(analysis.umap_plot.layout.annotations) == 2


def test_retrieve_zoom_returns_early_when_no_current_plot(
    make_analysis: AnalysisFactory,
) -> None:
    analysis = make_analysis()
    original = go.Figure()
    analysis.umap_plot = original
    analysis._retrieve_zoom(None)
    assert analysis.umap_plot is original


def test_retrieve_zoom_applies_provided_axis_ranges(
    make_analysis: AnalysisFactory,
) -> None:
    analysis = make_analysis()
    analysis.umap_plot = go.Figure()

    relayout_data = {
        "xaxis.range[0]": 0,
        "xaxis.range[1]": 1,
        "yaxis.range[0]": 0,
        "yaxis.range[1]": 2,
    }

    analysis._retrieve_zoom(relayout_data)

    assert analysis.umap_plot.layout.xaxis.range == (0, 1)
    assert analysis.umap_plot.layout.yaxis.range == (0, 2)


# ---------------------------------------------------------------------------
# make_cnv_plot success path
# ---------------------------------------------------------------------------


def test_make_cnv_plot_success_path_calls_get_cnv_plot(
    make_analysis: AnalysisFactory,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    ref_dir = tmp_path / "ref"
    ref_dir.mkdir()
    analysis = make_analysis(reference_dir=ref_dir)
    analysis.idat_handler.id_to_path["sample1"] = tmp_path / "sample1"
    monkeypatch.setattr(core, "is_valid_idat_basepath", lambda path: True)

    captured: dict[str, Any] = {}

    def fake_get_cnv_plot(**kwargs: Any) -> str:
        captured.update(kwargs)
        return "FAKE_FIGURE"

    monkeypatch.setattr(core, "get_cnv_plot", fake_get_cnv_plot)
    analysis.make_cnv_plot("sample1", genes_sel=["TP53"])

    assert analysis.cnv_plot == "FAKE_FIGURE"
    assert captured["genes_sel"] == ("TP53",)


# ---------------------------------------------------------------------------
# precompute_cnvs / get_cnv
# ---------------------------------------------------------------------------


def test_precompute_cnvs_writes_for_given_ids(
    make_analysis: AnalysisFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    analysis = make_analysis()
    analysis.idat_handler.ids = ["s1", "s2"]  # type: ignore[misc]
    analysis.idat_handler.id_to_path = {
        "s1": Path("/x/s1"),
        "s2": Path("/x/s2"),
    }
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        core, "write_cnv_to_disk", lambda **kw: captured.update(kw)
    )
    analysis.precompute_cnvs(["s1"])
    assert captured["sample_path"] == [Path("/x/s1")]


def test_precompute_cnvs_defaults_to_all_idat_handler_ids(
    make_analysis: AnalysisFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    analysis = make_analysis()
    analysis.idat_handler.ids = ["s1", "s2"]  # type: ignore[misc]
    analysis.idat_handler.id_to_path = {
        "s1": Path("/x/s1"),
        "s2": Path("/x/s2"),
    }
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        core, "write_cnv_to_disk", lambda **kw: captured.update(kw)
    )
    analysis.precompute_cnvs()
    assert captured["sample_path"] == [Path("/x/s1"), Path("/x/s2")]


def test_get_cnv_returns_data_when_zip_exists(
    make_analysis: AnalysisFactory,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    analysis = make_analysis()
    analysis.idat_handler.id_to_path["sample1"] = tmp_path / "sample1"
    analysis.idat_handler.id_to_basename["sample1"] = "sample1_basename"
    monkeypatch.setattr(core, "write_cnv_to_disk", lambda **kw: None)
    zip_path = analysis.cnv_dir / ("sample1_basename" + core.ZIP_ENDING)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    zip_path.write_text("zip-content")
    monkeypatch.setattr(
        core,
        "read_cnv_data_from_disk",
        lambda cnv_dir, basename, extract: ("bins_df", "detail_df", "seg_df"),
    )
    result = analysis.get_cnv("sample1")
    assert result == ("bins_df", "detail_df", "seg_df")


def test_get_cnv_returns_none_tuple_when_zip_missing(
    make_analysis: AnalysisFactory,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    analysis = make_analysis()
    analysis.idat_handler.id_to_path["sample1"] = tmp_path / "sample1"
    analysis.idat_handler.id_to_basename["sample1"] = "sample1_basename"
    monkeypatch.setattr(core, "write_cnv_to_disk", lambda **kw: None)
    result = analysis.get_cnv("sample1", extract=["bins"])
    assert result == (None,)


# ---------------------------------------------------------------------------
# cn_summary success path
# ---------------------------------------------------------------------------


def test_cn_summary_returns_plot_and_dataframe(
    make_analysis: AnalysisFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(core, "_get_cgsegment", lambda: object())
    analysis = make_analysis(do_seg=True)
    analysis.idat_handler.id_to_basename = {"s1": "s1_base", "s2": "s2_base"}
    monkeypatch.setattr(analysis, "precompute_cnvs", lambda ids: None)
    monkeypatch.setattr(
        core, "get_cn_summary", lambda cnv_dir, basenames: ("PLOT", "DF")
    )
    plot, df = analysis.cn_summary(["s1", "s2"])
    assert plot == "PLOT"
    assert df == "DF"


# ---------------------------------------------------------------------------
# classify / _load_training_data success paths
# ---------------------------------------------------------------------------


def test_load_training_data_uses_cached_cpgs_when_values_only(
    make_analysis: AnalysisFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    analysis = make_analysis()
    analysis.clf_dir.mkdir(parents=True, exist_ok=True)
    cpgs_path = analysis.clf_dir / "cpgs.npy"
    np.save(cpgs_path, np.array(["cpg1", "cpg2"], dtype=str))

    captured: dict[str, Any] = {}

    def fake_get_betas(**kwargs: Any) -> pd.DataFrame:
        captured.update(kwargs)
        return pd.DataFrame({"cpg1": [0.5], "cpg2": [0.6]}, index=["s1"])

    monkeypatch.setattr(core, "get_betas", fake_get_betas)

    X, y, values = analysis._load_training_data(["s1"], values_only=True)

    assert X is None
    assert y is None
    assert values is not None
    assert list(values.columns) == ["cpg1", "cpg2"]
    assert list(captured["cpgs"]) == ["cpg1", "cpg2"]


def test_load_training_data_values_only_with_no_ids_returns_none_values(
    make_analysis: AnalysisFactory,
) -> None:
    analysis = make_analysis()
    analysis.clf_dir.mkdir(parents=True, exist_ok=True)
    np.save(analysis.clf_dir / "cpgs.npy", np.array(["cpg1"], dtype=str))
    X, y, values = analysis._load_training_data(None, values_only=True)
    assert (X, y, values) == (None, None, None)


def test_load_training_data_raises_when_no_feature_matrix_available(
    make_analysis: AnalysisFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    analysis = make_analysis(load_full_betas=False)
    monkeypatch.setattr(analysis, "set_betas", lambda: None)
    analysis.betas_sel = None
    analysis.feature_matrix = None
    monkeypatch.setattr(
        analysis.idat_handler,
        "features",
        lambda *a, **k: pd.Series(dtype=object),
    )
    with pytest.raises(ValueError, match="No valid feature matrix"):
        analysis._load_training_data(None)


# ---------------------------------------------------------------------------
# mlh1_report_pages
# ---------------------------------------------------------------------------


def test_mlh1_report_pages_generates_report_per_sample(
    make_analysis: AnalysisFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(core, "MLH1_CPGS", ["cpg1", "cpg2"])
    analysis = make_analysis()

    def fake_set_betas() -> None:
        analysis.betas_all = pd.DataFrame(
            {"cpg1": [0.1, 0.2], "cpg2": [0.3, 0.4]}, index=["s1", "s2"]
        )

    monkeypatch.setattr(analysis, "set_betas", fake_set_betas)
    analysis.idat_handler.id_to_path = {
        "s1": Path("/x/s1"),
        "s2": Path("/x/s2"),
    }
    monkeypatch.setattr(core.ArrayType, "from_idat", lambda path: "450k")

    class FakeManifestDF:
        IlmnID = ["cpg1", "cpg2"]

    class FakeManifest:
        def __init__(self, array_type: str) -> None:
            self.data_frame = FakeManifestDF()

    monkeypatch.setattr(core, "Manifest", FakeManifest)
    monkeypatch.setattr(
        core,
        "make_single_mlh1_report_page",
        lambda probes: f"<html>{list(probes.index)}</html>",
    )

    result = analysis.mlh1_report_pages(["s1", "s2"])

    assert len(result) == 2
    assert all("html" in page for page in result)


# ---------------------------------------------------------------------------
# get_app / run_app
# ---------------------------------------------------------------------------


def test_get_app_builds_dash_app_with_layout_and_callbacks(
    make_analysis: AnalysisFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    analysis = make_analysis()
    captured: dict[str, bool] = {}
    monkeypatch.setattr(
        core,
        "build_layout",
        lambda app, self_: captured.setdefault("layout_called", True),
    )
    monkeypatch.setattr(
        core,
        "register_callbacks",
        lambda app, self_: captured.setdefault("callbacks_called", True),
    )
    app = analysis.get_app()
    assert captured.get("layout_called") is True
    assert captured.get("callbacks_called") is True
    assert app is not None


def test_run_app_starts_dash_server_with_expected_settings(
    make_analysis: AnalysisFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    analysis = make_analysis(host="127.0.0.1", port=12345, debug=False)

    class FakeApp:
        def __init__(self) -> None:
            self.run_kwargs: dict[str, Any] | None = None

        def run(self, **kwargs: Any) -> None:
            self.run_kwargs = kwargs

    fake_app = FakeApp()
    monkeypatch.setattr(analysis, "get_app", lambda: fake_app)
    monkeypatch.setattr(core, "get_free_port", lambda port: port)

    analysis.run_app(open_tab=False)

    assert analysis.app is fake_app
    assert fake_app.run_kwargs is not None
    assert fake_app.run_kwargs["host"] == "127.0.0.1"
    assert fake_app.run_kwargs["port"] == 12345


def test_run_app_opens_browser_tab_when_requested(
    make_analysis: AnalysisFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    analysis = make_analysis()

    class FakeApp:
        def run(self, **kwargs: Any) -> None:
            pass

    monkeypatch.setattr(analysis, "get_app", lambda: FakeApp())
    monkeypatch.setattr(core, "get_free_port", lambda port: port)

    opened: dict[str, str] = {}
    monkeypatch.setattr(
        core.webbrowser,
        "open_new_tab",
        lambda url: opened.setdefault("url", url),
    )

    class ImmediateTimer:
        def __init__(
            self, interval: float, function: Callable[[], None]
        ) -> None:
            self.function = function

        def start(self) -> None:
            self.function()

    monkeypatch.setattr(core.threading, "Timer", ImmediateTimer)

    analysis.run_app(open_tab=True)

    assert opened["url"].startswith("http://")


# ---------------------------------------------------------------------------
# __repr__: DataFrame attribute formatting
# ---------------------------------------------------------------------------


def test_repr_formats_dataframe_attribute(
    make_analysis: AnalysisFactory,
) -> None:
    analysis = make_analysis()
    analysis.betas_sel = pd.DataFrame({"a": [1, 2]})
    text = repr(analysis)
    assert "betas_sel" in text
