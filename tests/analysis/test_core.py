"""Pytest for module-level helpers in mepylome.analysis.core."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mepylome.analysis.core import (
    DualOutput,
    extract_sub_dataframe,
    get_balanced_indices,
    get_cpgs_from_file,
)

# ---------------------------------------------------------------------------
# Tests for extract_sub_dataframe
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


# ---------------------------------------------------------------------------
# Tests for get_balanced_indices
# ---------------------------------------------------------------------------


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
    """A class with zero effective members (via duplicate-free input) raises."""
    labels = ["a", "b"]
    with pytest.raises(ValueError, match="Only 1 sample"):
        get_balanced_indices(labels)


# ---------------------------------------------------------------------------
# Tests for get_cpgs_from_file
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Tests for DualOutput
# ---------------------------------------------------------------------------


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
