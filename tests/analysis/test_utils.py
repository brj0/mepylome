"""Pytest for module-level helpers in mepylome.analysis.utils."""

from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mepylome.analysis.utils import (
    INVALID_PATH,
    METHYLATION_CLASS,
    TEST_CASE,
    BetasHandler,
    IdatHandler,
    ProgressBar,
    check_memory,
    convert_to_sentrix_ids,
    extract_sentrix_id,
    guess_annotation_file,
    interpret_methylation,
    read_dataframe,
)

# ---------------------------------------------------------------------------
# Tests for ProgressBar
# ---------------------------------------------------------------------------


def test_progress_bar_initial_state() -> None:
    """A fresh ProgressBar starts at 0 with the given max_value/text."""
    pbar = ProgressBar(max_value=10, text="loading")
    assert pbar.cur_value == 0
    assert pbar.max_value == 10
    assert pbar.text == "loading"


def test_progress_bar_increment_caps_at_max() -> None:
    """Incrementing past max_value clamps at max_value."""
    pbar = ProgressBar(max_value=5)
    pbar.increment(3)
    assert pbar.cur_value == 3
    pbar.increment(10)
    assert pbar.cur_value == 5


def test_progress_bar_get_progress_percentage() -> None:
    """get_progress returns an integer percentage of completion."""
    pbar = ProgressBar(max_value=4)
    pbar.increment(1)
    assert pbar.get_progress() == 25


def test_progress_bar_get_progress_zero_max_returns_100() -> None:
    """A ProgressBar with max_value 0 is considered fully complete."""
    pbar = ProgressBar(max_value=0)
    assert pbar.get_progress() == 100


def test_progress_bar_get_text_in_progress() -> None:
    """get_text shows 'cur/max text' while not yet complete."""
    pbar = ProgressBar(max_value=10, text="files")
    pbar.increment(3)
    assert pbar.get_text() == "3/10 files"


def test_progress_bar_get_text_complete() -> None:
    """get_text shows '100 %' once cur_value equals max_value."""
    pbar = ProgressBar(max_value=2)
    pbar.increment(2)
    assert pbar.get_text() == "100 %"


def test_progress_bar_reset() -> None:
    """reset() overwrites cur_value, max_value and text."""
    pbar = ProgressBar(max_value=10, text="old")
    pbar.increment(5)
    pbar.reset(max_value=20, cur_value=4, text="new")
    assert pbar.cur_value == 4
    assert pbar.max_value == 20
    assert pbar.text == "new"


def test_progress_bar_str_and_repr() -> None:
    """__str__ and __repr__ both produce the same descriptive text."""
    pbar = ProgressBar(max_value=10)
    pbar.increment(5)
    assert str(pbar) == repr(pbar)
    assert "cur_value: 5" in str(pbar)
    assert "max_value: 10" in str(pbar)


# ---------------------------------------------------------------------------
# Tests for read_dataframe
# ---------------------------------------------------------------------------


def test_read_dataframe_csv(tmp_path: Path) -> None:
    """A comma-separated CSV is read correctly."""
    path = tmp_path / "data.csv"
    path.write_text("a,b\n1,2\n3,4\n")
    df = read_dataframe(path)
    assert list(df.columns) == ["a", "b"]
    assert df.shape == (2, 2)


def test_read_dataframe_tsv(tmp_path: Path) -> None:
    """A tab-separated TSV is read correctly."""
    path = tmp_path / "data.tsv"
    path.write_text("a\tb\n1\t2\n")
    df = read_dataframe(path)
    assert list(df.columns) == ["a", "b"]
    assert df.shape == (1, 2)


def test_read_dataframe_unsupported_extension_raises(tmp_path: Path) -> None:
    """An unsupported file extension raises ValueError."""
    path = tmp_path / "data.txt"
    path.write_text("a,b\n1,2\n")
    with pytest.raises(ValueError, match="Unsupported file format"):
        read_dataframe(path)


def test_read_dataframe_passes_kwargs(tmp_path: Path) -> None:
    """Extra kwargs are forwarded to the underlying pandas reader."""
    path = tmp_path / "data.csv"
    path.write_text("a,b\n1,2\n")
    df = read_dataframe(path, usecols=["a"])
    assert list(df.columns) == ["a"]


def test_read_dataframe_accepts_str_path(tmp_path: Path) -> None:
    """A plain string path works just as well as a Path object."""
    path = tmp_path / "data.csv"
    path.write_text("a,b\n1,2\n")
    df = read_dataframe(str(path))
    assert df.shape == (1, 2)


# ---------------------------------------------------------------------------
# Tests for guess_annotation_file
# ---------------------------------------------------------------------------


def test_guess_annotation_file_finds_supported_file(tmp_path: Path) -> None:
    """A csv file in the directory is returned."""
    (tmp_path / "annotation.csv").write_text("a,b\n1,2\n")
    result = guess_annotation_file(tmp_path)
    assert result == tmp_path / "annotation.csv"


def test_guess_annotation_file_no_match_returns_invalid_path(
    tmp_path: Path,
) -> None:
    """If no spreadsheet-like file exists, INVALID_PATH is returned."""
    (tmp_path / "notes.txt").write_text("hello")
    result = guess_annotation_file(tmp_path)
    assert result == INVALID_PATH


def test_guess_annotation_file_prefers_shallowest_path(
    tmp_path: Path,
) -> None:
    """A file directly in the directory is preferred over a nested one."""
    nested = tmp_path / "sub"
    nested.mkdir()
    (nested / "a_nested.csv").write_text("a\n1\n")
    (tmp_path / "z_top.csv").write_text("a\n1\n")
    result = guess_annotation_file(tmp_path)
    assert result == tmp_path / "z_top.csv"


def test_guess_annotation_file_breaks_ties_by_name(tmp_path: Path) -> None:
    """Among files at the same depth, alphabetically first name wins."""
    (tmp_path / "b_file.csv").write_text("a\n1\n")
    (tmp_path / "a_file.csv").write_text("a\n1\n")
    result = guess_annotation_file(tmp_path)
    assert result == tmp_path / "a_file.csv"


# ---------------------------------------------------------------------------
# Tests for extract_sentrix_id
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("200925700125_R07C01", "200925700125_R07C01"),
        ("prefix_200925700125_R07C01_suffix", "200925700125_R07C01"),
        ("200925700125_R07C01.idat", "200925700125_R07C01"),
        ("sample_no_sentrix_id", "sample_no_sentrix_id"),
    ],
)
def test_extract_sentrix_id(text: str, expected: str) -> None:
    """Sentrix IDs are correctly extracted, or text is returned unchanged."""
    assert extract_sentrix_id(text) == expected


def test_extract_sentrix_id_uses_last_match() -> None:
    """When multiple Sentrix-like patterns exist, the last one is used."""
    text = "200925700125_R07C01_and_201530470054_R04C01"
    assert extract_sentrix_id(text) == "201530470054_R04C01"


def test_extract_sentrix_id_non_string_input() -> None:
    """Non-string input is stringified before matching."""
    assert extract_sentrix_id(12345) == 12345


# ---------------------------------------------------------------------------
# Tests for convert_to_sentrix_ids
# ---------------------------------------------------------------------------


def test_convert_to_sentrix_ids_none_returns_none() -> None:
    """None input returns None."""
    assert convert_to_sentrix_ids(None) is None


def test_convert_to_sentrix_ids_list() -> None:
    """A list of IDs is converted element-wise, preserving order."""
    data = ["prefix_200925700125_R07C01", "201530470054_R04C01_suffix"]
    result = convert_to_sentrix_ids(data)
    assert result == ["200925700125_R07C01", "201530470054_R04C01"]


def test_convert_to_sentrix_ids_set() -> None:
    """A set of IDs is converted, returning a set."""
    data = {"prefix_200925700125_R07C01"}
    result = convert_to_sentrix_ids(data)
    assert result == {"200925700125_R07C01"}


def test_convert_to_sentrix_ids_dict_converts_keys_only() -> None:
    """Dict keys are converted to Sentrix IDs, values are kept unchanged."""
    data = {"prefix_200925700125_R07C01": "value1"}
    result = convert_to_sentrix_ids(data)
    assert result == {"200925700125_R07C01": "value1"}


# ---------------------------------------------------------------------------
# Tests for check_memory
# ---------------------------------------------------------------------------


def test_check_memory_passes_when_sufficient() -> None:
    """No exception is raised when memory required is well within limits."""
    check_memory(nrows=1, ncols=1, dtype=np.float32)


def test_check_memory_raises_when_insufficient() -> None:
    """A MemoryError is raised if the requested array is too large."""
    with patch("mepylome.analysis.utils.psutil.virtual_memory") as mock_vm:
        mock_vm.return_value.available = 100  # 100 bytes available
        with pytest.raises(MemoryError, match="Not enough free memory"):
            check_memory(nrows=1_000_000, ncols=1_000_000, dtype=np.float32)


# ---------------------------------------------------------------------------
# Tests for interpret_methylation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "median, expected",
    [
        (0.0, "Unmethylated"),
        (0.19, "Unmethylated"),
        (0.2, "Unclear"),
        (0.3, "Unclear"),
        (0.4, "Unclear"),
        (0.41, "Methylated"),
        (1.0, "Methylated"),
    ],
)
def test_interpret_methylation(median: float, expected: str) -> None:
    """Boundary and interior values map to the correct interpretation."""
    assert interpret_methylation(median) == expected


# ---------------------------------------------------------------------------
# Tests for IdatHandler
# ---------------------------------------------------------------------------


def _fake_idat_basepaths(directory: Path, only_valid: bool = False) -> list:
    """Stand-in for mepylome.analysis.utils.idat_basepaths in tests.

    Always returns three samples named sample_1/2/3 relative to the given
    directory, except when the directory is literally named "test" (used to
    simulate a separate test_dir with non-overlapping sample IDs).
    """
    directory = Path(directory)
    if not directory.exists():
        return []
    if directory.name == "test":
        names = ("test_sample_1", "test_sample_2", "test_sample_3")
    else:
        names = ("sample_1", "sample_2", "sample_3")
    return [directory / name for name in names]


@pytest.fixture
def patched_idat_basepaths() -> Generator[MagicMock, None, None]:
    """Patch idat_basepaths so IdatHandler doesn't need real IDAT files."""
    with patch(
        "mepylome.analysis.utils.idat_basepaths",
        side_effect=_fake_idat_basepaths,
    ) as mock_fn:
        yield mock_fn


def test_idat_handler_no_annotation_uses_empty_classes(
    tmp_path: Path, patched_idat_basepaths: MagicMock
) -> None:
    """Without an annotation file, samples get an empty Methylation_Class."""
    handler = IdatHandler(analysis_dir=tmp_path)
    assert sorted(handler.ids) == ["sample_1", "sample_2", "sample_3"]
    assert METHYLATION_CLASS in handler.samples_annotated.columns
    assert (handler.samples_annotated[METHYLATION_CLASS] == "").all()


def test_idat_handler_len_matches_id_count(
    tmp_path: Path, patched_idat_basepaths: MagicMock
) -> None:
    """__len__ reflects the number of discovered IDAT samples."""
    handler = IdatHandler(analysis_dir=tmp_path)
    assert len(handler) == 3


def test_idat_handler_with_annotation_file(
    tmp_path: Path, patched_idat_basepaths: MagicMock
) -> None:
    """Annotation columns are joined onto the sample table by matching ID."""
    annotation_path = tmp_path / "annotation.csv"
    annotation_path.write_text(
        "ID,Methylation_Class\nsample_1,Tumor_A\nsample_2,Tumor_B\n"
    )
    handler = IdatHandler(analysis_dir=tmp_path, annotation=annotation_path)
    assert (
        handler.samples_annotated.loc["sample_1", "Methylation_Class"]
        == "Tumor_A"
    )
    assert (
        handler.samples_annotated.loc["sample_2", "Methylation_Class"]
        == "Tumor_B"
    )
    # sample_3 has no annotation row, so it's filled with an empty string.
    assert handler.samples_annotated.loc["sample_3", "Methylation_Class"] == ""


def test_idat_handler_analysis_ids_restricts_samples(
    tmp_path: Path, patched_idat_basepaths: MagicMock
) -> None:
    """Providing analysis_ids restricts which samples are kept."""
    handler = IdatHandler(
        analysis_dir=tmp_path, analysis_ids=["sample_1", "sample_2"]
    )
    assert sorted(handler.ids) == ["sample_1", "sample_2"]


def test_idat_handler_analysis_ids_missing_raises(
    tmp_path: Path, patched_idat_basepaths: MagicMock
) -> None:
    """A requested analysis_id not found on disk raises ValueError."""
    with pytest.raises(ValueError, match="not found on disk"):
        IdatHandler(analysis_dir=tmp_path, analysis_ids=["does_not_exist"])


def test_idat_handler_test_dir_adds_test_case_column(
    tmp_path: Path, patched_idat_basepaths: MagicMock
) -> None:
    """Samples from test_dir are marked True in the Test_Case column."""
    analysis_dir = tmp_path / "analysis"
    test_dir = tmp_path / "test"
    analysis_dir.mkdir()
    test_dir.mkdir()
    handler = IdatHandler(
        analysis_dir=analysis_dir,
        test_dir=test_dir,
        test_ids=["test_sample_1", "test_sample_2", "test_sample_3"],
    )
    assert TEST_CASE in handler.samples_annotated.columns
    test_rows = handler.samples_annotated.loc[handler.test_ids, TEST_CASE]
    assert test_rows.all()


def test_idat_handler_overlapping_dirs_warns(
    tmp_path: Path, patched_idat_basepaths: MagicMock
) -> None:
    """A warning is raised if analysis_dir and test_dir share samples."""
    same_dir = tmp_path
    with pytest.warns(UserWarning, match="share"):
        IdatHandler(analysis_dir=same_dir, test_dir=same_dir)


def test_idat_handler_properties(
    tmp_path: Path, patched_idat_basepaths: MagicMock
) -> None:
    """ids, idat_basenames, paths and columns expose consistent data."""
    handler = IdatHandler(analysis_dir=tmp_path)
    assert set(handler.ids) == set(handler.idat_basenames)
    assert len(handler.paths) == len(handler.ids)
    assert METHYLATION_CLASS in handler.columns


def test_idat_handler_features_default_column(
    tmp_path: Path, patched_idat_basepaths: MagicMock
) -> None:
    """features() with no args uses the first/selected column."""
    handler = IdatHandler(analysis_dir=tmp_path)
    features = handler.features()
    assert set(features.index) == set(handler.ids)


def test_idat_handler_features_multiple_columns_joined(
    tmp_path: Path, patched_idat_basepaths: MagicMock
) -> None:
    """features() joins multiple columns with the given separator."""
    annotation_path = tmp_path / "annotation.csv"
    annotation_path.write_text("ID,GEO,CNVs\nsample_1,SGT_103,Balanced\n")
    handler = IdatHandler(analysis_dir=tmp_path, annotation=annotation_path)
    features = handler.features(columns=["GEO", "CNVs"], separator="|")
    assert features.loc["sample_1"] == "SGT_103|Balanced"


def test_idat_handler_init_parameters_returns_raw_args(
    tmp_path: Path, patched_idat_basepaths: MagicMock
) -> None:
    """init_parameters() returns the originally-passed constructor args."""
    handler = IdatHandler(analysis_dir=tmp_path, overlap=True)
    params = handler.init_parameters()
    assert params["analysis_dir"] == tmp_path
    assert params["overlap"] is True


def test_idat_handler_str_contains_class_name(
    tmp_path: Path, patched_idat_basepaths: MagicMock
) -> None:
    """__str__/__repr__ include the class name and attribute listing."""
    handler = IdatHandler(analysis_dir=tmp_path)
    text = str(handler)
    assert "IdatHandler" in text
    assert text == repr(handler)


def test_idat_handler_overlap_filters_to_annotated_samples(
    tmp_path: Path, patched_idat_basepaths: MagicMock
) -> None:
    """With overlap=True, only samples present in annotation are kept."""
    annotation_path = tmp_path / "annotation.csv"
    annotation_path.write_text("ID,Methylation_Class\nsample_1,Tumor_A\n")
    handler = IdatHandler(
        analysis_dir=tmp_path, annotation=annotation_path, overlap=True
    )
    assert handler.analysis_ids == ["sample_1"]


# ---------------------------------------------------------------------------
# Tests for BetasHandler
# ---------------------------------------------------------------------------


def test_betas_handler_initial_state_empty(tmp_path: Path) -> None:
    """A freshly-created BetasHandler with no files has empty paths."""
    handler = BetasHandler(tmp_path)
    assert handler.paths == {}
    assert handler.invalid_paths == {}
    assert handler.filenames == []
    assert handler.invalid_filenames == []


def test_betas_handler_add_writes_file(tmp_path: Path) -> None:
    """add() writes beta values to disk under an array-type subdirectory."""
    from mepylome.dtypes.arrays import ArrayType

    handler = BetasHandler(tmp_path, array_cpgs={})
    betas = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    handler.add(betas, "sample_1", ArrayType.ILLUMINA_EPIC)
    expected_path = tmp_path / "epic" / "sample_1"
    assert expected_path.exists()
    written = np.fromfile(expected_path, dtype=np.float32)
    np.testing.assert_array_almost_equal(written, betas)


def test_betas_handler_add_error_writes_message(tmp_path: Path) -> None:
    """add_error() writes the error message to the error directory."""
    handler = BetasHandler(tmp_path)
    handler.add_error("bad_sample", "boom went the parser")
    error_path = tmp_path / "error" / "bad_sample"
    assert error_path.exists()
    assert error_path.read_text() == "boom went the parser"


def test_betas_handler_update_discovers_new_files(tmp_path: Path) -> None:
    """update() picks up files written to disk after construction."""
    from mepylome.dtypes.arrays import ArrayType

    handler = BetasHandler(tmp_path, array_cpgs={})
    assert handler.filenames == []
    betas = np.array([0.5], dtype=np.float32)
    handler.add(betas, "sample_new", ArrayType.ILLUMINA_EPIC)
    handler.update()
    assert "sample_new" in handler.filenames


def test_betas_handler_update_discovers_invalid_files(tmp_path: Path) -> None:
    """update() picks up error files written to the error directory."""
    handler = BetasHandler(tmp_path)
    handler.add_error("broken_sample", "some error")
    handler.update()
    assert "broken_sample" in handler.invalid_filenames


def test_betas_handler_get_returns_betas_for_requested_ids(
    tmp_path: Path,
) -> None:
    """get() builds a DataFrame of beta values indexed by sample ID."""
    from mepylome.dtypes.arrays import ArrayType

    cpgs = np.array(["cg1", "cg2", "cg3"])
    handler = BetasHandler(
        tmp_path, array_cpgs={ArrayType.ILLUMINA_EPIC: cpgs}
    )
    betas = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    handler.add(betas, "sample_1", ArrayType.ILLUMINA_EPIC)
    handler.update()

    idat_handler = MagicMock()
    idat_handler.idat_basenames = ["sample_1"]
    idat_handler.basename_to_id = {"sample_1": "id_1"}

    result = handler.get(idat_handler=idat_handler, cpgs=cpgs, parallel=False)
    assert list(result.index) == ["id_1"]
    np.testing.assert_array_almost_equal(result.loc["id_1"].values, betas)


def test_betas_handler_get_skips_invalid_filenames(tmp_path: Path) -> None:
    """get() excludes samples whose filename is marked invalid."""
    from mepylome.dtypes.arrays import ArrayType

    cpgs = np.array(["cg1"])
    handler = BetasHandler(
        tmp_path, array_cpgs={ArrayType.ILLUMINA_EPIC: cpgs}
    )
    handler.add(
        np.array([0.7], dtype=np.float32), "good", ArrayType.ILLUMINA_EPIC
    )
    handler.add_error("bad", "broken")
    handler.update()

    idat_handler = MagicMock()
    idat_handler.idat_basenames = ["good", "bad"]
    idat_handler.basename_to_id = {"good": "id_good", "bad": "id_bad"}

    result = handler.get(idat_handler=idat_handler, cpgs=cpgs, parallel=False)
    assert list(result.index) == ["id_good"]


def test_betas_handler_columnwise_variance_two_samples(
    tmp_path: Path,
) -> None:
    """columnwise_variance computes sample variance across stored betas."""
    from mepylome.dtypes.arrays import ArrayType

    cpgs = np.array(["cg1", "cg2"])
    handler = BetasHandler(
        tmp_path, array_cpgs={ArrayType.ILLUMINA_EPIC: cpgs}
    )
    handler.add(
        np.array([0.0, 1.0], dtype=np.float32), "s1", ArrayType.ILLUMINA_EPIC
    )
    handler.add(
        np.array([2.0, 1.0], dtype=np.float32), "s2", ArrayType.ILLUMINA_EPIC
    )
    handler.update()

    idat_handler = MagicMock()
    idat_handler.idat_basenames = ["s1", "s2"]

    variance = handler.columnwise_variance(
        idat_handler=idat_handler, cpgs=cpgs, parallel=False
    )
    # cg1 values: [0, 2] -> sample variance 2.0; cg2 values: [1, 1] -> 0.0
    np.testing.assert_allclose(variance, [2.0, 0.0], atol=1e-4)


def test_betas_handler_columnwise_variance_single_sample_returns_zeros(
    tmp_path: Path,
) -> None:
    """With fewer than 2 samples, variance is defined as all zeros."""
    from mepylome.dtypes.arrays import ArrayType

    cpgs = np.array(["cg1"])
    handler = BetasHandler(
        tmp_path, array_cpgs={ArrayType.ILLUMINA_EPIC: cpgs}
    )
    handler.add(
        np.array([0.3], dtype=np.float32), "s1", ArrayType.ILLUMINA_EPIC
    )
    handler.update()

    idat_handler = MagicMock()
    idat_handler.idat_basenames = ["s1"]

    variance = handler.columnwise_variance(
        idat_handler=idat_handler, cpgs=cpgs, parallel=False
    )
    np.testing.assert_array_equal(variance, [0.0])


def test_betas_handler_array_cpgs_ignores_error_dir(tmp_path: Path) -> None:
    """The 'error' subdirectory is never treated as an ArrayType directory."""
    handler = BetasHandler(tmp_path, array_cpgs={})
    handler.add_error("bad", "broken")
    # Accessing the property should not raise despite 'error' dir existing.
    assert handler.array_cpgs == {}


def test_betas_handler_array_cpgs_skips_invalid_directory_names(
    tmp_path: Path,
) -> None:
    """Subdirectories that aren't valid ArrayType values are skipped."""
    (tmp_path / "not_a_real_array_type").mkdir()
    handler = BetasHandler(tmp_path, array_cpgs={})
    assert handler.array_cpgs == {}
