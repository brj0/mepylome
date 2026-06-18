"""Pytest tests for mepylome/cli.py."""

import argparse
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mepylome.cli import (
    SmartFormatter,
    absolute_path,
    get_app_version,
    parse_args,
    print_welcome_message,
    start_mepylome,
)

# ---------------------------------------------------------------------------
# get_app_version
# ---------------------------------------------------------------------------


def test_get_app_version_returns_string() -> None:
    assert isinstance(get_app_version(), str)


def test_get_app_version_returns_unknown_on_missing_package() -> None:
    with patch("mepylome.cli.version", side_effect=PackageNotFoundError):
        assert get_app_version() == "unknown"


def test_get_app_version_returns_mocked_version() -> None:
    with patch("mepylome.cli.version", return_value="1.2.3"):
        assert get_app_version() == "1.2.3"


# ---------------------------------------------------------------------------
# print_welcome_message
# ---------------------------------------------------------------------------


def test_print_welcome_message_outputs_something(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with patch("mepylome.cli.get_app_version", return_value="0.0.0"):
        print_welcome_message()
    assert len(capsys.readouterr().out) > 50


def test_print_welcome_message_includes_version(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with patch("mepylome.cli.get_app_version", return_value="9.8.7"):
        print_welcome_message()
    assert "9.8.7" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# absolute_path
# ---------------------------------------------------------------------------


def test_absolute_path_returns_path_object(tmp_path: Path) -> None:
    assert isinstance(absolute_path(str(tmp_path)), Path)


def test_absolute_path_relative_becomes_absolute() -> None:
    assert absolute_path("some/relative/path").is_absolute()


def test_absolute_path_accepts_path_object(tmp_path: Path) -> None:
    result = absolute_path(tmp_path)
    assert isinstance(result, Path)
    assert result.is_absolute()


def test_absolute_path_preserves_absolute(tmp_path: Path) -> None:
    assert absolute_path(tmp_path) == tmp_path.absolute()


# ---------------------------------------------------------------------------
# SmartFormatter
# ---------------------------------------------------------------------------


@pytest.fixture
def formatter() -> SmartFormatter:
    return SmartFormatter(prog="test")


def test_smart_formatter_split_lines_basic(
    formatter: SmartFormatter,
) -> None:
    lines = formatter._split_lines("hello world", 80)
    assert isinstance(lines, list)
    assert len(lines) >= 1


def test_smart_formatter_split_lines_preserves_newlines(
    formatter: SmartFormatter,
) -> None:
    lines = formatter._split_lines("line one\nline two", 80)
    assert len(lines) == 2


def test_smart_formatter_split_lines_wraps_long_text(
    formatter: SmartFormatter,
) -> None:
    lines = formatter._split_lines("word " * 30, 40)
    assert len(lines) > 1


def test_smart_formatter_fill_text_returns_string(
    formatter: SmartFormatter,
) -> None:
    assert isinstance(formatter._fill_text("some text here", 80, "  "), str)


def test_smart_formatter_fill_text_respects_indent(
    formatter: SmartFormatter,
) -> None:
    result = formatter._fill_text("some text", 80, ">>")
    for line in result.splitlines():
        assert line.startswith(">>")


def test_smart_formatter_format_action_adds_extra_newline(
    formatter: SmartFormatter,
) -> None:
    action = argparse.Action(option_strings=["--foo"], dest="foo")
    action.help = "Some help text"
    assert formatter._format_action(action).endswith("\n\n")


# ---------------------------------------------------------------------------
# parse_args – defaults
# ---------------------------------------------------------------------------


def test_parse_args_returns_namespace() -> None:
    with patch("sys.argv", ["mepylome"]):
        assert isinstance(parse_args(), argparse.Namespace)


def test_parse_args_default_prep() -> None:
    with patch("sys.argv", ["mepylome"]):
        assert parse_args().prep == "illumina"


def test_parse_args_default_cpgs() -> None:
    with patch("sys.argv", ["mepylome"]):
        assert parse_args().cpgs == "auto"


def test_parse_args_default_n_cpgs() -> None:
    with patch("sys.argv", ["mepylome"]):
        assert parse_args().n_cpgs == 25000


def test_parse_args_default_host() -> None:
    with patch("sys.argv", ["mepylome"]):
        assert parse_args().host == "localhost"


def test_parse_args_default_port() -> None:
    with patch("sys.argv", ["mepylome"]):
        assert parse_args().port == 8050


def test_parse_args_default_debug_false() -> None:
    with patch("sys.argv", ["mepylome"]):
        assert parse_args().debug is False


def test_parse_args_default_do_seg_false() -> None:
    with patch("sys.argv", ["mepylome"]):
        assert parse_args().do_seg is False


def test_parse_args_default_precalculate_cnv_false() -> None:
    with patch("sys.argv", ["mepylome"]):
        assert parse_args().precalculate_cnv is False


def test_parse_args_default_load_full_betas_true() -> None:
    with patch("sys.argv", ["mepylome"]):
        assert parse_args().load_full_betas is True


def test_parse_args_default_command_none() -> None:
    with patch("sys.argv", ["mepylome"]):
        assert parse_args().command is None


# ---------------------------------------------------------------------------
# parse_args – flags and values
# ---------------------------------------------------------------------------


def test_parse_args_debug_flag() -> None:
    with patch("sys.argv", ["mepylome", "--debug"]):
        assert parse_args().debug is True


def test_parse_args_do_seg_short_flag() -> None:
    with patch("sys.argv", ["mepylome", "-s"]):
        assert parse_args().do_seg is True


def test_parse_args_precalculate_cnv_flag() -> None:
    with patch("sys.argv", ["mepylome", "--precalculate_cnv"]):
        assert parse_args().precalculate_cnv is True


def test_parse_args_no_load_full_betas_flag() -> None:
    with patch("sys.argv", ["mepylome", "--no_load_full_betas"]):
        assert parse_args().load_full_betas is False


def test_parse_args_tutorial_flag() -> None:
    with patch("sys.argv", ["mepylome", "--tutorial"]):
        assert parse_args().tutorial is True


def test_parse_args_use_gpu_flag() -> None:
    with patch("sys.argv", ["mepylome", "--use_gpu"]):
        assert parse_args().use_gpu is True


def test_parse_args_overlap_flag() -> None:
    with patch("sys.argv", ["mepylome", "--overlap"]):
        assert parse_args().overlap is True


def test_parse_args_host_option() -> None:
    with patch("sys.argv", ["mepylome", "--host", "0.0.0.0"]):
        assert parse_args().host == "0.0.0.0"


def test_parse_args_port_option() -> None:
    with patch("sys.argv", ["mepylome", "--port", "9090"]):
        assert parse_args().port == 9090


def test_parse_args_n_cpgs_option() -> None:
    with patch("sys.argv", ["mepylome", "--n_cpgs", "5000"]):
        assert parse_args().n_cpgs == 5000


def test_parse_args_n_jobs_cnv_option() -> None:
    with patch("sys.argv", ["mepylome", "--n_jobs_cnv", "4"]):
        assert parse_args().n_jobs_cnv == 4


@pytest.mark.parametrize("choice", ["illumina", "swan", "noob"])
def test_parse_args_prep_valid_choices(choice: str) -> None:
    with patch("sys.argv", ["mepylome", "--prep", choice]):
        assert parse_args().prep == choice


def test_parse_args_prep_invalid_choice_exits() -> None:
    with (
        patch("sys.argv", ["mepylome", "--prep", "invalid"]),
        pytest.raises(SystemExit),
    ):
        parse_args()


@pytest.mark.parametrize("choice", ["top", "random"])
def test_parse_args_cpg_selection_valid_choices(choice: str) -> None:
    with patch("sys.argv", ["mepylome", "--cpg_selection", choice]):
        assert parse_args().cpg_selection == choice


def test_parse_args_cpg_selection_invalid_exits() -> None:
    with (
        patch("sys.argv", ["mepylome", "--cpg_selection", "bad"]),
        pytest.raises(SystemExit),
    ):
        parse_args()


def test_parse_args_analysis_dir_is_path(tmp_path: Path) -> None:
    with patch("sys.argv", ["mepylome", "-a", str(tmp_path)]):
        assert isinstance(parse_args().analysis_dir, Path)


def test_parse_args_reference_dir_is_path(tmp_path: Path) -> None:
    with patch("sys.argv", ["mepylome", "-r", str(tmp_path)]):
        assert isinstance(parse_args().reference_dir, Path)


def test_parse_args_output_dir_is_path(tmp_path: Path) -> None:
    with patch("sys.argv", ["mepylome", "--output_dir", str(tmp_path)]):
        assert isinstance(parse_args().output_dir, Path)


def test_parse_args_cpgs_value() -> None:
    with patch("sys.argv", ["mepylome", "--cpgs", "450k"]):
        assert parse_args().cpgs == "450k"


def test_parse_args_version_exits_cleanly() -> None:
    with (
        patch("sys.argv", ["mepylome", "--version"]),
        pytest.raises(SystemExit) as exc_info,
    ):
        parse_args()
    assert exc_info.value.code == 0


# ---------------------------------------------------------------------------
# parse_args – download subcommand
# ---------------------------------------------------------------------------


def test_parse_args_download_command_set() -> None:
    with patch("sys.argv", ["mepylome", "download", "-d", "GSE12345"]):
        assert parse_args().command == "download"


def test_parse_args_download_single_dataset() -> None:
    with patch("sys.argv", ["mepylome", "download", "--dataset", "GSE12345"]):
        assert parse_args().dataset == ["GSE12345"]


def test_parse_args_download_multiple_datasets() -> None:
    with patch("sys.argv", ["mepylome", "download", "-d", "GSE111", "GSE222"]):
        assert parse_args().dataset == ["GSE111", "GSE222"]


def test_parse_args_download_save_dir_is_path() -> None:
    with patch("sys.argv", ["mepylome", "download"]):
        assert isinstance(parse_args().save_dir, Path)


def test_parse_args_download_save_dir_custom(tmp_path: Path) -> None:
    with patch("sys.argv", ["mepylome", "download", "-s", str(tmp_path)]):
        assert isinstance(parse_args().save_dir, Path)


def test_parse_args_download_idat_flag() -> None:
    with patch("sys.argv", ["mepylome", "download", "--idat"]):
        assert parse_args().idat is True


def test_parse_args_download_metadata_flag() -> None:
    with patch("sys.argv", ["mepylome", "download", "--metadata"]):
        assert parse_args().metadata is True


def test_parse_args_download_idat_default_false() -> None:
    with patch("sys.argv", ["mepylome", "download"]):
        assert parse_args().idat is False


def test_parse_args_download_metadata_default_false() -> None:
    with patch("sys.argv", ["mepylome", "download"]):
        assert parse_args().metadata is False


def test_parse_args_download_tcga_cart_is_path(tmp_path: Path) -> None:
    cart = tmp_path / "cart.json"
    cart.touch()
    with patch("sys.argv", ["mepylome", "download", "-c", str(cart)]):
        assert isinstance(parse_args().tcga_cart, Path)


def test_parse_args_download_tcga_clinical_is_path(tmp_path: Path) -> None:
    clin = tmp_path / "clinical.tsv"
    clin.touch()
    with patch("sys.argv", ["mepylome", "download", "-l", str(clin)]):
        assert isinstance(parse_args().tcga_clinical, Path)


# ---------------------------------------------------------------------------
# start_mepylome – download branch helpers
# ---------------------------------------------------------------------------


def _download_args(**overrides: object) -> argparse.Namespace:
    defaults = dict(
        command="download",
        dataset=["GSE1"],
        save_dir=Path("."),
        idat=False,
        metadata=False,
        tcga_cart=None,
        tcga_clinical=None,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_start_mepylome_download_both_when_neither_flag_set() -> None:
    mock_dl = MagicMock()
    with (
        patch("mepylome.cli.parse_args", return_value=_download_args()),
        patch("mepylome.utils.downloader.download_idats", mock_dl),
    ):
        start_mepylome()
    _, kwargs = mock_dl.call_args
    assert kwargs["idat"] is True
    assert kwargs["metadata"] is True


def test_start_mepylome_download_idat_only() -> None:
    mock_dl = MagicMock()
    with (
        patch(
            "mepylome.cli.parse_args",
            return_value=_download_args(idat=True),
        ),
        patch("mepylome.utils.downloader.download_idats", mock_dl),
    ):
        start_mepylome()
    _, kwargs = mock_dl.call_args
    assert kwargs["idat"] is True
    assert kwargs["metadata"] is False


def test_start_mepylome_download_metadata_only() -> None:
    mock_dl = MagicMock()
    with (
        patch(
            "mepylome.cli.parse_args",
            return_value=_download_args(metadata=True),
        ),
        patch("mepylome.utils.downloader.download_idats", mock_dl),
    ):
        start_mepylome()
    _, kwargs = mock_dl.call_args
    assert kwargs["idat"] is False
    assert kwargs["metadata"] is True


def test_start_mepylome_download_dataset_forwarded() -> None:
    mock_dl = MagicMock()
    with (
        patch(
            "mepylome.cli.parse_args",
            return_value=_download_args(dataset=["GSE111", "GSE222"]),
        ),
        patch("mepylome.utils.downloader.download_idats", mock_dl),
    ):
        start_mepylome()
    _, kwargs = mock_dl.call_args
    assert "GSE111" in kwargs["dataset"]
    assert "GSE222" in kwargs["dataset"]


def test_start_mepylome_download_tcga_metadata_missing_cart_raises(
    tmp_path: Path,
) -> None:
    clinical = tmp_path / "clinical.tsv"
    clinical.touch()
    with (
        patch(
            "mepylome.cli.parse_args",
            return_value=_download_args(
                metadata=True, tcga_cart=None, tcga_clinical=clinical
            ),
        ),
        pytest.raises(ValueError, match="tcga_cart"),
    ):
        start_mepylome()


def test_start_mepylome_download_tcga_metadata_missing_clinical_raises(
    tmp_path: Path,
) -> None:
    cart = tmp_path / "cart.json"
    cart.touch()
    with (
        patch(
            "mepylome.cli.parse_args",
            return_value=_download_args(
                metadata=True, tcga_cart=cart, tcga_clinical=None
            ),
        ),
        pytest.raises(ValueError, match="tcga_clinical"),
    ):
        start_mepylome()


def test_start_mepylome_download_tcga_dict_added_to_dataset(
    tmp_path: Path,
) -> None:
    cart = tmp_path / "cart.json"
    cart.touch()
    mock_dl = MagicMock()
    with (
        patch(
            "mepylome.cli.parse_args",
            return_value=_download_args(idat=True, tcga_cart=cart),
        ),
        patch("mepylome.utils.downloader.download_idats", mock_dl),
    ):
        start_mepylome()
    _, kwargs = mock_dl.call_args
    tcga_entries = [
        d
        for d in kwargs["dataset"]
        if isinstance(d, dict) and d.get("source") == "tcga"
    ]
    assert len(tcga_entries) == 1
    assert tcga_entries[0]["metadata_cart"] == cart


def test_start_mepylome_download_tcga_full_includes_clinical(
    tmp_path: Path,
) -> None:
    cart = tmp_path / "cart.json"
    cart.touch()
    clinical = tmp_path / "clinical.tsv"
    clinical.touch()
    mock_dl = MagicMock()
    with (
        patch(
            "mepylome.cli.parse_args",
            return_value=_download_args(
                tcga_cart=cart, tcga_clinical=clinical
            ),
        ),
        patch("mepylome.utils.downloader.download_idats", mock_dl),
    ):
        start_mepylome()
    _, kwargs = mock_dl.call_args
    tcga_entry = next(d for d in kwargs["dataset"] if isinstance(d, dict))
    assert "metadata_clinical" in tcga_entry


# ---------------------------------------------------------------------------
# start_mepylome – analysis branch
# ---------------------------------------------------------------------------


def _analysis_args(**overrides: object) -> argparse.Namespace:
    defaults: dict[str, object] = dict(
        command=None,
        tutorial=False,
        prep="illumina",
        cpgs="auto",
        n_cpgs=25000,
        n_jobs_cnv=None,
        host="localhost",
        port=8050,
        debug=False,
        do_seg=False,
        precalculate_cnv=False,
        load_full_betas=True,
        overlap=False,
        use_gpu=False,
        cpg_selection="top",
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_start_mepylome_analysis_creates_and_runs() -> None:
    mock_cls = MagicMock()
    mock_instance = MagicMock()
    mock_cls.return_value = mock_instance
    with (
        patch("mepylome.cli.parse_args", return_value=_analysis_args()),
        patch("mepylome.cli.print_welcome_message"),
        patch("mepylome.analysis.core.MethylAnalysis", mock_cls),
    ):
        start_mepylome()
    mock_cls.assert_called_once()
    mock_instance.run_app.assert_called_once_with(open_tab=True)


def test_start_mepylome_analysis_none_values_excluded() -> None:
    mock_cls = MagicMock()
    mock_cls.return_value = MagicMock()
    with (
        patch(
            "mepylome.cli.parse_args",
            return_value=_analysis_args(n_jobs_cnv=None),
        ),
        patch("mepylome.cli.print_welcome_message"),
        patch("mepylome.analysis.core.MethylAnalysis", mock_cls),
    ):
        start_mepylome()
    _, kwargs = mock_cls.call_args
    assert "n_jobs_cnv" not in kwargs


def test_start_mepylome_analysis_tutorial_key_removed() -> None:
    mock_cls = MagicMock()
    mock_cls.return_value = MagicMock()
    with (
        patch(
            "mepylome.cli.parse_args",
            return_value=_analysis_args(tutorial=True),
        ),
        patch("mepylome.cli.print_welcome_message"),
        patch("mepylome.analysis.core.MethylAnalysis", mock_cls),
        patch("mepylome.utils.setup_tutorial_files"),
        patch.object(Path, "exists", return_value=True),
    ):
        start_mepylome()
    _, kwargs = mock_cls.call_args
    assert "tutorial" not in kwargs
