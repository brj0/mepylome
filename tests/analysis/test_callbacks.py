"""Tests for the mepylome analysis Dash app layout and callbacks."""

from collections.abc import Callable
from pathlib import Path
from typing import Any  # or from typing import Any if preferred
from unittest.mock import MagicMock, patch

import pytest
from dash import html, no_update

from mepylome.analysis.callbacks import (
    build_layout,
    register_callbacks,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_analysis() -> MagicMock:
    """Completely mocked MethylAnalysis instance with layout defaults."""
    analysis = MagicMock()
    analysis.ids = ["sample1", "sample2"]
    analysis.ids_to_highlight = []
    analysis.idat_handler.columns = ["col1", "col2"]
    analysis.idat_handler.annotation = Path("/path/to/annotation.csv")
    analysis.idat_handler.selected_columns = ["col1"]
    analysis.analysis_dir = Path("/path/to/analysis")
    analysis.reference_dir = Path("/path/to/reference")
    analysis.test_dir = Path("/path/to/test")
    analysis.output_dir = Path("/path/to/output")
    analysis.cnv_dir = Path("/path/to/cnv")
    analysis.cpgs = ["cg00000029", "cg00000103"]
    analysis.n_cpgs = 1000
    analysis.prep = "functional"
    analysis.precalculate_cnv = False
    analysis.cpg_selection = "top"
    analysis.umap_parms = {
        "n_neighbors": 15,
        "metric": "euclidean",
        "min_dist": 0.1,
    }
    analysis._use_discrete_colors = True
    analysis.classifiers = ["classifier_1", "classifier_2"]
    analysis.umap_plot = {"data": []}
    analysis.cnv_plot = {"data": []}
    analysis.cnv_id = "sample1"

    # Progress bars
    analysis._prog_bar.get_progress.return_value = 50
    analysis._prog_bar.get_text.return_value = "Running UMAP..."
    analysis._clf_log = Path("/path/to/clf.log")

    return analysis


@pytest.fixture
def mock_dash_app() -> MagicMock:
    """Mocked Dash instance that captures registered layouts and callbacks."""
    app = MagicMock()
    app.callback_registry = {}

    def mock_callback(
        *args: Any, **kwargs: Any
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            app.callback_registry[func.__name__] = func
            return func

        return decorator

    app.callback = mock_callback
    return app


# =============================================================================
# Layout Tests
# =============================================================================


@patch("mepylome.analysis.callbacks.get_side_navigation")
@patch("mepylome.analysis.callbacks.get_navbar")
def test_build_layout(
    mock_navbar: MagicMock,
    mock_side_nav: MagicMock,
    mock_dash_app: MagicMock,
    mock_analysis: MagicMock,
) -> None:
    """Test that the application layout is correctly generated and assigned."""
    mock_navbar.return_value = html.Div("Navbar")
    mock_side_nav.return_value = html.Div("Sidebar")

    build_layout(mock_dash_app, mock_analysis)

    assert mock_dash_app.layout is not None
    mock_side_nav.assert_called_once()
    mock_navbar.assert_called_once()


# =============================================================================
# Callback Registration & Logic Tests
# =============================================================================


def test_register_callbacks_populates_registry(
    mock_dash_app: MagicMock, mock_analysis: MagicMock
) -> None:
    """Verify all expected callback handlers are correctly registered."""
    register_callbacks(mock_dash_app, mock_analysis)

    expected_callbacks = {
        "update_plots",
        "validate_analysis_path",
        "validate_annotation_file",
        "validate_reference_path",
        "validate_test_path",
        "validate_output_path",
        "on_umap_start_button_click",
        "toggle_console_out",
        "toggle_column_dropdown",
        "update_umap_parms",
        "precalculate_cnv_wrapper",
        "update_progress",
        "update_output",
        "on_clf_start_button_click",
    }
    assert expected_callbacks.issubset(mock_dash_app.callback_registry.keys())


@patch("mepylome.analysis.callbacks.callback_context")
def test_update_plots_trigger_ids_highlight(
    mock_ctx: MagicMock,
    mock_dash_app: MagicMock,
    mock_analysis: MagicMock,
) -> None:
    """Test plot updates when filtering or highlighting specific sample IDs."""
    register_callbacks(mock_dash_app, mock_analysis)
    update_plots = mock_dash_app.callback_registry["update_plots"]

    mock_ctx.triggered = [{"prop_id": "ids-to-highlight.value"}]

    umap, cnv, err = update_plots(None, ["sample1"], None, None, None, None)

    assert umap == mock_analysis.umap_plot
    assert cnv == no_update
    assert err == ""
    assert mock_analysis.ids_to_highlight == ["sample1"]
    mock_analysis._umap_plot_highlight.assert_called_once()


@patch("os.access")
def test_validate_analysis_path_valid_dir(
    mock_access: MagicMock,
    mock_dash_app: MagicMock,
    mock_analysis: MagicMock,
) -> None:
    """Verify validation logic accepts accessible directory paths."""
    register_callbacks(mock_dash_app, mock_analysis)
    validate_path = mock_dash_app.callback_registry["validate_analysis_path"]

    mock_access.return_value = True
    mock_analysis.idat_handler.annotation = "/mock/anno.csv"

    with patch("pathlib.Path.is_dir", return_value=True):
        valid, err, anno = validate_path("/valid/path")

        assert valid is True
        assert err == ""
        assert anno == "/mock/anno.csv"
        assert mock_analysis.analysis_dir == Path("/valid/path")


@patch("os.access")
def test_validate_analysis_path_protected_dir(
    mock_access: MagicMock,
    mock_dash_app: MagicMock,
    mock_analysis: MagicMock,
) -> None:
    """Verify validation flags errors on unreadable or protected paths."""
    register_callbacks(mock_dash_app, mock_analysis)
    validate_path = mock_dash_app.callback_registry["validate_analysis_path"]

    mock_access.return_value = False

    with patch("pathlib.Path.is_dir", return_value=True):
        valid, err, anno = validate_path("/protected/path")

        assert valid is False
        assert "Protected directory" in err
        assert anno == no_update


def test_toggle_column_dropdown(
    mock_dash_app: MagicMock, mock_analysis: MagicMock
) -> None:
    """Test dynamic visibility changes for balancing dropdown elements."""
    register_callbacks(mock_dash_app, mock_analysis)
    toggle_dropdown = mock_dash_app.callback_registry["toggle_column_dropdown"]

    assert toggle_dropdown("balanced") == {"display": "block"}
    assert toggle_dropdown("top") == {"display": "none"}


def test_update_umap_parms(
    mock_dash_app: MagicMock, mock_analysis: MagicMock
) -> None:
    """Ensure UMAP parameters map accurately onto the analysis instance."""
    register_callbacks(mock_dash_app, mock_analysis)
    update_parms = mock_dash_app.callback_registry["update_umap_parms"]

    update_parms(30, "cosine", 0.2)

    assert mock_analysis.umap_parms["n_neighbors"] == 30
    assert mock_analysis.umap_parms["metric"] == "cosine"
    assert mock_analysis.umap_parms["min_dist"] == 0.2


def test_on_umap_start_button_click_validation_errors(
    mock_dash_app: MagicMock, mock_analysis: MagicMock
) -> None:
    """Confirm the UMAP pipeline stops early if directory validity is false."""
    register_callbacks(mock_dash_app, mock_analysis)
    on_start = mock_dash_app.callback_registry["on_umap_start_button_click"]

    umap, ids, err, state = on_start(
        n_clicks=1,
        n_cpgs=5000,
        analysis_dir=".",
        annotation=".",
        reference_dir=".",
        test_dir=".",
        output_dir=".",
        prep="functional",
        analysis_dir_valid=False,
        test_dir_valid=True,
        output_dir_valid=True,
        precalculate_cnv=False,
        cpg_selection="top",
        balancing_feature="feat",
    )
    assert err == "Invalid Analysis path."
    assert umap == no_update


@patch("mepylome.analysis.callbacks.ensure_directory_exists")
def test_on_umap_start_button_click_success(
    mock_ensure_dir: MagicMock,
    mock_dash_app: MagicMock,
    mock_analysis: MagicMock,
) -> None:
    """Ensure UMAP generation runs cleanly when parameters are valid."""
    register_callbacks(mock_dash_app, mock_analysis)
    on_start = mock_dash_app.callback_registry["on_umap_start_button_click"]

    umap, ids, err, state = on_start(
        n_clicks=1,
        n_cpgs=5000,
        analysis_dir="/a",
        annotation="/b",
        reference_dir="/c",
        test_dir="/d",
        output_dir="/e",
        prep="functional",
        analysis_dir_valid=True,
        test_dir_valid=True,
        output_dir_valid=True,
        precalculate_cnv=["ON"],
        cpg_selection="balanced",
        balancing_feature="Age",
    )

    assert state == {"status": "umap_done"}
    assert mock_analysis.n_cpgs == 5000
    assert mock_analysis.balancing_feature == "Age"
    mock_analysis.make_umap.assert_called_once()
