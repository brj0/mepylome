"""Dash layout construction and callback registration for MethylAnalysis."""

from __future__ import annotations

import base64
import logging
import os
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import dash_bootstrap_components as dbc
from dash import (
    Dash,
    Input,
    Output,
    State,
    callback_context,
    dcc,
    html,
    no_update,
)

from mepylome import LOG_FILE
from mepylome.analysis.methyl_aux import INVALID_PATH
from mepylome.analysis.methyl_layout import ON, get_navbar, get_side_navigation
from mepylome.analysis.methyl_plots import EMPTY_FIGURE
from mepylome.dtypes import PrepType
from mepylome.utils import (
    MEPYLOME_TMP_DIR,
    ensure_directory_exists,
)

if TYPE_CHECKING:
    from mepylome.analysis.methyl import MethylAnalysis

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path(MEPYLOME_TMP_DIR, "analysis")


def build_layout(app: Dash, analysis: MethylAnalysis) -> None:
    """Assemble and assign the Dash application layout.

    Constructs the full page layout from the navbar and side navigation panel,
    then assigns it to ``app.layout``. Called once during ``get_app()``.

    Args:
        app: The Dash application instance to populate.
        analysis: The active MethylAnalysis instance whose current state
            (directories, cpgs, umap parameters) seeds the initial UI values.
    """
    side_navigation = get_side_navigation(
        sample_ids=analysis.ids,
        ids_to_highlight=analysis.ids_to_highlight,
        annotation_columns=analysis.idat_handler.columns,
        analysis_dir=analysis.analysis_dir,
        annotation=analysis.idat_handler.annotation,
        reference_dir=analysis.reference_dir,
        test_dir=analysis.test_dir,
        output_dir=analysis.output_dir,
        cpgs=analysis.cpgs,
        n_cpgs=analysis.n_cpgs,
        prep=analysis.prep,
        precalculate=analysis.precalculate_cnv,
        cpg_selection=analysis.cpg_selection,
        n_neighbors=analysis.umap_parms["n_neighbors"],
        metric=analysis.umap_parms["metric"],
        min_dist=analysis.umap_parms["min_dist"],
        use_discrete_colors=analysis._use_discrete_colors,
        custom_clfs=analysis.classifiers,
    )
    dash_plots = dbc.Col(
        [
            dcc.Graph(
                id="umap-plot",
                figure=analysis.umap_plot,
                config={
                    "scrollZoom": True,
                    "doubleClick": "autosize",
                    "modeBarButtonsToRemove": ["lasso2d", "select"],
                    "displaylogo": False,
                },
                style={"height": "70vh"},
            ),
            html.Div(id="umap-error"),
            dcc.Graph(
                id="cnv-plot",
                figure=analysis.cnv_plot,
                config={
                    "scrollZoom": True,
                    "doubleClick": "reset",
                    "modeBarButtonsToRemove": ["lasso2d", "select"],
                    "displaylogo": False,
                },
            ),
        ],
        width={"size": 10},
    )
    app.layout = html.Div(
        [
            get_navbar(),
            dbc.Container(
                [
                    dbc.Row(
                        [side_navigation, dash_plots],
                        style={"margin-top": "20px"},
                    ),
                ],
                fluid=True,
            ),
        ],
    )


def register_callbacks(app: Dash, analysis: MethylAnalysis) -> None:
    """Register all Dash callbacks on the application.

    Binds all interactive callbacks to ``app``. Each callback closes over
    ``analysis`` and reads or mutates its state in response to user input. Must
    be called after ``build_layout`` so all referenced component IDs exist in
    the layout.

    Args:
        app: The Dash application instance to register callbacks on.
        analysis: The active MethylAnalysis instance. Callbacks read and write
            its attributes directly (e.g. ``analysis.prep``,
            ``analysis.umap_plot``) to keep analysis state in sync with the UI.
    """

    @app.callback(
        [
            Output("umap-plot", "figure"),
            Output("cnv-plot", "figure"),
            Output("umap-error", "children"),
        ],
        [
            Input("umap-plot", "clickData"),
            Input("ids-to-highlight", "value"),
            Input("umap-annotation-color", "value"),
            Input("umap-color-scheme", "value"),
            Input("selected-genes", "value"),
        ],
        State("umap-plot", "figure"),
    )
    def update_plots(
        click_data: dict[str, Any] | None,
        ids_to_highlight: list[str] | None,
        umap_color_columns: list[str] | None,
        umap_color_scheme: str | None,
        genes_sel: list[str] | None,
        curr_umap_plot: dict[str, Any] | None,
    ) -> tuple[Any, Any, Any]:
        def update_umap_plot() -> tuple[Any, Any, Any]:
            try:
                analysis.make_umap_plot()
                analysis._umap_plot_highlight(cnv_id=analysis.cnv_id)
                analysis._retrieve_zoom(curr_umap_plot)
                return analysis.umap_plot, no_update, ""
            except AttributeError:
                return no_update, no_update, no_update

        genes_sel = genes_sel or []
        trigger = callback_context.triggered[0]["prop_id"].split(".")[0]
        analysis.ids_to_highlight = ids_to_highlight or []
        if trigger == "ids-to-highlight":
            analysis._umap_plot_highlight()
            analysis._retrieve_zoom(curr_umap_plot)
            return analysis.umap_plot, no_update, ""
        if (
            trigger == "umap-annotation-color"
            and umap_color_columns is not None
        ):
            analysis.idat_handler.selected_columns = umap_color_columns
            update_umap_plot()
        if trigger == "umap-color-scheme" and umap_color_scheme is not None:
            analysis._use_discrete_colors = umap_color_scheme == "discrete"
            update_umap_plot()

        if trigger == "umap-plot" and isinstance(click_data, dict):
            points = click_data.get("points")
            if isinstance(points, list):
                first_point = points[0] if points else {}
                sample_id = first_point.get("hovertext")
                if sample_id is None:
                    return no_update, no_update, ""
                analysis._umap_plot_highlight(cnv_id=sample_id)
                analysis._retrieve_zoom(curr_umap_plot)
                try:
                    analysis.make_cnv_plot(sample_id, genes_sel)
                    return analysis.umap_plot, analysis.cnv_plot, ""
                except FileNotFoundError as exc:
                    logger.info("umap failed: %s", exc)
                    logger.info("sample_id: %s", sample_id)
                    logger.info("MethylAnalysis: %s", analysis)

                    error_message = (
                        f"{exc} - There is probably no CNV neutral "
                        "reference set for the array type of the "
                        "selected sample. To solve this, add missing CNV "
                        f"neutral sets to '{analysis.reference_dir}' and "
                        f"remove the error file in '{analysis.cnv_dir}'."
                    )
                    return analysis.umap_plot, EMPTY_FIGURE, error_message
                except Exception as exc:
                    logger.info("umap failed: %s", exc)
                    logger.info("sample_id: %s", sample_id)
                    logger.info("MethylAnalysis: %s", analysis)
                    return analysis.umap_plot, EMPTY_FIGURE, str(exc)
        if trigger == "selected-genes" and genes_sel is not None:
            try:
                assert analysis.cnv_id is not None
                analysis.make_cnv_plot(analysis.cnv_id, genes_sel)
                return no_update, analysis.cnv_plot, ""
            except Exception as exc:
                logger.info("selected-genes failed: %s", exc)
                logger.info("analysis.cnv_id: %s", analysis.cnv_id)
                logger.info("genes_sel: %s", genes_sel)
                return no_update, no_update, str(exc)
        return analysis.umap_plot, analysis.cnv_plot, ""

    @app.callback(
        [
            Output("analysis-dir", "valid"),
            Output("analysis-path-validation", "children"),
            Output("annotation-file", "value"),
        ],
        [Input("analysis-dir", "value")],
        prevent_initial_call=False,
    )
    def validate_analysis_path(input_path: str) -> tuple[bool, str, Any]:
        try:
            path = Path(input_path).expanduser()

            # Directory exists but not writable
            if path.is_dir() and not os.access(path, os.W_OK):
                return False, f"Protected directory: {path}", no_update

            # Directory exists and writable
            if path.is_dir():
                analysis.analysis_dir = path
                try:
                    annotation_str = str(analysis.idat_handler.annotation)
                except Exception:
                    analysis._idat_handler = None
                    analysis.annotation = INVALID_PATH
                    annotation_str = str(INVALID_PATH)

                return True, "", annotation_str

            return False, f"Not a directory: {path}", no_update

        except Exception as exc:
            logger.info(
                "An error occurred (1) (validate_analysis_path): %s", exc
            )
            return False, "Invalid path format", no_update

    @app.callback(
        [
            Output("annotation-file", "valid"),
            Output("annotation-file-validation", "children"),
            Output("umap-annotation-color", "options"),
            Output("umap-annotation-color", "value"),
        ],
        [Input("annotation-file", "value")],
        prevent_initial_call=False,
    )
    def validate_annotation_file(
        input_path: str,
    ) -> tuple[bool, str, Any, Any]:
        selection = analysis.idat_handler.selected_columns
        try:
            path = Path(input_path).expanduser()
            if path.exists() and not os.access(path, os.W_OK):
                return (
                    False,
                    f"Protected file: {path}",
                    no_update,
                    selection,
                )
            if path.exists():
                analysis.annotation = path
                return True, "", analysis.idat_handler.columns, selection
            return False, f"Not a file: {path}", no_update, selection
        except Exception as exc:
            logger.info(
                "An error occurred (1) (validate_annotation_file): %s", exc
            )
            return False, "Invalid path format", no_update, selection

    @app.callback(
        [
            Output("reference-dir", "valid"),
            Output("reference-path-validation", "children"),
        ],
        [Input("reference-dir", "value")],
        prevent_initial_call=False,
    )
    def validate_reference_path(input_path: str) -> tuple[bool, str]:
        try:
            path = Path(input_path).expanduser()
            if path.is_dir() and not os.access(path, os.W_OK):
                return False, f"Protected directory: {path}"
            if path.is_dir():
                analysis.reference_dir = path
                return True, ""
            return False, f"Not a directory: {path}"
        except Exception as exc:
            logger.info(
                "An error occurred (1) (validate_reference_path): %s", exc
            )
            return False, "Invalid path format"

    @app.callback(
        [
            Output("test-dir", "valid"),
            Output("test-path-validation", "children"),
        ],
        [Input("test-dir", "value")],
        prevent_initial_call=False,
    )
    def validate_test_path(input_path: str) -> tuple[bool, str]:
        try:
            path = Path(input_path).expanduser()
            if path.is_dir() and not os.access(path, os.W_OK):
                return False, f"Protected directory: {path}"
            if path.is_dir():
                analysis.test_dir = path
                return True, ""
            return False, f"Not a directory: {path}"
        except Exception as exc:
            logger.info("An error occurred (1) (validate_test_path): %s", exc)
            return False, "Invalid path format"

    @app.callback(
        [
            Output("output-dir", "valid"),
            Output("output-path-validation", "children"),
        ],
        [Input("output-dir", "value")],
        prevent_initial_call=False,
    )
    def validate_output_path(input_path: str) -> tuple[bool, str]:
        try:
            path = Path(input_path).expanduser()
            if path == DEFAULT_OUTPUT_DIR:
                analysis.output_dir = path
                return True, ""
            if path.is_dir() and not os.access(path, os.W_OK):
                return False, f"Protected directory: {path}"
            if path.is_dir():
                analysis.output_dir = path
                return True, ""
            return False, f"Not a directory: {path}"
        except Exception as exc:
            logger.info(
                "An error occurred (2) (validate_output_path): %s", exc
            )
            return False, "Invalid path format"

    @app.callback(
        [
            Output("umap-plot", "figure", allow_duplicate=True),
            Output("ids-to-highlight", "options"),
            Output("output-div", "children"),
            Output("running-state", "data"),
        ],
        [
            Input("start-button", "n_clicks"),
        ],
        [
            State("num-cpgs", "value"),
            State("analysis-dir", "value"),
            State("annotation-file", "value"),
            State("reference-dir", "value"),
            State("test-dir", "value"),
            State("output-dir", "value"),
            State("preprocessing-method", "value"),
            State("analysis-dir", "valid"),
            State("test-dir", "valid"),
            State("output-dir", "valid"),
            State("precalculate-cnv", "value"),
            State("cpg-selection", "value"),
            State("balancing-column", "value"),
        ],
        prevent_initial_call=True,
        running=[
            (Output("start-button", "disabled"), True, False),
        ],
    )
    def on_umap_start_button_click(
        n_clicks: int | None,
        n_cpgs: int | None,
        analysis_dir: str,
        annotation: str,
        reference_dir: str,
        test_dir: str,
        output_dir: str,
        prep: PrepType | None,
        analysis_dir_valid: bool | None,
        test_dir_valid: bool | None,
        output_dir_valid: bool | None,
        precalculate_cnv: bool | None,
        cpg_selection: str | None,
        balancing_feature: str | None,
    ) -> tuple[Any, Any, Any, dict[str, str]]:
        if not n_clicks:
            return no_update, no_update, "", {}

        error_message = None

        if n_cpgs is None:
            error_message = "Invalid no. of CpGs."
        elif not analysis_dir_valid:
            error_message = "Invalid Analysis path."
        elif not test_dir_valid:
            error_message = "Invalid Test path."
        elif not output_dir_valid:
            error_message = "Invalid Output path."
        elif prep is None:
            error_message = "Invalid preprocessing method."
        elif precalculate_cnv is None:
            error_message = "Invalid precalculation method."
        elif cpg_selection is None:
            error_message = "Invalid CpG selection method."
        elif balancing_feature is None:
            error_message = "Invalid balancing features."

        if error_message:
            return no_update, no_update, error_message, {}

        assert n_cpgs is not None
        assert prep is not None
        assert precalculate_cnv is not None
        assert cpg_selection is not None
        assert balancing_feature is not None

        analysis.n_cpgs = n_cpgs
        analysis.output_dir = Path(output_dir).expanduser()
        analysis.reference_dir = Path(reference_dir).expanduser()
        analysis.test_dir = Path(test_dir).expanduser()
        analysis.prep = prep
        analysis.precalculate_cnv = precalculate_cnv == ON
        analysis.cpg_selection = cpg_selection
        analysis.analysis_dir = Path(analysis_dir).expanduser()
        analysis.annotation = Path(annotation).expanduser()
        analysis.balancing_feature = (
            balancing_feature if cpg_selection == "balanced" else None
        )

        try:
            ensure_directory_exists(analysis.output_dir)
            analysis.make_umap()
        except Exception as exc:
            # BUG: Error 'no module named tqdm.auto' if mepylome is new
            # installed. This error disapears after running tutorial
            # Error was produced via cli with -a -A and -o -C 'epic'
            # --overlap -S 'top'. Maybe this error occurs only on Mac OS?
            logger.info("An error occurred (3): %s", exc)
        else:
            return (
                analysis.umap_plot,
                analysis.ids,
                no_update,
                {"status": "umap_done"},
            )
        return no_update, no_update, "", {}

    @app.callback(
        Output("console-out-setting", "style"),
        [Input("toggle-button-setting", "n_clicks")],
        [State("console-out-setting", "style")],
    )
    def toggle_console_out(
        n_clicks: int,
        current_style: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            **current_style,
            "display": "flex" if n_clicks % 2 == 0 else "none",
        }

    @app.callback(
        Output("balancing-column-container", "style"),
        Input("cpg-selection", "value"),
    )
    def toggle_column_dropdown(
        selected_method: str | None,
    ) -> dict[str, str]:
        """Show column dropdown only if 'balanced' is selected."""
        if selected_method == "balanced":
            return {"display": "block"}
        return {"display": "none"}

    @app.callback(
        [
            Input("umap-n_neighbors", "value"),
            Input("umap-metric", "value"),
            Input("umap-min_dist", "value"),
        ],
        prevent_initial_call=True,
    )
    def update_umap_parms(
        n_neighbors: int,
        metric: str,
        min_dist: float,
    ) -> None:
        analysis.umap_parms["n_neighbors"] = n_neighbors
        analysis.umap_parms["metric"] = metric
        analysis.umap_parms["min_dist"] = min_dist

    @app.callback(
        [
            # This ensures button is enabled in Colab
            Output("start-button", "disabled")
        ],
        [Input("running-state", "data")],
        prevent_initial_call=True,
        running=[
            (Output("start-button", "disabled"), True, False),
        ],
    )
    def precalculate_cnv_wrapper(
        state: dict[str, Any] | None,
    ) -> list[bool]:
        if (
            state
            and state.get("status") == "umap_done"
            and analysis.precalculate_cnv
        ):
            analysis.precompute_cnvs()
        return [False]

    @app.callback(
        [
            Output("umap-progress-bar", "value"),
            Output("umap-progress-bar", "label"),
            Output("console-out-setting", "children"),
            Output("console-out-upload", "children"),
            Output("clf-out", "children"),
        ],
        [Input("clock", "n_intervals")],
    )
    def update_progress(n: int) -> tuple[int, str, str, str, list[str]]:
        progress = analysis._prog_bar.get_progress()
        out_str = analysis._prog_bar.get_text()
        with LOG_FILE.open("r") as file:
            log_str = ""
            lines = file.readlines()
            n_top = 50
            last_lines = lines if len(lines) <= n_top else lines[-n_top:]
            for line in last_lines:
                log_str = log_str + line
        with analysis._clf_log.open("r") as file:
            clf_str = file.readlines()
        return progress, out_str, log_str, log_str, clf_str

    @app.callback(
        Output("output-idat-upload", "children"),
        Input("upload-idat", "contents"),
        State("upload-idat", "filename"),
    )
    def update_output(
        list_of_contents: list[str],
        list_of_names: list[str],
    ) -> list[html.Div] | None:
        logger.info("Uploading files...")

        def parse_contents(contents: str, filename: str) -> html.Div:
            file_path = analysis.test_dir / filename
            content_string = contents.split(",")[1]
            decoded = base64.b64decode(content_string)
            with file_path.open("wb") as f:
                f.write(decoded)
            logger.info("Upload of %s completed", filename)
            return html.Div(
                [
                    html.H6(filename),
                ]
            )

        if list_of_contents is not None:
            children = []
            for c, n in zip(list_of_contents, list_of_names, strict=True):
                children.append(parse_contents(c, n))
            # Reload idat handler now that there are new files
            analysis._idat_handler = None
            analysis._update_paths()
            # Update cpgs as uploaded files may have different array types
            analysis.cpgs = analysis._get_cpgs()
            return children

        return no_update

    @app.callback(
        Output("clf-error-out", "children"),
        [
            Input("clf-start-button", "n_clicks"),
        ],
        [
            State("clf-clf-dropdown", "value"),
        ],
        prevent_initial_call=True,
        running=[
            (Output("clf-start-button", "disabled"), True, False),
        ],
    )
    def on_clf_start_button_click(
        n_clicks: int | None,
        clf_list: Sequence[str] | None,
    ) -> str | Any | None:
        if not n_clicks:
            return no_update

        error_message = None
        if clf_list is None or len(clf_list) == 0:
            error_message = "No classifiers selected."
        elif analysis.cnv_id is None:
            error_message = "No sample selected."
        if error_message:
            return error_message

        try:
            assert clf_list is not None
            parsed_clf_list = [
                analysis.classifiers[int(x)] if x.isdigit() else x
                for x in clf_list
            ]
            _ = analysis.classify(
                ids=analysis.cnv_id, clf_list=parsed_clf_list
            )
        except Exception as exc:
            logger.info("An error occurred (4): %s", exc)
            return f"{exc}"
        return ""
