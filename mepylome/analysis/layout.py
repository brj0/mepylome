"""Dash layout builders for the mepylome browser application."""

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import dash_bootstrap_components as dbc
import numpy as np
from dash import dcc, html

from mepylome.dtypes import Annotation, PrepType

logger = logging.getLogger(__name__)

OFF = "off"
ON = "on"

UMAP_METRICS = [
    "manhattan",
    "euclidean",
    "cosine",
    "chebyshev",
    "canberra",
    "braycurtis",
    "correlation",
    "hamming",
    "jaccard",
    "dice",
    "kulsinski",
    "ll_dirichlet",
    "hellinger",
    "rogerstanimoto",
    "sokalmichener",
    "sokalsneath",
    "yule",
]


def get_all_genes() -> list[str]:
    """Returns a list of names for all genes."""
    return Annotation.default_genes()["Name"].tolist()


def get_navbar() -> dbc.Navbar:
    """Returns a navigation bar with a logo and title."""
    logo = html.Img(src="/assets/mepylome.svg", height="30px")
    title = dbc.NavbarBrand("Methylation Analysis", className="ms-2")

    return dbc.Navbar(
        dbc.Container(
            [
                html.A(
                    dbc.Row(
                        [
                            dbc.Col(logo),
                            dbc.Col(title),
                        ],
                        align="center",
                        className="g-0",
                    ),
                    style={"textDecoration": "none"},
                ),
                dbc.NavbarToggler(id="navbar-toggler", n_clicks=0),
                html.Div(id="dummy-output", hidden=True),
            ]
        ),
        color="dark",
        dark=True,
    )


def get_side_navigation(
    *,
    sample_ids: Sequence[str],
    ids_to_highlight: Sequence[str],
    annotation_columns: Sequence[str],
    analysis_dir: Path,
    annotation: Path,
    reference_dir: Path,
    test_dir: Path,
    output_dir: Path,
    cpgs: np.ndarray,
    n_cpgs: int,
    prep: PrepType,
    precalculate: bool,
    cpg_selection: str,
    n_neighbors: int,
    metric: str,
    min_dist: float,
    use_discrete_colors: bool,
    custom_clfs: Sequence[dict[str, Any]],
) -> Any:
    """Generates a side navigation panel for setting up parameters."""
    n_cpgs_max = np.inf if len(cpgs) == 0 else len(cpgs)
    n_cpgs_max_str = "" if n_cpgs_max == np.inf else f" (max. {n_cpgs_max})"
    color_scheme = "discrete" if use_discrete_colors else "continuous"
    clf_options = {
        "vtl-kbest(k=10000)-et": "ExtraTreesClassifier",
        "vtl-kbest(k=10000)-lr(max_iter=10000)": "LogisticRegression",
        "vtl-kbest(k=10000)-rf": "RandomForestClassifier",
        "vtl-kbest(k=10000)-svc": "SVC(kernel='rbf')",
        "vtl-pca_auto-lr(max_iter=10000)": "PCALogisticRegression",
        "vtl-pca_auto-et": "PCAExtraTreesClassifier",
        "vtl-knn(weights='distance')": "KNeighborsClassifier",
        **{str(i): clf["name"] for i, clf in enumerate(custom_clfs)},
    }
    tabs = [
        dbc.Tab(
            label="Setting",
            tab_id="tab-setting",
            children=[
                dcc.Store(id="running-state"),
                dcc.Interval(
                    id="clock",
                    interval=500,
                    n_intervals=0,
                    max_intervals=-1,
                ),
                html.Div(
                    dbc.Button(
                        "Console",
                        id="toggle-button-setting",
                        n_clicks=0,
                        size="sm",
                    ),
                    className="d-grid gap-2",
                ),
                html.Div(
                    id="console-out-setting",
                    style={
                        "overflow": "hidden",
                        "fontFamily": "monospace",
                        "fontSize": "9px",
                        "whiteSpace": "pre-wrap",
                        "height": "40vh",
                        "display": "flex",
                        "flexDirection": "column-reverse",
                    },
                ),
                html.Br(),
                dbc.Progress(value=0, id="umap-progress-bar"),
                html.Br(),
                dbc.Row(
                    [
                        dbc.Col(
                            dbc.Button(
                                "Start",
                                id="start-button",
                                color="primary",
                                disabled=False,
                            ),
                            width={"size": 6},
                        ),
                    ],
                ),
                html.Div(id="output-div"),
                html.Br(),
                html.H6(f"Number of CpG sites{n_cpgs_max_str}"),
                dcc.Input(
                    id="num-cpgs",
                    type="number",
                    min=1,
                    max=n_cpgs_max,
                    step=1,
                    value=min(n_cpgs, len(cpgs) or n_cpgs),
                ),
                html.Br(),
                html.Br(),
                html.H6("Analysis directory"),
                dbc.Input(
                    id="analysis-dir",
                    valid=False,
                    value=str(analysis_dir),
                    type="text",
                ),
                html.Div(id="analysis-path-validation"),
                html.Br(),
                html.H6("Annotation file"),
                dbc.Input(
                    id="annotation-file",
                    valid=False,
                    value=str(annotation),
                    type="text",
                ),
                html.Div(id="annotation-file-validation"),
                html.Br(),
                html.H6("Reference directory (CNV neutral cases)"),
                dbc.Input(
                    id="reference-dir",
                    value=str(reference_dir),
                    type="text",
                ),
                html.Div(id="reference-path-validation"),
                html.Br(),
                html.H6("Test directory (new cases for analysis)"),
                dbc.Input(
                    id="test-dir",
                    value=str(test_dir),
                    type="text",
                ),
                html.Div(id="test-path-validation"),
                html.Br(),
                html.H6("Output directory"),
                dbc.Input(
                    id="output-dir",
                    valid=False,
                    value=str(output_dir),
                    type="text",
                ),
                html.Div(id="output-path-validation"),
                html.Br(),
                html.H6("IDAT preprocessing method"),
                dcc.Dropdown(
                    id="preprocessing-method",
                    options={
                        "illumina": "Illumina",
                        "swan": "SWAN",
                        "noob": "NOOB",
                        "raw": "No preprocessing",
                    },
                    value=prep,
                    clearable=False,
                    multi=False,
                ),
                html.Br(),
                html.H6("How should CpG's be selected"),
                dcc.Dropdown(
                    id="cpg-selection",
                    options={
                        "random": "Random",
                        "top": "Top varying (all samples)",
                        "balanced": ("Top varying (balanced selection)"),
                    },
                    value=cpg_selection,
                    clearable=False,
                    multi=False,
                ),
                html.Br(),
                html.Div(
                    [
                        html.H6("Select Balancing Column:"),
                        dcc.Dropdown(
                            id="balancing-column",
                            options=annotation_columns,
                            value=annotation_columns[0],
                            clearable=False,
                            multi=False,
                            placeholder="Choose a column for balancing...",
                        ),
                        html.Br(),
                    ],
                    id="balancing-column-container",
                    style={"display": "none"},
                ),
                html.H6("Calculate CNV"),
                dcc.Dropdown(
                    id="precalculate-cnv",
                    options={
                        ON: "Precalculate all (much longer!)",
                        OFF: "When clicking on dots",
                    },
                    value=ON if precalculate else OFF,
                    clearable=False,
                    multi=False,
                ),
            ],
        ),
        dbc.Tab(
            label="UMAP",
            tab_id="tab-umap",
            children=[
                html.Br(),
                html.H5("Determine UMAP settings."),
                html.Br(),
                html.H6(
                    "n_neighbors",
                    style={"font-family": "monospace"},
                ),
                dcc.Input(
                    id="umap-n_neighbors",
                    type="number",
                    min=2,
                    max=100,
                    step=1,
                    value=n_neighbors,
                ),
                html.Br(),
                html.Br(),
                html.H6("metric", style={"font-family": "monospace"}),
                dcc.Dropdown(
                    id="umap-metric",
                    value=metric,
                    options=UMAP_METRICS,
                ),
                html.Br(),
                html.H6("min_dist", style={"font-family": "monospace"}),
                dcc.Input(
                    id="umap-min_dist",
                    type="number",
                    min=0,
                    value=min_dist,
                ),
                html.Br(),
            ],
        ),
        dbc.Tab(
            label="Highlight",
            tab_id="tab-highlight",
            children=[
                html.Br(),
                html.Br(),
                html.H6("Sample IDs to highlight in UMAP"),
                dcc.Dropdown(
                    id="ids-to-highlight",
                    options=sample_ids,
                    value=ids_to_highlight,
                    multi=True,
                ),
                html.Br(),
                html.H6("Umap coloring"),
                dcc.Dropdown(
                    id="umap-annotation-color",
                    options=annotation_columns,
                    value=annotation_columns[0],
                    multi=True,
                ),
                html.Br(),
                html.H6("Color scheme"),
                dcc.Dropdown(
                    id="umap-color-scheme",
                    options={
                        "discrete": "Discrete Colors",
                        "continuous": "Continuous Colors",
                    },
                    value=color_scheme,
                    multi=False,
                ),
                html.Br(),
                html.H6("Genes to highlight in CNV"),
                dcc.Dropdown(
                    id="selected-genes",
                    options=get_all_genes(),
                    multi=True,
                ),
            ],
        ),
        dbc.Tab(
            label="Upload",
            tab_id="tab-upload",
            children=[
                html.Br(),
                html.Span(
                    "Do not upload too many files at once, else it "
                    "might not work!"
                ),
                html.Br(),
                dcc.Upload(
                    html.Div(
                        [
                            html.Span("Drag & Drop or "),
                            html.A("Select IDAT File pairs"),
                            html.Br(),
                        ],
                    ),
                    style={
                        "width": "100%",
                        "height": "60px",
                        "lineHeight": "60px",
                        "borderWidth": "1px",
                        "borderStyle": "dashed",
                        "borderRadius": "5px",
                        "textAlign": "center",
                    },
                    multiple=True,
                    id="upload-idat",
                ),
                html.Br(),
                html.Div(
                    dbc.Button(
                        "Console",
                        id="toggle-button-upload",
                        n_clicks=0,
                        size="sm",
                    ),
                    className="d-grid gap-2",
                ),
                html.Div(
                    id="console-out-upload",
                    style={
                        "overflow": "hidden",
                        "fontFamily": "monospace",
                        "fontSize": "9px",
                        "whiteSpace": "pre-wrap",
                        "height": "40vh",
                        "display": "flex",
                        "flexDirection": "column-reverse",
                    },
                ),
                html.Br(),
                html.Div(id="output-idat-upload"),
            ],
        ),
        dbc.Tab(
            label="Classify",
            tab_id="tab-classify",
            children=[
                html.Br(),
                html.H6("Classifiers to use"),
                dcc.Dropdown(
                    id="clf-clf-dropdown",
                    options=clf_options,
                    value=[
                        "vtl-kbest(k=10000)-lr(max_iter=10000)",
                        "vtl-kbest(k=10000)-et",
                    ],
                    multi=True,
                ),
                html.Br(),
                dbc.Row(
                    [
                        dbc.Col(
                            dbc.Button(
                                "Start",
                                id="clf-start-button",
                                color="primary",
                            ),
                            width={"size": 6},
                        ),
                    ],
                ),
                html.Div(id="clf-error-out"),
                html.Br(),
                html.Div(
                    id="clf-out",
                    style={
                        "overflow": "hidden",
                        "fontFamily": "monospace",
                        "fontSize": "9px",
                        "whiteSpace": "pre-wrap",
                        "display": "flex",
                        "flexDirection": "column-reverse",
                    },
                ),
            ],
        ),
    ]
    return dbc.Col(
        [dbc.Tabs(tabs, id="side-tab", active_tab="tab-setting")],
        width={"size": 2},
    )
