"""Methylation analysis tools including a Dash-based browser application.

This module provides a comprehensive set of tools for conducting methylation
analysis. The core functionality is encapsulated in the ``MethylAnalysis``
class, which manages the methylation analysis process and executes an
interactive web application for the exploration of methylation data.
"""

import base64
import logging
import os
import re
import sys
import threading
import time
import webbrowser
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly
import plotly.graph_objects as go
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
from sklearn.model_selection import (
    StratifiedKFold,
)

from mepylome.analysis.methyl_aux import (
    INVALID_PATH,
    IdatHandler,
    ProgressBar,
    get_betas,
    guess_annotation_file,
    read_dataframe,
)
from mepylome.analysis.methyl_clf import (
    fit_and_evaluate_clf,
)
from mepylome.analysis.methyl_plots import (
    EMPTY_FIGURE,
    get_cnv_plot,
    umap_plot_from_data,
    write_cnv_to_disk,
)
from mepylome.dtypes import (
    ZIP_ENDING,
    Annotation,
    ArrayType,
    Manifest,
    _get_cgsegment,
    _overlap_indices,
    get_cn_summary,
    input_args_id,
    is_valid_idat_basepath,
    read_cnv_data_from_disk,
)
from mepylome.utils import (
    MEPYLOME_TMP_DIR,
    ensure_directory_exists,
    get_free_port,
    log,
)

DEFAULT_OUTPUT_DIR = Path(MEPYLOME_TMP_DIR, "analysis")
DEFAULT_N_CPGS = 25000
ON = "on"
OFF = "off"
UMAP_METRICS = [
    "euclidean",
    "manhattan",
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


class DualOutput:
    """Enables to simultaneously write output to the terminal and file."""

    def __init__(self, filename):
        self.terminal = sys.stdout
        # Clean the file
        with open(filename, "w"):
            pass
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


LOG_DIR = MEPYLOME_TMP_DIR / "log"
ensure_directory_exists(MEPYLOME_TMP_DIR)
ensure_directory_exists(LOG_DIR)


def _log_file(suffix):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = uuid4().hex[:8]
    return LOG_DIR / f"{suffix}-{timestamp}-{unique_id}.log"


# Save console output to file
LOG_FILE = _log_file("stdout")
LOG_FILE.touch(exist_ok=True)
sys.stdout = DualOutput(LOG_FILE)


def get_all_genes():
    """Returns a list of names for all genes."""
    return Annotation.default_genes().df.Name.tolist()


def get_navbar():
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
    sample_ids,
    ids_to_highlight,
    annotation_columns,
    analysis_dir,
    annotation,
    reference_dir,
    output_dir,
    cpgs,
    n_cpgs,
    prep,
    precalculate,
    cpg_selection,
    n_neighbors,
    metric,
    min_dist,
    use_discrete_colors,
    custom_clfs,
):
    """Generates a side navigation panel for setting up parameters."""
    n_cpgs_max = np.inf if len(cpgs) == 0 else len(cpgs)
    n_cpgs_max_str = "" if n_cpgs_max == np.inf else f" (max. {n_cpgs_max})"
    color_scheme = "discrete" if use_discrete_colors else "continuous"
    clf_options = {
        "none-kbest-et": "ExtraTreesClassifier",
        "none-kbest-lr": "LinearRegression",
        "none-kbest-rf": "RandomForestClassifier",
        "none-kbest-svc_rbf": "SVC(kernel='rbf')",
        "none-pca-lr": "PCALinearRegression",
        "none-pca-et": "PCAExtraTreesClassifier",
        "none-none-knn": "KNeighborsClassifier",
        **{str(i): clf["name"] for i, clf in enumerate(custom_clfs)},
    }
    tabs = [
        dbc.Tab(
            label="Setting",
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
                        id="toggle-button",
                        n_clicks=0,
                        size="sm",
                    ),
                    className="d-grid gap-2",
                ),
                html.Div(
                    id="console-out",
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
                    valid=True,
                    value=str(analysis_dir),
                    type="text",
                ),
                html.Div(id="analysis-path-validation"),
                html.Br(),
                html.H6("Annotation file"),
                dbc.Input(
                    id="annotation-file",
                    valid=True,
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
                html.H6("Output directory"),
                dbc.Input(
                    id="output-dir",
                    valid=True,
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
                    multi=False,
                ),
                html.Br(),
                html.H6("Calculate CNV"),
                dcc.Dropdown(
                    id="precalculate-cnv",
                    options={
                        ON: "Precalculate all (much longer!)",
                        OFF: "When clicking on dots",
                    },
                    value=ON if precalculate else OFF,
                    multi=False,
                ),
                html.Br(),
                html.H6("How should CpG's be selected"),
                dcc.Dropdown(
                    id="cpg-selection",
                    options={
                        "random": "By random",
                        "top": "Take most varying CpG's (memory intensive!)",
                    },
                    value=cpg_selection,
                    multi=False,
                ),
            ],
        ),
        dbc.Tab(
            label="UMAP",
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
            children=[
                html.Br(),
                html.Br(),
                dcc.Upload(
                    [
                        "Drag & Drop or ",
                        html.A("Select IDAT File pairs"),
                    ],
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
                html.Div(id="output-idat-upload"),
            ],
        ),
        dbc.Tab(
            label="Classify",
            children=[
                html.Br(),
                html.H6("Classifiers to use"),
                dcc.Dropdown(
                    id="clf-clf-dropdown",
                    options=clf_options,
                    value=["none-kbest-lr", "none-kbest-et"],
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
    return dbc.Col([dbc.Tabs(tabs)], width={"size": 2})


def extract_sub_dataframe(data_frame, columns, fill=0.49):
    """Extracts a sub-dataframe based on the intersection of provided columns.

    Args:
        data_frame (pd.DataFrame): The original DataFrame.
        columns (list or array): The column names to be extracted.
        fill (float, optional): The value to fill for non-overlapping columns.
            Defaults to 0.49.

    Returns:
        pd.DataFrame: The sub-dataframe with the specified columns.
    """
    result_np = np.full((len(data_frame.index), len(columns)), fill)
    left_idx, right_idx = _overlap_indices(columns, data_frame.columns)
    result_np[:, left_idx] = data_frame.values[:, right_idx]
    return pd.DataFrame(result_np, columns=columns, index=data_frame.index)


def reordered_cpgs_by_variance(data_frame):
    """Reorders CpGs by descending column variance."""
    variances = np.var(data_frame.values, axis=0)
    sorted_columns = np.argsort(-variances, kind="stable")
    return data_frame.columns[sorted_columns]


def get_cpgs_from_file(input_path):
    """Reads CpG sites from a file and return a numpy array."""
    try:
        path = Path(input_path).expanduser()
        cpgs_df = read_dataframe(path, header=None)
        cpgs = [cpg for cpg in cpgs_df.values.flatten() if not pd.isna(cpg)]
        return np.array(cpgs)
    except Exception:
        return None


class MethylAnalysis:
    """Main class for methylation analysis including a GUI application.

    Main class for methylation analysis, providing methods for
    setting up analysis parameters, reading data, and running a Dash-based
    web application for data visualization.

    Args:
        analysis_dir (str or Path): Directory containing IDAT files for
            analysis.

        annotation (str or Path): Path to an annotation spreadsheet where
            Sentrix IDs are preferably listed in the first column. If not
            provided, the system will attempt to infer the correct column
            automatically, and if an annotation file is missing, it will try to
            detect a spreadsheet in the `analysis_dir` if available.

        reference_dir (str or Path): Directory containing reference IDAT files.

        output_dir (str or Path): Directory where output files will be saved
            (default: "/tmp/mepylome/analysis").

        test_dir (Path or None): Directory for test files, including new cases
            for analysis or validation. Files uploaded via the GUI will be
            placed here. If set to `None`, the application will automatically
            use a temporary directory.

        prep (str): Prepreparation method used for data preparation (default:
            "illumina").

        cpgs (str or np.ndarray): Array of CpG sites to analyze, or one of
            '450k', 'epic', 'epicv2', or 'auto' for automatic detection. It can
            also be a path to a CSV file containing the CpG sites.

        cpg_blacklist (set or list, optional): A list or set of CpG sites to
            exclude. Default is None.

        n_cpgs (int): Number of CpG sites to select for UMAP (default: 25000).

        classifiers (object or list of objects, optional): Classifier model(s)
            (default: None).
            Each classifier can be provided as:

            - A dictionary containing:

                - 'model' (object): The classifier model object as defined
                  below (required).
                - 'name' (str, optional): A name for the classifier (default:
                  "Custom_Classifier_<index>").
                - 'cv' (int or cross-validation generator, optional):
                  Cross-validation strategy (default: `self.cv_default`).

            - A classifier model object (e.g., `RandomForestClassifier()`,
              `none-kbest-rf`), in which case the 'name' and 'cv' are
              automatically generated (see above). A classifier model can be
              one of:

                - A scikit-learn classifier object (trained or untrained).
                - A string in the format `"scaler-selector-classifier"`. See
                  the documentation of `fit_and_evaluate_clf` in
                  `mepylome.analysis.methyl_clf` for all valid values.
                - A custom class, that inherits from `TrainedClassifier`.

        cv_default (int or cross-validation generator, optional): Determines
            the default cross-validation splitting strategy (default: 5).

        n_jobs (int): Number of parallel processes to run for classifying
            (default: 1). Choose -1 for using all available cores.

        precalculate_cnv (bool): Flag to precalculate CNV information by
            invoking 'precompute_cnvs' (default: False).

        load_full_betas (bool): Flag to load beta values for all CpG sites
            into memory (default: True).

        feature_matrix (pandas.DataFrame or numpy.ndarray, optional): A
            user-provided feature matrix to be used for UMAP dimensionality
            reduction. If provided, this matrix will be used instead of
            `betas_top`. If not provided (default is None), the `betas_top`
            containing methylation beta values will be used for UMAP.

        overlap (bool): Flag to analyze only samples that are both in the
            analysis directory and within the annotation file (default: False).

        sample_ids (list, optional): A list of sample IDs. If provided, the
            analysis will be restricted to these samples only. If `None`, the
            analysis will include all available samples.

        cpg_selection (str): Method to select CpG sites ('top' or 'random')
            (default: 'top'). For 'top', CpG sites are selected based on their
            variation, taking the most varying ones. For 'random', CpG sites
            are randomly selected.

        do_seg (bool): Flag to enable segmentation analysis on CNV data
            (default: False).

        host (str): Host address for the Dash application (default:
            'localhost').

        port (int): Port number for the Dash application (default: 8050).

        debug (bool): Flag to enable debug mode for the Dash application
            (default: False).

        umap_parms (dict): Parameters for UMAP algorithm (default: {'metric':
            'euclidean', 'min_dist': 0.1, 'n_neighbors': 15, 'verbose': True}).

        verbose (bool): Flag to enable verbose logging (default: True).

    Note:
        Many parameters can be modified within the GUI application after
        initialization, but not all.

    Attributes:
        analysis_dir (Path): Path to the directory containing IDAT files for
            analysis.

        annotation (Path): Path to an annotation spreadsheet where
            Sentrix IDs are preferably listed in the first column. If not
            provided, the system will attempt to infer the correct column
            automatically, and if an annotation file is missing, it will try to
            detect a spreadsheet in the `analysis_dir` if available.

        overlap (bool): Flag to analyze only samples that are both in the
            analysis directory and within the annotation file (default: False).

        sample_ids (list, optional): A list of sample IDs. If provided, the
            analysis will be restricted to these samples only. If `None`, the
            analysis will include all available samples.

        n_cpgs (int): Number of CpG sites to select for UMAP (default: 25000).

        n_jobs (int): Number of parallel processes to run for classifying
            (default: 1). Choose -1 for using all available cores.

        reference_dir (Path): Path to the reference directory containing
            reference IDAT files.

        output_dir (Path): Path to the directory where output files will be
            saved (default: "/tmp/mepylome/analysis").

        test_dir (Path or None): Directory for test files, including new cases
            for analysis or validation. Files uploaded via the GUI will be
            placed here. If set to `None`, the application will automatically
            use a temporary directory.

        prep (str): Prepreparation method used for data preparation (default:
            "illumina").

        cpg_selection (str): Method to select CpG sites ('top' or 'random')
            (default: 'top'). For 'top', CpG sites are selected based on their
            variation, taking the most varying ones. For 'random', CpG sites
            are randomly selected.

        host (str): Host address for the Dash application (default:
            'localhost').

        port (int): Port number for the Dash application (default: 8050).

        debug (bool): Flag to enable debug mode for the Dash application
            (default: False).

        verbose (bool): Flag to enable verbose logging (default: True).

        cnv_dir (Path): Directory for CNV (Copy Number Variation) data,
            initially set to None.

        umap_dir (Path): Directory for UMAP (Uniform Manifold Approximation
            and Projection) data, initially set to None.

        umap_cpgs (numpy.array): CpG sites for UMAP analysis, initially set to
            None.

        precalculate_cnv (bool): Flag to precalculate CNV information by
            invoking 'precalculate_all_cnvs' (default: False).

        load_full_betas (bool): Flag to load beta values for all CpG sites into
            memory (default: True).

        betas_top (pandas.DataFrame): Dataframe containing top variable beta
            values, initially set to None.

        betas_all (pandas.DataFrame): Dataframe containing beta values for all
            CpG sites, initially set to None.

        feature_matrix (pandas.DataFrame or numpy.ndarray, optional): A
            user-provided feature matrix to be used for UMAP dimensionality
            reduction. If provided, this matrix will be used instead of
            `betas_top` for UMAP plots and instead of `betas_all` for
            classifying (default: None).

        betas_path (Path): Path to the betas directory, initially set to
            None.

        umap_plot (plotly.Figure): Plot for UMAP, initially set to
            EMPTY_FIGURE.

        umap_plot_path (Path): Path to the CSV file containing the UMAP
            plot data, initially set to None.

        umap_df (pandas.DataFrame): Dataframe containing UMAP data, initially
            set to None.

        umap_parms (dict): Parameters for UMAP algorithm (default: {'metric':
            'euclidean', 'min_dist': 0.1, 'n_neighbors': 15, 'verbose': True}).

        raw_umap_plot (plotly.Figure): Raw UMAP plot data, initially set to
            None.

        cnv_plot (plotly.Figure): Plot for CNV (Copy Number Variation)
            visualization, initially set to EMPTY_FIGURE.

        cnv_id (str): ID for CNV (Copy Number Variation) sample, initially
            set to None.

        dropdown_id (list): ID for dropdown selection, initially set to
            None.

        ids (list): List of IDs, initially empty.

        ids_to_highlight (list): IDs to highlight in the plot, initially
            set to None.

        app (dash.dash.Dash): Dash application object, initially set to None.


    Raises:
        ValueError: If `cpg_selection` is not 'top' or 'random'.

    Examples:
        >>> # Basic usage
        >>> from mepylome import MethylAnalysis
        >>> analysis0 = MethylAnalysis()
        >>> analysis0.run_app()
        >>> # Usage if directories are known in advance
        >>> analysis1 = MethylAnalysis(
        >>>     analysis_dir='/path/to/idat/dir',
        >>>     reference_dir='/path/to/reference/idat/dir',
        >>>     annotation='/path/to/annotation/spread/sheat/with/2/cols',
        >>>     output_dir='/path/to/mepylome/output',
        >>> )
        >>> analysis1.run_app()
    """

    def __init__(
        self,
        analysis_dir=INVALID_PATH,
        *,
        annotation=INVALID_PATH,
        reference_dir=INVALID_PATH,
        output_dir=DEFAULT_OUTPUT_DIR,
        test_dir=INVALID_PATH,
        prep="illumina",
        cpgs="auto",
        cpg_blacklist=None,
        n_cpgs=DEFAULT_N_CPGS,
        classifiers=None,
        cv_default=5,
        n_jobs=1,
        precalculate_cnv=False,
        load_full_betas=True,
        feature_matrix=None,
        overlap=False,
        sample_ids=None,
        cpg_selection="top",
        do_seg=False,
        host="localhost",
        port=8050,
        debug=False,
        umap_parms=None,
        verbose=True,
    ):
        self.umap_cpgs = None
        self.analysis_dir = Path(analysis_dir).expanduser()
        self.annotation = Path(annotation).expanduser()
        self.overlap = overlap
        self.sample_ids = None if sample_ids is None else list(sample_ids)
        self._idat_handler = None
        self.n_cpgs = n_cpgs
        self.n_jobs = n_jobs
        self.cpg_blacklist = set(cpg_blacklist or [])
        self.cpg_selection = cpg_selection
        self.host = host
        self.port = port
        self.debug = debug
        self.reference_dir = Path(reference_dir).expanduser()
        self.output_dir = Path(output_dir).expanduser()
        self.test_dir = Path(test_dir).expanduser()
        self.cnv_dir = None
        self.umap_dir = None
        self.clf_dir = None
        self.cv_default = cv_default
        self.prep = prep
        self.precalculate_cnv = precalculate_cnv
        self.load_full_betas = load_full_betas
        self.feature_matrix = feature_matrix
        self.umap_plot = EMPTY_FIGURE
        self.umap_plot_path = None
        self.betas_top = None
        self.betas_all = None
        self.betas_path = None
        self.umap_df = None
        self.umap_parms = MethylAnalysis._get_umap_parms(umap_parms)
        self.verbose = verbose
        self.cnv_plot = EMPTY_FIGURE
        self.raw_umap_plot = None
        self.cnv_id = None
        self.dropdown_id = None
        self.ids = []
        self.ids_to_highlight = None
        self.app = None

        ensure_directory_exists(self.output_dir)

        self._classifiers = classifiers
        self._prog_bar = ProgressBar()
        self._use_discrete_colors = True
        self._internal_cpgs_hash = None
        self._old_selected_columns = None
        self._sorted_cpgs = None
        self._clf_log = _log_file(f"{self.analysis_dir.name}-clf")
        self._clf_log.touch(exist_ok=True)
        self._testdir_provided = self.test_dir != INVALID_PATH

        if self.cpg_selection not in ["top", "random"]:
            msg = "Invalid 'cpg_selection' (expected: 'top' or 'random')"
            raise ValueError(msg)

        if not self.load_full_betas and self.cpg_selection == "top":
            msg = (
                "If 'load_full_betas' is disabled, 'cpg_selection' must be "
                " set to 'random'"
            )
            raise ValueError(msg)

        if self.annotation == INVALID_PATH:
            self.annotation = guess_annotation_file(
                self.analysis_dir, self.verbose
            )
        if not self.annotation.exists() and self.verbose:
            log("[MethylAnalysis] No annotation file found")
        if self.verbose:
            log("[MethylAnalysis] Try to import cbseg or linear_segment...")
        self.do_seg = False if _get_cgsegment(verbose=True) is None else do_seg

        # Set test dir, as it is needed by _get_cpgs
        self._set_test_dir()
        self._cpgs = self._get_cpgs(cpgs)

        self._prev_vars = self._get_vars_or_hashes()
        self._update_paths()

        self.read_umap_plot_from_disk()

        if self.verbose:
            log("[MethylAnalysis] Initialization completed.")

    @property
    def cpgs(self):
        """Get the CpG sites to analyze."""
        return self._cpgs

    @cpgs.setter
    def cpgs(self, cpgs):
        """Set the CpG sites for analysis.

        Args:
            cpgs (str or np.ndarray): An array of CpG sites to analyze, or one
                of the following strings: '450k', 'epic', 'epicv2', or 'auto'
                for automatic detection. It can also be a path to a CSV file
                containing the CpG sites.

        Raises:
            ValueError: If the provided cpgs value is not a valid type or
                format.
        """
        self._cpgs = self._get_cpgs(cpgs)

    @property
    def idat_handler(self):
        """Handles the management of IDAT files and associated metadata.

        Returns:
            IdatHandler: An instance of IdatHandler configured with current
            settings.
        """

        def _to_list(sample_ids):
            if sample_ids is None:
                return []
            if isinstance(sample_ids, (pd.Series, np.ndarray)):
                return sample_ids.tolist()
            return list(sample_ids)

        if self._idat_handler is not None:
            self._old_selected_columns = self._idat_handler.selected_columns

        if (
            self._idat_handler is None
            or self.analysis_dir != self._idat_handler.idat_dir
            or self.annotation != self._idat_handler.annotation_file
            or self.overlap != self._idat_handler.overlap
            or self.test_dir != self._idat_handler.test_dir
            or _to_list(self.sample_ids)
            != _to_list(self._idat_handler.sample_ids)
        ):
            self._idat_handler = IdatHandler(
                idat_dir=self.analysis_dir,
                annotation_file=self.annotation,
                overlap=self.overlap,
                test_dir=self.test_dir,
                sample_ids=self.sample_ids,
            )

        # Restore old selected columns if they are still valid
        if self._old_selected_columns and all(
            x in self._idat_handler.samples_annotated.columns
            for x in self._old_selected_columns
        ):
            self._idat_handler.selected_columns = self._old_selected_columns

        return self._idat_handler

    @idat_handler.setter
    def idat_handler(self, value):
        self._idat_handler = value

    @property
    def classifiers(self):
        """Retrieves the configuration for classifiers.

        This property returns a list of dictionaries, where each dictionary
        includes:

        - 'name' (str): A human-readable name for the classifier
          (e.g., 'Random Forest').
        - 'model' (object): The classifier model instance.
        - 'cv' (int or cross-validation generator): Determines the
          cross-validation splitting strategy.

        Returns:
            list of dict: Classifier configurations.
        """
        return self._get_classifiers(self._classifiers)

    @classifiers.setter
    def classifiers(self, classifiers):
        """Sets the configuration for classifiers.

        This setter accepts either a single classifier model or a list of
        classifier models. If a model is provided without additional
        configuration, it will be automatically wrapped in a dictionary with
        default values for 'name' and 'cv'.

        Args:
            classifiers (object or list of objects): A classifier model or a
                list of classifier models and configurations. This argument is
                handled the same way as `self.classifiers`. For full details on
                the format and options, refer to the docstring for
                `self.classifiers`.

        Examples:
            >>> clf = {
            >>>     "model": RandomForestClassifier(),
            >>>     "name": "Custom RF",
            >>>     "cv": 10,
            >>> }
            >>> analysis.classifiers = ["none-kbest-rf", clf]
            >>> analysis.classifiers
            [{'model': 'none-kbest-rf', 'name': 'Custom_Classifier_0', 'cv':
            StratifiedKFold(n_splits=5, random_state=None, shuffle=True)},
            {'model': RandomForestClassifier(), 'name': 'Custom RF', 'cv':
            StratifiedKFold(n_splits=10, random_state=None, shuffle=True)}]

        """
        self._classifiers = classifiers

    def _get_classifiers(self, classifiers):
        if classifiers is None:
            return []

        if not isinstance(classifiers, list):
            classifiers = [classifiers]

        result = []
        for i, clf in enumerate(classifiers):
            clf_copy = clf.copy() if isinstance(clf, dict) else {"model": clf}
            clf_copy["name"] = clf_copy.get("name", f"Custom_Classifier_{i}")
            cv = clf_copy.get("cv", self.cv_default)
            if isinstance(cv, int):
                cv = StratifiedKFold(n_splits=cv, shuffle=True)
            clf_copy["cv"] = cv

            if "model" not in clf_copy:
                msg = "Each classifier in 'classifiers' must have a 'model'."
                raise ValueError(msg)
            result.append(clf_copy)

        return result

    @staticmethod
    def _get_umap_parms(umap_parms):
        """Returns UMAP parameters with defaults if not provided."""
        umap_parms = {} if umap_parms is None else umap_parms
        default = {
            "metric": "euclidean",
            "min_dist": 0.1,
            "n_neighbors": 15,
            "verbose": True,
        }
        return {**default, **umap_parms}

    def _get_cpgs_hash(self):
        """Returns or computes and caches the hash of the CpG array."""
        if self._internal_cpgs_hash is None:
            self._internal_cpgs_hash = input_args_id(self.cpgs)
        return self._internal_cpgs_hash

    def _get_test_files_hash(self):
        """Returns the hash of the test files."""
        if not self.test_dir.exists():
            return ""
        return input_args_id(
            extra_hash=sorted(str(x) for x in self.test_dir.iterdir())
        )

    def _get_vars_or_hashes(self):
        """Returns current variables and hashes."""
        return {
            "analysis_dir": self.analysis_dir,
            "prep": self.prep,
            "n_cpgs": self.n_cpgs,
            "cpg_selection": self.cpg_selection,
            "cpgs": self._get_cpgs_hash(),
            "test_files": self._get_test_files_hash(),
            "sample_ids": self.sample_ids,
        }

    def _get_cpgs(self, input_var="auto"):
        """Returns CpG sites based on the provided input variable."""
        self._internal_cpgs_hash = None

        def exclude_blacklist(cpgs):
            return np.sort(np.array(list(set(cpgs) - self.cpg_blacklist)))

        if self.verbose:
            log("[MethylAnalysis] Determine CpG sites...")

        if isinstance(input_var, (np.ndarray, set, list, pd.Index)):
            return exclude_blacklist(input_var)

        cpgs_from_file = get_cpgs_from_file(input_var)
        if cpgs_from_file is not None:
            return exclude_blacklist(cpgs_from_file)

        valid_str_parms = ["auto", "450k", "epic", "epicv2"]

        if isinstance(input_var, str):
            input_var = input_var.split("+")

        if "auto" in input_var:

            def get_grn_idat_path(base_path):
                idat_path = Path(str(base_path) + "_Grn.idat")
                if not idat_path.exists():
                    idat_path = Path(str(base_path) + "_Grn.idat.gz")
                return idat_path

            input_var = {
                str(ArrayType.from_idat(get_grn_idat_path(path)))
                for path in self.idat_handler.paths
            }
            if self.verbose:
                log(
                    f"[MethylAnalysis] The following array types were "
                    f"detected: {input_var}"
                )
            input_var = list(input_var - {str(ArrayType.UNKNOWN)})

        if all(x in valid_str_parms for x in input_var):
            if not input_var:
                return np.array([])

            if "all" in input_var:
                input_var = valid_str_parms

            if self.verbose:
                log(
                    "[MethylAnalysis] Load manifests and "
                    "calculate CpG overlap..."
                )

            cpg_sets = [
                set(Manifest(array_type).methylation_probes)
                for array_type in valid_str_parms[1:]
                if array_type in input_var
            ]
            cpgs = set.intersection(*cpg_sets)
            return exclude_blacklist(cpgs)

        msg = (
            "'cpgs' must be one of the following:\n"
            "- a list, set, or array of CpG sites\n"
            "- a '+' joined string of valid parameters: {valid_str_parms}"
        )
        raise ValueError(msg)

    def _set_test_dir(self):
        if self._testdir_provided:
            ensure_directory_exists(self.test_dir)
            return
        test_hash_key = input_args_id(
            self.analysis_dir,
            "test",
        )
        self.test_dir = self.output_dir / f"{test_hash_key}"
        ensure_directory_exists(self.test_dir)

    def _update_paths(self):
        """Update file paths and directories based on current settings.

        This method recalculates and updates various internal paths and
        directories whenever changes occur in related attributes like
        'analysis_dir', 'prep', 'n_cpgs', 'cpg_selection', and uploaded files
        in 'test_dir'. It ensures that the correct paths are set for storing
        beta values, copy number variation data, uploaded files, and UMAP
        plots.
        """
        if self.verbose:
            log("[MethylAnalysis] Update filepaths...")
        if not self.output_dir.exists():
            return
        if self._prev_vars["analysis_dir"] != self.analysis_dir:
            self.cpgs = self._get_cpgs()

        # betas dir
        betas_hash_key = input_args_id(
            self.analysis_dir,
            "betas",
            self.prep,
        )
        self.betas_path = self.output_dir / f"{betas_hash_key}"

        # cnv dir
        cnv_hash_key = input_args_id(
            self.analysis_dir,
            "cnv",
            self.reference_dir,
            self.prep,
            self.do_seg,
        )
        self.cnv_dir = self.output_dir / f"{cnv_hash_key}"
        ensure_directory_exists(self.cnv_dir)

        # test dir
        self._set_test_dir()

        # umap dir
        cur_vars = self._get_vars_or_hashes()
        umap_hash_key = input_args_id(
            self.analysis_dir,
            "umap",
            self.prep,
            self.n_cpgs,
            self.cpg_selection,
            extra_hash=[
                cur_vars["cpgs"],
                cur_vars["test_files"],
                self.annotation,
            ],
        )
        self.umap_dir = self.output_dir / f"{umap_hash_key}"
        self.umap_plot_path = self.umap_dir / "umap_plot.csv"
        ensure_directory_exists(self.umap_dir)

        # clf dir
        clf_label = re.sub(
            r"\W+", "", "_".join(self.idat_handler.selected_columns)
        )
        clf_hash_key = input_args_id(
            self.analysis_dir,
            "clf",
            clf_label,
            self.feature_matrix,
            self.prep,
            extra_hash=[
                cur_vars["cpgs"],
                self.annotation,
            ],
        )
        self.clf_dir = self.output_dir / f"{clf_hash_key}"

        # Reset betas_top if necessary
        dependencies = [
            "analysis_dir",
            "prep",
            "n_cpgs",
            "cpg_selection",
            "cpgs",
            "test_files",
            "sample_ids",
        ]
        if any(self._prev_vars[arg] != cur_vars[arg] for arg in dependencies):
            self.betas_top = None

        # Reset betas_all if necessary
        dependencies = [
            "analysis_dir",
            "prep",
            "cpg_selection",
            "cpgs",
            "test_files",
            "sample_ids",
        ]
        if any(self._prev_vars[arg] != cur_vars[arg] for arg in dependencies):
            self.betas_all = None

        # Update variables/hashes
        self._prev_vars = cur_vars

    def make_umap(self):
        """Generates the UMAP plot.

        This method extracts the beta values required for UMAP computation,
        computes the UMAP 2D embedding, and creates and displays the UMAP plot
        based on the computed embedding.
        """
        self._prog_bar.reset(len(self.idat_handler), text="(betas)")
        self.set_betas()
        self._prog_bar.reset(1, 1)
        self.compute_umap()
        self.make_umap_plot()

    def compute_umap(self):
        """Applies the UMAP algorithm on 'betas_top'.

        Saves the 2D embedding in 'umap_df' and and on disk.

        Raises:
            AttributeError: If a dimension mismatch occurs, or if 'betas_top'
                is not set.
        """
        if self.betas_top is None and self.feature_matrix is None:
            msg = "'betas_top' is not set. First run 'set_betas'"
            raise AttributeError(msg)
        if self.verbose:
            log("[MethylAnalysis] Importing umap library...")
        import umap

        matrix_to_use = (
            self.betas_top
            if self.feature_matrix is None
            else self.feature_matrix
        )
        if len(self.idat_handler.ids) != len(matrix_to_use):
            if self.feature_matrix is not None:
                msg = (
                    "Dimension mismatch 0: 'feature_matrix' has not the same "
                    "number of rows as there are samples in "
                    "'idat_handler.ids'."
                )
            else:
                msg = (
                    "Dimension mismatch 1: Analysis files may have changed. "
                    f"Try to delete cached files in {self.output_dir}."
                )
            raise AttributeError(msg)
        if self.verbose:
            shape = matrix_to_use.shape
            log(
                "[MethylAnalysis] Starting UMAP for matrix with shape "
                f"{shape}..."
            )
        umap_2d = umap.UMAP(**self.umap_parms).fit_transform(matrix_to_use)
        umap_df = pd.DataFrame(
            umap_2d,
            columns=["Umap_x", "Umap_y"],
            index=self.idat_handler.ids,
        )
        self.umap_df = pd.concat(
            [
                umap_df,
                self.idat_handler.samples_annotated,
            ],
            axis=1,
        )
        self.umap_df.to_csv(self.umap_plot_path, sep="\t", index=True)

    def make_umap_plot(self):
        """Generates a UMAP plot from the given 2D embedding.

        Generates the UMAP plot from the data provided in 'umap_df'. The
        scatter plot color is based on selected columns in
        'idat_handler.selected_columns'.

        Raises:
            AttributeError: If a dimension mismatch occurs, or if 'umap_df' is
                not set.
        """
        if self.verbose:
            log("[MethylAnalysis] Make UMAP plot...")
        if self.umap_df is None:
            msg = "'umap_df' not set. Run 'make_umap' instead."
            raise AttributeError(msg)
        self.ids = self.umap_df.index
        umap_color = self.idat_handler.features()
        if len(umap_color) != len(self.umap_df):
            msg = (
                "Dimension mismatch 2: Analysis files may have changed. "
                f"Try to delete cached files in {self.output_dir}."
            )
            raise AttributeError(msg)
        self.umap_df["Umap_color"] = [str(x) for x in umap_color]
        self.umap_plot = umap_plot_from_data(
            self.umap_df, self._use_discrete_colors
        )
        self.umap_plot = self.umap_plot.update_layout(
            margin={"l": 0, "r": 0, "t": 30, "b": 0},
        )
        self.raw_umap_plot = self.umap_plot
        self.dropdown_id = None
        self._umap_plot_highlight()

    def read_umap_plot_from_disk(self):
        """Reads UMAP plot from disk if available from previous analysis."""
        if self.umap_plot_path is not None and self.umap_plot_path.exists():
            if self.verbose:
                log("[MethylAnalysis] Read umap plot from disk...")
            self.umap_df = pd.read_csv(
                self.umap_plot_path, sep="\t", index_col=0
            )
            self.umap_df = self.umap_df.fillna("")
            try:
                self.make_umap_plot()
            except AttributeError:
                log("[MethylAnalysis] Probable dimension mismatch.")

    def set_betas(self):
        """Sets the beta values DataFrame ('betas_top') for further analysis.

        This method reads the IDAT files located in 'analysis_dir', extracts
        the beta values, and saves them locally in 'output_dir'. Depending on
        the configuration ('cpg_selection' and 'load_full_betas' flags), it
        either extracts a subset of CpGs for UMAP computation or loads all CpGs
        for subsequent processing into memory.

        Raises:
            ValueError: If no valid samples are found.
        """
        if len(self.idat_handler) == 0:
            msg = "No valid samples found"
            raise ValueError(msg)

        self._update_paths()

        def get_random_cpgs():
            if self.verbose:
                log("[MethylAnalysis] Get random CpG's...")
            rng = np.random.default_rng()
            random_idx = np.sort(
                rng.choice(len(self.cpgs), self.n_cpgs, replace=False)
            )
            return self.cpgs[random_idx]

        def _get_betas(cpgs):
            if self.verbose:
                log("[MethylAnalysis] Get beta values...")
            return get_betas(
                idat_handler=self.idat_handler,
                cpgs=cpgs,
                prep=self.prep,
                betas_path=self.betas_path,
                pbar=self._prog_bar,
            )

        def _extract_sub_dataframe():
            if self.verbose:
                log("[MethylAnalysis] Extract beta values...")
            if self.cpg_selection == "random":
                return extract_sub_dataframe(self.betas_all, self.umap_cpgs)
            return self.betas_all[self._sorted_cpgs[: self.n_cpgs]]

        if self.cpg_selection == "random":
            self.umap_cpgs = get_random_cpgs()

        if self.betas_all is not None:
            self.betas_top = _extract_sub_dataframe()

        elif self.load_full_betas or self.cpg_selection == "top":
            self.betas_all = _get_betas(self.cpgs)
            self._sorted_cpgs = reordered_cpgs_by_variance(self.betas_all)
            self.betas_top = _extract_sub_dataframe()

        else:
            self.betas_top = _get_betas(self.umap_cpgs)

    def _get_coordinates(self, sample_id):
        """Returns UMAP 2D embedding coordinates."""
        return self.umap_df[self.umap_df.index == sample_id].iloc[0][
            ["Umap_x", "Umap_y"]
        ]

    def _umap_plot_highlight(self, cnv_id=None):
        """Highlights the selected samples and the sample selected for CNV."""
        if cnv_id is not None:
            self.cnv_id = cnv_id
        self.dropdown_id = (
            [] if self.ids_to_highlight is None else self.ids_to_highlight
        )
        self.umap_plot = go.Figure(self.raw_umap_plot)
        for id_ in self.dropdown_id:
            x, y = self._get_coordinates(id_)
            self.umap_plot.add_annotation(
                x=x,
                y=y,
                text=f"{id_}",
                showarrow=True,
                arrowhead=2,
                arrowcolor="blue",
                font={"color": "blue"},
            )
        if self.cnv_id is not None:
            x, y = self._get_coordinates(self.cnv_id)
            self.umap_plot.add_annotation(
                x=x,
                y=y,
                text=f"CNV: {self.cnv_id}",
                showarrow=True,
                arrowhead=2,
            )

    def _retrieve_zoom(self, current_plot):
        """Retrieves and applies the zoom level to the UMAP plot."""
        self.umap_plot.layout.xaxis = current_plot["layout"]["xaxis"]
        self.umap_plot.layout.yaxis = current_plot["layout"]["yaxis"]

    def make_cnv_plot(self, sample_id, genes_sel=None):
        """Generates a copy number variation (CNV) plot for a specific sample.

        This method generates a CNV plot for the specified sample and
        optionally highlights specific genes within the plot.

        Args:
            sample_id (str): ID of the sample for which CNV plot is generated.
            genes_sel (list or None, optional): List of specific genes to
                highlight in the plot.

        Raises:
            FileNotFoundError: If the specified sample ID is not found in the
                analysis directory or if the reference directory does not
                exist.
        """
        idat_basepath = self.idat_handler.id_to_path[sample_id]
        if not is_valid_idat_basepath(idat_basepath):
            msg = f"Sample {sample_id} not found in {self.analysis_dir}"
            raise FileNotFoundError(msg)
        if not self.reference_dir.exists():
            msg = f"Reference dir {self.reference_dir} does not exist"
            raise FileNotFoundError(msg)
        genes_sel = () if genes_sel is None else tuple(genes_sel)
        if self.verbose:
            log(f"[MethylAnalysis] Make CNV for {sample_id}...")
        self.cnv_plot = get_cnv_plot(
            sample_path=idat_basepath,
            reference_dir=self.reference_dir,
            prep=self.prep,
            cnv_dir=self.cnv_dir,
            genes_sel=genes_sel,
            do_seg=self.do_seg,
            verbose=self.verbose,
        )

    def precompute_cnvs(self, sample_ids=None):
        """Precalculates CNVs for all samples and saves them to disk.

        This method performs CNV analysis, and writes the output to the
        configured CNV directory. If `sample_ids` is not provided, the method
        will compute CNVs for all samples found in the `analysis_dir`.

        Args:
            sample_ids (list, optional): A list of sample IDs to process. If
                `None`, the function will compute CNVs for all samples in the
                `analysis_dir`. Default is `None`.

        Note:
            Precalculating CNVs improves performance but requires additional
            memory space in the output directory.
        """
        if self.verbose:
            log("[MethylAnalysis] Precalculate CNV's...")
        self._update_paths()
        if sample_ids is None:
            sample_ids = self.idat_handler.ids
        self._prog_bar.reset(len(sample_ids), text="(CNV)")
        write_cnv_to_disk(
            sample_path=[self.idat_handler.id_to_path[x] for x in sample_ids],
            reference_dir=self.reference_dir,
            cnv_dir=self.cnv_dir,
            prep=self.prep,
            do_seg=self.do_seg,
            pbar=self._prog_bar,
            verbose=self.verbose,
        )
        self._prog_bar.reset(1, 1)

    def get_cnv(self, sample_id, extract=None):
        """Retrieves the CNV information for a specified sample.

        This method locates the IDAT file corresponding to the provided
        `sample_id`, processes it to generate CNV data if not already
        available, and reads the resulting CNV information from disk.

        Args:
            sample_id (str): The identifier for the sample whose CNV data is to
                be retrieved.
            extract (list): Specifies the data to extract from the CNV
                analysis. Available options include:
                - "bins": Raw CNV data at the bin level.
                - "detail": Detailed CNV information (generally genes).
                - "segments": Segmented CNV regions.
                - "metadata": CNV analysis metadata.

        Returns:
            tuple: A tuple containing the following elements:
                - bins (DataFrame): Data representing CNV bins.
                - detail (DataFrame): Gene CNV information.
                - segments (DataFrame): Segmented CNV data.

                If CNV data is not found or cannot be generated, returns None
                for each extract value.
        """
        if extract is None:
            extract = ["bins", "detail", "segments"]
        write_cnv_to_disk(
            sample_path=[self.idat_handler.id_to_path[sample_id]],
            reference_dir=self.reference_dir,
            cnv_dir=self.cnv_dir,
            prep=self.prep,
            do_seg=self.do_seg,
            pbar=self._prog_bar,
            verbose=self.verbose,
        )
        basename = self.idat_handler.id_to_basename[sample_id]
        if (self.cnv_dir / (basename + ZIP_ENDING)).exists():
            return read_cnv_data_from_disk(
                self.cnv_dir,
                basename,
                extract=extract,
            )
        return (None,) * len(extract)

    def cn_summary(self, sample_ids):
        if not self.do_seg:
            msg = "To use CN-summary plots you must set 'do_seg' to 'True'."
            raise ValueError(msg)
        self.precompute_cnvs(sample_ids)
        basenames = [self.idat_handler.id_to_basename[x] for x in sample_ids]
        plot, df_cn_summary = get_cn_summary(self.cnv_dir, basenames)
        return plot, df_cn_summary

    def classify(self, *, ids=None, values=None, clf_list):
        """Classify samples using specified classifiers.

        This method performs classification on given samples, defined either by
        `ids` or by `values`, using one or more supervised classifiers. The
        labels for classification are derived from the `selected_columns`.
        Classification can either use a provided `feature_matrix` (custom
        features), or default to CpG methylation data (`betas_all`). All
        samples in `analysis_dir` resp. those in `sample_ids` with valid label
        will be used for learning.


        Classifiers are applied to the data, and the method returns their
        predictions and performance reports.

        Args:
            ids (list, tuple, np.ndarray, or None): Sample IDs for
                prediction/classification. If `values` is provided, `ids` must
                be `None`.
            values (pd.DataFrame, np.ndarray, or None): Feature matrix for
                prediction/classification. If `ids` is provided, `values` must
                be `None`.
            clf_list (object or list of objects): A classifier model or a
                list of classifier models and configurations. This argument is
                handled the same way as `self.classifiers`. For full details on
                the format and options, refer to the docstring for
                `self.classifiers`.

        Returns:
            list: A list containing:
                - pd.DataFrame: The predicted labels with probabilities.
                - sklearn object or TrainedClassifier: The classifier object
                  used.
                - list: Evaluation metrics for the classifier.

        Outputs:
            Log file: Contains training time, classifier performance metrics,
                and evaluation results for each classifier.

        Raises:
            ValueError: If not exactly one if `ids` or `values` is set.
        """
        if (ids is not None) and (values is not None):
            msg = "Provide only one of 'ids' or 'values'."
            raise ValueError(msg)

        if (ids is None) and (values is None):
            msg = "Provide either 'ids' or 'values'."
            raise ValueError(msg)

        if ids and not isinstance(ids, (list, tuple, np.ndarray)):
            ids = [ids]

        self._update_paths()
        self.set_betas()
        ensure_directory_exists(self.clf_dir)

        # Clean the file
        with self._clf_log.open("w"):
            pass

        y = self.idat_handler.features()
        if self.verbose:
            log("[MethylAnalysis] Start classifying...")

        if self.feature_matrix is not None:
            X = pd.DataFrame(self.feature_matrix, index=self.betas_all.index)
        elif self.load_full_betas:
            X = self.betas_all
        else:
            msg = (
                "For classification either all CpGs must be loaded into "
                "memory (enable 'load_full_betas') or 'feature_matrix' "
                "must be provided."
            )
            raise ValueError(msg)

        if ids and len(ids):
            values = X.loc[ids]

        def _invalid_class(cls):
            return isinstance(cls, str) and cls.strip("|") == ""

        # Remove all test samples and samples with unknown classification.
        test_indices = set(X.index.get_indexer(self.idat_handler.test_ids))
        valid_indices = [
            i
            for i, x in enumerate(y)
            if not _invalid_class(x) and i not in test_indices
        ]
        X = X.iloc[valid_indices]
        y = [y[i] for i in valid_indices]

        def _log(string):
            with self._clf_log.open("a") as f:
                f.write(string)
            print(string, end="")

        # Stop if there is no data to fit.
        if X.empty:
            _log("No data to fit.\n")
            return []

        results = []
        clfs = self._get_classifiers(clf_list)
        for i, clf in enumerate(clfs):
            if self.verbose:
                _log(f"Start training classifier {i + 1}/{len(clfs)}...\n")
            start_time = time.time()
            clf_result = fit_and_evaluate_clf(
                X=X,
                y=y,
                X_test=values,
                id_test=ids,
                directory=self.clf_dir,
                clf=clf["model"],
                cv=clf["cv"],
                n_jobs=self.n_jobs,
            )
            elapsed_time = time.time() - start_time
            if self.verbose:
                _log(f"Time used for classification: {elapsed_time:.2f} s\n\n")
            if self.verbose and len(clf_result.reports) == 1:
                _log(clf_result.reports[0] + "\n\n\n")
            results.append(clf_result)

        return results[0] if len(results) == 1 else results

    def get_app(self):
        """Returns a Dash application object for methylation analysis."""
        current_dir = Path(__file__).resolve().parent
        assets_folder = current_dir.parent / "data" / "assets"
        app = Dash(
            __name__,
            update_title=None,
            assets_folder=assets_folder,
            external_stylesheets=[dbc.themes.BOOTSTRAP],
        )
        app._favicon = "favicon.svg"
        app.title = "mepylome"
        side_navigation = get_side_navigation(
            sample_ids=self.ids,
            ids_to_highlight=self.ids_to_highlight,
            annotation_columns=self.idat_handler.columns,
            analysis_dir=self.analysis_dir,
            annotation=self.idat_handler.annotation_file,
            reference_dir=self.reference_dir,
            output_dir=self.output_dir,
            cpgs=self.cpgs,
            n_cpgs=self.n_cpgs,
            prep=self.prep,
            precalculate=self.precalculate_cnv,
            cpg_selection=self.cpg_selection,
            n_neighbors=self.umap_parms["n_neighbors"],
            metric=self.umap_parms["metric"],
            min_dist=self.umap_parms["min_dist"],
            use_discrete_colors=self._use_discrete_colors,
            custom_clfs=self.classifiers,
        )
        dash_plots = dbc.Col(
            [
                dcc.Graph(
                    id="umap-plot",
                    figure=self.umap_plot,
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
                    figure=self.cnv_plot,
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
            click_data,
            ids_to_highlight,
            umap_color_columns,
            umap_color_scheme,
            genes_sel,
            curr_umap_plot,
        ):
            def update_umap_plot():
                try:
                    self.make_umap_plot()
                    self._umap_plot_highlight(cnv_id=self.cnv_id)
                    self._retrieve_zoom(curr_umap_plot)
                    return self.umap_plot, no_update, ""
                except AttributeError:
                    return no_update, no_update, no_update

            genes_sel = tuple(genes_sel) if genes_sel else ()
            trigger = callback_context.triggered[0]["prop_id"].split(".")[0]
            self.ids_to_highlight = ids_to_highlight
            if trigger == "ids-to-highlight" and ids_to_highlight is not None:
                self._umap_plot_highlight()
                self._retrieve_zoom(curr_umap_plot)
                return self.umap_plot, no_update, ""
            if (
                trigger == "umap-annotation-color"
                and umap_color_columns is not None
            ):
                self.idat_handler.selected_columns = umap_color_columns
                update_umap_plot()
            if (
                trigger == "umap-color-scheme"
                and umap_color_scheme is not None
            ):
                self._use_discrete_colors = umap_color_scheme == "discrete"
                update_umap_plot()

            if trigger == "umap-plot" and isinstance(click_data, dict):
                points = click_data.get("points")
                if isinstance(points, list):
                    first_point = points[0] if points else {}
                    sample_id = first_point.get("hovertext")
                    if sample_id is None:
                        return no_update, no_update, ""
                    self._umap_plot_highlight(cnv_id=sample_id)
                    self._retrieve_zoom(curr_umap_plot)
                    try:
                        self.make_cnv_plot(sample_id, genes_sel)
                        return self.umap_plot, self.cnv_plot, ""
                    except Exception as exc:
                        log("[MethylAnalysis] umap failed:", exc)
                        log("[MethylAnalysis] sample_id:", sample_id)
                        log("[MethylAnalysis] MethylAnalysis:", self)
                        return self.umap_plot, EMPTY_FIGURE, str(exc)
            if trigger == "selected-genes" and genes_sel is not None:
                try:
                    self.make_cnv_plot(self.cnv_id, genes_sel)
                    return no_update, self.cnv_plot, ""
                except Exception as exc:
                    log("[MethylAnalysis] selected-genes failed:", exc)
                    log("[MethylAnalysis] self.cnv_id:", self.cnv_id)
                    log("[MethylAnalysis] genes_sel:", genes_sel)
                    return no_update, no_update, str(exc)
            return self.umap_plot, self.cnv_plot, ""

        @app.callback(
            [
                Output("analysis-dir", "valid"),
                Output("analysis-path-validation", "children"),
                Output("annotation-file", "value"),
            ],
            [Input("analysis-dir", "value")],
            prevent_initial_call=False,
        )
        def validate_analysis_path(input_path):
            try:
                path = Path(input_path).expanduser()
                if path.is_dir() and not os.access(path, os.W_OK):
                    return False, f"Protected directory: {path}", no_update
                if path.is_dir():
                    self.analysis_dir = path
                    return True, "", str(self.idat_handler.annotation_file)
                return False, f"Not a directory: {path}", no_update

            except Exception:
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
        def validate_annotation_file(input_path):
            selection = self.idat_handler.selected_columns
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
                    self.annotation = path
                    return True, "", self.idat_handler.columns, selection
                return False, f"Not a file: {path}", no_update, selection
            except Exception:
                return False, "Invalid path format", no_update, selection

        @app.callback(
            [
                Output("reference-dir", "valid"),
                Output("reference-path-validation", "children"),
            ],
            [Input("reference-dir", "value")],
            prevent_initial_call=False,
        )
        def validate_reference_path(input_path):
            try:
                path = Path(input_path).expanduser()
                if path.is_dir() and not os.access(path, os.W_OK):
                    return False, f"Protected directory: {path}"
                if path.is_dir():
                    self.reference_dir = path
                    return True, ""
                return False, f"Not a directory: {path}"
            except Exception as exc:
                log(
                    f"[MethylAnalysis] An error occured (1) "
                    f"(validate_reference_path): {exc}"
                )
                return False, "Invalid path format"

        @app.callback(
            [
                Output("output-dir", "valid"),
                Output("output-path-validation", "children"),
            ],
            [Input("output-dir", "value")],
            prevent_initial_call=False,
        )
        def validate_output_path(input_path):
            try:
                path = Path(input_path).expanduser()
                if path == DEFAULT_OUTPUT_DIR:
                    self.output_dir = path
                    return True, ""
                if path.is_dir() and not os.access(path, os.W_OK):
                    return False, f"Protected directory: {path}"
                if path.is_dir():
                    self.output_dir = path
                    return True, ""
                return False, f"Not a directory: {path}"
            except Exception as exc:
                log(
                    f"[MethylAnalysis] An error occured (2) "
                    f"(validate_output_path): {exc}"
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
                State("output-dir", "value"),
                State("preprocessing-method", "value"),
                State("analysis-dir", "valid"),
                State("output-dir", "valid"),
                State("precalculate-cnv", "value"),
                State("cpg-selection", "value"),
            ],
            prevent_initial_call=True,
            running=[
                (Output("start-button", "disabled"), True, False),
            ],
        )
        def on_umap_start_button_click(
            n_clicks,
            n_cpgs,
            analysis_dir,
            annotation,
            reference_dir,
            output_dir,
            prep,
            analysis_dir_valid,
            output_dir_valid,
            precalculate_cnv,
            cpg_selection,
        ):
            if not n_clicks:
                return no_update, no_update, "", {}

            error_message = None

            if n_cpgs is None:
                error_message = "Invalid no. of CpGs."
            elif not analysis_dir_valid:
                error_message = "Invalid analysis path."
            elif not output_dir_valid:
                error_message = "Invalid Output path."
            elif prep is None:
                error_message = "Invalid preprocessing method."
            elif precalculate_cnv is None:
                error_message = "Invalid precalculation method."
            elif cpg_selection is None:
                error_message = "Invalid CpG selection method."

            if error_message:
                return no_update, no_update, error_message, {}

            self.n_cpgs = n_cpgs
            self.output_dir = Path(output_dir).expanduser()
            self.reference_dir = Path(reference_dir).expanduser()
            self.prep = prep
            self.precalculate_cnv = precalculate_cnv == ON
            self.cpg_selection = cpg_selection
            self.analysis_dir = Path(analysis_dir).expanduser()
            self.annotation = Path(annotation).expanduser()

            try:
                ensure_directory_exists(self.output_dir)
                self.make_umap()
            except Exception as exc:
                log(f"[MethylAnalysis] An error occured (3): {exc}")
            else:
                return (
                    self.umap_plot,
                    self.ids,
                    no_update,
                    {"status": "umap_done"},
                )
            return no_update, no_update, "", {}

        @app.callback(
            Output("console-out", "style"),
            [Input("toggle-button", "n_clicks")],
            [State("console-out", "style")],
        )
        def toggle_console_out(n_clicks, current_style):
            if n_clicks % 2 == 0:
                return {**current_style, "display": "flex"}
            return {**current_style, "display": "none"}

        @app.callback(
            [
                Input("umap-n_neighbors", "value"),
                Input("umap-metric", "value"),
                Input("umap-min_dist", "value"),
            ],
            prevent_initial_call=True,
        )
        def update_umap_parms(
            n_neighbors,
            metric,
            min_dist,
        ):
            self.umap_parms["n_neighbors"] = n_neighbors
            self.umap_parms["metric"] = metric
            self.umap_parms["min_dist"] = min_dist

        @app.callback(
            [
                # Add a dummy output component
                Output("dummy-output", "children")
            ],
            [Input("running-state", "data")],
            running=[
                (Output("start-button", "disabled"), True, False),
            ],
        )
        def precalculate_cnv_wrapper(state):
            if (
                state
                and state.get("status") == "umap_done"
                and self.precalculate_cnv
            ):
                self.precompute_cnvs()
            # Dummy update
            return no_update

        @app.callback(
            [
                Output("umap-progress-bar", "value"),
                Output("umap-progress-bar", "label"),
                Output("console-out", "children"),
                Output("clf-out", "children"),
            ],
            [Input("clock", "n_intervals")],
        )
        def update_progress(n):
            progress = self._prog_bar.get_progress()
            out_str = self._prog_bar.get_text()
            with LOG_FILE.open("r") as file:
                log_str = ""
                lines = file.readlines()
                n_top = 50
                last_lines = lines if len(lines) <= n_top else lines[-n_top:]
                for line in last_lines:
                    log_str = log_str + line
            with self._clf_log.open("r") as file:
                clf_str = file.readlines()
            return progress, out_str, log_str, clf_str

        @app.callback(
            Output("output-idat-upload", "children"),
            Input("upload-idat", "contents"),
            State("upload-idat", "filename"),
            State("upload-idat", "last_modified"),
        )
        def update_output(list_of_contents, list_of_names, list_of_dates):
            def parse_contents(contents, filename, date):
                file_path = self.test_dir / filename
                content_string = contents.split(",")[1]
                decoded = base64.b64decode(content_string)
                with file_path.open("wb") as f:
                    f.write(decoded)
                log(f"[MethylAnalysis] Upload of {filename} completed.")
                return html.Div(
                    [
                        html.H6(filename),
                    ]
                )

            if list_of_contents is not None:
                children = [
                    parse_contents(c, n, d)
                    for c, n, d in zip(
                        list_of_contents, list_of_names, list_of_dates
                    )
                ]
                self.idat_handler = None
                self._update_paths()
                return children

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
            n_clicks,
            clf_list,
        ):
            if not n_clicks:
                return no_update

            error_message = None
            if clf_list is None or len(clf_list) == 0:
                error_message = "No classifiers selected."
            elif self.cnv_id is None:
                error_message = "No sample selected."
            if error_message:
                return error_message

            try:
                parsed_clf_list = [
                    self.classifiers[int(x)] if x.isdigit() else x
                    for x in clf_list
                ]
                print("PARSED", parsed_clf_list)
                _ = self.classify(ids=self.cnv_id, clf_list=parsed_clf_list)
            except Exception as exc:
                log(f"[MethylAnalysis] An error occured (4): {exc}")
                return f"{exc}"
            return ""

        return app

    def run_app(self, *, open_tab=False):
        """Runs the mepylome Dash application.

        Args:
            open_tab (bool, optional): Whether to automatically open a new
                browser tab with the application URL. Defaults to False.
        """
        self.app = self.get_app()
        free_port = get_free_port(self.port)
        if open_tab:

            def open_browser_tab():
                webbrowser.open_new_tab(f"http://{self.host}:{free_port}")

            threading.Timer(1, open_browser_tab).start()

        # Don't show all the flask logging statements.
        flask_logger = logging.getLogger("werkzeug")
        flask_logger.setLevel(logging.ERROR)

        self.app.run(
            debug=self.debug,
            host=self.host,
            use_reloader=False,
            port=free_port,
        )

    def __repr__(self):
        title = f"{self.__class__.__name__}()"
        header = title + "\n" + "*" * len(title)
        lines = [header]

        def format_value(value):
            length_info = ""
            if isinstance(value, (pd.DataFrame, pd.Series, pd.Index)):
                display_value = str(value)
            elif isinstance(value, np.ndarray):
                display_value = str(value)
                length_info = f"\n\n[{len(value)} items]"
            elif hasattr(value, "__len__"):
                display_value = str(value)[:80] + (
                    "..." if len(str(value)) > 80 else ""
                )
                if len(str(value)) > 80:
                    length_info = f"\n\n[{len(value)} items]"
            elif isinstance(value, (plotly.graph_objs.Figure)):
                data_str = (
                    str(value.data[0])[:70].replace("\n", " ")
                    if value.data
                    else "No data"
                )
                layout_str = str(value.layout)[:70].replace("\n", " ")
                data_str += "..." if len(data_str) == 70 else ""
                layout_str += "..." if len(layout_str) == 70 else ""
                display_value = (
                    f"Figure(\n"
                    f"    data: {data_str}\n"
                    f"    layout: {layout_str}\n"
                    f")"
                )
            else:
                display_value = str(value)[:80] + (
                    "..." if len(str(value)) > 80 else ""
                )
            return display_value, length_info

        for attr, value in sorted(self.__dict__.items()):
            display_value, length_info = format_value(value)
            lines.append(f"{attr}:\n{display_value}{length_info}")
        return "\n\n".join(lines)
