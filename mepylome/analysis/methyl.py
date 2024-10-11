"""Methylation analysis tools including a Dash-based browser application.

This module provides a comprehensive set of tools for conducting methylation
analysis. The core functionality is encapsulated in the ``MethylAnalysis``
class, which manages the methylation analysis process and executes an
interactive web application for the exploration of methylation data.
"""

import base64
import hashlib
import logging
import os
import sys
import threading
import webbrowser
from pathlib import Path

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
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

from mepylome.analysis.methyl_aux import (
    IdatHandler,
    ProgressBar,
    get_betas,
    read_dataframe,
)
from mepylome.analysis.methyl_clf import (
    fit_and_evaluate_classifiers,
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
    idat_basepaths,
    is_valid_idat_basepath,
    read_cnv_data_from_disk,
)
from mepylome.utils import (
    MEPYLOME_TMP_DIR,
    Timer,
    ensure_directory_exists,
    log,
)

timer = Timer()


DEFAULT_OUTPUT_DIR = Path(MEPYLOME_TMP_DIR, "analysis")
INVALID_PATH = Path("None")

DEFAULT_N_CPGS = 25000
NEUTRAL_BETA = 0.49

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


# Save console output to file
ensure_directory_exists(MEPYLOME_TMP_DIR)
LOG_FILE = MEPYLOME_TMP_DIR / "stdout.log"
CLF_FILE = MEPYLOME_TMP_DIR / "clf.log"
sys.stdout = DualOutput(LOG_FILE)

# Clean the file
with CLF_FILE.open("w"):
    pass


def get_all_genes():
    """Returns a list of names for all genes."""
    return Annotation.default_genes().df.Name.tolist()


def get_navbar():
    """Returns a navigation bar with a logo and title."""
    return dbc.Navbar(
        dbc.Container(
            [
                html.A(
                    dbc.Row(
                        [
                            dbc.Col(
                                html.Img(
                                    src="/assets/mepylome.svg",
                                    height="30px",
                                )
                            ),
                            dbc.Col(
                                dbc.NavbarBrand(
                                    "Methylation Analysis",
                                    className="ms-2",
                                )
                            ),
                        ],
                        align="center",
                        className="g-0",
                    ),
                    style={"textDecoration": "none"},
                ),
                dbc.NavbarToggler(id="navbar-toggler", n_clicks=0),
                html.Div(id="dummy-output", style={"display": "none"}),
            ]
        ),
        color="dark",
        dark=True,
    )


def get_side_navigation(
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
):
    """Generates a side navigation panel for setting up parameters."""
    n_cpgs_max = np.inf if len(cpgs) == 0 else len(cpgs)
    n_cpgs_max_str = "" if n_cpgs_max == np.inf else f" (max. {n_cpgs_max})"
    color_scheme = "discrete" if use_discrete_colors else "continuous"
    return dbc.Col(
        [
            dbc.Tabs(
                [
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
                                value=n_cpgs,
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
                                    "top": (
                                        "Take most varying CpG's (memory "
                                        "intensive!)"
                                    ),
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
                            html.H6(
                                "metric", style={"font-family": "monospace"}
                            ),
                            dcc.Dropdown(
                                id="umap-metric",
                                value=metric,
                                options=UMAP_METRICS,
                            ),
                            html.Br(),
                            html.H6(
                                "min_dist", style={"font-family": "monospace"}
                            ),
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
                            html.H6("Which CpG's should be used"),
                            dcc.Dropdown(
                                id="clf-cpg-dropdown",
                                options={
                                    "not_all": "Use the same CpGs as in UMAP",
                                    "all": "Use all CpGs (memory intensive)",
                                },
                                value="not_all",
                                multi=False,
                            ),
                            html.Br(),
                            html.H6("Classifiers to use"),
                            dcc.Dropdown(
                                id="clf-clf-dropdown",
                                options={
                                    "rf": "Random Forest",
                                    "knn": "k-Nearest Neighbors",
                                    "nn": "Neural Network",
                                    "svm": "Support Vector Machine",
                                },
                                value=["rf", "knn"],
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
                ],
            ),
        ],
        width={"size": 2},
    )


def guess_annotation_file(directory, verbose=False):
    """Returns the first spreadsheat file in the given directory."""
    if verbose:
        log("[MethylAnalysis] Try to read annotation file...")
    supported_extensions = [".csv", ".tsv", ".ods", ".xls", ".xlsx"]
    for file in directory.glob("*"):
        if file.suffix in supported_extensions:
            return file
    return INVALID_PATH


def input_args_id(*args, extra_hash=None):
    """Returns a unique identifier for a set of arguments."""
    hasher = hashlib.md5()
    components = []
    for arg in args:
        if isinstance(arg, np.ndarray):
            hasher.update(str(arg.tolist()).encode())
        elif isinstance(arg, Path):
            hasher.update(str(arg).encode())
            components.append(arg.name)
        else:
            hasher.update(str(arg).encode())
            components.append(str(arg))
    extra_hash = [] if extra_hash is None else extra_hash
    hasher.update(",".join([str(x) for x in extra_hash]).encode())
    components.append(hasher.hexdigest())
    return "-".join(components)


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


def reorder_columns_by_variance(data_frame):
    """Reorders data frame by descending column variance."""
    variances = data_frame.var()
    sorted_columns = variances.sort_values(ascending=False).index
    return data_frame[sorted_columns]


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
            Sentrix IDs are listed in the first column.

        reference_dir (str or Path): Directory containing reference IDAT files.

        output_dir (str or Path): Directory where output files will be saved
            (default: "/tmp/mepylome/analysis").

        prep (str): Prepreparation method used for data preparation (default:
            "illumina").

        cpgs (str or np.ndarray): Array of CpG sites to analyze, or one of
            '450k', 'epic', 'epicv2', or 'auto' for automatic detection. It can
            also be a path to a CSV file containing the CpG sites.


        cpg_blacklist (set or list, optional): A list or set of CpG sites to
            exclude. Default is None.

        n_cpgs (int): Number of CpG sites to select for UMAP (default: 25000).

        precalculate_cnv (bool): Flag to precalculate CNV information by
            invoking 'precompute_cnvs' (default: False).

        load_full_betas (bool): Flag to load beta values for all CpG sites
            into memory (default: False).

        feature_matrix (pandas.DataFrame or numpy.ndarray, optional): A
            user-provided feature matrix to be used for UMAP dimensionality
            reduction. If provided, this matrix will be used instead of
            `betas_df`. If not provided (default is None), the `betas_df`
            containing methylation beta values will be used for UMAP.

        overlap (bool): Flag to analyze only samples that are both in the
            analysis directory and within the annotation file (default: False).

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

        annotation (Path): Path to an annotation spreadsheet where Sentrix IDs
            are listed in the first column.

        overlap (bool): Flag to analyze only samples that are both in the
            analysis directory and within the annotation file (default: False).

        n_cpgs (int): Number of CpG sites to select for UMAP (default: 25000).

        reference_dir (Path): Path to the reference directory containing
            reference IDAT files.

        output_dir (Path): Path to the directory where output files will be
            saved (default: "/tmp/mepylome/analysis").

        upload_dir (NoneType): Directory for uploaded files, initially set to
            invalid path.

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

        cnv_dir (NoneType): Directory for CNV (Copy Number Variation) data,
            initially set to None.

        umap_dir (NoneType): Directory for UMAP (Uniform Manifold Approximation
            and Projection) data, initially set to None.

        umap_cpgs (NoneType): CpG sites for UMAP analysis, initially set to
            None.

        precalculate_cnv (bool): Flag to precalculate CNV information by
            invoking 'precalculate_all_cnvs' (default: False).

        load_full_betas (bool): Flag to load beta values for all CpG sites into
            memory (default: False).

        betas_df (NoneType): Dataframe containing beta values, initially set to
            None.

        feature_matrix (pandas.DataFrame or numpy.ndarray, optional): A
            user-provided feature matrix to be used for UMAP dimensionality
            reduction. If provided, this matrix will be used instead of
            `betas_df`. If not provided (default is None), the `betas_df`
            containing methylation beta values will be used for UMAP.

        betas_df_all_cpgs (NoneType): Dataframe containing beta values for all
            CpG sites, initially set to None.

        betas_path (NoneType): Path to the betas directory, initially set to
            None.

        umap_plot (Figure): Plot for UMAP, initially set to EMPTY_FIGURE.

        umap_plot_path (NoneType): Path to the CSV file containing the UMAP
            plot data, initially set to None.

        umap_df (NoneType): Dataframe containing UMAP data, initially set to
            None.

        umap_parms (dict): Parameters for UMAP algorithm (default: {'metric':
            'euclidean', 'min_dist': 0.1, 'n_neighbors': 15, 'verbose': True}).

        raw_umap_plot (NoneType): Raw UMAP plot data, initially set to None.

        cnv_plot (Figure): Plot for CNV (Copy Number Variation) visualization,
            initially set to EMPTY_FIGURE.

        cnv_id (NoneType): ID for CNV (Copy Number Variation) sample, initially
            set to None.

        dropdown_id (NoneType): ID for dropdown selection, initially set to
            None.

        ids (list): List of IDs, initially empty.

        ids_to_highlight (NoneType): IDs to highlight in the plot, initially
            set to None.

        app (NoneType): Dash application object, initially set to None.


    Raises:
        FileNotFoundError: If `annotation` file does not exist and cannot be
            guessed.

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
        prep="illumina",
        cpgs="auto",
        cpg_blacklist=None,
        n_cpgs=DEFAULT_N_CPGS,
        precalculate_cnv=False,
        load_full_betas=False,
        feature_matrix=None,
        overlap=False,
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
        self._idat_handler = None
        self.n_cpgs = n_cpgs
        self.cpg_blacklist = set(cpg_blacklist or [])
        self.cpg_selection = cpg_selection
        self.host = host
        self.port = port
        self.debug = debug
        self.reference_dir = Path(reference_dir).expanduser()
        self.output_dir = Path(output_dir).expanduser()
        self.upload_dir = INVALID_PATH
        self.cnv_dir = None
        self.umap_dir = None
        self.prep = prep
        self.precalculate_cnv = precalculate_cnv
        self.load_full_betas = load_full_betas
        self.feature_matrix = feature_matrix
        self.umap_plot = EMPTY_FIGURE
        self.umap_plot_path = None
        self.betas_df = None
        self.betas_df_all_cpgs = None
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

        self._prog_bar = ProgressBar()
        self._use_discrete_colors = True
        self._internal_cpgs_hash = None

        if self.cpg_selection not in ["top", "random"]:
            msg = "Invalid 'cpg_selection' (expected: 'top' or 'random')"
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

        self.cpgs = self._get_cpgs(cpgs)
        self._prev_vars = self._get_vars_or_hashes()
        self._update_paths()
        self.read_umap_plot_from_disk()

    @property
    def idat_handler(self):
        """Handles the management of IDAT files and associated metadata.

        Returns:
            IdatHandler: An instance of IdatHandler configured with current
            settings.
        """
        if (
            self._idat_handler is None
            or self.analysis_dir != self._idat_handler.idat_dir
            or self.annotation != self._idat_handler.annotation_file
            or self.overlap != self._idat_handler.overlap
            or self.upload_dir != self._idat_handler.upload_dir
        ):
            self._idat_handler = IdatHandler(
                self.analysis_dir,
                self.annotation,
                overlap=self.overlap,
                upload_dir=self.upload_dir,
            )
        return self._idat_handler

    @idat_handler.setter
    def idat_handler(self, value):
        self._idat_handler = value

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

    def _get_uploaded_files_hash(self):
        """Returns the hash of the uploaded files."""
        if not self.upload_dir.exists():
            return ""
        return input_args_id(
            extra_hash=sorted(str(x) for x in self.upload_dir.iterdir())
        )

    def _get_vars_or_hashes(self):
        """Returns current variables and hashes."""
        return {
            "analysis_dir": self.analysis_dir,
            "prep": self.prep,
            "n_cpgs": self.n_cpgs,
            "cpg_selection": self.cpg_selection,
            "cpgs": self._get_cpgs_hash(),
            "uploaded_files": self._get_uploaded_files_hash(),
        }

    def _get_cpgs(self, input_var="auto"):
        """Returns CpG sites based on the provided input variable."""
        self._internal_cpgs_hash = None

        def exclude_blacklist(cpgs):
            return np.sort(np.array(list(set(cpgs) - self.cpg_blacklist)))

        if self.verbose:
            log("[MethylAnalysis] Determine CpG sites...")

        if isinstance(input_var, (np.ndarray, set, list)):
            return exclude_blacklist(input_var)

        cpgs_from_file = get_cpgs_from_file(input_var)
        if cpgs_from_file is not None:
            return exclude_blacklist(cpgs_from_file)

        valid_str_parms = ["auto", "450k", "epic", "epicv2"]

        if isinstance(input_var, str):
            input_var = input_var.split("+")

        if "auto" in input_var:
            input_var = {
                str(ArrayType.from_idat(str(path) + "_Grn.idat"))
                for path in self.idat_handler.sample_paths.values()
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

    def _update_paths(self):
        """Update file paths and directories based on current settings.

        This method recalculates and updates various internal paths and
        directories whenever changes occur in related attributes like
        'analysis_dir', 'prep', 'n_cpgs', 'cpg_selection', and uploaded files
        in 'upload_dir'. It ensures that the correct paths are set for storing
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
            "betas",
            self.analysis_dir,
            self.prep,
        )
        self.betas_path = self.output_dir / f"{betas_hash_key}"

        # cnv dir
        cnv_hash_key = input_args_id(
            "cnv",
            self.analysis_dir,
            self.reference_dir,
            self.prep,
            self.do_seg,
        )
        self.cnv_dir = self.output_dir / f"{cnv_hash_key}"
        ensure_directory_exists(self.cnv_dir)

        # upload dir
        upload_hash_key = input_args_id(
            "upload",
            self.analysis_dir,
        )
        self.upload_dir = self.output_dir / f"{upload_hash_key}"
        ensure_directory_exists(self.upload_dir)

        # umap dir
        cur_vars = self._get_vars_or_hashes()
        umap_hash_key = input_args_id(
            "umap",
            self.analysis_dir,
            self.prep,
            self.n_cpgs,
            self.cpg_selection,
            extra_hash=[cur_vars["cpgs"], cur_vars["uploaded_files"]],
        )
        self.umap_dir = self.output_dir / f"{umap_hash_key}"
        self.umap_plot_path = self.umap_dir / "umap_plot.csv"
        ensure_directory_exists(self.umap_dir)

        # Reset betas_df if necessary
        dependencies = [
            "analysis_dir",
            "prep",
            "n_cpgs",
            "cpg_selection",
            "cpgs",
            "uploaded_files",
        ]
        if any(self._prev_vars[arg] != cur_vars[arg] for arg in dependencies):
            self.betas_df = None

        # Reset betas_df_all_cpgs if necessary
        dependencies = [
            "analysis_dir",
            "prep",
            "cpg_selection",
            "cpgs",
            "uploaded_files",
        ]
        if any(self._prev_vars[arg] != cur_vars[arg] for arg in dependencies):
            self.betas_df_all_cpgs = None

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
        """Applies the UMAP algorithm on 'betas_df'.

        Saves the 2D embedding in 'umap_df' and and on disk.

        Raises:
            AttributeError: If a dimension mismatch occurs, or if 'betas_df' is
                not set.
        """
        if self.betas_df is None and self.feature_matrix is None:
            msg = "'betas_df' is not set. First run 'set_betas'"
            raise AttributeError(msg)
        if self.verbose:
            log("[MethylAnalysis] Importing umap library...")
        import umap

        matrix_to_use = (
            self.betas_df
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
                self.idat_handler.annotated_samples,
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
        umap_color = self.idat_handler.compound_class(
            self.idat_handler.selected_columns
        )
        if len(umap_color) != len(self.umap_df):
            msg = (
                "Dimension mismatch 2: Analysis files may have changed. "
                f"Try to delete cached files in {self.output_dir}."
            )
            raise AttributeError(msg)
        self.umap_df["Umap_color"] = umap_color
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
        """Sets the beta values DataFrame ('betas_df') for further analysis.

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
                return extract_sub_dataframe(
                    self.betas_df_all_cpgs, self.umap_cpgs
                )
            return self.betas_df_all_cpgs.iloc[:, : self.n_cpgs]

        if self.cpg_selection == "random":
            self.umap_cpgs = get_random_cpgs()

        if self.betas_df_all_cpgs is not None:
            self.betas_df = _extract_sub_dataframe()

        elif self.load_full_betas or self.cpg_selection == "top":
            self.betas_df_all_cpgs = _get_betas(self.cpgs)
            self.betas_df_all_cpgs = reorder_columns_by_variance(
                self.betas_df_all_cpgs
            )
            self.betas_df = _extract_sub_dataframe()

        else:
            self.betas_df = _get_betas(self.umap_cpgs)

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
        idat_basepath = self.idat_handler.sample_paths[sample_id]
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
            sample_ids = [x.name for x in idat_basepaths(self.analysis_dir)]
        self._prog_bar.reset(len(sample_ids), text="(CNV)")
        write_cnv_to_disk(
            sample_path=[
                self.idat_handler.sample_paths[x] for x in sample_ids
            ],
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
            sample_path=[self.idat_handler.sample_paths[sample_id]],
            reference_dir=self.reference_dir,
            cnv_dir=self.cnv_dir,
            prep=self.prep,
            do_seg=self.do_seg,
            pbar=self._prog_bar,
            verbose=self.verbose,
        )
        if (self.cnv_dir / (sample_id + ZIP_ENDING)).exists():
            return read_cnv_data_from_disk(
                self.cnv_dir,
                sample_id,
                extract=extract,
            )
        return (None,) * len(extract)

    def cn_summary(self, sample_ids):
        if not self.do_seg:
            msg = "To use CN-summary plots you must set 'do_seg' to 'True'."
            raise ValueError(msg)
        self.precompute_cnvs(sample_ids)
        plot, df_cn_summary = get_cn_summary(self.cnv_dir, sample_ids)
        return plot, df_cn_summary

    def classify(self, sample_id, clf_list=None, use_all_cpgs=False):
        """Classify the sample using specified classifiers.

        This method classifies the sample using a list of supervised
        classifiers. The possible classes are selected from 'selected_columns'.
        Returns trained classifiers with their evaluations and prints
        classification to console.

        Args:
            sample_id (str): Sample to classify.
            clf_list (list, optional): List of classifier objects to use for
                classification. Defaults to None.
            use_all_cpgs (bool, optional): Whether to use all CpGs
                ('betas_df_all_cpgs') or just the ones selected in the UMAP
                plot ('betas_df'). Defaults to False

        Returns:
            list[dict]: List of dictionaries containing trained classifiers
                with their evaluations.

        Raises:
            ValueError: If `sample_id` is not set.
        """
        if sample_id is None:
            msg = "Must set 'sample_id' before calling classify()."
            raise ValueError(msg)

        if use_all_cpgs:
            self.load_full_betas = True

        self._update_paths()
        self.set_betas()
        classes_ = self.idat_handler.compound_class(
            self.idat_handler.selected_columns
        )
        log("[MethylAnalysis] Start classifying...")

        if self.feature_matrix is not None:
            betas = pd.DataFrame(
                self.feature_matrix, index=self.betas_df.index
            )
        elif use_all_cpgs:
            betas = self.betas_df_all_cpgs
        else:
            betas = self.betas_df

        sample_index = betas.index.tolist().index(sample_id)

        def _empty_class(cls):
            return cls.strip("|") == ""

        # Remove all samples with unknown classification.
        valid_indices = [
            i
            for i, x in enumerate(classes_)
            if (not _empty_class(x) or i == sample_index)
            and not (i > sample_index and betas.index[i] == sample_id)
        ]
        classes_ = [classes_[i] for i in valid_indices]
        betas = betas.iloc[valid_indices]

        return fit_and_evaluate_classifiers(
            betas_df=betas,
            classes_=classes_,
            log_file=CLF_FILE,
            sample_id=sample_id,
            clf_list=clf_list,
        )

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
            self.ids,
            self.ids_to_highlight,
            self.idat_handler.properties,
            self.analysis_dir,
            self.idat_handler.annotation_file,
            self.reference_dir,
            self.output_dir,
            self.cpgs,
            self.n_cpgs,
            self.prep,
            self.precalculate_cnv,
            self.cpg_selection,
            self.umap_parms["n_neighbors"],
            self.umap_parms["metric"],
            self.umap_parms["min_dist"],
            self._use_discrete_colors,
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
                except AttributeError:
                    return no_update, no_update, no_update
                else:
                    return self.umap_plot, no_update, ""

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
                    except Exception as exc:
                        log("[MethylAnalysis] umap failed:", exc)
                        log("[MethylAnalysis] sample_id:", sample_id)
                        log("[MethylAnalysis] MethylAnalysis:", self)
                        return no_update, no_update, str(exc)
                    else:
                        return self.umap_plot, self.cnv_plot, ""
            if trigger == "selected-genes" and genes_sel is not None:
                try:
                    self.make_cnv_plot(self.cnv_id, genes_sel)
                except Exception as exc:
                    log("[MethylAnalysis] selected-genes failed:", exc)
                    log("[MethylAnalysis] self.cnv_id:", self.cnv_id)
                    log("[MethylAnalysis] genes_sel:", genes_sel)
                    return no_update, no_update, str(exc)
                else:
                    return no_update, self.cnv_plot, ""
            return self.umap_plot, self.cnv_plot, ""

        @app.callback(
            [
                Output("analysis-dir", "valid"),
                Output("analysis-path-validation", "children"),
            ],
            [Input("analysis-dir", "value")],
            prevent_initial_call=False,
        )
        def validate_analysis_path(input_path):
            try:
                path = Path(input_path).expanduser()
                if path.is_dir() and not os.access(path, os.W_OK):
                    return False, f"Protected directory: {path}"
                if path.is_dir():
                    self.analysis_dir = path
                    return True, ""
            except Exception:
                return False, "Invalid path format"
            else:
                return False, f"Not a directory: {path}"

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
                    return True, "", self.idat_handler.properties, selection
            except Exception:
                return False, "Invalid path format", no_update, selection
            else:
                return False, f"Not a file: {path}", no_update, selection

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
            except Exception as exc:
                log(
                    f"[MethylAnalysis] An error occured (1) "
                    f"(validate_reference_path): {exc}"
                )
                return False, "Invalid path format"
            else:
                return False, f"Not a directory: {path}"

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
            except Exception as exc:
                log(
                    f"[MethylAnalysis] An error occured (2) "
                    f"(validate_output_path): {exc}"
                )
                return False, "Invalid path format"
            else:
                return False, f"Not a directory: {path}"

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
            else:
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
                N_TOP = 50
                last_lines = lines if len(lines) <= N_TOP else lines[-N_TOP:]
                for line in last_lines:
                    log_str = log_str + line
            with CLF_FILE.open("r") as file:
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
                file_path = self.upload_dir / filename
                content_type, content_string = contents.split(",")
                decoded = base64.b64decode(content_string)
                with open(file_path, "wb") as f:
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
                State("clf-cpg-dropdown", "value"),
                State("clf-clf-dropdown", "value"),
            ],
            prevent_initial_call=True,
            running=[
                (Output("clf-start-button", "disabled"), True, False),
            ],
        )
        def on_clf_start_button_click(
            n_clicks,
            cpgs_to_use,
            clf_list,
        ):
            if not n_clicks:
                return no_update

            error_message = None
            if cpgs_to_use is None:
                error_message = "Invalid CpGs selection method."
            elif clf_list is None or len(clf_list) == 0:
                error_message = "No classifiers selected."
            elif self.cnv_id is None:
                error_message = "No sample selected."
            if error_message:
                return error_message

            use_all_cpgs = cpgs_to_use == "all"

            try:
                _ = self.classify(self.cnv_id, clf_list, use_all_cpgs)
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
        if open_tab:

            def open_browser_tab():
                webbrowser.open_new_tab(f"http://{self.host}:{self.port}")

            threading.Timer(1, open_browser_tab).start()

        # Don't show all the flask logging statements.
        flask_logger = logging.getLogger("werkzeug")
        flask_logger.setLevel(logging.ERROR)

        self.app.run(debug=self.debug, host=self.host, use_reloader=False)

    def __repr__(self):
        title = "MethylAnalysis():"
        lines = [
            title + "\n" + "*" * len(title),
            f"analysis_dir:\n{self.analysis_dir}",
            f"annotation:\n{self.annotation}",
            f"app:\n{self.app}",
            f"betas_df:\n{self.betas_df}",
            f"betas_df_all_cpgs:\n{self.betas_df_all_cpgs}",
            f"betas_path:\n{self.betas_path}",
            f"cnv_dir:\n{self.cnv_dir}",
            f"cnv_id:\n{self.cnv_id}",
            f"cnv_plot:\n{str(self.cnv_plot)[:80]}...",
            f"cpg_selection:\n{self.cpg_selection}",
            f"cpgs:\n{self.cpgs}",
            f"debug:\n{self.debug}",
            f"do_seg:\n{self.do_seg}",
            f"dropdown_id:\n{self.dropdown_id}",
            f"host:\n{self.host}",
            f"ids:\n{self.ids}",
            f"ids_to_highlight:\n{self.ids_to_highlight}",
            f"load_full_betas:\n{self.load_full_betas}",
            f"n_cpgs:\n{self.n_cpgs}",
            f"output_dir:\n{self.output_dir}",
            f"overlap:\n{self.overlap}",
            f"port:\n{self.port}",
            f"precalculate_cnv:\n{self.precalculate_cnv}",
            f"prep:\n{self.prep}",
            f"raw_umap_plot:\n{str(self.raw_umap_plot)[:80]}...",
            f"reference_dir:\n{self.reference_dir}",
            f"selected_columns:\n{self.idat_handler.selected_columns}",
            f"umap_cpgs:\n{self.umap_cpgs}",
            f"umap_df:\n{self.umap_df}",
            f"umap_dir:\n{self.umap_dir}",
            f"umap_plot_path:\n{self.umap_plot_path}",
            f"upload_dir:\n{self.upload_dir}",
            f"verbose:\n{self.verbose}",
        ]
        return "\n\n".join(lines)
