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
from collections import Counter
from pathlib import Path

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

from mepylome import LOG_FILE
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
    CONFIG,
    MEPYLOME_TMP_DIR,
    ensure_directory_exists,
    get_free_port,
    make_log_file,
)

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path(MEPYLOME_TMP_DIR, "analysis")
DEFAULT_N_CPGS = 25000
ON = "on"
OFF = "off"
ZIP_ENDING = CONFIG["suffixes"]["cnv_zip"]
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
VERBOSITY_LEVELS = {
    0: logging.WARNING,
    1: logging.INFO,
    2: logging.DEBUG,
}


class DualOutput:
    """Enables to simultaneously write output to the terminal and file.

    Should only be used with `while` as it leads to print problems with
    jupyter notebooks.
    """

    def __init__(self, filename):
        self.terminal = sys.stdout
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

    def __enter__(self):
        self.original_stdout = sys.stdout
        sys.stdout = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.original_stdout
        self.close()


LOG_DIR = MEPYLOME_TMP_DIR / "log"
ensure_directory_exists(MEPYLOME_TMP_DIR)
ensure_directory_exists(LOG_DIR)


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
        "none-kbest-lr": "LogisticRegression",
        "none-kbest-rf": "RandomForestClassifier",
        "none-kbest-svc_rbf": "SVC(kernel='rbf')",
        "none-pca-lr": "PCALogisticRegression",
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
                html.Span(
                    "Do not upload too many files at once, else it "
                    "might not work!"
                ),
                html.Br(),
                dcc.Upload(
                    [
                        "Drag & Drop or ",
                        html.A("Select IDAT File pairs"),
                        html.Br(),
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


def get_balanced_indices(feature_labels, seed=None):
    """Returns indices of a balanced selection of feature labels.

    Args:
        feature_labels (list or array-like): Labels of features to balance.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        np.ndarray: Sorted indices of the balanced selection.
    """
    feature_labels = np.array(feature_labels)
    class_frequencies = Counter(feature_labels)
    min_class_size = min(class_frequencies.values())
    if min_class_size <= 1:
        problematic_classes = [
            cls for cls, count in class_frequencies.items() if count <= 1
        ]
        msg = f"Only 1 sample for the following classes: {problematic_classes}"
        raise ValueError(msg)
    class_to_indices = {
        label: np.where(feature_labels == label)[0]
        for label in class_frequencies
    }
    rng = np.random.default_rng(seed)
    balanced_sample_indices = np.hstack(
        [
            rng.choice(class_to_indices[label], min_class_size, replace=False)
            for label in class_frequencies
        ]
    )
    balanced_sample_indices.sort()
    logger.info("Balanced classes with %s samples each", min_class_size)
    return balanced_sample_indices


def reordered_cpgs_by_variance(data_frame):
    """Reorders CpGs by descending column variance."""
    logger.info("Reordering CpG's...")
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

        annotation (str or Path): Path to an annotation spreadsheet used to map
            sample files located in both `analysis_dir` and `test_dir`. One of
            the columns must contain the ID corresponding to the IDAT files
            (such as SentrixID or ID from files downloaded from GEO). If not
            provided, the system will attempt to identify the correct column
            automatically. If the annotation file is missing, it will search
            for a spreadsheet within the `analysis_dir` if available. (default:
            None)

        reference_dir (str or Path): Directory containing CNV neutral reference
            IDAT files. Must be provided if you wanna generate CNV plots.
            (default: None)

        output_dir (str or Path): Directory where output files will be saved
            (default: "/tmp/mepylome/analysis").

        test_dir (Path or None): Directory for test files, including new cases
            for analysis or validation. Files uploaded via the GUI will be
            placed here. If set to `None`, the application will automatically
            use a temporary directory. (default: None)

        prep (str): Prepreparation method used for methylation microarrays:
            'illumina', 'swan', or 'noob (default: 'illumina').

        cpgs (str, np.ndarray, list, set, or Path, optional): Specifies the CpG
            sites to analyze. Possible values:

            1. A list, set, or NumPy array of official Illumina CpG site names.
            2. A path to a CSV file containing the CpG sites.
            3. A string specifying a predefined array type:

                - `'450k'`   : The CpG sites from the Illumina 450k array.
                - `'epic'`   : The CpG sites from the Illumina EPIC array.
                - `'epicv2'` : The CpG sites from the Illumina EPIC v2 array.
                - `'msa48'`  : The CpG sites from the Illumina MSA array.

            4. A `'+'`-joined string of the options above combining multiple
            array types, returning the intersection of their CpG sites. For
            example:

                - `'450k+epic'`  : CpG sites both in the 450k and EPIC arrays.
                - `'epic+epicv2'`: CpG sites both in the EPIC and EPICv2
                  arrays.

            5. `'auto'` (default): Automatically detects all array types from
            IDAT files in `analysis_dir` and returns the intersection of CpG
            sites. This process may take longer as all files need to be read
            and, if necessary, decompressed.

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

        precalculate_cnv (bool): If set to `True`, CNV data will be
            precalculated before the main analysis. This process takes
            approximately 2-5 seconds per case initially, but it will improve
            performance during runtime by reducing computation time. (default:
            False)

        load_full_betas (bool): Flag to load beta values for all CpG sites
            into memory (default: True).

        feature_matrix (pandas.DataFrame or numpy.ndarray, optional): A
            user-provided feature matrix to be used for UMAP dimensionality
            reduction. If provided, this matrix will be used instead of
            `betas_sel`. If not provided (default is None), the `betas_sel`
            containing methylation beta values will be used for UMAP. (default:
            None)

        overlap (bool): Flag to analyze only samples that are both in the
            analysis directory and within the annotation file (default: False).

        analysis_ids (list, optional): A list of sample IDs. If provided, the
            analysis will be restricted to these samples only. If `None`, the
            analysis will include all available samples. (default: None)

        test_ids (list, optional): A list of sample IDs within `test_dir`.
            - If provided, only these samples will be used.
            - If `None`, all available IDAT files in `test_dir` will be used.
            (default: None)

        cpg_selection (str): Method to select CpG sites for UMAP ('top',
            'random', or 'balanced') (default: 'top').

            - 'top': Selects CpG sites with the highest variance.
            - 'random': Selects CpG sites randomly.
            - 'balanced': Selects the most varying CpG sites while ensuring a
              balanced distribution across groups based on
              `balancing_feature`.  This method takes an **equal number of
              sample files from `self.analysis_dir`**  for each group defined
              by `balancing_feature`.  It is especially useful when the
              dataset is **imbalanced**, where some  groups have significantly
              more samples than others.

        balancing_feature (str): Column in `self.annotation` used for
            balancing when `cpg_selection='balanced'`. The balancing feature
            determines the groups/categories used to create a stratified
            selection of CpG sites.

        do_seg (bool): If set, enables segmentation analysis on CNV data and
            adds horizontal segmentation lines to the CNV plot. This will take
            an additional 2-5 seconds per sample. (default: False)

        host (str): Host address for the Dash application (default:
            'localhost').

        port (int): Port number for the Dash application (default: 8050).

        debug (bool): Flag to enable debug mode for the Dash application
            (default: False).

        umap_parms (dict): Parameters for UMAP algorithm (default: {'metric':
            'manhattan', 'min_dist': 0.1, 'n_neighbors': 15, 'verbose': True}).

        verbose (int): Sets the (global) logging verbosity level:
            - 0: Errors and warnings only.
            - 1: Info, warnings, and errors (default).
            - 2: Debug, info, warnings, and errors.

    Note:
        Many parameters can be modified within the GUI application after
        initialization, but not all.

    Attributes:
        analysis_dir (Path): Path to the directory containing IDAT files for
            analysis.

        annotation (str or Path): Path to an annotation spreadsheet used to map
            sample files located in both `analysis_dir` and `test_dir`.

        overlap (bool): Flag to analyze only samples that are both in the
            analysis directory and within the annotation file (default: False).

        analysis_ids (list): A list of sample IDs in 'analysis_dir'
            that will be used.

        test_ids (list): A list of sample IDs in 'test_dir' that will
            be used.

        n_cpgs (int): Number of CpG sites to select for UMAP (default: 25000).

        n_jobs (int): Number of parallel processes to run for classifying
            (default: 1). Choose -1 for using all available cores.

        reference_dir (str or Path): Directory containing CNV neutral reference
            IDAT files. Must be provided if you wanna generate CNV plots.

        output_dir (Path): Path to the directory where output files will be
            saved (default: "/tmp/mepylome/analysis").

        test_dir (Path or None): Directory for test files, including new cases
            for analysis or validation. Files uploaded via the GUI will be
            placed here. If set to `None`, the application will automatically
            use a temporary directory.

        prep (str): Prepreparation method used for methylation microarrays:
            'illumina', 'swan', or 'noob (default: 'illumina').

        cpg_selection (str): Method to select CpG sites for UMAP ('top',
            'random', or 'balanced') (default: 'top').

            - 'top': Selects CpG sites with the highest variance.
            - 'random': Selects CpG sites randomly.
            - 'balanced': Selects the most varying CpG sites while ensuring a
              balanced distribution across groups based on
              `balancing_feature`.  This method takes an **equal number of
              sample files from `self.analysis_dir`**  for each group defined
              by `balancing_feature`.  It is especially useful when the
              dataset is **imbalanced**, where some  groups have significantly
              more samples than others.

        balancing_feature (str): Column in `annotation` used for
            balancing when `cpg_selection='balanced'`. The balancing feature
            determines the groups/categories used to create a stratified
            selection of CpG sites.

        host (str): Host address for the Dash application (default:
            'localhost').

        port (int): Port number for the Dash application (default: 8050).

        debug (bool): Flag to enable debug mode for the Dash application
            (default: False).

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

        betas_sel (pandas.DataFrame): DataFrame containing a selected subset of
            beta values used for dimensionality reduction. Initially set to
            None.

        betas_all (pandas.DataFrame): Dataframe containing beta values for all
            CpG sites, initially set to None.

        feature_matrix (pandas.DataFrame or numpy.ndarray, optional): A
            user-provided feature matrix to be used for UMAP dimensionality
            reduction. If provided, this matrix will be used instead of
            `betas_sel` for UMAP plots and instead of `betas_all` for
            classifying (default: None).

        betas_dir (Path): Path to the betas directory, initially set to
            None.

        umap_plot (plotly.Figure): Plot for UMAP, initially set to
            EMPTY_FIGURE.

        umap_plot_path (Path): Path to the CSV file containing the UMAP
            plot data, initially set to None.

        umap_df (pandas.DataFrame): Dataframe containing UMAP data, initially
            set to None.

        umap_parms (dict): Parameters for UMAP algorithm (default: {'metric':
            'manhattan', 'min_dist': 0.1, 'n_neighbors': 15, 'verbose': True}).

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
        ValueError: If `cpg_selection` is not 'top', 'balanced', or 'random'.

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
        analysis_ids=None,
        test_ids=None,
        cpg_selection="top",
        do_seg=False,
        host="localhost",
        port=8050,
        debug=False,
        umap_parms=None,
        verbose=1,
        balancing_feature=None,
    ):
        self.analysis_dir = Path(analysis_dir).expanduser()
        self.annotation = Path(annotation).expanduser()
        self.app = None
        self.balancing_feature = balancing_feature or []
        self.betas_all = None
        self.betas_dir = None
        self.betas_sel = None
        self.clf_dir = None
        self.cnv_dir = None
        self.cnv_id = None
        self.cnv_plot = EMPTY_FIGURE
        self.cpg_blacklist = set(cpg_blacklist or [])
        self.cpg_selection = cpg_selection
        self.cv_default = cv_default
        self.debug = debug
        self.dropdown_id = None
        self.feature_matrix = feature_matrix
        self.host = host
        self.ids = []
        self.ids_to_highlight = None
        self.load_full_betas = load_full_betas
        self.n_cpgs = n_cpgs
        self.n_jobs = n_jobs
        self.output_dir = Path(output_dir).expanduser()
        self.overlap = overlap
        self.port = port
        self.precalculate_cnv = precalculate_cnv
        self.prep = prep
        self.raw_umap_plot = None
        self.reference_dir = Path(reference_dir).expanduser()
        self.analysis_ids = (
            None if analysis_ids is None else list(analysis_ids)
        )
        self.test_dir = Path(test_dir).expanduser()
        self.test_ids = test_ids
        self.umap_cpgs = None
        self.umap_df = None
        self.umap_dir = None
        self.umap_parms = MethylAnalysis._get_umap_parms(umap_parms)
        self.umap_plot = EMPTY_FIGURE
        self.umap_plot_path = None

        self._balanced_sorted_cpgs = None
        self._classifiers = classifiers
        self._clf_log = make_log_file(f"{self.analysis_dir.name}-clf")
        self._idat_handler = None
        self._internal_cpgs_hash = None
        self._old_selected_columns = None
        self._prog_bar = ProgressBar()
        self._sorted_cpgs = None
        self._testdir_provided = self.test_dir != INVALID_PATH
        self._use_discrete_colors = True

        ensure_directory_exists(self.output_dir)

        # Set logging level dynamically
        main_logger = logging.getLogger("mepylome")
        main_logger.setLevel(VERBOSITY_LEVELS.get(verbose, logging.INFO))
        for handler in main_logger.handlers:
            handler.setLevel(VERBOSITY_LEVELS.get(verbose, logging.INFO))

        if self.cpg_selection not in ["top", "balanced", "random"]:
            msg = (
                "Invalid 'cpg_selection' (expected: 'top', 'balanced', or "
                "'random')"
            )
            raise ValueError(msg)

        if not self.load_full_betas and self.cpg_selection != "random":
            msg = (
                "If 'load_full_betas' is disabled, 'cpg_selection' must be "
                " set to 'random'"
            )
            raise ValueError(msg)

        if self.annotation != INVALID_PATH and not self.annotation.exists():
            logger.warning(
                "Warning: The provided annotation file '%s' does not exist",
                self.annotation,
            )
        if self.annotation == INVALID_PATH or not self.annotation.exists():
            self.annotation = guess_annotation_file(self.analysis_dir)
        if not self.annotation.exists():
            logger.info("No annotation file found")

        logger.info("Try to import cbseg, linear_segment or ruptures...")
        self.do_seg = False if _get_cgsegment() is None else do_seg

        # Set test dir, as it is needed by _get_cpgs
        self._set_test_dir()
        self._cpgs = self._get_cpgs(cpgs)

        self._prev_vars = self._get_vars_or_hashes()
        self._update_paths()

        self.read_umap_plot_from_disk()

        logger.info("Initialization completed")

    @property
    def cpgs(self):
        """Array of CpG sites to analyze, sorted in order.

        When setting, the input should be the same as the `cpgs` argument in
        the constructor (`__init__`).

        Raises:
            ValueError: If the provided `cpgs` value is not a valid type or
            format.
        """
        return self._cpgs

    @cpgs.setter
    def cpgs(self, cpgs):
        """Set the CpG sites for analysis."""
        self._cpgs = self._get_cpgs(cpgs)

    @property
    def idat_handler(self):
        """Handles the management of IDAT files and associated metadata.

        Returns:
            IdatHandler: An instance of IdatHandler configured with current
            settings.
        """
        if self._idat_handler is not None:
            self._old_selected_columns = self._idat_handler.selected_columns

        new_parameters = {
            "analysis_dir": self.analysis_dir,
            "annotation": self.annotation,
            "overlap": self.overlap,
            "test_dir": self.test_dir,
            "analysis_ids": self.analysis_ids,
            "test_ids": self.test_ids,
        }

        # Reinitialize IdatHandler if values have changed
        if (
            self._idat_handler is None
            or self._idat_handler.parameters() != new_parameters
        ):
            self._idat_handler = IdatHandler(**new_parameters)

            # Update the attributes and log changes.
            for attr, current in new_parameters.items():
                new = getattr(self._idat_handler, attr)
                if current != new:
                    logger.info("Updating '%s'", attr)
                    setattr(self, attr, new)

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
            "metric": "manhattan",
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
            "analysis_ids": self.analysis_ids,
            "test_ids": self.test_ids,
        }

    def _get_cpgs(self, input_var="auto"):
        """Returns CpG sites based on the provided input variable."""
        self._internal_cpgs_hash = None

        def exclude_blacklist(cpgs):
            return np.sort(np.array(list(set(cpgs) - self.cpg_blacklist)))

        logger.info("Determine CpG sites...")

        if isinstance(input_var, (np.ndarray, set, list, pd.Index)):
            return exclude_blacklist(input_var)

        cpgs_from_file = get_cpgs_from_file(input_var)
        if cpgs_from_file is not None:
            return exclude_blacklist(cpgs_from_file)

        if input_var == "auto":
            logger.info("Automatically determine array types...")
            input_var = {
                str(ArrayType.from_idat(path))
                for path in self.idat_handler.paths
            }
            logger.info(
                "The following array types were detected: %s", input_var
            )
            input_var = input_var - {str(ArrayType.UNKNOWN)}

        elif isinstance(input_var, str):
            input_var = set(input_var.split("+"))
            logger.info(
                "The following array types were provided: %s", input_var
            )

        supported_types = {str(x) for x in ArrayType} - {
            str(ArrayType.UNKNOWN),
            str(ArrayType.ILLUMINA_27K),
        }
        if input_var.issubset(supported_types):
            if not input_var:
                return np.array([])

            logger.info("Load manifests and calculate CpG overlap...")

            cpg_sets = [
                set(Manifest(array_type).methylation_probes)
                for array_type in supported_types
                if array_type in input_var
            ]
            cpgs = set.intersection(*cpg_sets)
            return exclude_blacklist(cpgs)

        mismatches = ", ".join(input_var - supported_types)
        types_str = ", ".join(supported_types)
        msg = (
            "'cpgs' must be one of the following:\n"
            "- a list, set, or array of CpG sites\n"
            f"- a '+' joined string of valid parameters: {types_str}\n"
            f"Received invalid input: {mismatches}"
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
        logger.info("Update filepaths...")
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
        self.betas_dir = self.output_dir / f"{betas_hash_key}"

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
            self.balancing_feature,
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
                self.analysis_ids,
            ],
        )
        self.clf_dir = self.output_dir / f"{clf_hash_key}"

        # Reset betas_sel if necessary
        dependencies = [
            "analysis_dir",
            "cpg_selection",
            "cpgs",
            "n_cpgs",
            "prep",
            "analysis_ids",
            "test_ids",
            "test_files",
        ]
        if any(self._prev_vars[arg] != cur_vars[arg] for arg in dependencies):
            self.betas_sel = None

        # Reset betas_all if necessary
        dependencies = [
            "analysis_dir",
            "cpgs",
            "prep",
            "analysis_ids",
            "test_ids",
            "test_files",
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
        """Applies the UMAP algorithm on 'betas_sel'.

        Saves the 2D embedding in 'umap_df' and and on disk.

        Raises:
            AttributeError: If a dimension mismatch occurs, or if 'betas_sel'
                is not set.
        """
        if self.betas_sel is None and self.feature_matrix is None:
            msg = "'betas_sel' is not set. First run 'set_betas'"
            raise AttributeError(msg)
        logger.info("Importing umap library...")
        import umap

        matrix_to_use = (
            self.betas_sel
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
        logger.info(
            "Starting UMAP for matrix with shape %s...", matrix_to_use.shape
        )
        with DualOutput(LOG_FILE):
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
        logger.info("Make UMAP plot...")
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
            logger.info("Read umap plot from disk...")
            self.umap_df = pd.read_csv(
                self.umap_plot_path, sep="\t", index_col=0
            )
            self.umap_df = self.umap_df.fillna("")
            try:
                self.make_umap_plot()
            except AttributeError:
                logger.info("Probable dimension mismatch")

    def set_betas(self):
        """Sets the beta values DataFrame ('betas_sel') for further analysis.

        This method reads the IDAT files located in 'analysis_dir', extracts
        the beta values, and saves them locally in 'output_dir'. Depending on
        the configuration ('cpg_selection' and 'load_full_betas' flags), it
        either extracts a subset of CpGs for UMAP computation or loads all CpGs
        for subsequent processing into memory.

        Raises:
            ValueError: If no valid samples are found.
        """
        if not self.idat_handler:
            msg = "No valid samples found"
            raise ValueError(msg)

        self._update_paths()

        def get_random_cpgs():
            logger.info("Selecting random CpG's...")
            return np.sort(
                np.random.default_rng().choice(
                    self.cpgs, self.n_cpgs, replace=False
                )
            )

        def _get_betas(cpgs):
            logger.info("Retrieving beta values...")
            return get_betas(
                idat_handler=self.idat_handler,
                cpgs=cpgs,
                prep=self.prep,
                betas_dir=self.betas_dir,
                pbar=self._prog_bar,
            )

        # Load all beta values if necessary
        if self.betas_all is None and (
            self.load_full_betas or self.cpg_selection in ["top", "balanced"]
        ):
            self.betas_all = _get_betas(self.cpgs)
            self._sorted_cpgs = reordered_cpgs_by_variance(self.betas_all)

        # Handle CpG selection
        if self.cpg_selection == "random":
            self.umap_cpgs = get_random_cpgs()
            self.betas_sel = (
                extract_sub_dataframe(self.betas_all, self.umap_cpgs)
                if self.betas_all is not None
                else _get_betas(self.umap_cpgs)
            )

        elif self.cpg_selection == "top":
            self.umap_cpgs = self._sorted_cpgs[: self.n_cpgs]
            self.betas_sel = self.betas_all[self.umap_cpgs]

        elif self.cpg_selection == "balanced":
            features = self.idat_handler.features(self.balancing_feature)
            balanced_indices = get_balanced_indices(features, seed=42)
            self._balanced_sorted_cpgs = reordered_cpgs_by_variance(
                self.betas_all.iloc[balanced_indices]
            )
            self.umap_cpgs = self._balanced_sorted_cpgs[: self.n_cpgs]
            self.betas_sel = self.betas_all[self.umap_cpgs]

        else:
            msg = f"Invalid 'cpg_selection': {self.cpg_selection}"
            raise ValueError(msg)

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
                text=f"{self.cnv_id}",
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
        logger.info("Make CNV for %s...", sample_id)
        self.cnv_plot = get_cnv_plot(
            sample_path=idat_basepath,
            reference_dir=self.reference_dir,
            prep=self.prep,
            cnv_dir=self.cnv_dir,
            genes_sel=genes_sel,
            do_seg=self.do_seg,
        )

    def precompute_cnvs(self, ids=None):
        """Precalculates CNVs for all samples and saves them to disk.

        This method performs CNV analysis, and writes the output to the
        configured CNV directory. If `ids` is not provided, the method
        will compute CNVs for all samples found in the `analysis_dir`.

        Args:
            ids (list, optional): A list of sample IDs to process. If
                `None`, the function will compute CNVs for all samples in the
                `analysis_dir`. Default is `None`.

        Note:
            Precalculating CNVs improves performance but requires additional
            memory space in the output directory.
        """
        logger.info("Precalculate CNV's...")
        self._update_paths()
        if ids is None:
            ids = self.idat_handler.ids
        self._prog_bar.reset(len(ids), text="(CNV)")
        write_cnv_to_disk(
            sample_path=[self.idat_handler.id_to_path[x] for x in ids],
            reference_dir=self.reference_dir,
            cnv_dir=self.cnv_dir,
            prep=self.prep,
            do_seg=self.do_seg,
            pbar=self._prog_bar,
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
        )
        basename = self.idat_handler.id_to_basename[sample_id]
        if (self.cnv_dir / (basename + ZIP_ENDING)).exists():
            return read_cnv_data_from_disk(
                self.cnv_dir,
                basename,
                extract=extract,
            )
        return (None,) * len(extract)

    def cn_summary(self, ids):
        if not self.do_seg:
            msg = "To use CN-summary plots you must set 'do_seg' to 'True'."
            raise ValueError(msg)
        self.precompute_cnvs(ids)
        basenames = [self.idat_handler.id_to_basename[x] for x in ids]
        plot, df_cn_summary = get_cn_summary(self.cnv_dir, basenames)
        return plot, df_cn_summary

    def classify(self, *, ids=None, values=None, clf_list):
        """Classify samples using specified classifiers.

        This method performs classification on given samples, defined either by
        `ids` or by `values`, using one or more supervised classifiers. The
        labels for classification are derived from the `selected_columns`.
        Classification can either use a provided `feature_matrix` (custom
        features), or default to CpG methylation data (`betas_all`). All
        samples in `analysis_dir` resp. those in `analysis_ids` with valid
        label will be used for learning.

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
            list[ClassifierResult]: A list of ClassifierResult objects, each
            containing the following attributes:

                - prediction (pd.DataFrame): A DataFrame containing the
                  predicted labels with their associated probabilities.
                - model (sklearn.base.BaseEstimator or TrainedClassifier): The
                  trained classifier object used for prediction.
                - metrics (dict): A dictionary of evaluation metrics for the
                  classifier, such as accuracy, precision, recall, etc.
                - reports (dict): A dictionary containing textual and HTML
                  reports of the classifier's performance. The keys are:

                    - "txt": A plain-text report (e.g., classification report).
                    - "html": An HTML-formatted report for richer
                      visualization.

        Outputs:
            Log file: Contains training time, classifier performance metrics,
                and evaluation results for each classifier.

        Raises:
            ValueError: If not exactly one if `ids` or `values` is set.
        """
        if sum(x is not None for x in (ids, values)) != 1:
            msg = "Provide exactly one of 'ids' or 'values'."
            raise ValueError(msg)

        if ids and not isinstance(ids, (list, tuple, np.ndarray)):
            ids = [ids]

        # Clean the clf log file
        with self._clf_log.open("w"):
            pass

        self._update_paths()
        ensure_directory_exists(self.clf_dir)

        clfs = self._get_classifiers(clf_list)

        def _clf_path(clf):
            filename = input_args_id(clf["model"], clf["cv"]) + ".pkl"
            return self.clf_dir / filename

        all_classifiers_trained = all(_clf_path(clf).exists() for clf in clfs)

        # Only load data to memory if training is needed
        if all_classifiers_trained:
            shape = (len(self.analysis_ids), len(self.cpgs))
            X = type("EmptyDataFrame", (), {"shape": shape})
            y = None
            if ids and len(ids):
                values = get_betas(
                    idat_handler=self.idat_handler,
                    ids=ids,
                    cpgs=self.cpgs,
                    prep=self.prep,
                    betas_dir=self.betas_dir,
                    pbar=self._prog_bar,
                )

        else:
            X, y, _values = self._load_training_data(ids)
            values = _values if _values is not None else values

        def _log(string):
            with self._clf_log.open("a") as f:
                f.write(string)
            print(string, end="")

        # Stop if there is no data to fit.
        if isinstance(X, pd.DataFrame) and X.empty:
            _log("No data to fit.\n")
            return []

        results = []
        for i, clf in enumerate(clfs):
            if logger.isEnabledFor(logging.INFO):
                _log(f"Start training classifier {i + 1}/{len(clfs)}...\n")
            start_time = time.time()
            clf_result = fit_and_evaluate_clf(
                X=X,
                y=y,
                X_test=values,
                id_test=ids,
                save_path=_clf_path(clf),
                clf=clf["model"],
                cv=clf["cv"],
                n_jobs=self.n_jobs,
            )
            elapsed_time = time.time() - start_time
            if logger.isEnabledFor(logging.INFO):
                _log(f"Time used for classification: {elapsed_time:.2f} s\n\n")
            if (
                logger.isEnabledFor(logging.INFO)
                and len(clf_result.reports["txt"]) == 1
            ):
                _log(clf_result.reports["txt"][0] + "\n\n\n")
            results.append(clf_result)

        return results

    def _load_training_data(self, ids):
        """Load training data for classification."""
        self.set_betas()

        y = self.idat_handler.features()
        logger.info("Start classifying...")

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

        values = X.loc[ids] if ids else None

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

        return X, y, values

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
            annotation=self.idat_handler.annotation,
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
                    except FileNotFoundError as exc:
                        logger.info("umap failed: %s", exc)
                        logger.info("sample_id: %s", sample_id)
                        logger.info("MethylAnalysis: %s", self)

                        error_message = (
                            f"{exc} - There is probably no CNV neutral "
                            "reference set for the array type of the "
                            "selected sample. To solve this, add missing CNV "
                            f"neutral sets to '{self.reference_dir}' and "
                            f"remove the error file in '{self.cnv_dir}'."
                        )
                        return self.umap_plot, EMPTY_FIGURE, error_message
                    except Exception as exc:
                        logger.info("umap failed: %s", exc)
                        logger.info("sample_id: %s", sample_id)
                        logger.info("MethylAnalysis: %s", self)
                        return self.umap_plot, EMPTY_FIGURE, str(exc)
            if trigger == "selected-genes" and genes_sel is not None:
                try:
                    self.make_cnv_plot(self.cnv_id, genes_sel)
                    return no_update, self.cnv_plot, ""
                except Exception as exc:
                    logger.info("selected-genes failed: %s", exc)
                    logger.info("self.cnv_id: %s", self.cnv_id)
                    logger.info("genes_sel: %s", genes_sel)
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
                    return True, "", str(self.idat_handler.annotation)
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
                logger.info(
                    "An error occured (1) (validate_reference_path): %s", exc
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
                logger.info(
                    "An error occured (2) (validate_output_path): %s", exc
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
                State("balancing-column", "value"),
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
            balancing_feature,
        ):
            if not n_clicks:
                return no_update, no_update, "", {}

            error_message = None

            if n_cpgs is None:
                error_message = "Invalid no. of CpGs."
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
            self.balancing_feature = (
                balancing_feature if cpg_selection == "balanced" else []
            )

            try:
                ensure_directory_exists(self.output_dir)
                self.make_umap()
            except Exception as exc:
                # BUG: Error 'no module named tqdm.auto' if mepylome is new
                # installed. This error disapears after running tutorial
                # Error was produced via cli with -a -A and -o -C 'epic'
                # --overlap -S 'top'. Maybe this error occurs only on Mac OS?
                logger.info("An error occured (3): %s", exc)
            else:
                return (
                    self.umap_plot,
                    self.ids,
                    no_update,
                    {"status": "umap_done"},
                )
            return no_update, no_update, "", {}

        @app.callback(
            Output("console-out-setting", "style"),
            [Input("toggle-button-setting", "n_clicks")],
            [State("console-out-setting", "style")],
        )
        def toggle_console_out(n_clicks, current_style):
            return {
                **current_style,
                "display": "flex" if n_clicks % 2 == 0 else "none",
            }

        @app.callback(
            Output("balancing-column-container", "style"),
            Input("cpg-selection", "value"),
        )
        def toggle_column_dropdown(selected_method):
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
            n_neighbors,
            metric,
            min_dist,
        ):
            self.umap_parms["n_neighbors"] = n_neighbors
            self.umap_parms["metric"] = metric
            self.umap_parms["min_dist"] = min_dist

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
        def precalculate_cnv_wrapper(state):
            if (
                state
                and state.get("status") == "umap_done"
                and self.precalculate_cnv
            ):
                self.precompute_cnvs()
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
            return progress, out_str, log_str, log_str, clf_str

        @app.callback(
            Output("output-idat-upload", "children"),
            Input("upload-idat", "contents"),
            State("upload-idat", "filename"),
            State("upload-idat", "last_modified"),
        )
        def update_output(list_of_contents, list_of_names, list_of_dates):
            logger.info("Uploading files...")

            def parse_contents(contents, filename, date):
                file_path = self.test_dir / filename
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
                for c, n, d in zip(
                    list_of_contents, list_of_names, list_of_dates
                ):
                    children.append(parse_contents(c, n, d))
                self.idat_handler = None
                self._update_paths()
                self.cpgs = self._get_cpgs()
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
                _ = self.classify(ids=self.cnv_id, clf_list=parsed_clf_list)
            except Exception as exc:
                logger.info("An error occured (4): %s", exc)
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
