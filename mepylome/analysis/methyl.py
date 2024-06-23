"""Methylation analysis tools including a Dash-based browser application.

This module provides a comprehensive set of tools for conducting methylation
analysis. The core functionality is encapsulated in the ``MethylAnalysis``
class, which manages the methylation analysis process and executes an
interactive web application for the exploration of methylation data.
"""

import colorsys
import hashlib
import logging
import os
import pickle
import subprocess
import tempfile
import threading
import webbrowser
from functools import lru_cache, partial
from itertools import repeat
from multiprocessing import Pool
from pathlib import Path

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
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
from tqdm import tqdm

from mepylome.dtypes import (
    CNV,
    IMPORTANT_GENES,
    ZIP_ENDING,
    Annotation,
    ArrayType,
    Manifest,
    MethylData,
    RawData,
    ReferenceMethylData,
    cnv_plot_from_data,
    idat_basepaths,
    is_valid_idat_basepath,
    _overlap_indices,
    _get_cgsegment,
    read_cnv_data_from_disk,
)
from mepylome.utils import Timer, ensure_directory_exists

timer = Timer()


MEPYLOME_TMP_DIR = Path(tempfile.gettempdir()) / "mepylome"

DEFAULT_OUTPUT_DIR = Path(MEPYLOME_TMP_DIR, "methylation_analysis")
INVALID_PATH = Path("None")

PLOTLY_RENDER_MODE = "webgl"
DEFAULT_N_CPGS = 25000
NEUTRAL_BETA = 0.49
ERROR_ENDING = "_error.txt"

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


class ProgressBar:
    """A thread-safe progress bar.

    Attributes:
        cur_value (int): The current value of the progress bar.
        max_value (int): The maximum value of the progress bar.
        text (str): Optional text to display alongside the progress.
        lock (threading.Lock): A lock to ensure thread safety.
    """

    def __init__(self, max_value=100, text=""):
        self.cur_value = 0
        self.max_value = int(max_value)
        self.text = str(text)
        self.lock = threading.Lock()

    def reset(self, max_value=100, cur_value=0, text=""):
        with self.lock:
            if self.max_value == 0:
                msg = "ProgressBar cannot be initailized with 0 samples."
                raise ValueError(msg)
            self.cur_value = cur_value
            self.max_value = int(max_value)
            self.text = str(text)

    def increment(self, n=1):
        with self.lock:
            self.cur_value = min(self.cur_value + n, self.max_value)

    def get_progress(self):
        with self.lock:
            if self.max_value == 0:
                print("ProgressBar: max_value is 0")
                return 100
            return self.cur_value * 100 // self.max_value

    def get_text(self):
        with self.lock:
            if self.cur_value == self.max_value:
                out_str = "100 %"
            else:
                out_str = (
                    f"{self.cur_value}/{self.max_value} {self.text}".rstrip()
                )
            return out_str

    def __str__(self):
        lines = [
            "ProgressBar(",
            f"    cur_value: {self.cur_value}",
            f"    max_value: {self.max_value}",
            f"    progress: {self.get_progress()}",
            ")",
        ]
        return "\n".join(lines)

    def __repr__(self):
        return str(self)


def hash_from_str(string):
    """Calculates a pseudorandom int from a string."""
    hash_str = hashlib.md5(string.encode()).hexdigest()
    return int(hash_str, 16)


def random_color(string, i, n_strings, rand):
    """Generate a random RGB color for a given string-name.

    Ensures the color is unique and distributed across the hue spectrum.

    Args:
        string (str): The input string to generate the color for.
        i (int): The index of the current string-name.
        n_strings (int): The total number of string-names.
        rand (int): A random offset

    Returns:
        tuple: The RGB color as a tuple of integers.
    """
    hash_value = hash_from_str(string)
    hue = (360 * i // n_strings + rand) % 360
    saturation = (hash_value & 0xFFFF) % 91 + 10
    lightness = (hash_value >> 16 & 0xFFFF) % 41 + 30
    # hsl has to be transformed to rgb for plotly, since otherwise not all
    # colors are displayed correctly, probably due to plotly bug.
    rgb_frac = colorsys.hls_to_rgb(
        hue / 360, lightness / 100, saturation / 100
    )
    return tuple(int(255 * x) for x in rgb_frac)


def discrete_colors(names):
    """Returns a colorscheme for all methylation classes."""
    sorted_names = sorted(names, key=hash_from_str)
    n_names = len(sorted_names)
    rand = hash_from_str("-".join(sorted_names))
    return {
        var: f"rgb{random_color(var, i, n_names, rand)}"
        for i, var in enumerate(sorted_names)
    }


def umap_plot_from_data(umap_df):
    """Create and return umap plot from UMAP data.

    Args:
        umap_df: pandas data frame containing UMAP matrix and
            attributes. First row,w corresponds to sample.

    Returns:
        UMAP plot as plotly object.
    """
    methyl_classes = np.sort(umap_df["Umap_color"].unique())
    color_map = discrete_colors(methyl_classes)
    category_orders = {"Umap_color": methyl_classes}
    umap_plot = px.scatter(
        umap_df,
        x="Umap_x",
        y="Umap_y",
        labels={
            "Umap_x": "UMAP 0",
            "Umap_y": "UMAP 1",
            "Umap_color": "Class",
        },
        title="",
        color="Umap_color",
        color_discrete_map=color_map,
        hover_name=umap_df.index,
        category_orders=category_orders,
        hover_data=umap_df.columns,
        render_mode=PLOTLY_RENDER_MODE,
        template="simple_white",
    )
    umap_plot.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
        mirror=True,
    )
    umap_plot.update_xaxes(
        mirror=True,
    )
    return umap_plot


class IdatHandler:
    """A class for handling IDAT files with annotation.

    Includes reading annotation from various file formats and provides
    description lookups for methylation classes.
    """

    def __init__(self, idat_dir, annotation_file=None, *, overlap=False):
        self.idat_dir = Path(idat_dir)
        if annotation_file is None:
            annotation_file = guess_annotation_file(self.idat_dir)
        self.annotation_file = Path(annotation_file)
        self.overlap = overlap

        self.sample_paths = {
            x.name: x
            for x in idat_basepaths(self.idat_dir)
            if is_valid_idat_basepath(x)
        }
        self.annotation_df = None
        self.annotated_samples = self.load_annotated_samples()
        if self.overlap and self.annotated_samples is not None:
            ids_annotation = self.annotation_df.index
            self.sample_paths = {
                x: i
                for x, i in self.sample_paths.items()
                if x in ids_annotation
            }
            self.annotated_samples = self.annotated_samples.loc[
                self.sample_paths.keys()
            ]
        self.selected_columns = self.annotated_samples.columns[0]

    def __len__(self):
        return len(self.sample_paths)

    def load_annotated_samples(self):
        result_df = pd.DataFrame(index=self.sample_paths.keys())
        if self.annotation_file.exists():
            if self.annotation_file.suffix == ".xlsx":
                self.annotation_df = pd.read_excel(
                    self.annotation_file,
                    engine="openpyxl",
                    index_col=0,
                )
            elif self.annotation_file.suffix in [".csv", ".tsv"]:
                sep = "\t" if self.annotation_file.suffix == ".tsv" else ","
                self.annotation_df = pd.read_csv(
                    self.annotation_file, sep=sep, index_col=0
                )
            else:
                self.annotation_df = result_df
            result_df = result_df.join(self.annotation_df)
            result_df = result_df.fillna("")
        if len(result_df.columns) == 0:
            result_df["Methylation_Class"] = "NO_DIAGNOSIS"
        return result_df.fillna("")

    @property
    def ids(self):
        return self.annotated_samples.index.tolist()

    @property
    def properties(self):
        return self.annotated_samples.columns.tolist()

    def compound_class(self, columns=None):
        if columns is None:
            return self.annotated_samples.iloc[:, 0].tolist()
        if not isinstance(columns, list):
            columns = [columns]
        return (
            self.annotated_samples[columns]
            .apply(lambda row: "|".join(row.values.astype(str)), axis=1)
            .tolist()
        )

    def __str__(self):
        lines = [
            "IdatHandler():",
            f"annotation_file: '{self.annotation_file}'",
            f"annotated_samples:\n{self.annotated_samples}",
            f"selected_columns:\n{self.selected_columns}",
        ]
        return "\n".join(lines)

    def __repr__(self):
        return str(self)


def extract_beta(data):
    """Extracts beta values for specified CpGs from an IDAT file."""
    idat_file, cpgs, prep = data
    try:
        methyl = MethylData(file=idat_file, prep=prep)
        betas_450k_df = methyl.betas_at(cpgs=cpgs, fill=NEUTRAL_BETA)
        betas = betas_450k_df.values.ravel()
    except ValueError as e:
        return (idat_file, e), ArrayType.INVALID
    else:
        return betas, methyl.array_type


def methyl_mtx_from_all_cpgs(betas_list, cpgs, fill=NEUTRAL_BETA):
    """Creates a methylation matrix from a list of beta values.

    Args:
        betas_list (list): List of tuples containing beta values and array
            types.
        cpgs (list): List of CpGs to include in the matrix.
        fill (float, optional): Value to fill missing beta values. Defaults to
            NEUTRAL_BETA.

    Returns:
        np.ndarray: A 2D array representing the methylation matrix for the
            specified CpGs.
    """
    array_types = {x[1] for x in betas_list}
    all_cpgs = {
        array_type: Manifest(array_type).methylation_probes
        for array_type in array_types
    }
    left_idx = {}
    right_idx = {}
    for array_type in array_types:
        left_idx[array_type], right_idx[array_type] = _overlap_indices(
            cpgs, all_cpgs[array_type]
        )
    converted = np.full((len(betas_list), len(cpgs)), fill)
    for i, (betas, array_type) in enumerate(betas_list):
        converted[i, left_idx[array_type]] = betas[right_idx[array_type]]
    converted[np.isnan(converted)] = fill
    return converted


def get_betas(pbar, idat_handler, cpgs, prep, *, save=False, betas_path=None):
    """Extracts and processes beta values from IDAT files.

    Args:
        pbar (ProgressBar): Progress bar for tracking progress.
        idat_handler (IdatHandler): Handler for IDAT file paths and metadata.
        cpgs (list): List of CpGs to include in the output matrix.
        prep (str): Prepreparation method for the MethylData.
        save (bool, optional): Whether to save the extracted betas to a file.
            Defaults to False.
        betas_path (Path, optional): Path to save/load the betas. Defaults to
            None.

    Returns:
        pd.DataFrame: DataFrame containing the beta values for the specified
            CpGs.
    """
    if betas_path is not None and betas_path.exists():
        with betas_path.open("rb") as f:
            betas_list = pickle.load(f)
    else:
        # Load all manifests before parallelization
        Manifest.load(["450k", "epic", "epicv2"])
        betas_list = []
        _cpgs = None if save else cpgs
        with Pool() as pool, tqdm(
            total=len(idat_handler), desc="Reading IDAT files"
        ) as tqdm_bar:
            for betas in pool.imap(
                extract_beta,
                zip(
                    idat_handler.sample_paths.values(),
                    repeat(_cpgs),
                    repeat(prep),
                ),
            ):
                betas_list.append(betas)
                pbar.increment()
                tqdm_bar.update(1)
        if save:
            with betas_path.open("wb") as f:
                pickle.dump(betas_list, f)
    valid_idx = [
        i for i, x in enumerate(betas_list) if x[1] != ArrayType.INVALID
    ]
    valid_ids = [idat_handler.ids[i] for i in valid_idx]
    all_cpgs_in_betas = betas_path is not None and (
        betas_path.exists() or save
    )
    if all_cpgs_in_betas:
        valid_betas = [betas_list[i] for i in valid_idx]
        methyl_mtx = methyl_mtx_from_all_cpgs(valid_betas, cpgs)
    else:
        valid_betas = [betas_list[i][0] for i in valid_idx]
        methyl_mtx = np.vstack(valid_betas)
    return pd.DataFrame(methyl_mtx, index=valid_ids, columns=cpgs)


@lru_cache(maxsize=None)
def get_reference_methyl_data(reference_dir, prep):
    """Loads and caches CNV-neutral reference data."""
    try:
        reference = ReferenceMethylData(file=reference_dir, prep=prep)
    except FileNotFoundError as exc:
        msg = f"File in reference dir {reference_dir} not found: {exc}"
        raise FileNotFoundError(msg) from exc
    return reference


def write_single_cnv_to_disk(
    idat_basename, reference_dir, cnv_dir, prep, do_seg
):
    """Performs CNV analysis on a single sample and writes results to disk."""
    sample_id = idat_basename.name
    try:
        sample_methyl = MethylData(file=idat_basename)
        reference = get_reference_methyl_data(reference_dir, prep)
        cnv = CNV.set_all(sample_methyl, reference, do_seg=do_seg)
        cnv_filename = sample_id + ZIP_ENDING
        cnv.write(Path(cnv_dir, cnv_filename))
    except Exception as exc:
        cnv_filename = sample_id + ERROR_ENDING
        files_on_disk = [
            f"{x}, size={x.stat().st_size} B"
            for x in idat_basename.parent.glob(f"{sample_id}*")
        ]
        error_message = (
            "During processing '"
            + sample_id
            + "' the following exception occurred:\n\n"
            + str(exc)
            + "\n\nCorresponding files on disk:\n"
            + "\n".join(files_on_disk)
            + "\n\n\nTo recalculate, delete this file."
        )
        print(error_message)
        with Path(cnv_dir, cnv_filename).open("w") as f:
            f.write(error_message)


def write_cnv_to_disk(
    sample_path, reference_dir, cnv_dir, prep, do_seg, pbar=None
):
    """Generate and save CNV-analysis output files for given samples.

    Saves CNV data with a ZIP_ENDING extension, or an error message with an
    ERROR_ENDING extension. Processes unseen samples, avoiding existing CNV and
    samples that lead to an error. Uses single-threading for one sample, and
    multi-threading for multiple samples.

    Args:
        sample_path (list): Paths to sample IDAT files.
        reference_dir (str): Directory with CNV neural reference data.
        cnv_dir (str): Directory to save CNV data.
        prep (str): Prepreparation method for MethylData.
        do_seg (bool): If segments should be calculated as well (slow)
        pbar (optional): Progress bar for tracking progress.
    """
    new_idat_paths = [
        x
        for x in sample_path
        if not Path(cnv_dir, str(x.name) + ZIP_ENDING).exists()
        and not Path(cnv_dir, str(x.name) + ERROR_ENDING).exists()
    ]
    if len(new_idat_paths) == 0:
        return
    # Load the reference into memory before parallelization to prevent loading
    # it for each core.
    Manifest.load()
    _ = get_reference_methyl_data(reference_dir, prep)
    _write_single_cnv_to_disk = partial(
        write_single_cnv_to_disk,
        reference_dir=reference_dir,
        cnv_dir=cnv_dir,
        prep=prep,
        do_seg=do_seg,
    )
    # Pooling is slower if there is only 1 sample
    if len(new_idat_paths) == 1:
        _write_single_cnv_to_disk(new_idat_paths[0])
    else:
        with Pool() as pool, tqdm(
            total=len(new_idat_paths), desc="Generating CNV files"
        ) as tqdm_bar:
            for _ in pool.imap(_write_single_cnv_to_disk, new_idat_paths):
                if pbar is not None:
                    pbar.increment()
                _ = tqdm_bar.update(1)


@lru_cache
def get_cnv_plot(
    sample_path,
    reference_dir,
    prep,
    cnv_dir,
    genes_sel,
    do_seg,
):
    """Generate and return a CNV plot for a given sample.

    Args:
        sample_path (Path): Path to the sample IDAT file.
        reference_dir (str): Directory with reference data.
        prep (str): Prepreparation method for MethylData.
        cnv_dir (str): Directory to save CNV data.
        genes_sel (list): List of genes to highlight in the plot.
        do_seg (bool): If segments should be calculated as well (slow)

    Returns:
        plotly.graph_objs.Figure: CNV plotly figure.
    """
    sample_id = sample_path.name
    write_cnv_to_disk([sample_path], reference_dir, cnv_dir, prep, do_seg)
    bins, detail, segments = read_cnv_data_from_disk(cnv_dir, sample_id)
    plot = cnv_plot_from_data(
        sample_id,
        bins,
        detail,
        segments,
        IMPORTANT_GENES,
        list(genes_sel),
    )
    return plot.update_layout(
        margin={"l": 0, "r": 0, "t": 30, "b": 0},
    )


def get_all_genes():
    """Returns a list of names for all genes."""
    return Annotation.default_genes().df.Name.tolist()


EMPTY_FIGURE = go.Figure()
EMPTY_FIGURE.update_layout(
    xaxis={"visible": False},
    yaxis={"visible": False},
    plot_bgcolor="rgba(0, 0, 0, 0)",
    paper_bgcolor="rgba(0, 0, 0, 0)",
    showlegend=False,
    margin={"l": 0, "r": 0, "t": 0, "b": 0},
)
EMPTY_FIGURE = go.Figure(layout=go.Layout(yaxis={"range": [-2, 2]}))


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
):
    """Generates a side navigation panel for setting up parameters."""
    n_cpgs_max = np.inf if len(cpgs) == 0 else len(cpgs)
    n_cpgs_max_str = "" if n_cpgs_max == np.inf else f" (max. {n_cpgs_max})"
    return dbc.Col(
        [
            dbc.Tabs(
                [
                    dbc.Tab(
                        label="Settings",
                        children=[
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
                                    "top": "Take best (most varying) CpG's",
                                },
                                value=cpg_selection,
                                multi=False,
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
                            dcc.Store(id="running-state"),
                            html.Div(id="output-div"),
                            dcc.Interval(
                                id="clock",
                                interval=1000,
                                n_intervals=0,
                                max_intervals=-1,
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
                                max=1,
                                step=0.01,
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
                            html.H6("Genes to highlight in CNV"),
                            dcc.Dropdown(
                                id="selected-genes",
                                options=get_all_genes(),
                                multi=True,
                            ),
                        ],
                    ),
                ]
            ),
        ],
        width={"size": 2},
    )


def guess_annotation_file(directory):
    """Returns the first csv or xlsx file in the given directory."""
    files = list(directory.glob("*"))
    csv_files = [f for f in files if f.suffix == ".csv"]
    xlsx_files = [f for f in files if f.suffix == ".xlsx"]
    if csv_files:
        return csv_files[0]
    if xlsx_files:
        return xlsx_files[0]
    return INVALID_PATH


def input_args_id(*args):
    """Returns a unique identifier for a set of arguments."""
    result = "-".join(
        str(arg.tolist()) if isinstance(arg, np.ndarray) else str(arg)
        for arg in args
    )
    return hashlib.sha256(result.encode()).hexdigest()


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


class MethylAnalysis:
    """Main class for methylation analysis including a GUI application.

    Main class for methylation analysis, providing methods for
    setting up analysis parameters, reading data, and running a Dash-based
    web application for data visualization.

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
        n_cpgs=DEFAULT_N_CPGS,
        precalculate_cnv=False,
        save_betas=False,
        overlap=False,
        cpg_selection="top",
        do_seg=False,
        host="localhost",
        port=8050,
        debug=False,
        umap_parms=None,
    ):
        self.umap_cpgs = None
        self.analysis_dir = Path(analysis_dir).expanduser()
        self._old_analysis_dir = self.analysis_dir
        self.idat_paths = None
        self.annotation = Path(annotation).expanduser()
        if self.annotation == INVALID_PATH:
            self.annotation = guess_annotation_file(self.analysis_dir)
        if not self.annotation.exists():
            print("No annotation file found")
        self.overlap = overlap
        self._idat_handler = None
        self.cpgs = self.get_cpgs(cpgs)
        self.n_cpgs = n_cpgs
        self.cpg_selection = cpg_selection
        if self.cpg_selection not in ["top", "random"]:
            msg = "Invalid 'cpg_selection' (expected: 'top' or 'random')"
            raise ValueError(msg)
        self.do_seg = False if _get_cgsegment() is None else do_seg
        self.host = host
        self.port = port
        self.debug = debug
        self.reference_dir = Path(reference_dir).expanduser()
        self.output_dir = Path(output_dir).expanduser()
        self.cnv_dir = None
        self.umap_dir = None
        self.prep = prep
        self.precalculate_cnv = precalculate_cnv
        self.save_betas = save_betas
        self.umap_plot = EMPTY_FIGURE
        self.umap_plot_path = None
        self.betas_df = None
        self.betas_df_all_cpgs = None
        self.betas_path = None
        self.umap_df = None
        self.umap_parms = {} if umap_parms is None else umap_parms
        self.cnv_plot = EMPTY_FIGURE
        self.raw_umap_plot = None
        self.cnv_id = None
        self.dropdown_id = None
        self.ids = []
        self.ids_to_highlight = None
        self.prog_bar = ProgressBar()
        self.app = None

        self.set_umap_parms()
        self.update_paths()
        self.read_umap_plot_from_disk()

    @property
    def idat_handler(self):
        if (
            self._idat_handler is None
            or self.analysis_dir != self._idat_handler.idat_dir
            or self.annotation != self._idat_handler.annotation_file
            or self.overlap != self._idat_handler.overlap
        ):
            self._idat_handler = IdatHandler(
                self.analysis_dir, self.annotation, overlap=self.overlap
            )
        return self._idat_handler

    @idat_handler.setter
    def idat_handler(self, value):
        self._idat_handler = value

    def set_umap_parms(self):
        default = {
            "metric": "euclidean",
            "min_dist": 0.1,
            "n_neighbors": 15,
            "verbose": True,
        }
        self.umap_parms = {**default, **self.umap_parms}

    def get_cpgs(self, input_var="auto"):
        if isinstance(input_var, np.ndarray):
            return np.sort(input_var)
        valid_parms = ["auto", "450k", "epic", "epicv2"]

        if isinstance(input_var, str):
            input_var = [input_var]

        if "auto" in input_var:
            input_var = {
                RawData(path).array_type.value
                for path in self.idat_handler.sample_paths.values()
            }
            input_var = list(input_var)
            print(
                f"The following array types were detected: {input_var}"
            )

        if all(x in valid_parms for x in input_var):
            if not input_var:
                return np.array([])

            input_var = valid_parms if "all" in input_var else input_var
            cpg_sets = [
                set(Manifest(array_type).methylation_probes)
                for array_type in valid_parms[1:]
                if array_type in input_var
            ]
            cpgs = set.intersection(*cpg_sets)
            return np.sort(np.array(list(cpgs)))

        msg = f"'cpgs' must be a numpy array or list of {valid_parms}"
        raise ValueError(msg)

    def update_paths(self):
        timer.start()
        if not self.output_dir.exists():
            return
        old_betas_path = self.betas_path
        umap_hash_key = input_args_id(
            self.cpgs,
            self.n_cpgs,
            self.analysis_dir,
            self.prep,
            self.cpg_selection,
        )
        betas_hash_key = input_args_id(
            self.analysis_dir,
            self.prep,
        )
        cnv_hash_key = input_args_id(
            self.analysis_dir,
            self.reference_dir,
            self.prep,
            self.do_seg,
        )
        umap_dir_name = (
            f"umap-{self.analysis_dir.name}"
            f"-{self.n_cpgs}-{self.prep}-{self.cpg_selection}"
            f"-{umap_hash_key}"
        )
        self.umap_dir = Path(self.output_dir, f"{umap_dir_name}")
        ensure_directory_exists(self.umap_dir)

        self.cnv_dir = Path(
            self.output_dir,
            f"cnv-{self.analysis_dir.name}-{self.prep}-{cnv_hash_key}",
        )
        ensure_directory_exists(self.cnv_dir)

        self.umap_plot_path = Path(self.umap_dir, "umap_plot.csv")
        self.betas_path = Path(
            self.output_dir,
            f"betas-{self.analysis_dir.name}-{self.prep}-{betas_hash_key}.pkl",
        )
        if old_betas_path != self.betas_path:
            self.betas_df_all_cpgs = None

        if self._old_analysis_dir != self.analysis_dir:
            self._old_analysis_dir = self.analysis_dir
            self.cpgs = self.get_cpgs()
            self.betas_df_all_cpgs = None
            self.betas_df = None

    def make_umap(self):
        self.prog_bar.reset(len(self.idat_handler), text="(betas)")
        self.set_betas()
        self.prog_bar.reset(1, 1)
        umap_2d = self.compute_umap()
        self.make_umap_plot(umap_2d)

    def compute_umap(self):
        import umap

        return umap.UMAP(**self.umap_parms).fit_transform(self.betas_df)

    def make_umap_plot(self, umap_2d=None):
        if umap_2d is not None:
            if len(self.idat_handler.ids) != len(umap_2d):
                msg = (
                    "Dimension mismatch: Analysis files may have changed. "
                    f"Try to delete cached files in {self.output_dir}."
                )
                raise AttributeError(msg)
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

        if self.umap_df is None:
            msg = "'umap_df' not set"
            raise AttributeError(msg)
        self.ids = self.umap_df.index
        umap_color = self.idat_handler.compound_class(
            self.idat_handler.selected_columns
        )
        if len(umap_color) != len(self.umap_df):
            msg = (
                "Dimension mismatch: Analysis files may have changed. "
                f"Try to delete cached files in {self.output_dir}."
            )
            raise AttributeError(msg)
        self.umap_df["Umap_color"] = umap_color
        self.umap_plot = umap_plot_from_data(self.umap_df)
        self.umap_plot = self.umap_plot.update_layout(
            margin={"l": 0, "r": 0, "t": 30, "b": 0},
        )
        self.raw_umap_plot = self.umap_plot
        self.dropdown_id = None
        self.umap_plot_highlight()

    def read_umap_plot_from_disk(self):
        self.update_paths()
        if self.umap_plot_path is not None and self.umap_plot_path.exists():
            self.umap_df = pd.read_csv(
                self.umap_plot_path, sep="\t", index_col=0
            )
            self.umap_df = self.umap_df.fillna("")
            self.make_umap_plot()
            return True
        return False

    def set_betas(self):
        if len(self.idat_handler) == 0:
            msg = "No valid samples found"
            raise ValueError(msg)

        self.update_paths()

        def get_random_cpgs():
            rng = np.random.default_rng()
            random_idx = np.sort(
                rng.choice(len(self.cpgs), self.n_cpgs, replace=False)
            )
            return self.cpgs[random_idx]

        def _get_betas(cpgs):
            return get_betas(
                self.prog_bar,
                self.idat_handler,
                cpgs,
                self.prep,
                save=self.save_betas,
                betas_path=self.betas_path,
            )

        def _extract_sub_dataframe():
            if self.cpg_selection == "random":
                return extract_sub_dataframe(
                    self.betas_df_all_cpgs, self.umap_cpgs
                )
            return self.betas_df_all_cpgs.iloc[:, : self.n_cpgs]

        if self.cpg_selection == "random":
            self.umap_cpgs = get_random_cpgs()

        if self.betas_df_all_cpgs is not None:
            self.betas_df = _extract_sub_dataframe()

        elif self.save_betas or self.cpg_selection == "top":
            self.betas_df_all_cpgs = _get_betas(self.cpgs)
            self.betas_df_all_cpgs = reorder_columns_by_variance(
                self.betas_df_all_cpgs
            )
            self.betas_df = _extract_sub_dataframe()

        else:
            self.betas_df = _get_betas(self.umap_cpgs)

    def get_coordinates(self, sample_id):
        return self.umap_df[self.umap_df.index == sample_id].iloc[0][
            ["Umap_x", "Umap_y"]
        ]

    def umap_plot_highlight(self, cnv_id=None):
        if cnv_id is not None:
            self.cnv_id = cnv_id
        self.dropdown_id = (
            [] if self.ids_to_highlight is None else self.ids_to_highlight
        )
        self.umap_plot = go.Figure(self.raw_umap_plot)
        for id_ in self.dropdown_id:
            x, y = self.get_coordinates(id_)
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
            x, y = self.get_coordinates(self.cnv_id)
            self.umap_plot.add_annotation(
                x=x,
                y=y,
                text=f"CNV: {self.cnv_id}",
                showarrow=True,
                arrowhead=2,
            )

    def retrieve_zoom(self, current_plot):
        self.umap_plot.layout.xaxis = current_plot["layout"]["xaxis"]
        self.umap_plot.layout.yaxis = current_plot["layout"]["yaxis"]

    def make_cnv_plot(self, sample_id, genes_sel=None):
        idat_basepath = self.idat_handler.sample_paths[sample_id]
        if not is_valid_idat_basepath(idat_basepath):
            msg = f"Sample {sample_id} not found in {self.analysis_dir}"
            raise FileNotFoundError(msg)
        if not self.reference_dir.exists():
            msg = f"Reference dir {self.reference_dir} does not exist"
            raise FileNotFoundError(msg)
        genes_sel = () if genes_sel is None else tuple(genes_sel)
        self.cnv_plot = get_cnv_plot(
            idat_basepath,
            self.reference_dir,
            self.prep,
            self.cnv_dir,
            genes_sel,
            self.do_seg,
        )

    def precalculate_all_cnvs(self):
        self.update_paths()
        sample_ids = [
            x.name
            for x in idat_basepaths(self.analysis_dir)
            if not Path(self.cnv_dir, x.name + ZIP_ENDING).exists()
            and not Path(self.cnv_dir, str(x) + ERROR_ENDING).exists()
        ]
        self.prog_bar.reset(len(sample_ids), text="(CNV)")
        write_cnv_to_disk(
            list(self.idat_handler.sample_paths.values()),
            self.reference_dir,
            self.cnv_dir,
            self.prep,
            self.prog_bar,
        )
        self.prog_bar.reset(1, 1)

    def get_app(self):
        current_dir = Path(__file__).resolve().parent
        assets_folder = current_dir.parent / "data" / "assets"
        app = Dash(
            __name__,
            assets_folder=assets_folder,
            external_stylesheets=[dbc.themes.BOOTSTRAP],
        )
        app._favicon = "favicon.svg"
        app.title = "mepylome"
        side_navigation = get_side_navigation(
            self.ids,
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
                Input("selected-genes", "value"),
            ],
            State("umap-plot", "figure"),
        )
        def update_plots(
            click_data,
            ids_to_highlight,
            umap_color_columns,
            genes_sel,
            curr_umap_plot,
        ):
            genes_sel = tuple(genes_sel) if genes_sel else ()
            trigger = callback_context.triggered[0]["prop_id"].split(".")[0]
            self.ids_to_highlight = ids_to_highlight
            if trigger == "ids-to-highlight" and ids_to_highlight is not None:
                self.umap_plot_highlight()
                self.retrieve_zoom(curr_umap_plot)
                return self.umap_plot, no_update, ""
            if (
                trigger == "umap-annotation-color"
                and umap_color_columns is not None
            ):
                self.idat_handler.selected_columns = umap_color_columns
                try:
                    self.make_umap_plot()
                    self.umap_plot_highlight(cnv_id=self.cnv_id)
                    self.retrieve_zoom(curr_umap_plot)
                except AttributeError:
                    return no_update, no_update, no_update
                else:
                    return self.umap_plot, no_update, ""

            if trigger == "umap-plot" and isinstance(click_data, dict):
                points = click_data.get("points")
                if isinstance(points, list):
                    first_point = points[0] if points else {}
                    sample_id = first_point.get("hovertext")
                    if sample_id is None:
                        return no_update, no_update, ""
                    self.umap_plot_highlight(cnv_id=sample_id)
                    self.retrieve_zoom(curr_umap_plot)
                    try:
                        self.make_cnv_plot(sample_id, genes_sel)
                    except Exception as e:
                        print("umap failed:", e)
                        print("sample_id:", sample_id)
                        print("MethylAnalysis:", self)
                        return no_update, no_update, str(e)
                    else:
                        return self.umap_plot, self.cnv_plot, ""
            if trigger == "selected-genes" and genes_sel is not None:
                try:
                    self.make_cnv_plot(self.cnv_id, genes_sel)
                except Exception as e:
                    print("selected-genes failed:", e)
                    print("self.cnv_id:", self.cnv_id)
                    print("genes_sel:", genes_sel)
                    return no_update, no_update, str(e)
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
            ],
            [Input("annotation-file", "value")],
            prevent_initial_call=False,
        )
        def validate_annotation_file(input_path):
            try:
                path = Path(input_path).expanduser()
                if path.exists() and not os.access(path, os.W_OK):
                    return False, f"Protected file: {path}", no_update
                if path.exists():
                    self.annotation = path
                    return True, "", self.idat_handler.properties
            except Exception:
                return False, "Invalid path format", no_update
            else:
                return False, f"Not a file: {path}", no_update

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
                print(f"An error occured (validate_reference_path): {exc}")
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
                print(f"An error occured (validate_output_path): {exc}")
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
        def on_start_button_click(
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
                print(f"An error occurred: {exc}")
            else:
                return (
                    self.umap_plot,
                    self.ids,
                    no_update,
                    {"status": "umap_done"},
                )
            return no_update, no_update, "", {}

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
                self.precalculate_all_cnvs()
            # Dummy update
            return no_update

        @app.callback(
            [
                Output("umap-progress-bar", "value"),
                Output("umap-progress-bar", "label"),
            ],
            [Input("clock", "n_intervals")],
        )
        def progress_bar_update(n):
            progress = self.prog_bar.get_progress()
            out_str = self.prog_bar.get_text()
            return progress, out_str

        return app

    def run_app(self, *, open_tab=False):
        self.app = self.get_app()
        if open_tab:

            def open_browser_tab():
                webbrowser.open_new_tab(f"http://{self.host}:{self.port}")

            threading.Timer(1, open_browser_tab).start()

        # Don't show all the flask logging statements.
        logger = logging.getLogger("werkzeug")
        logger.setLevel(logging.ERROR)

        self.app.run(debug=self.debug, host=self.host, use_reloader=False)

    def __repr__(self):
        title = "MethylAnalysis():"
        lines = [
            title + "\n" + "*" * len(title),
            f"cnv_id:\n{self.cnv_id}",
            f"n_cpgs:\n{self.n_cpgs}",
            f"analysis_dir:\n{self.analysis_dir}",
            f"reference_dir:\n{self.reference_dir}",
            f"output_dir:\n{self.output_dir}",
            f"umap_dir:\n{self.umap_dir}",
            f"cnv_dir:\n{self.cnv_dir}",
            f"umap_dir:\n{self.umap_dir}",
            f"selected_columns:\n{self.idat_handler.selected_columns}",
            f"annotation:\n{self.annotation}",
            f"prep:\n{self.prep}",
            f"precalculate_cnv:\n{self.precalculate_cnv}",
            f"cpg_selection:\n{self.cpg_selection}",
            f"umap_df:\n{self.umap_df}",
        ]
        return "\n\n".join(lines)
