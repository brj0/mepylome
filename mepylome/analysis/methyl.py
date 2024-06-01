"""Methylation analysis tools including a Dash-based browser application.

This package provides a set of tools for conducting methylation analysis. The
core functionality is encapsulated in the `MethylAnalysis` class, which
facilitates analysis setup, management, and the execution of a web application
for interactive exploration of methylation data.


Classes:
    MethylAnalysis: Main class for methylation analysis, providing methods for
        setting up analysis parameters, reading data, and running a Dash-based
        web application for data visualization.

Usage:
    from mepylome import MethylAnalysis

    # Basic usage
    analysis0 = MethylAnalysis()
    analysis0.run_app()

    # Usage if directories are known in advance
    analysis1 = MethylAnalysis(
        analysis_dir='/path/to/idat/dir',
        reference_dir='/path/to/reference/idat/dir',
        annotation='/path/to/annotation/spread/sheat/with/2/cols',
        output_dir='/path/to/mepylome/output,
    )
    analysis1.run_app()

"""

import colorsys
import hashlib
import os
import pathlib
import pickle
import subprocess
import threading
import webbrowser
from functools import lru_cache
from itertools import repeat
from multiprocessing import Pool
from pathlib import Path

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import pkg_resources
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
    COLOR_MAP,
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
    overlap_indices,
    read_cnv_data_from_disk,
)
from mepylome.utils import Timer, ensure_directory_exists

timer = Timer()

PLOTLY_RENDER_MODE = "webgl"

HOME_DIR = pathlib.Path.home()
DEFAULT_ANALYSIS_DIR = Path(HOME_DIR, "Documents", "mepylome", "analysis")
DEFAULT_ANNOTATION_FILE = Path(DEFAULT_ANALYSIS_DIR, "annotation.csv")
DEFAULT_REFERENCE_DIR = Path(HOME_DIR, "Documents", "mepylome", "reference")
DEFAULT_OUTPUT_DIR = Path(HOME_DIR, "Documents", "mepylome", "output")
DEFAULT_N_CPGS = 25000

ERROR_ENDING = "_error.txt"

ON = "on"
OFF = "off"

NEUTRAL_BETA = 0.49

# Contains CpGs common to both 450k and EPIC arrays, excluding those on sex
# chromosomes and cross-reactive probes (as identified in Chen et al., 2013).
CPG_450K_EPIC_OVERLAP = (
    "/applications/reference_data/betaEPIC450Kmix_bin/index.csv"
)
with open(CPG_450K_EPIC_OVERLAP) as f:
    CPG_450K = np.array(f.read().splitlines())


ACRONYMS = pkg_resources.resource_filename(
    "mepylome", "data/methylation_class_acronyms.tsv.gz"
)


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
                raise ValueError(
                    "ProgressBar cannot be initailized with 0 samples."
                )
            self.cur_value = cur_value
            self.max_value = int(max_value)
            self.text = str(text)

    def increment(self, n=1):
        with self.lock:
            self.cur_value = min(self.cur_value + n, self.max_value)

    def get_progress(self):
        with self.lock:
            progress = self.cur_value * 100 // self.max_value
            return progress

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


def random_color(var):
    """Pseudorandom color based on hashed string.

    Args:
        var: string to hash
    Returns:
        Tripple of rgb color.
    """
    hash_str = hashlib.md5(bytes(var, "utf-8")).digest()
    hash1 = int.from_bytes(hash_str[:8], byteorder="big")
    hash2 = int.from_bytes(hash_str[8:12], byteorder="big")
    hash3 = int.from_bytes(hash_str[12:], byteorder="big")
    hue = hash1 % 365
    saturation = hash2 % 91 + 10
    lightness = hash3 % 41 + 30
    # hsl has to be transformed to rgb for plotly, since otherwise not all
    # colors are displayed correctly, probably due to plotly bug.
    rgb_frac = colorsys.hls_to_rgb(
        hue / 364, lightness / 100, saturation / 100
    )
    rgb = tuple(int(255 * x) for x in rgb_frac)
    return rgb


def discrete_colors(names):
    """Returns a colorscheme for all methylation classes.

    Generate a pseudorandom color scheme based on precalculated values to
    enhance readability for neighboring methylation groups.

    Args:
        names (list of str): List of string elements (corresponds to
            methylation class acronyms).

    Returns:
        dict: Dictionary mapping each string element to its corresponding
            color.
    """
    return {
        var: COLOR_MAP[var] if var in COLOR_MAP else f"rgb{random_color(var)}"
        for var in set(names)
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
    # methyl_classes = umap_df[color_columns].iloc[:,0]
    # umap_df["Umap_color"] = umap_df[color_columns].iloc[:,0]
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


def adjust_columns(df, num_cols):
    if len(df.columns) < num_cols:
        for i in range(num_cols - len(df.columns)):
            df[f"new_col_{i}"] = ""
    return df


def description_from_acronym(df_col):
    acronyms_df = pd.read_csv(ACRONYMS, sep="\t")
    acronym_to_description = dict(
        zip(acronyms_df.Methylation_Class, acronyms_df.Description)
    )

    def get_description(mc):
        """Returns description of a methylation class from acronym."""
        if mc in [np.nan, ""]:
            return ""
        mc = mc.upper()
        if mc in acronym_to_description:
            return acronym_to_description[mc]
        substrings = [
            a for a in acronym_to_description if mc.startswith(a + "_")
        ]
        substrings.sort(key=len)
        if substrings:
            return acronym_to_description[substrings[-1]]
        return ""

    return df_col.apply(get_description)


class IdatFiles:
    """A class for handling IDAT file annotations.

    Includes reading from various file formats and provides description
    lookups for methylation classes.
    """

    def __init__(self, idat_dir, annotation=None, overlap=False):
        if annotation is None:
            annotation = guess_annotation_file(idat_dir)
        self.annotation = Path(annotation).expanduser()
        self.path = {
            x.name: x
            for x in idat_basepaths(idat_dir)
            if is_valid_idat_basepath(x)
        }
        self.annotation_df = None
        self.annotated_ids = self.get_annotated_ids()
        if overlap and self.annotated_ids is not None:
            ids_annotation = self.annotation_df.index
            self.path = {
                x: i for x, i in self.path.items() if x in ids_annotation
            }
            self.annotated_ids = self.annotated_ids.loc[self.path.keys()]

    def __len__(self):
        return len(self.path)

    def get_annotated_ids(self):
        df = pd.DataFrame(index=self.path.keys())
        if self.annotation.exists():
            if self.annotation.suffix == ".xlsx":
                self.annotation_df = pd.read_excel(
                    self.annotation,
                    engine="openpyxl",
                    index_col=0,
                )
            elif self.annotation.suffix in [".csv", ".tsv"]:
                sep = "\t" if self.annotation.suffix == ".tsv" else ","
                self.annotation_df = pd.read_csv(
                    self.annotation, sep=sep, index_col=0
                )
            else:
                raise ValueError(
                    "Annotation files must be as 'xlsx', 'tsv' or 'csv' file"
                )
            df = df.join(self.annotation_df)
            df = df.fillna("")
        if len(df.columns) == 0:
            df["Methylation_Class"] = ""
            df["Description"] = ""
        if len(df.columns) == 1:
            df["Description"] = description_from_acronym(df.iloc[:, 0])
        df = df.fillna("")
        return df

    @property
    def ids(self):
        return self.annotated_ids.index.tolist()

    @property
    def properties(self):
        return self.annotated_ids.columns.tolist()

    @property
    def methylation_class(self, columns=None):
        return self.annotated_ids.iloc[:, 0].tolist()

    @property
    def description(self, columns=None):
        if columns is None:
            return self.annotated_ids.iloc[:, 1].tolist()
        return self.annotated_ids[columns].tolist()

    def compound_class(self, columns=None):
        if columns is None:
            return self.annotated_ids.iloc[:, 0].tolist()
        if not isinstance(columns, list):
            columns = [columns]
        return (
            self.annotated_ids[columns]
            .apply(lambda row: "|".join(row.values.astype(str)), axis=1)
            .tolist()
        )

    def __str__(self):
        lines = [
            "IdatFiles():",
            f"annotation: '{self.annotation}'",
            f"annotated_ids:\n{self.annotated_ids}",
        ]
        return "\n".join(lines)

    def __repr__(self):
        return str(self)


def extract_beta(data):
    idat_file, cpgs, prep = data
    try:
        methyl = MethylData(file=idat_file, prep=prep)
        betas_450k_df = methyl.converted_beta(cpgs=cpgs, fill=NEUTRAL_BETA)
        betas = betas_450k_df.values.ravel()
        return betas, methyl.array_type
    except ValueError as e:
        return (idat_file, e), ArrayType.INVALID


def methyl_mtx_from_all_cpgs(betas_list, cpgs, fill=NEUTRAL_BETA):
    array_types = set(x[1] for x in betas_list)
    all_cpgs = {
        array_type: Manifest(array_type).get_cpgs()
        for array_type in array_types
    }
    left_idx = {}
    right_idx = {}
    for array_type in array_types:
        left_idx[array_type], right_idx[array_type] = overlap_indices(
            cpgs, all_cpgs[array_type]
        )
    converted = np.full((len(betas_list), len(cpgs)), fill)
    for i, (beta, array_type) in enumerate(betas_list):
        converted[i, left_idx[array_type]] = beta[right_idx[array_type]]
    converted[np.isnan(converted)] = fill
    return converted
    # return pd.DataFrame(converted.T, columns=cpgs, index=cpgs)


def get_betas(pbar, idat_files, cpgs, prep, save=False, betas_path=None):
    if betas_path is not None and betas_path.exists():
        with open(betas_path, "rb") as f:
            betas_list = pickle.load(f)
    else:
        # Load all manifests before parallelization
        Manifest.load(["450k", "epic", "epicv2"])
        betas_list = []
        _cpgs = None if save else cpgs
        with Pool() as pool, tqdm(
            total=len(idat_files), desc="Reading IDAT files"
        ) as tqdm_bar:
            for betas in pool.imap(
                extract_beta,
                zip(idat_files.path.values(), repeat(_cpgs), repeat(prep)),
            ):
                betas_list.append(betas)
                pbar.increment()
                tqdm_bar.update(1)
        if save:
            with open(betas_path, "wb") as f:
                pickle.dump(betas_list, f)
    valid_idx = [
        i for i, x in enumerate(betas_list) if x[1] != ArrayType.INVALID
    ]
    valid_ids = [idat_files.ids[i] for i in valid_idx]
    all_cpgs_in_betas = betas_path is not None and (
        betas_path.exists() or save
    )
    if all_cpgs_in_betas:
        valid_betas = [betas_list[i] for i in valid_idx]
        methyl_mtx = methyl_mtx_from_all_cpgs(valid_betas, cpgs)
    else:
        valid_betas = [betas_list[i][0] for i in valid_idx]
        methyl_mtx = np.vstack(valid_betas)
    betas_df = pd.DataFrame(methyl_mtx, index=valid_ids, columns=cpgs)
    return betas_df


@lru_cache(maxsize=None)
def get_reference_methyl_data(reference_dir, prep):
    return ReferenceMethylData(files=reference_dir, prep=prep)


def write_single_cnv_to_disk(args):
    idat_basename, reference_dir, cnv_dir, prep = args
    sample_id = idat_basename.name
    try:
        sample_methyl = MethylData(file=idat_basename)
        reference = get_reference_methyl_data(reference_dir, prep)
        cnv = CNV.set_all(sample_methyl, reference)
        cnv_filename = sample_id + ZIP_ENDING
        cnv.write(Path(cnv_dir, cnv_filename))
    except Exception as error:
        cnv_filename = sample_id + ERROR_ENDING
        command = f"ls -lh {str(idat_basename)}*"
        files_on_disk = subprocess.check_output(command, shell=True).decode(
            "utf-8"
        )
        error_message = (
            "During processing '"
            + sample_id
            + "' the following error occurred:\n\n"
            + str(error)
            + "\n\nCorresponding files on disk:\n"
            + files_on_disk
            + "\n\n\nTo recalculate, delete this file."
        )
        with open(Path(cnv_dir, cnv_filename), "w") as f:
            f.write(error_message)


def write_cnv_to_disk(sample_path, reference_dir, cnv_dir, prep, pbar=None):
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
    # Pooling is slower if there is only 1 sample
    if len(new_idat_paths) == 1:
        write_single_cnv_to_disk(
            (
                new_idat_paths[0],
                reference_dir,
                cnv_dir,
                prep,
            )
        )
    else:
        with Pool() as pool, tqdm(
            total=len(new_idat_paths), desc="Generating CNV files"
        ) as tqdm_bar:
            for _ in pool.imap(
                write_single_cnv_to_disk,
                zip(
                    new_idat_paths,
                    repeat(reference_dir),
                    repeat(cnv_dir),
                    repeat(prep),
                ),
            ):
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
):
    sample_id = sample_path.name
    write_cnv_to_disk([sample_path], reference_dir, cnv_dir, prep)
    bins, detail, segments = read_cnv_data_from_disk(cnv_dir, sample_id)
    plot = cnv_plot_from_data(
        sample_id,
        bins,
        detail,
        segments,
        IMPORTANT_GENES,
        list(genes_sel),
    )
    plot = plot.update_layout(
        margin=dict(l=0, r=0, t=30, b=0),
    )
    return plot


def get_all_genes():
    genomic_annotation = Annotation(array_type=ArrayType.ILLUMINA_450K)
    detail = genomic_annotation.detail.df
    return detail.Name.tolist()


EMPTY_FIGURE = go.Figure()
EMPTY_FIGURE.update_layout(
    xaxis=dict(visible=False),
    yaxis=dict(visible=False),
    plot_bgcolor="rgba(0, 0, 0, 0)",
    paper_bgcolor="rgba(0, 0, 0, 0)",
    showlegend=False,
    margin=dict(l=0, r=0, t=0, b=0),
)
EMPTY_FIGURE = go.Figure(layout=go.Layout(yaxis=dict(range=[-2, 2])))


def get_navbar():
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
):
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
                            html.H6("Choose analysis directory"),
                            dbc.Input(
                                id="analysis-dir",
                                valid=True,
                                value=str(analysis_dir),
                                type="text",
                            ),
                            html.Div(id="analysis-path-validation"),
                            html.Br(),
                            html.H6("Choose annotation file"),
                            dbc.Input(
                                id="annotation-file",
                                valid=True,
                                value=str(annotation),
                                type="text",
                            ),
                            html.Div(id="annotation-file-validation"),
                            html.Br(),
                            html.H6(
                                "Choose reference directory (CNV neutral cases)"
                            ),
                            dbc.Input(
                                id="reference-dir",
                                value=str(reference_dir),
                                type="text",
                            ),
                            html.Div(id="reference-path-validation"),
                            html.Br(),
                            html.H6("Choose output directory"),
                            dbc.Input(
                                id="output-dir",
                                valid=True,
                                value=str(output_dir),
                                type="text",
                            ),
                            html.Div(id="output-path-validation"),
                            html.Br(),
                            html.H6("Choose IDAT preprocessing method"),
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
                                    ON: ("Precalculate all (much " "longer!)"),
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
                            html.Br(),
                            html.Br(),
                            dbc.Progress(value=0, id="umap-progress-bar"),
                        ],
                    ),
                    dbc.Tab(
                        label="Plot",
                        children=[
                            dcc.Location(id="url", refresh=False),
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
    directory_path = Path(directory)
    files = list(directory_path.glob("*"))
    csv_files = [f for f in files if f.suffix == ".csv"]
    xlsx_files = [f for f in files if f.suffix == ".xlsx"]
    xls_files = [f for f in files if f.suffix == ".xls"]
    if csv_files:
        return csv_files[0]
    if xlsx_files:
        return xlsx_files[0]
    if xls_files:
        return xls_files[0]
    return DEFAULT_ANNOTATION_FILE


def input_args_hash(*args):
    result = "-".join(
        [
            str(arg.tolist()[:10]) if isinstance(arg, np.ndarray) else str(arg)
            for arg in args
        ]
    )
    return hashlib.md5(result.encode()).hexdigest()[:32]


def extract_sub_dataframe(data_frame, arr, fill=0.49):
    result_np = np.full((len(data_frame.index), len(arr)), fill)
    left_idx, right_idx = overlap_indices(arr, data_frame.columns)
    result_np[:, left_idx] = data_frame.values[:, right_idx]
    result = pd.DataFrame(result_np, columns=arr, index=data_frame.index)
    return result


def reorder_columns_by_variance(data_frame):
    variances = data_frame.var()
    sorted_columns = variances.sort_values(ascending=False).index
    reordered = data_frame[sorted_columns]
    return reordered


class MethylAnalysis:
    def __init__(
        self,
        analysis_dir=DEFAULT_ANALYSIS_DIR,
        annotation=None,
        reference_dir=DEFAULT_REFERENCE_DIR,
        output_dir=DEFAULT_OUTPUT_DIR,
        prep="illumina",
        cpgs="auto",
        n_cpgs=DEFAULT_N_CPGS,
        precalculate_cnv=False,
        save_betas=False,
        overlap=False,
        cpg_selection="random",
        host="localhost",
        port=8050,
        debug=False,
    ):
        self.umap_cpgs = None
        self.analysis_dir = analysis_dir
        self.idat_paths = None
        if annotation is None:
            annotation = guess_annotation_file(analysis_dir)
        if not Path(annotation).exists():
            print("No annotation file found")
        self.annotation = annotation
        self.overlap = overlap
        self.idat_files = IdatFiles(
            self.analysis_dir, self.annotation, self.overlap
        )
        self.cpgs = self.get_cpgs(cpgs)
        self.n_cpgs = n_cpgs
        self.cpg_selection = cpg_selection
        if self.cpg_selection not in ["top", "random"]:
            raise ValueError(
                "Invalid 'cpg_selection' (expected: 'top' or 'random')"
            )
        self.host = host
        self.port = port
        self.debug = debug
        self.umap_color_columns = None
        self.reference_dir = reference_dir
        self.output_dir = output_dir
        ensure_directory_exists(self.output_dir)
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
        self.cnv_plot = EMPTY_FIGURE
        self.raw_umap_plot = None
        self.cnv_id = None
        self.dropdown_id = None
        self.ids = []
        self.ids_to_highlight = None
        self.prog_bar = ProgressBar()
        self.app = None

        self.update_output_paths()
        self.read_umap_plot_from_disk()

    def get_cpgs(self, input_var):
        if isinstance(input_var, np.ndarray):
            return input_var
        valid_parms = ["auto", "450k", "epic", "epicv2"]
        if isinstance(input_var, str):
            input_var = [input_var]
        if "auto" in input_var:
            input_var = set()
            for path in self.idat_files.path.values():
                input_var.add(RawData(path).array_type.value)
            input_var = list(input_var)
            print(f"The following array types were detected: {input_var}")
        if all(x in valid_parms for x in input_var):
            if len(input_var) == 0:
                return np.array([])
            input_var = valid_parms if "all" in input_var else input_var
            cpg_sets = []
            for array_type in valid_parms[1:]:
                if array_type in input_var:
                    cpg_sets.append(set(Manifest(array_type).get_cpgs()))
            cpgs = set.intersection(*cpg_sets)
            return np.array(list(cpgs))
        raise ValueError(
            f"'cpgs' must be numpy array or a list of strings in {valid_parms}"
        )

    def update_output_paths(self):
        old_betas_path = self.betas_path
        umap_hash_key = input_args_hash(
            np.sort(self.cpgs),
            self.n_cpgs,
            self.analysis_dir,
            self.prep,
            self.cpg_selection,
        )
        betas_hash_key = input_args_hash(
            self.analysis_dir,
            self.prep,
        )
        cnv_hash_key = input_args_hash(
            self.analysis_dir,
            self.reference_dir,
            self.prep,
        )
        umap_dir_name = (
            f"umap-{self.n_cpgs}-{self.prep}-{self.cpg_selection}-"
            f"{umap_hash_key}"
        )
        self.umap_dir = Path(self.output_dir, f"{umap_dir_name}")
        ensure_directory_exists(self.umap_dir)

        cnv_dir_name = f"cnv-{self.prep}-{cnv_hash_key}"
        self.cnv_dir = Path(self.output_dir, f"{cnv_dir_name}")
        ensure_directory_exists(self.cnv_dir)

        self.umap_plot_path = Path(self.umap_dir, "umap_plot.csv")
        self.betas_path = Path(
            self.output_dir, f"betas-{self.prep}-{betas_hash_key}.pkl"
        )
        if old_betas_path != self.betas_path:
            self.betas_df_all_cpgs = None

    def make_umap(self):
        self.update_output_paths()
        self.idat_files = IdatFiles(
            self.analysis_dir, self.annotation, self.overlap
        )
        # if self.read_umap_plot_from_disk():
        # return
        if len(self.idat_files) == 0:
            raise ValueError("No valid samples found")

        self.prog_bar.reset(len(self.idat_files), text="(betas)")
        self.set_betas()
        self.prog_bar.reset(1, 1)
        self.compute_umap()
        self.make_umap_plot()

    def compute_umap(self):
        import umap

        umap_2d = umap.UMAP(verbose=True).fit_transform(self.betas_df)
        umap_df = pd.DataFrame(
            umap_2d,
            columns=["Umap_x", "Umap_y"],
            index=self.idat_files.ids,
        )
        self.umap_df = pd.concat(
            [
                umap_df,
                self.idat_files.annotated_ids,
            ],
            axis=1,
        )
        if self.umap_color_columns is None:
            self.umap_color_columns = self.umap_df.columns[2]
        self.umap_df.to_csv(self.umap_plot_path, sep="\t", index=True)

    def set_betas(self):
        self.update_output_paths()

        def get_random_cpgs():
            random_idx = np.sort(
                np.random.choice(len(self.cpgs), self.n_cpgs, replace=False)
            )
            return self.cpgs[random_idx]

        def _get_betas(cpgs):
            return get_betas(
                self.prog_bar,
                self.idat_files,
                cpgs,
                self.prep,
                self.save_betas,
                self.betas_path,
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
            if self.cpg_selection == "top":
                self.betas_df_all_cpgs = reorder_columns_by_variance(
                    self.betas_df_all_cpgs
                )
            self.betas_df = _extract_sub_dataframe()

        else:
            self.betas_df = _get_betas(self.umap_cpgs)

    def read_umap_plot_from_disk(self):
        self.update_output_paths()
        if self.umap_plot_path.exists():
            self.umap_df = pd.read_csv(
                self.umap_plot_path, sep="\t", index_col=0
            )
            self.umap_df = self.umap_df.fillna("")
            self.make_umap_plot()
            return True
        return False

    def make_umap_plot(self):
        self.ids = self.umap_df.index
        self.umap_df["Umap_color"] = self.idat_files.compound_class(
            self.umap_color_columns
        )
        self.umap_plot = umap_plot_from_data(self.umap_df)
        self.umap_plot = self.umap_plot.update_layout(
            margin=dict(l=0, r=0, t=30, b=0),
        )
        self.raw_umap_plot = self.umap_plot
        # self.cnv_id = None
        self.dropdown_id = None
        self.umap_plot_highlight()

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
                font=dict(
                    color="blue",
                ),
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
        idat_basepath = self.idat_files.path[sample_id]
        if not is_valid_idat_basepath(idat_basepath):
            raise FileNotFoundError(
                f"Sample {sample_id} not found in {self.analysis_dir}"
            )
        genes_sel = () if genes_sel is None else tuple(genes_sel)
        self.cnv_plot = get_cnv_plot(
            idat_basepath,
            self.reference_dir,
            self.prep,
            self.cnv_dir,
            genes_sel,
        )

    def precalculate_all_cnvs(self):
        self.update_output_paths()
        sample_ids = [
            x.name
            for x in idat_basepaths(self.analysis_dir)
            if not Path(self.cnv_dir, x.name + ZIP_ENDING).exists()
            and not Path(self.cnv_dir, str(x) + ERROR_ENDING).exists()
        ]
        self.prog_bar.reset(len(sample_ids), text="(CNV)")
        write_cnv_to_disk(
            list(self.idat_files.path.values()),
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
            self.idat_files.properties,
            self.analysis_dir,
            self.idat_files.annotation,
            self.reference_dir,
            self.output_dir,
            self.cpgs,
            self.n_cpgs,
            self.prep,
            self.precalculate_cnv,
            self.cpg_selection,
        )
        dash_plots = dbc.Col(
            [
                dcc.Graph(
                    id="umap-plot",
                    figure=self.umap_plot,
                    config={
                        "scrollZoom": True,
                        "doubleClick": "reset+autosize",
                        "modeBarButtonsToRemove": ["lasso2d", "select"],
                        "displaylogo": False,
                    },
                    # style={"width": "80%", "height": "60vh"},
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
                    # style={"width": "100%", "height": "40vh"},
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
            State("cnv-plot", "figure"),
        )
        def update_plots(
            click_data,
            ids_to_highlight,
            umap_color_columns,
            genes_sel,
            curr_umap_plot,
            curr_cnv_plot,
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
                self.umap_color_columns = umap_color_columns
                self.make_umap_plot()
                self.umap_plot_highlight(cnv_id=self.cnv_id)
                self.retrieve_zoom(curr_umap_plot)
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
                        return self.umap_plot, self.cnv_plot, ""
                    except Exception as e:
                        print("umap failed:", e)
                        return no_update, no_update, str(e)
            if trigger == "selected-genes" and genes_sel is not None:
                try:
                    self.make_cnv_plot(self.cnv_id, genes_sel)
                    return no_update, self.cnv_plot, ""
                except Exception as e:
                    print("selected-genes failed:", e)
                    print("self.cnv_id:", self.cnv_id)
                    print("genes_sel:", genes_sel)
                    return no_update, no_update, str(e)
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
                path = Path(input_path)
                if path.is_dir() and not os.access(path, os.W_OK):
                    return False, f"Protected directory: {path}"
                if path.is_dir():
                    return True, ""
                return False, f"Not a directory: {path}"
            except Exception:
                return False, "Invalid path format"

        @app.callback(
            [
                Output("annotation-file", "valid"),
                Output("annotation-file-validation", "children"),
            ],
            [Input("annotation-file", "value")],
            prevent_initial_call=False,
        )
        def validate_annotation_file(input_path):
            try:
                path = Path(input_path)
                if path.exists() and not os.access(path, os.W_OK):
                    return False, f"Protected file: {path}"
                if path.exists():
                    return True, ""
                return False, f"Not a file: {path}"
            except Exception:
                return False, "Invalid path format"

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
                path = Path(input_path)
                if path.is_dir() and not os.access(path, os.W_OK):
                    return False, f"Protected directory: {path}"
                if path.is_dir():
                    return True, ""
                return False, f"Not a directory: {path}"
            except Exception:
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
                path = Path(input_path)
                if path == DEFAULT_OUTPUT_DIR:
                    return True, ""
                if path.is_dir() and not os.access(path, os.W_OK):
                    return False, f"Protected directory: {path}"
                if path.is_dir():
                    return True, ""
                return False, f"Not a directory: {path}"
            except Exception:
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
                State("reference-dir", "valid"),
                State("annotation-file", "valid"),
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
            reference_dir_valid,
            annotation_file_valid,
            precalculate_cnv,
            cpg_selection,
        ):
            if not n_clicks:
                return no_update, no_update, "", {}

            error_message = None

            if n_clicks is None:
                error_message = "Invalid no. of CpGs."
            elif not analysis_dir_valid:
                error_message = "Invalid analysis path."
            elif not output_dir_valid:
                error_message = "Invalid Output path."
            elif not reference_dir_valid:
                error_message = "Invalid Reference path."
            elif not annotation_file_valid:
                error_message = "Invalid annotation path."
            elif prep is None:
                error_message = "Invalid preprocessing method."
            elif precalculate_cnv is None:
                error_message = "Invalid precalculation method."
            elif cpg_selection is None:
                error_message = "Invalid CpG selection method."

            if error_message:
                return no_update, no_update, error_message, {}

            self.n_cpgs = n_cpgs
            self.output_dir = output_dir
            self.reference_dir = reference_dir
            self.prep = prep
            self.precalculate_cnv = precalculate_cnv == ON
            self.cpg_selection = cpg_selection
            if (
                self.analysis_dir != analysis_dir
                or self.annotation != annotation
            ):
                self.analysis_dir = analysis_dir
                self.annotation = annotation
                self.idat_files = IdatFiles(
                    self.analysis_dir, self.annotation, self.overlap
                )

            try:
                self.make_umap()
                return (
                    self.umap_plot,
                    self.ids,
                    no_update,
                    {"status": "umap_done"},
                )
            except Exception as e:
                print(f"An error occurred: {e}")
            return no_update, no_update, "", {}

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

    def run_app(self, open_tab=False):
        self.app = self.get_app()
        if open_tab:

            def open_browser_tab():
                webbrowser.open_new_tab(f"http://{self.host}:{self.port}")

            threading.Timer(1, open_browser_tab).start()
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
            f"umap_color_columns:\n{self.umap_color_columns}",
            f"annotation:\n{self.annotation}",
            f"prep:\n{self.prep}",
            f"precalculate_cnv:\n{self.precalculate_cnv}",
            f"cpg_selection:\n{self.cpg_selection}",
            f"umap_df:\n{self.umap_df}",
        ]
        return "\n\n".join(lines)
