import colorsys
import dash
import dash_daq as daq
import pathlib
from dash import callback_context
import hashlib
import importlib
import pkg_resources
import json
import random
from functools import lru_cache, wraps
from multiprocessing import Pool
import os
from pathlib import Path

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import umap
from dash import Dash, Input, Output, callback, dcc, html, no_update, State
from dash.exceptions import PreventUpdate
from tqdm import tqdm

from mepylome import CNV, Manifest, MethylData, RawData, idat_basepaths
from mepylome.dtypes import (
    CHROMOSOME_DATA,
    COLOR_MAP,
    IMPORTANT_GENES,
    ArrayType,
    Annotation,
    MANIFEST_TMP_DIR,
    ZIP_ENDING,
    ReferenceMethylData,
    cnv_plot,
    read_cnv_data_from_disk,
    cache,
    memoize,
)
from mepylome.utils import ensure_directory_exists, Timer

timer = Timer()

# IDAT_DIR = "/data/epidip_IDAT"
# IDAT_DIR = "/mnt/ws528695/data/epidip_IDAT"
IDAT_DIR = Path("~/MEGA/work/programming/data/epidip_IDAT").expanduser()
PLOTLY_RENDER_MODE = "webgl"
HOST = "localhost"

HOME_DIR = pathlib.Path.home()
DEFAULT_ANALYSIS_DIR = Path(HOME_DIR, "Documents", "mepylome", "analysis")
DEFAULT_ANNOTATION_FILE = Path(DEFAULT_ANALYSIS_DIR, "annotation.csv")
DEFAULT_REFERENCE_DIR = Path(HOME_DIR, "Documents", "mepylome", "reference")
DEFAULT_OUTPUT_DIR = Path(HOME_DIR, "Documents", "mepylome", "output")
DEFAULT_N_CPGS = 25000

# Contains CpGs common to both 450k and EPIC arrays, excluding those on sex
# chromosomes and cross-reactive probes (as identified in Chen et al., 2013).
CPG_450K_EPIC_OVERLAP = (
    "/applications/reference_data/betaEPIC450Kmix_bin/index.csv"
)
CNV_LINK = (
    "http://s1665.rootserver.io/umapplot01/%s_CNV_IFPBasel_annotations.pdf"
)
CNV_LINK = "http://localhost:8050/%s"

ACRONYMS = pkg_resources.resource_filename(
    "mepylome", "data/methylation_class_acronyms.tsv.gz"
)



ref_dir = "/data/ref_IDAT"
reference_methyl = ReferenceMethylData(ref_dir)
# reference_methyl[ArrayType.ILLUMINA_450K]


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
    """Pseudorandom color scheme based on precalculated values to improve
    readability for neighboring methylation groups.
    Args:
        names: List of strings.
    Returns:
        Dictionary of color scheme for all string elements.
    """
    color = {}
    for var in set(names):
        if var in COLOR_MAP.keys():
            color[var] = COLOR_MAP[var]
        else:
            color[var] = f"rgb{random_color(var)}"
    return color


def umap_plot_from_data(umap_df, sample=None, reference=None, close_up=False):
    """Create and return umap plot from UMAP data.
    Args:
        umap_df: pandas data frame containing UMAP matrix and
            attributes. First row,w corresponds to sample.
        sample: Sample data.
        reference: Reference data.
        close_up: Bool to indicate if only top matches should be plotted.
    Returns:
        UMAP plot as plotly object.
    """
    methyl_classes = umap_df.methylation_class[1:].to_list()
    methyl_classes.sort()
    if sample is None or sample.cpgs_only():
        color_map = discrete_colors(methyl_classes)
        category_orders = {"methylation_class": methyl_classes}
        add_sample = False
        title0 = ""
        title1 = ""
    else:
        color_map = {sample.name: "#ff0000", **discrete_colors(methyl_classes)}
        category_orders = {"methylation_class": [sample.name] + methyl_classes}
        add_sample = True
        umap_sample = umap_df.iloc[0]
        title0 = f"for {sample.name}"
        title1 = f", {len(sample.cpg_overlap)}"
    if reference is not None:
        umap_title = (
            f"UMAP {title0} <br><sup>Reference: {reference.name} "
            f"({len(reference.specimens)} cases){title1} CpGs </sup>"
        )
    else:
        umap_title = ""
    if close_up:
        umap_title = "Close-up " + umap_title
    umap_plot = px.scatter(
        umap_df,
        x="x",
        y="y",
        labels={
            # "x": "",
            "x": "UMAP 0",
            # "y": "",
            "y": "UMAP 1",
            "methylation_class": "WHO class",
        },
        title=umap_title,
        color="methylation_class",
        color_discrete_map=color_map,
        hover_name="id",
        category_orders=category_orders,
        hover_data=["description"],
        render_mode=PLOTLY_RENDER_MODE,
        template="simple_white",
        # template="none",
    )
    if add_sample:
        umap_plot.add_annotation(
            x=umap_sample["x"],
            y=umap_sample["y"],
            text=sample.name,
            showarrow=True,
            arrowhead=2,
        )
    umap_plot.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
        mirror=True,
    )
    umap_plot.update_xaxes(
        mirror=True,
    )
    umap_df_ref = umap_df[1:] if add_sample else umap_df
    links = [
        f"<a href='{url}' target='_blank'>&nbsp;</a>"
        for url in [CNV_LINK % id_ for id_ in umap_df_ref.id]
    ]
    # Add hyperlinks
    umap_plot.add_trace(
        go.Scatter(
            x=umap_df_ref.x,
            y=umap_df_ref.y,
            mode="text",
            name="CNV links",
            text=links,
            hoverinfo="skip",
            visible="legendonly",
        )
    )
    # If close-up add hyperlinks for all references and draw circle
    if close_up:
        umap_plot.update_traces(marker=dict(size=5))
        # Draw circle
        radius = umap_df["distance"].iloc[-1]
        umap_plot.add_shape(
            type="circle",
            x0=umap_sample["x"] - radius,
            y0=umap_sample["y"] - radius,
            x1=umap_sample["x"] + radius,
            y1=umap_sample["y"] + radius,
            fillcolor="rgba(0,0,0,0)",
            line_color="black",
            line_width=1.0,
        )
    return umap_plot


with open(CPG_450K_EPIC_OVERLAP, "r") as f:
    cpg_450k_epic = np.array(f.read().splitlines())

class IdatAnnotation():
    acronyms_df = pd.read_csv(ACRONYMS, sep="\t")
    acronym_to_description = dict(
        zip(acronyms_df.Methylation_Class, acronyms_df.Description)
    )
    def __init__(self, filepath):
        self.filepath = Path(filepath).expanduser()
        self.annotation = IdatAnnotation.get_annotation(self.filepath)
    @staticmethod
    def get_annotation(filepath):
        if filepath.suffix == ".xlsx":
            annotation = pd.read_excel(
                filepath,
                header=None,
                engine="openpyxl",
            )
        elif filepath.suffix in [".csv", ".tsv"]:
            sep = "\t" if filepath.suffix == ".tsv" else ","
            annotation = pd.read_csv(filepath, sep=sep)
        else:
            raise ValueError("Annotation files must be as 'xlsx', 'tsv' or 'csv' file")
        if annotation.shape[1] < 2:
            raise ValueError("The anottatino file must contain at least 2 columns")
        def get_description(mc):
            """Returns description of methylation class {mc}."""
            if mc in [np.nan, ""]:
                return ""
            mc = mc.upper()
            if mc in IdatAnnotation.acronym_to_description:
                return IdatAnnotation.acronym_to_description[mc]
            substrings = [
                a for a in IdatAnnotation.acronym_to_description
                if mc.startswith(a + "_")
            ]
            substrings.sort(key=len)
            if substrings:
                return IdatAnnotation.acronym_to_description[substrings[-1]]
            return ""
        if annotation.shape[1] == 2:
            annotation["Description"] = annotation.iloc[:,1].apply(get_description)
        if annotation.shape[1] == 3:
            missing_idx = annotation.iloc[:,2].isin(["", None, np.nan])
            annotation.loc[missing_idx, annotation.columns[2]] = annotation.loc[
                missing_idx, annotation.columns[1]].apply(get_description)
        annotation = annotation.iloc[:,:3]
        annotation.columns = ["Id", "Methylation_Class", "Description"]
        return annotation
    @property
    def id(self):
        return self.annotation.Id.tolist()
    @property
    def methylation_class(self):
        return self.annotation.Methylation_Class.tolist()
    @property
    def description(self):
        return self.annotation.Description.tolist()
    def __str__(self):
        lines = [
            f"IdatAnnotation():",
            f"filepath: '{self.filepath}'",
            f"annotation:\n{self.annotation}",
        ]
        return "\n".join(lines)
    def __repr__(self):
        return str(self)

filepath = "~/Documents/mepylome/analysis/AllIDATv2_20210804.xlsx"
filepath = "~/Documents/mepylome/analysis/annotation.csv"

idat_annotation = IdatAnnotation(filepath)



def extract_beta(data):
    idat_file, cpgs = data
    try:
        methyl = MethylData(file=idat_file)
        betas_450k_df = methyl.converted_beta(
            cpgs=cpgs, fill=0.49
        )
        betas = betas_450k_df.values.ravel()
        return betas
    except ValueError as e:
        return (idat_file, e)

def read_betas(idat_ids, idat_files, cpgs):
    # Load all manifests before parallelization
    Manifest.load(["450k", "epic", "epicv2"])
    with Pool() as pool:
        betas_list = list(
            tqdm(
                pool.imap(
                    extract_beta,
                    zip(idat_files, [cpgs] * len(idat_files)),
                ),
                total=len(idat_files),
                desc="Reading IDAT files",
            )
        )
    valid_ids = [
        i for i, x in enumerate(betas_list) if len(x) == len(cpgs)
    ]
    valid_betas = [betas_list[i] for i in valid_ids]
    methyl_mtx = np.vstack(valid_betas)
    cnv_df = pd.DataFrame(methyl_mtx, index=idat_ids)
    return cnv_df



@lru_cache()
def get_cnv_plot(files_dir, sample_id, genes_sel):
    timer.start()
    cnv_dir = Path(MANIFEST_TMP_DIR, "cnv")
    cnv_filename = sample_id + ZIP_ENDING
    if not Path(cnv_dir, cnv_filename).exists():
        idat_base = Path(files_dir, sample_id)
        sample_methyl = MethylData(file=idat_base)
        print("ID=", sample_id)
        cnv = CNV.set_all(sample_methyl, reference_methyl)
        ensure_directory_exists(cnv_dir)
        cnv.write(cnv_dir, cnv_filename)
    genes_fix = IMPORTANT_GENES
    bins, detail, segments = read_cnv_data_from_disk(sample_id, cnv_dir)
    plot = cnv_plot(
        sample_id,
        bins,
        detail,
        segments,
        genes_fix,
        list(genes_sel),
    )
    plot = plot.update_layout(
        margin=dict(l=0, r=0, t=30, b=0),
    )
    timer.stop("get_cnv_plot")
    return plot


def get_all_genes():
    genomic_annotation = Annotation(array_type=ArrayType.ILLUMINA_450K)
    detail = genomic_annotation.detail.df
    return detail.Name.tolist()


EMPTY_FIGURE = go.Figure()
EMPTY_FIGURE.update_layout(
    xaxis=dict(visible=False),  # Hide the x-axis
    yaxis=dict(visible=False),  # Hide the y-axis
    plot_bgcolor="rgba(0, 0, 0, 0)",  # Make the background transparent
    paper_bgcolor="rgba(0, 0, 0, 0)",  # Transparent plot paper
    showlegend=False,  # No legend
    margin=dict(l=0, r=0, t=0, b=0),  # No margins
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
            ]
        ),
        color="dark",
        dark=True,
    )


def get_side_navivation(sample_ids):
    return dbc.Col(
        [
            dbc.Tabs(
                [
                    dbc.Tab(
                        label="Plot",
                        children=[
                            dcc.Store(id="memory"),
                            dcc.Location(id="url", refresh=False),
                            html.Br(),
                            html.Br(),
                            html.H6("Sample IDs to highlight in UMAP"),
                            dcc.Dropdown(
                                id="ids-to-highlight",
                                options=sample_ids,
                                multi=True,
                                # placeholder="",
                            ),
                            html.Br(),
                            html.H6("Genes to highlight in CNV"),
                            dcc.Dropdown(
                                id="selected-genes",
                                options=get_all_genes(),
                                multi=True,
                                # placeholder=" in CNV plot",
                            ),
                        ],
                    ),
                    dbc.Tab(
                        label="Settings",
                        children=[
                            html.Br(),
                            html.H6(
                                f"Number of CpG sites (max. {len(cpg_450k_epic)})"
                            ),
                            html.Br(),
                            dcc.Input(
                                id="num-cpgs",
                                type="number",
                                min=1,
                                max=len(cpg_450k_epic),
                                step=1,
                                value=DEFAULT_N_CPGS,
                            ),
                            html.Br(),
                            html.Br(),
                            html.H6("Choose analysis directory"),
                            dbc.Input(
                                id="analysis-dir",
                                valid=True,
                                value=str(DEFAULT_ANALYSIS_DIR),
                                type="text",
                            ),
                            html.Div(id="analysis-path-validation"),
                            html.Br(),
                            html.H6("Choose reference directory"),
                            dbc.Input(
                                id="reference-dir",
                                valid=True,
                                value=str(DEFAULT_REFERENCE_DIR),
                                type="text",
                            ),
                            html.Div(id="reference-path-validation"),
                            html.Br(),
                            html.H6("Choose output directory"),
                            dbc.Input(
                                id="output-dir",
                                valid=True,
                                value=str(DEFAULT_OUTPUT_DIR),
                                type="text",
                            ),
                            html.Div(id="output-path-validation"),
                            html.Br(),
                            html.H6("Choose IDAT preprocessing method"),
                            dcc.Dropdown(
                                id="preprocessing-method",
                                options={
                                    "illumina": "Illumina (fast)",
                                    "swan": "SWAN",
                                    "noob": "NOOB",
                                    "raw": "No preprocessing",
                                },
                                value="illumina",
                                multi=False,
                            ),
                            html.Br(),
                            html.H6("Calculate CNV"),
                            dcc.Dropdown(
                                id="precalculate-cnv",
                                options={
                                    "on": (
                                        "Precalculate all (better performance, "
                                        "longer initialization)"
                                    ),
                                    "off": "When clicking on dots",
                                },
                                value="off",
                                multi=False,
                            ),
                            html.Br(),
                            html.Br(),
                            dbc.Col(
                                dbc.Button(
                                    "Start",
                                    id="start-button",
                                    color="primary",
                                ),
                            ),
                            html.Div(id="output-div"),
                        ],
                    ),
                ]
            ),
        ],
        width={"size": 2},
    )

def input_args_hash(*args):
    result = "-".join(
        [
            str(arg.tolist()) if isinstance(arg, np.ndarray)
            else str(arg) for arg in args
        ]
    )
    return hashlib.md5(result.encode()).hexdigest()[:32]


class MethylationAnalysis:
    def __init__(
        self,
        cpgs=cpg_450k_epic,
        n_cpgs=DEFAULT_N_CPGS,
        analysis_dir=DEFAULT_ANALYSIS_DIR,
        annotation=DEFAULT_ANNOTATION_FILE,
        reference_dir=DEFAULT_REFERENCE_DIR,
        output_dir=DEFAULT_OUTPUT_DIR,
        prep="illumina",
        precalculate_cnv=False,
    ):
        self.cpgs = cpgs
        self.n_cpgs = n_cpgs
        self.analysis_dir = analysis_dir
        self.annotation_path = annotation
        self.reference_dir = reference_dir
        self.output_dir = output_dir
        self.prep = prep
        self.precalculate_cnv = precalculate_cnv
        hash_key = input_args_hash(
            cpgs,
            n_cpgs,
            analysis_dir,
            annotation,
            reference_dir,
            prep,
        )
        print(hash_key)
        out_dir_name = f"{n_cpgs}-{prep}-{hash_key}"
        self.out_dir = Path(output_dir, out_dir_name)
        ensure_directory_exists(self.out_dir)
        self.umap_plot = EMPTY_FIGURE
        self.cnv_plot = EMPTY_FIGURE
        self.cnv_id = None
        self.dropdown_id = None
        self.umap_plot_path = Path(self.output_dir, out_dir_name, "umap_plot.csv")
        self.ids = []
        self.read_umap_plot_from_disk()
        # self.umap_2d_reduction()
        self.app = self.get_app()
    def get_app(self):
        app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        app._favicon = "favicon.svg"
        side_navigation = get_side_navivation(self.ids)
        dash_plots = dbc.Col(
            [
                dcc.Graph(
                    id="umap",
                    figure=self.umap_plot,
                    config={
                        "scrollZoom": False,
                        "doubleClick": "reset+autosize",
                        "modeBarButtonsToRemove": ["lasso2d", "select"],
                        "displaylogo": False,
                    },
                    # style={"width": "80%", "height": "60vh"},
                ),
                dcc.Graph(
                    id="cnv",
                    figure=self.cnv_plot,
                    config={
                        "scrollZoom": False,
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
                Output("umap", "figure"),
                Output("cnv", "figure"),
            ],
            [
                Input("umap", "clickData"),
                Input("ids-to-highlight", "value"),
                Input("selected-genes", "value"),
            ],
            State("umap", "figure"),
        )
        def update_plots(click_data, ids_to_highlight, genes_sel, curr_umap_plot):
            genes_sel = () if genes_sel is None else tuple(genes_sel)
            trigger = callback_context.triggered[0]["prop_id"].split(".")[0]
            print("\n\nTRIGGER:", trigger)
            print("ids_to_highlight:", ids_to_highlight)
            print("genes_sel:", genes_sel)
            # print(self.umap_df)
            if trigger == "ids-to-highlight" and ids_to_highlight is not None:
                self.highlight(dropdown_id=ids_to_highlight)
                self.retrieve_zoom(curr_umap_plot)
                return self.umap_plot, no_update
            if trigger == "umap" and isinstance(click_data, dict):
                points = click_data.get("points", [])
                if points and isinstance(points, list):
                    first_point = points[0] if points else {}
                    sample_id = first_point.get("hovertext", None)
                    if sample_id is None:
                        return no_update, no_update
                    print("sample_id:", sample_id)
                    print("---0---")
                    self.highlight(
                        cnv_id=sample_id, dropdown_id=ids_to_highlight
                    )
                    print("---1---")
                    self.retrieve_zoom(curr_umap_plot)
                    print("---2---")
                    self.cnv_plot = get_cnv_plot(self.analysis_dir, sample_id, genes_sel)
                    print("---3---")
                    return self.umap_plot, self.cnv_plot
            if trigger == "selected-genes" and genes_sel is not None:
                self.cnv_plot = get_cnv_plot(self.analysis_dir, self.cnv_id, genes_sel)
                genes_sel = genes_sel if genes_sel else []
                return no_update, self.cnv_plot
            return no_update, no_update
        @app.callback(
            [
                Output("analysis-dir", "valid"),
                Output("analysis-path-validation", "children"),
            ],
            [Input("analysis-dir", "value")],
            prevent_initial_call=True,
        )
        def validate_analysis_path(input_path):
            try:
                path = Path(input_path)
                if path == DEFAULT_ANALYSIS_DIR:
                    return True, f""
                elif path.is_dir() and not os.access(path, os.W_OK):
                    return False, f"Protected directory: {path}"
                elif path.is_dir():
                    return True, f""
                else:
                    return False, f"Not a directory: {path}"
            except Exception:
                return False, "Invalid path format"
        @app.callback(
            [
                Output("reference-dir", "valid"),
                Output("reference-path-validation", "children"),
            ],
            [Input("reference-dir", "value")],
            prevent_initial_call=True,
        )
        def validate_reference_path(input_path):
            try:
                path = Path(input_path)
                if path == DEFAULT_REFERENCE_DIR:
                    return True, f""
                elif path.is_dir() and not os.access(path, os.W_OK):
                    return False, f"Protected directory: {path}"
                elif path.is_dir():
                    return True, f""
                else:
                    return False, f"Not a directory: {path}"
            except Exception:
                return False, "Invalid path format"
        @app.callback(
            [
                Output("output-dir", "valid"),
                Output("output-path-validation", "children"),
            ],
            [Input("output-dir", "value")],
            prevent_initial_call=True,
        )
        def validate_reference_path(input_path):
            try:
                path = Path(input_path)
                if path == DEFAULT_OUTPUT_DIR:
                    return True, f""
                elif path.is_dir() and not os.access(path, os.W_OK):
                    return False, f"Protected directory: {path}"
                elif path.is_dir():
                    return True, f""
                else:
                    return False, f"Not a directory: {path}"
            except Exception:
                return False, "Invalid path format"
        @app.callback(
            Output("output-div", "children"),
            [Input("start-button", "n_clicks")],
            [
                State("analysis-dir", "value"),
                State("reference-dir", "value"),
                State("output-dir", "value"),
                State("preprocessing-method", "value"),
                State("analysis-dir", "valid"),
                State("reference-dir", "valid"),
            ],
            prevent_initial_call=True,
        )
        def on_start_button_click(
            n_clicks,
            analysis_dir,
            reference_dir,
            output_dir,
            prep,
            analysis_dir_valid,
            reference_dir_valid,
        ):
            print("BUTTON")
            if n_clicks:
                # Perform some action with the input values
                print(
                    f"Starting with number: {analysis_dir}, "
                    f" file path: {reference_dir}"
                    f" prep: {prep}"
                    f" analysis_dir_valid: {analysis_dir_valid}"
                    f" reference_dir_valid: {reference_dir_valid}"
                )
                return (
                    f"Starting with number: {analysis_dir}, "
                    f" file path: {reference_dir}"
                    f" prep: {prep}"
                    f" analysis_dir_valid: {analysis_dir_valid}"
                    f" reference_dir_valid: {reference_dir_valid}"
                )
            return "Start button not clicked"
        return app
    def umap_2d_reduction(self):
        self.annotation = IdatAnnotation(self.annotation_path)
        random_idx = sorted(random.sample(range(len(self.cpgs)), self.n_cpgs))
        random_cpg_sample = [self.cpgs[x] for x in random_idx]
        all_idat_files = idat_basepaths(self.analysis_dir)
        name_to_file = {x.name: x for x in all_idat_files}
        idat_files = [name_to_file.get(x, None) for x in self.annotation.id]
        overlap_idx = [i for i, x in enumerate(idat_files) if x is not None]
        idat_overlap = [idat_files[x] for x in overlap_idx]
        ids_overlap = [self.annotation.id[x] for x in overlap_idx]
        classes_overlap = [self.annotation.methylation_class[x] for x in overlap_idx]
        descripton_overlap = [self.annotation.description[x] for x in overlap_idx]
        cnv_df = read_betas(ids_overlap, idat_overlap, random_cpg_sample)
        umap_2d = umap.UMAP(verbose=True).fit_transform(cnv_df)
        id_to_mc = dict(zip(ids_overlap, classes_overlap))
        id_to_desription = dict(zip(ids_overlap, descripton_overlap))
        self.umap_df = pd.DataFrame(
            {
                "methylation_class": [id_to_mc[x] for x in ids_overlap],
                "description": [id_to_desription[x] for x in ids_overlap],
                "id": cnv_df.index,
                "x": umap_2d[:, 0],
                "y": umap_2d[:, 1],
            }
        )
        self.umap_df.to_csv(self.umap_plot_path, sep="\t", index=False)
        self.set_umap_plot()
    def read_umap_plot_from_disk(self):
        if self.umap_plot_path.exists():
            self.umap_df = pd.read_csv(self.umap_plot_path, sep="\t")
            self.set_umap_plot()
    def set_umap_plot(self):
        self.ids = self.umap_df.id
        self.umap_plot = umap_plot_from_data(self.umap_df)
        self.umap_plot = self.umap_plot.update_layout(
            margin=dict(l=0, r=0, t=30, b=0),
        )
        self.raw_plot = self.umap_plot
        self.cnv_id = None
        self.dropdown_id = None
    def get_coordinates(self, sample_id):
        return self.umap_df[self.umap_df.id == sample_id].iloc[0][["x", "y"]]
    def highlight(self, dropdown_id=None, cnv_id=None):
        if cnv_id is not None:
            self.cnv_id = cnv_id
        self.dropdown_id = [] if dropdown_id is None else dropdown_id
        self.umap_plot = go.Figure(self.raw_plot)
        if self.dropdown_id is not None:
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
    def run_app(self):
        self.app.run(debug=True, host=HOST, use_reloader=False)
    def __repr__(self):
        title = f"UMAPPlot():"
        lines = [
            title + "\n" + "*" * len(title),
            f"cnv_id:\n{self.cnv_id}",
            f"dropdown_id:\n{self.dropdown_id}",
            f"umap_df:\n{self.umap_df}",
        ]
        return "\n\n".join(lines)



self = MethylationAnalysis(analysis_dir=IDAT_DIR)
# self.umap_2d_reduction()
self.run_app()

# home
id_ = "6042324058_R04C01"

id_ = "7970376005_R03C01"


n_cpgs = DEFAULT_N_CPGS
analysis_dir = DEFAULT_ANALYSIS_DIR
reference_dir = DEFAULT_REFERENCE_DIR
prep = "illumina"
precalculate_cnv = False



