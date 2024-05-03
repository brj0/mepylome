import colorsys
import hashlib
import json
import random
import importlib
from functools import lru_cache, wraps
from multiprocessing import Pool
from pathlib import Path

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import umap
from dash import Dash, Input, Output, callback, dcc, html, no_update
from dash.exceptions import PreventUpdate
from nanodip import Reference
from tqdm import tqdm

from mepylome import CNV, Manifest, MethylData, RawData, idat_basepaths
from mepylome.dtypes import (
    CHROMOSOME_DATA,
    ReferenceMethylData,
    COLOR_MAP,
    IMPORTANT_GENES,
    MANIFEST_TMP_DIR,
    ZIP_ENDING,
    cnv_plot,
    read_cnv_data_from_disk,
)

# IDAT_DIR = "/data/epidip_IDAT"
IDAT_DIR = "/mnt/ws528695/data/epidip_IDAT"
# IDAT_DIR = Path("~/MEGA/work/programming/data/epidip_IDAT").expanduser()
PLOTLY_RENDER_MODE = "webgl"
HOST = "localhost"

# Contains CpGs common to both 450k and EPIC arrays, excluding those on sex
# chromosomes and cross-reactive probes (as identified in Chen et al., 2013).
CPG_450K_EPIC_OVERLAP = (
    "/applications/reference_data/betaEPIC450Kmix_bin/index.csv"
)
CNV_LINK = (
    "http://s1665.rootserver.io/umapplot01/%s_CNV_IFPBasel_annotations.pdf"
)
CNV_LINK = "http://localhost:8050/%s"


from mepylome import idat_basepaths

ref_dir = "/data/ref_IDAT"
ref_idat_basepaths = idat_basepaths(ref_dir)


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
            "x": "",
            # "x": "UMAP 0",
            "y": "",
            # "y": "UMAP 1",
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
            arrowhead=1,
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

# reference = Reference("AllIDATv2_20210804_HPAP_Sarc")
reference = Reference("AllIDATv2_20210804")
NR_CPGS = 9000
random_cpg_sample = random.sample(reference.cpg_sites, NR_CPGS)


files = IDAT_DIR

cpgs = cpg_450k_epic
cpgs = random_cpg_sample

ids = reference.specimens
classes = reference.methylation_class
description = reference.description


def extract_beta(data):
    idat_file, cpg_mask = data
    try:
        methyl = MethylData(file=idat_file)
        betas_450k_df = methyl.converted_beta(cpgs=cpg_450k_epic, fill=0.49)
        betas = betas_450k_df.values.ravel()
        return betas[cpg_mask]
    except ValueError as e:
        return (idat_file, e)


class UMAPPlot:
    def __init__(self, cpgs, files, ids, classes, description=None):
        all_idat_files = idat_basepaths(files)
        name_to_file = {x.name: x for x in all_idat_files}
        idat_files = [name_to_file.get(x, None) for x in ids]
        overlap_idx = [i for i, x in enumerate(idat_files) if x is not None]
        idat_overlap = [idat_files[x] for x in overlap_idx]
        ids_overlap = [ids[x] for x in overlap_idx]
        classes_overlap = [classes[x] for x in overlap_idx]
        descripton_overlap = [description[x] for x in overlap_idx]
        # Load all manifests
        Manifest.load(["450k", "epic", "epicv2"])
        cpg_mask = np.isin(cpg_450k_epic, cpgs)
        with Pool() as pool:
            betas_450k_results = list(
                tqdm(
                    pool.imap(
                        extract_beta,
                        zip(idat_overlap, [cpg_mask] * len(idat_overlap)),
                    ),
                    total=len(idat_overlap),
                    desc="Reading IDAT files",
                )
            )
        valid_betas = [x for x in betas_450k_results if len(x) == len(cpgs)]
        valid_ids = [
            i for i, x in enumerate(betas_450k_results) if len(x) == NR_CPGS
        ]
        methyl_mtx = np.vstack(valid_betas)
        reference_ids = [Path(idat_overlap[x]).name for x in valid_ids]
        cnv_df = pd.DataFrame(methyl_mtx, index=ids_overlap)
        umap_2d = umap.UMAP(verbose=True).fit_transform(methyl_mtx)
        # NR_CPGS = len(cpg_450k_epic)
        # cpg_mask = np.ones(len(cpg_450k_epic), dtype=bool)
        # idat_files = idat_files[:200]
        id_to_mc = dict(zip(ids_overlap, classes_overlap))
        id_to_desription = dict(zip(ids_overlap, descripton_overlap))
        umap_df = pd.DataFrame(
            {
                "methylation_class": [id_to_mc[x] for x in ids_overlap],
                "description": [id_to_desription[x] for x in ids_overlap],
                "id": cnv_df.index,
                "x": umap_2d[:, 0],
                "y": umap_2d[:, 1],
            }
        )
        self.umap_plt = umap_plot_from_data(umap_df)
        self.umap_plt = self.umap_plt.update_layout(
            margin=dict(l=0, r=0, t=30, b=0),
        )
        # self.umap_plt.show(config=dict({"scrollZoom": True}))


styles = {"pre": {"border": "thin lightgrey solid", "overflowX": "scroll"}}


umap = UMAPPlot(cpgs, files, ids, classes, description)

# umap.umap_plt.show(config=dict({"scrollZoom": True}))


@lru_cache()
def get_cnv_plot(sample_id):
    cnv_dir = Path(MANIFEST_TMP_DIR, "cnv")
    cnv_filename = sample_id + ZIP_ENDING
    if not Path(cnv_dir, cnv_filename).exists():
        idat_base = Path(files, sample_id)
        sample_methyl = MethylData(file=idat_base)
        print("ID=", sample_id)
        cnv = CNV.set_all(sample_methyl, ref_methyl, annotation)
        ensure_directory_exists(cnv_dir)
        cnv.write(cnv_dir, cnv_filename)
    genes_fix = IMPORTANT_GENES
    genes_sel = []
    bins, detail, segments = read_cnv_data_from_disk(sample_id, cnv_dir)
    plot = cnv_plot(
        sample_id,
        bins,
        detail,
        segments,
        genes_fix,
        genes_sel,
    )
    plot = plot.update_layout(
        margin=dict(l=0, r=0, t=30, b=0),
    )
    return plot


umap_fig = umap.umap_plt
cnv_fig = go.Figure(layout=go.Layout(yaxis=dict(range=[-2, 2])))
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
LOGO = app.get_asset_url("mepylome.svg")
navbar = dbc.Navbar(
    dbc.Container(
        [
            html.A(
                dbc.Row(
                    [
                        dbc.Col(html.Img(src=LOGO, height="30px")),
                        dbc.Col(dbc.NavbarBrand("Navbar", className="ms-2")),
                    ],
                    align="center",
                    className="g-0",
                ),
                style={"textDecoration": "none"},
            ),
            dbc.NavbarToggler(id="navbar-toggler", n_clicks=0),
            # dbc.Collapse(
            # search_bar,
            # id="navbar-collapse",
            # is_open=False,
            # navbar=True,
            # ),
        ]
    ),
    color="dark",
    dark=True,
)
side_navigation = dbc.Col(
    [
        dcc.Store(id="memory"),
        dcc.Location(id="url", refresh=False),
        html.Div(
            [
                dcc.Markdown(
                    """
                    **Click Data**
                    Click on points in the graph.
                """
                ),
                html.Pre(id="click-data", style=styles["pre"]),
            ],
            # className="six columns",
        ),
    ],
    width={"size": 2},
)
plots = dbc.Col(
    [
        dcc.Graph(
            id="umap",
            figure=umap_fig,
            config={
                "scrollZoom": False,
                "doubleClick": "reset",
                "modeBarButtonsToRemove": ["lasso2d", "select"],
                "displaylogo": False,
            },
            # style={"height": "75vh"},
            # style={"width": "70vw", "height": "90vh"},
        ),
        dcc.Graph(
            id="cnv",
            figure={},
            config={
                "scrollZoom": False,
                "doubleClick": "reset",
                "modeBarButtonsToRemove": ["select"],
                # "modeBarButtonsToRemove": ["lasso2d", "select"],
                "displaylogo": False,
            },
            # style={"width": "30vw", "height": "45vh"},
            # style={"width": "90%", "height": "65vh"},
        ),
    ],
    width={"size": 10},
)
# app.layout = dbc.Container(
# dbc.Row([side_navigation, plots]),
# fluid=True,
# )
app.layout = html.Div(
    [
        navbar,
        dbc.Container(
            [
                dbc.Row(
                    [side_navigation, plots],
                    style={"margin-top": "20px"},
                ),
            ],
            fluid=True,
        ),
    ],
)


# app.layout = html.Div(
# style={"display": "flex"},  # Use Flexbox for horizontal layout
# children=[
# # Left column
# html.Div(
# children=[
# dcc.Store(id="memory"),
# dcc.Location(id="url", refresh=False),
# html.P(id="err", style={"color": "red"}),
# html.Div(id="page"),
# # dcc.Dropdown(
# # options=[],
# # value=None,
# # placeholder="Select genes to highlight...",
# # id="dropdown",
# # multi=True,
# # style={"width": "30vw"},
# # ),
# html.Div(
# [
# dcc.Markdown(
# """
# **Click Data**
# Click on points in the graph.
# """
# ),
# html.Pre(id="click-data", style=styles["pre"]),
# ],
# # className="six columns",
# ),
# ],
# style={"flex": 2},  # Flex value determines the relative width
# ),
# # Right column
# html.Div(
# children=[
# dcc.Graph(
# id="umap",
# figure=umap_fig,
# config={
# "scrollZoom": False,
# "doubleClick": "reset",
# "modeBarButtonsToRemove": ["lasso2d", "select"],
# "displaylogo": False,
# },
# style={"height": "75vh"},
# # style={"width": "70vw", "height": "90vh"},
# ),
# dcc.Graph(
# id="cnv",
# figure=cnv_fig,
# config={
# "scrollZoom": False,
# "doubleClick": "reset",
# "modeBarButtonsToRemove": ["select"],
# # "modeBarButtonsToRemove": ["lasso2d", "select"],
# "displaylogo": False,
# },
# # style={"width": "30vw", "height": "45vh"},
# style={"width": "90%", "height": "65vh"},
# ),
# ],
# style={"flex": 8},  # Flex value for the right column
# ),
# # className="six columns",
# ],
# )
@callback(
    [
        Output("click-data", "children"),
        Output("cnv", "figure"),
    ],
    Input("umap", "clickData"),
)
def display_click_data(click_data):
    # print("click: ---0---")
    if click_data is None:
        return no_update, no_update
    sample_id = None
    if isinstance(click_data, dict):
        # print("click: ---1---")
        points = click_data.get("points", [])
        if points and isinstance(points, list):
            # print("click: ---2---")
            first_point = points[0] if points else {}
            sample_id = first_point.get("hovertext", None)
            # try:
            plot = get_cnv_plot(sample_id)
            # print("click: ---3---")
            return json.dumps(click_data, indent=2), plot
            # except:
            # pass
    # print("click: ---4---")
    # print()
    return no_update, no_update


app.run(debug=True, host=HOST, use_reloader=False)

# app.css.append_css({
# "external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"
# })


def open_browser_tab():
    webbrowser.open_new_tab(f"http://{HOST}:{PORT}/{init_sample_id}")


# home
id_ = "6042324058_R04C01"

id_ = "7970376005_R03C01"


ref0 = "/data/ref_IDAT/cnvrefidat_450k/3999997083_R02C02_Grn.idat"
ref1 = "/data/ref_IDAT/cnvrefidat_450k/5775446049_R06C01_Red.idat"
GENES = pkg_resources.resource_filename("mepylome", "data/hg19_genes.tsv.gz")
GAPS = pkg_resources.resource_filename("mepylome", "data/gaps.csv.gz")
refs_raw = RawData([ref0, ref1])
ref_methyl = MethylData(refs_raw)
sample_raw = RawData(smp0)
gap = pr.PyRanges(pd.read_csv(GAPS))
gap.Start -= 1
genes_df = pd.read_csv(GENES, sep="\t")
genes_df.Start -= 1
genes = pr.PyRanges(genes_df)
genes = genes[["Name"]]
annotation = Annotation(manifest, gap=gap, detail=genes)


df = pd.DataFrame(
    {
        "UMAP_X": np.random.rand(100),
        "UMAP_Y": np.random.rand(100),
        "Sample ID": ["Sample" + str(i) for i in range(100)],
    }
)

external_stylesheets = [dbc.themes.BOOTSTRAP]
app = Dash(__name__, external_stylesheets=external_stylesheets)


left_column = dbc.Col(
    [
        html.H4("Options"),
        dcc.Dropdown(
            id="sample-dropdown",
            options=[{"label": f"Sample {i}", "value": i} for i in range(10)],
            multi=True,
            placeholder="Select Sample IDs",
        ),
        html.Br(),
        dbc.ButtonGroup(
            [
                dbc.Button("25k CpGs", color="primary"),
                dbc.Button("50k CpGs", color="secondary"),
                dbc.Button("75k CpGs", color="success"),
            ],
            size="lg",
        ),
        html.Br(),
        html.Br(),
        dcc.Dropdown(
            id="reference-dropdown",
            options=[
                {"label": f"Reference {i}", "value": i} for i in range(5)
            ],
            placeholder="Select Reference Set",
        ),
        html.Br(),
        dcc.Tabs(
            [
                dcc.Tab(
                    label="Settings",
                    children=[
                        dcc.Input(
                            type="text",
                            placeholder="Enter IDAT directory",
                            id="idat-dir",
                        ),
                        dcc.Input(
                            type="text",
                            placeholder="Enter save directory",
                            id="save-dir",
                        ),
                        dbc.Button("Start", id="start-btn", color="primary"),
                    ],
                )
            ],
        ),
    ],
    width=3,
)

# Right column with plots
right_column = dbc.Col(
    [
        html.H4("Plots"),
        dcc.Graph(
            id="umap-plot",
            figure=px.scatter(
                df,
                x="UMAP_X",
                y="UMAP_Y",
                color="Sample ID",
                title="UMAP Plot",
            ),
        ),
        html.Div(id="cnv-plot", children=[]),  # Placeholder for CNV plot
    ],
    width=9,
)

# Main layout with the navbar and columns
app.layout = html.Div(
    [
        navbar,
        dbc.Container(
            [
                dbc.Row(
                    [left_column, right_column],
                    style={"margin-top": "20px"},
                ),
            ]
        ),
    ],
)

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True, use_reloader=False)


ref_dir = "/data/ref_IDAT"
reference_methyl = ReferenceMethylData(ref_dir)
reference_methyl[ArrayType.ILLUMINA_450K]


cnv = CNV.set_all(sample_methyl, reference_methyl)



