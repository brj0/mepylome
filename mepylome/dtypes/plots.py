from dash import Dash, html, dcc, callback, Output, Input, no_update
from dash.exceptions import PreventUpdate
from mepylome.dtypes import MANIFEST_TMP_DIR, IMPORTANT_GENES, CHROMOSOME_DATA

from plotly.io import write_json, from_json

import webbrowser
import threading
import bisect
import threading
import io
import json
import numpy as np
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time
import zipfile
from pathlib import Path


PORT = 8050
HOST = "localhost"
PLOTLY_RENDER_MODE = "webgl"
ZIP_ENDING = "_cnv.zip"
THRESHOLD_BALANCED = 0.1

CNV_GRID = Path(MANIFEST_TMP_DIR, "grid.json")


class Genome:
    """Data container for reference genome data."""

    def __init__(self):
        self.chrom = pd.DataFrame(CHROMOSOME_DATA)
        self.chrom["offset"] = [0] + np.cumsum(self.chrom["len"]).tolist()[:-1]
        self.chrom["center"] = self.chrom["offset"] + self.chrom["len"] // 2
        self.chrom["centromere_offset"] = (
            self.chrom["offset"]
            + (self.chrom["centromere_start"] + self.chrom["centromere_end"])
            // 2
        )
        self.length = (
            self.chrom["offset"].iloc[-1] + self.chrom["len"].iloc[-1]
        )

    def __iter__(self):
        """Enables looping over chromosomes."""
        return self.chrom.itertuples()

    def __len__(self):
        return self.length

    def __str__(self):
        """Prints overview of object for debugging purposes."""
        lines = [
            "Genome object:",
            f"length: {self.length}",
            f"chrom:\n{self.chrom}",
        ]
        return "\n".join(lines)

    def __repr__(self):
        return str(self)


genome = Genome()

offset = {i.name: i.offset for i in genome}


def get_df_from_zip(zip_file_path, extract=["bins", "detail", "segments"]):
    """Reads the CNV data from a zip file without extraxtion on disk."""
    zip_file = os.path.basename(zip_file_path)
    csv_file_templ = zip_file.replace(ZIP_ENDING, "_cnv_%s.csv")
    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        extracted_dfs = []
        for filename in extract:
            with zip_ref.open(csv_file_templ % filename) as file_:
                extracted_dfs.append(pd.read_csv(io.TextIOWrapper(file_)))
    extracted_dfs = rename_cols(extracted_dfs)
    if len(extract) == 1:
        return extracted_dfs[0]
    return extracted_dfs


# TODO del
def rename_cols(df_list):
    translate = {
        "chrom": "Chromosome",
        "end": "End",
        "len": "Len",
        "mean": "Mean",
        "median": "Median",
        "name": "Name",
        "nprobes": "N_probes",
        "start": "Start",
        "value": "Median",
        "x_mid": "X_mid",
    }
    return [df.rename(columns=translate) for df in df_list]


def add_offset(df, chrom_nm, col):
    return df[col] + df[chrom_nm].map(offset)


def get_x_mid(df):
    return df["Chromosome"].map(offset) + (df["Start"] + df["End"]) // 2


def cnv_grid(genome):
    """Returns chromosome grid layout for CNV Plot as plotly object and
    saves it on disk. If available grid is directly read from disk.
    """
    # Check if grid exists and return if available.
    if os.path.exists(CNV_GRID):
        with open(CNV_GRID, "r") as f:
            grid = from_json(f.read())
        return grid

    grid = go.Figure()
    grid.update_layout(
        coloraxis_showscale=False,
        xaxis=dict(
            linecolor="black",
            linewidth=1,
            mirror=True,
            range=[0, len(genome)],
            showgrid=False,
            ticklen=10,
            tickmode="array",
            ticks="outside",
            tickson="boundaries",
            ticktext=genome.chrom.name,
            tickvals=genome.chrom.center,
            zeroline=False,
        ),
        yaxis=dict(
            linecolor="black",
            linewidth=1,
            mirror=True,
            showline=True,
        ),
        template="simple_white",
    )
    # Vertical line: centromere.
    for i in genome.chrom.centromere_offset:
        grid.add_vline(x=i, line_color="black", line_dash="dot", line_width=1)
    # Vertical line: chromosomes.
    for i in genome.chrom.offset.tolist() + [len(genome)]:
        grid.add_vline(x=i, line_color="black", line_width=1)
    # Save to disk
    grid.write_json(CNV_GRID)
    return grid


def cnv_bins_plot(data_frame, title, labels, genome):
    """Create CNV plot from CNV data.
    Args:
        data_frame: DataFrame with columns 'x', 'y', and 'hover_data'
        title: Title of plot
        labels: Touple of labels for x-axis and y-axis.
        genome: Reference Genome.
    """
    t0 = time.time()
    grid = cnv_grid(genome)
    # Draw horizontal line.
    grid.add_hline(y=0, line_color="black", line_width=1)
    plot = px.scatter(
        data_frame=data_frame,
        x="x",
        y="y",
        labels={
            "x": labels[0],
            "y": labels[1],
        },
        title=title,
        color="y",
        range_color=[-0.4, 0.4],
        color_continuous_scale="RdBu",
        render_mode=PLOTLY_RENDER_MODE,
        hover_data=["hover_data"],
    )
    plot.update_traces(
        hovertemplate="<b>Value:</b> %{y} <br><b>Genes:</b> %{customdata[0]}",
    )
    plot.update_layout(grid.layout, yaxis_range=[-2, 2])
    return plot


def add_segments(plot, seg_df):
    seg_lines = [
        go.Scattergl(
            x=[seg["X_start"], seg["X_end"]],
            y=[seg["Mean"], seg["Mean"]],
            mode="lines",
            line=dict(color="darkslategrey"),
            showlegend=False,
            name="",
        )
        for _, seg in seg_df.iterrows()
    ]
    plot.add_traces(seg_lines)
    return plot


def add_genes(plot, genes):
    scatter_genes = go.Scattergl(
        customdata=genes[
            [
                "Name",  # 0
                "Chromosome",  # 1
                "Start",  # 2
                "End",  # 3
                "Len",  # 4
                "N_probes",  # 5
            ]
        ],
        hovertemplate=(
            "<b> %{customdata[0]} </b> "
            "%{customdata[1]}:%{customdata[2]:,}-%{customdata[3]:,} "
            "(hg19) <br>"
            "Value = %{y}<br>"
            "Gene length: %{customdata[4]:,} base pairs <br>"
            "Number or probes: %{customdata[5]} <br>"
        ),
        name="",
        marker_color="rgba(0,0,0,1)",
        mode="markers+text",
        # marker_symbol="diamond",
        marker_symbol="cross",
        textfont_color="rgba(0,0,0,1)",
        showlegend=False,
        text=genes.Name,
        textposition="top center",
        x=genes.X_mid,
        y=genes.Median,
        error_x=dict(
            type="data",
            symmetric=False,
            array=genes.Len / 2,
            arrayminus=genes.Len / 2,
        ),
    )
    plot.add_trace(scatter_genes)
    return plot


def add_highlited_bins(plot, highlighted_bins):
    highlighted_bins_scatter = go.Scattergl(
        x=highlighted_bins.X_mid,
        y=highlighted_bins.Median,
        marker_color="magenta",
        mode="markers+text",
        showlegend=False,
    )
    plot.add_trace(highlighted_bins_scatter)
    return plot


def find_genes_within_bins(bins, detail):
    bins["X_start"] = add_offset(bins, "Chromosome", "Start")
    bins["X_end"] = add_offset(bins, "Chromosome", "End")
    bins = bins.sort_values("X_start")  # TODO del
    detail["X_start"] = add_offset(detail, "Chromosome", "Start")
    detail["X_end"] = add_offset(detail, "Chromosome", "End")
    detail["Start_bin"] = np.digitize(detail.X_start, bins.X_start) - 1
    detail["Start_bin_"] = np.digitize(detail.X_start, bins.X_end, right=True)
    detail["End_bin"] = np.digitize(detail.X_end, bins.X_start) - 1
    # Identify genes that start within gaps between bins
    detail.loc[(detail["Start_bin"] < detail["Start_bin_"]), "Start_bin"] += 1
    detail["Range"] = [
        np.arange(start, end)
        for start, end in zip(detail["Start_bin"], detail["End_bin"] + 1)
    ]
    df_long_format = detail[["Name", "Range"]].explode("Range")
    bins["Genes"] = (
        df_long_format.dropna()
        # By sorting first, resultig string will be sorted too
        .sort_values(by="Name")
        .groupby("Range")["Name"]
        .agg(lambda x: ", ".join(x))
        .reindex(range(len(bins)), fill_value="")
        .reset_index(drop=True)
    )
    detail = detail.drop(
        columns=["Start_bin", "Start_bin_", "End_bin"],
    )
    return bins, detail


def _find_genes_within_bins(bins, detail):
    """Used for debugging: Does (almost) the same as find_genes_within_bins with
    much easier code but 20x slower.
    """
    bins["X_start"] = add_offset(bins, "Chromosome", "Start")
    bins["X_end"] = add_offset(bins, "Chromosome", "End")
    detail["X_start"] = add_offset(detail, "Chromosome", "Start")
    detail["X_end"] = add_offset(detail, "Chromosome", "End")
    bins["Genes"] = None
    bins["Genes"] = bins.apply(
        lambda bin_row: ", ".join(
            detail.loc[
                # The the intervals have a nonempty intersection
                ~(
                    ((bin_row["X_end"] < detail["X_start"]))
                    | ((detail["X_end"] < bin_row["X_start"]))
                ),
                "Name",
            ].sort_values()
        ),
        axis=1,
    )


def read_cnv_data_from_disk(sample_id, cnv_dir):
    sample_zip = sample_id + ZIP_ENDING
    bins, detail, segments = get_df_from_zip(
        os.path.join(cnv_dir, sample_zip),
        extract=["bins", "detail", "segments"],
    )
    # TODO del
    # bins = probes
    # Calculate some x-values
    bins["X_mid"] = get_x_mid(bins)
    detail["X_mid"] = get_x_mid(detail)
    segments["X_start"] = add_offset(segments, "Chromosome", "Start")
    segments["X_end"] = add_offset(segments, "Chromosome", "End")
    detail["Len"] = detail["End"] - detail["Start"]
    return bins, detail, segments


def cnv_plot(sample_id, bins, detail, segments, genes_fix, genes_sel):
    # Base scatterplot
    bins, detail = find_genes_within_bins(bins, detail)
    scatter_df = bins[["X_mid", "Median", "Genes"]]
    scatter_df.columns = ["x", "y", "hover_data"]
    plot = cnv_bins_plot(
        data_frame=scatter_df,
        title=f"Sample ID: {sample_id}",
        labels=("", ""),
        genome=genome,
    )
    # Highlight bins adjacent to the added genes
    genes_sel = genes_sel if genes_sel else []
    genes_x_range = (
        detail[detail["Name"].isin(genes_sel)]["Range"].explode().tolist()
    )
    highlighted_bins = bins.loc[genes_x_range, ["X_mid", "Median"]]
    plot = add_highlited_bins(plot, highlighted_bins)
    # Draw the segments
    plot = add_segments(plot, segments)
    # Add all added and important genes
    genes_to_plot = genes_fix + genes_sel
    gene_detail = detail[detail["Name"].isin(genes_to_plot)].copy()
    plot = add_genes(plot, gene_detail)
    # plot = plot.update_layout(
        # margin=dict(l=0, r=0, t=30, b=0),
    # )
    return plot


class CNVPlot:
    def __init__(self, cnv_dir, cnv_file, genes=IMPORTANT_GENES):
        self.cnv_dir = cnv_dir
        self.genes_fix = genes
        # TODO
        init_sample_id = cnv_file.replace(ZIP_ENDING, "")
        fig = go.Figure(layout=go.Layout(yaxis=dict(range=[-2, 2])))
        app = Dash(__name__)
        app.layout = html.Div(
            [
                dcc.Store(id="memory"),
                dcc.Location(id="url", refresh=False),
                html.P(id="err", style={"color": "red"}),
                html.Div(id="page"),
                dcc.Dropdown(
                    options=[],
                    value=None,
                    placeholder="Select genes to highlight...",
                    id="dropdown",
                    multi=True,
                ),
                dcc.Graph(
                    id="graph",
                    figure=fig,
                    config=dict(
                        {
                            "scrollZoom": False,
                            "doubleClick": "reset",
                            "modeBarButtonsToRemove": ["lasso2d", "select"],
                        },
                        displaylogo=False,
                    ),
                    style={"width": "80hh", "height": "90vh"},
                ),
            ]
        )

        @callback(
            [
                Output("graph", "figure"),
                Output("dropdown", "options"),
                Output("err", "children"),
            ],
            [
                Input("dropdown", "value"),
                Input("url", "pathname"),
            ],
        )
        def update_graph(genes_sel, url_path):
            sample_id = url_path[1:]
            try:
                bins, detail, segments = read_cnv_data_from_disk(
                    sample_id, self.cnv_dir
                )
            except FileNotFoundError:
                return no_update, no_update, f"Sample ID {sample_id} not found"
            plot = cnv_plot(
                sample_id,
                bins,
                detail,
                segments,
                self.genes_fix,
                genes_sel,
            )
            return plot, detail.sort_values(by="Name").Name, ""

        def open_browser_tab():
            webbrowser.open_new_tab(f"http://{HOST}:{PORT}/{init_sample_id}")

        threading.Timer(1, open_browser_tab).start()
        app.run(debug=True, host=HOST, use_reloader=False)
