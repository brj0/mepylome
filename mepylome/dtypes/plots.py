"""Contains utilities to generate CNV plots."""

import io
import threading
import webbrowser
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from mepylome.dtypes.genetic_data import CHROMOSOME_DATA, IMPORTANT_GENES
from mepylome.dtypes.manifests import MEPYLOME_TMP_DIR
from mepylome.utils.files import ensure_directory_exists

PLOTLY_RENDER_MODE = "webgl"
ZIP_ENDING = "_cnv.zip"
THRESHOLD_BALANCED = 0.1

ensure_directory_exists(MEPYLOME_TMP_DIR)
CNV_GRID = Path(MEPYLOME_TMP_DIR, "cnv_grid.json")


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


reference_genome = Genome()

offset = {i.name: i.offset for i in reference_genome}


def get_df_from_zip(zip_file_path, extract=None):
    """Reads the CNV data from a zip file without extraxtion on disk."""
    if extract is None:
        extract = ["bins", "detail", "segments"]
    zip_file_path = Path(zip_file_path).expanduser()
    zip_file = zip_file_path.name
    csv_file_templ = zip_file.replace(ZIP_ENDING, "_cnv_%s.csv")
    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        extracted_dfs = []
        for filename in extract:
            try:
                with zip_ref.open(csv_file_templ % filename) as file_:
                    extracted_dfs.append(pd.read_csv(io.TextIOWrapper(file_)))
            except KeyError:
                extracted_dfs.append(None)
    if len(extract) == 1:
        return extracted_dfs[0]
    return extracted_dfs


def add_offset(df, chrom_nm, col):
    """Calculates x-values of the CNV-plot."""
    return df[col] + df[chrom_nm].map(offset)


def get_x_mid(df):
    """Calculates the midpoint x-values of all chromosomes."""
    return df["Chromosome"].map(offset) + (df["Start"] + df["End"]) // 2


def cnv_grid(genome):
    """Returns chromosome grid for CNV Plot as a cached plotly object."""
    from plotly.io import from_json

    # Check if grid exists and return if available.
    if CNV_GRID.exists():
        with CNV_GRID.open() as f:
            return from_json(f.read())

    grid = go.Figure()
    grid.update_layout(
        coloraxis_showscale=False,
        xaxis={
            "linecolor": "black",
            "linewidth": 1,
            "mirror": True,
            "range": [0, len(genome)],
            "showgrid": False,
            "ticklen": 10,
            "tickmode": "array",
            "ticks": "outside",
            "tickson": "boundaries",
            "ticktext": genome.chrom.name,
            "tickvals": genome.chrom.center,
            "zeroline": False,
        },
        yaxis={
            "linecolor": "black",
            "linewidth": 1,
            "mirror": True,
            "showline": True,
        },
        template="simple_white",
    )
    # Vertical line: centromere.
    for i in genome.chrom.centromere_offset:
        grid.add_vline(x=i, line_color="black", line_dash="dot", line_width=1)
    # Vertical line: chromosomes.
    for i in [*genome.chrom.offset.tolist(), len(genome)]:
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
    """Adds segments calculated by Circular Binary Segmentation (CBG)."""
    if seg_df is None:
        return plot
    seg_lines = [
        go.Scattergl(
            x=[seg["X_start"], seg["X_end"]],
            y=[seg["Mean"], seg["Mean"]],
            mode="lines",
            line={"color": "darkslategrey"},
            showlegend=False,
            name="",
        )
        for _, seg in seg_df.iterrows()
    ]
    plot.add_traces(seg_lines)
    return plot


def add_genes(plot, genes):
    """Add genes to the plot as bars with central crosses.

    Args:
        plot: The plot to which genes will be added.
        genes (list): A list of genes to be added to the plot.
    """
    # Draw NaN's with value 0
    genes["Median"] = genes["Median"].fillna(0)
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
        marker_symbol="cross",
        textfont_color="rgba(0,0,0,1)",
        showlegend=False,
        text=genes.Name,
        textposition="top center",
        x=genes.X_mid,
        y=genes.Median,
        error_x={
            "type": "data",
            "symmetric": False,
            "array": genes.Len / 2,
            "arrayminus": genes.Len / 2,
        },
    )
    plot.add_trace(scatter_genes)
    return plot


def add_highlited_bins(plot, highlighted_bins):
    """Changes the color of the specified bins."""
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
    """Determine genes overlapping with each bin and add them as a column.

    Args:
        bins (DataFrame): DataFrame containing bin information.
        detail (DataFrame): DataFrame containing gene information.

    Returns:
        Tuple(DataFrame, DataFrame): A tuple containing the bins DataFrame with
            assigned genes and the modified detail DataFrame.

    Raises:
        ValueError: If the 'bins' DataFrame is not sorted.
    """
    bins["X_start"] = add_offset(bins, "Chromosome", "Start")
    bins["X_end"] = add_offset(bins, "Chromosome", "End")
    if not bins.X_start.is_monotonic_increasing:
        msg = "'bins' is not sorted"
        raise ValueError(msg)
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
    """Does (almost) the same as find_genes_within_bins (but 20x slower).

    Function is used for debugging as it is much easier to read.
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
                    (bin_row["X_end"] < detail["X_start"])
                    | (detail["X_end"] < bin_row["X_start"])
                ),
                "Name",
            ].sort_values()
        ),
        axis=1,
    )


def read_cnv_data_from_disk(cnv_dir, sample_id):
    """Reads the components of a zip-file generated by CNV.write."""
    sample_zip = sample_id + ZIP_ENDING
    bins, detail, segments = get_df_from_zip(
        cnv_dir / sample_zip,
        extract=["bins", "detail", "segments"],
    )
    # Calculate some plot x-values
    if bins is not None:
        bins["X_mid"] = get_x_mid(bins)
    if detail is not None:
        detail["X_mid"] = get_x_mid(detail)
        detail["Len"] = detail["End"] - detail["Start"]
    if segments is not None:
        segments["X_start"] = add_offset(segments, "Chromosome", "Start")
        segments["X_end"] = add_offset(segments, "Chromosome", "End")
    return bins, detail, segments


def cnv_plot_from_data(
    sample_id, bins, detail, segments, genes_fix, genes_sel
):
    """Generate a CNV plot from data calculated by the class CNV.

    Args:
        sample_id (str): The sample ID for the plot.
        bins (DataFrame): DataFrame containing bin information.
        detail (DataFrame): DataFrame containing gene information.
        segments (DataFrame): DataFrame containing segment information.
        genes_fix (list): List of genes to include in the plot.
        genes_sel (list): List of genes to include in the plot and highlight
            all associated bins.

    Returns:
        Plotly Figure: A Plotly figure representing the CNV plot.
    """
    # Base scatterplot
    bins, detail = find_genes_within_bins(bins, detail)
    scatter_df = bins[["X_mid", "Median", "Genes"]]
    scatter_df.columns = ["x", "y", "hover_data"]
    plot = cnv_bins_plot(
        data_frame=scatter_df,
        title=f"Sample ID: {sample_id}",
        labels=("", ""),
        genome=reference_genome,
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
    return add_genes(plot, gene_detail)


class CNVPlot:
    """Creates an interactive plot for visualizing CNV data using Dash.

    This class sets up a Dash application to visualize copy number variation
    (CNV) data, including highlighting interactively genes.

    Attributes:
        cnv_dir (str): Path to the directory containing CNV data.
        cnv_file (str): Name of the CNV file in the directory.
        genes_fix (list): List of genes to highlight. Defaults to
            IMPORTANT_GENES.
        host (str): Host address for running the Dash app. Defaults to
            "localhost".
        port (int): Port number for running the Dash app. Defaults to 8050.
        app (Dash): Dash application object created for the CNV plot.
    """
    def __init__(
        self,
        cnv_dir,
        cnv_file,
        genes=IMPORTANT_GENES,
        host="localhost",
        port=8050,
    ):
        """Initializes a CNVPlot object.

        Args:
            cnv_dir (str): Path to the directory containing CNV data.
            cnv_file (str): Name of the CNV file in the directory.
            genes (list, optional): List of genes to highlight. Defaults to
                IMPORTANT_GENES.
            host (str, optional): Host address for running the Dash app.
                Defaults to "localhost".
            port (int, optional): Port number for running the Dash app.
                Defaults to 8050.
        """
        self.cnv_dir = cnv_dir
        self.cnv_file = cnv_file
        self.genes_fix = genes
        self.host = host
        self.port = port
        self.app = self.get_app()

    def get_app(self):
        """Generates the dash app."""
        from dash import Dash, Input, Output, callback, dcc, html, no_update

        current_dir = Path(__file__).resolve().parent
        assets_folder = current_dir.parent / "data" / "assets"
        app = Dash(__name__, assets_folder=assets_folder)
        app._favicon = "favicon.svg"
        app.title = "mepylome"
        fig = go.Figure(layout=go.Layout(yaxis={"range": [-2, 2]}))
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
                    self.cnv_dir, sample_id
                )
            except FileNotFoundError:
                return no_update, no_update, f"Sample ID {sample_id} not found"
            plot = cnv_plot_from_data(
                sample_id,
                bins,
                detail,
                segments,
                self.genes_fix,
                genes_sel,
            )
            return plot, detail.sort_values(by="Name").Name, ""

        return app

    def run_app(self):
        """Opens new tab and runs app in browser."""
        init_sample_id = self.cnv_file.replace(ZIP_ENDING, "")

        def open_browser_tab():
            webbrowser.open_new_tab(
                f"http://{self.host}:{self.port}/{init_sample_id}"
            )

        threading.Timer(1, open_browser_tab).start()
        self.app.run(debug=True, host=self.host, use_reloader=False)
