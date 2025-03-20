"""Contains utilities to generate CNV plots."""

import io
import logging
import threading
import webbrowser
import zipfile
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import plotly
import plotly.graph_objects as go

from mepylome.dtypes.cache import memoize
from mepylome.utils.files import ensure_directory_exists
from mepylome.utils.varia import CONFIG, MEPYLOME_TMP_DIR, get_free_port

logger = logging.getLogger(__name__)

PLOTLY_RENDER_MODE = "webgl"
ZIP_ENDING = CONFIG["suffixes"]["cnv_zip"]
THRESHOLD_BALANCED = 0.1
CHROMOSOME_DATA = CONFIG["genes"]["chromosome_data"]

ensure_directory_exists(MEPYLOME_TMP_DIR)
CNV_GRID = Path(MEPYLOME_TMP_DIR, f"cnv_grid_v{plotly.__version__}.json")


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
    extract = extract or ["bins", "detail", "segments"]
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
    return extracted_dfs


def add_offset(df, chrom_nm, col):
    """Calculates x-values of the CNV-plot."""
    return df[col] + df[chrom_nm].map(offset)


def get_x_mid(df):
    """Calculates the midpoint x-values of all chromosomes."""
    return df["Chromosome"].map(offset) + (df["Start"] + df["End"]) // 2


@lru_cache
def cnv_grid():
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
            "range": [0, len(reference_genome)],
            "showgrid": False,
            "ticklen": 10,
            "tickmode": "array",
            "ticks": "outside",
            "tickson": "boundaries",
            "ticktext": reference_genome.chrom.name,
            "tickvals": reference_genome.chrom.center,
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
    for i in reference_genome.chrom.centromere_offset:
        grid.add_vline(x=i, line_color="black", line_dash="dot", line_width=1)
    # Vertical line: chromosomes.
    for i in [*reference_genome.chrom.offset.tolist(), len(reference_genome)]:
        grid.add_vline(x=i, line_color="black", line_width=1)
    # Draw horizontal line.
    grid.add_hline(y=0, line_color="black", line_width=1)
    # Save to disk
    grid.write_json(CNV_GRID)
    return grid


def cnv_bins_plot(data_frame, title, labels):
    """Create CNV plot from CNV data.

    Args:
        data_frame: DataFrame with columns 'x', 'y', and 'hover_data'
        title: Title of plot
        labels: Touple of labels for x-axis and y-axis.
    """
    scatter_trace = go.Scattergl(
        x=data_frame["x"],
        y=data_frame["y"],
        mode="markers",
        marker={
            "color": data_frame["y"],
            "colorscale": "RdBu_r",
            "cmin": -0.4,
            "cmax": 0.4,
        },
        text=data_frame["hover_data"],
        hovertemplate=(
            "<b>Value:</b> %{y}<br><b>Genes:</b> %{text}<extra></extra>"
        ),
        showlegend=False,
    )
    plot = go.Figure(cnv_grid())
    plot.add_trace(scatter_trace)
    plot.update_layout(
        title=title,
        yaxis_range=[-2, 2],
        xaxis_title=labels[0],
        yaxis_title=labels[1],
    )
    return plot


def add_segments(plot, seg_df):
    """Adds segments calculated by Circular Binary Segmentation (CBG)."""
    if seg_df is None:
        return plot
    seg_lines = [
        go.Scattergl(
            x=[seg["X_start"], seg["X_end"]],
            y=[seg["Median"], seg["Median"]],
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


@memoize
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
    bins_X_start = add_offset(bins, "Chromosome", "Start")
    bins_X_end = add_offset(bins, "Chromosome", "End")

    if not bins_X_start.is_monotonic_increasing:
        msg = "'bins' is not sorted"
        raise ValueError(msg)

    detail_X_start = add_offset(detail, "Chromosome", "Start")
    detail_X_end = add_offset(detail, "Chromosome", "End")

    detail_Start_bin = np.digitize(detail_X_start, bins_X_start) - 1
    detail_Start_bin_ = np.digitize(detail_X_start, bins_X_end, right=True)
    detail_End_bin = np.digitize(detail_X_end, bins_X_start) - 1

    # Identify genes that start within gaps between bins
    detail_Start_bin[detail_Start_bin < detail_Start_bin_] += 1

    detail_Range = [
        np.arange(start, end)
        for start, end in zip(detail_Start_bin, detail_End_bin + 1)
    ]
    lengths = [len(x) for x in detail_Range]
    df_expanded = pd.DataFrame(
        {
            "Name": np.repeat(detail["Name"].values, lengths),
            "Range": np.concatenate(detail_Range),
        }
    )
    bins_Genes = (
        df_expanded
        # By sorting first, resultig string will be sorted too
        .sort_values(by="Name")
        .groupby("Range")["Name"]
        .agg(lambda x: ", ".join(x))
        .reindex(range(len(bins)), fill_value="")
        .reset_index(drop=True)
    )
    return bins_Genes, detail_Range


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


def read_cnv_data_from_disk(cnv_dir, sample_id, extract=None):
    """Reads the components of a zip-file generated by CNV.write."""
    extract = extract or ["bins", "detail", "segments"]
    if isinstance(extract, str):
        extract = [extract]

    sample_zip = sample_id + ZIP_ENDING
    unzipped = get_df_from_zip(
        Path(cnv_dir) / sample_zip,
        extract=extract,
    )

    results = dict(zip(extract, unzipped))
    # Calculate some plot x-values
    if results.get("bins") is not None:
        results["bins"]["X_mid"] = get_x_mid(results["bins"])
    if results.get("detail") is not None:
        results["detail"]["X_mid"] = get_x_mid(results["detail"])
        results["detail"]["Len"] = (
            results["detail"]["End"] - results["detail"]["Start"]
        )
    if results.get("segments") is not None:
        results["segments"]["X_start"] = add_offset(
            results["segments"], "Chromosome", "Start"
        )
        results["segments"]["X_end"] = add_offset(
            results["segments"], "Chromosome", "End"
        )

    if len(results) == 1:
        return next(iter(results.values()))
    return tuple(results.values())


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
    logger.info("Make CNV plot: prepare data...")
    bins_Genes, detail_Range = find_genes_within_bins(
        bins[["Chromosome", "Start", "End"]],
        detail[["Chromosome", "Start", "End", "Name"]],
    )
    bins["Genes"] = bins_Genes
    detail["Range"] = detail_Range
    scatter_df = bins[["X_mid", "Median", "Genes"]]
    scatter_df.columns = ["x", "y", "hover_data"]

    # Base scatterplot
    logger.info("Make CNV plot: bins...")
    plot = cnv_bins_plot(
        data_frame=scatter_df,
        title=f"Sample ID: {sample_id}",
        labels=("", ""),
    )

    # Highlight bins adjacent to the added genes
    logger.info("Make CNV plot: genes...")
    genes_sel = genes_sel if genes_sel else []
    genes_x_range = (
        detail[detail["Name"].isin(genes_sel)]["Range"].explode().tolist()
    )
    highlighted_bins = bins.loc[genes_x_range, ["X_mid", "Median"]]
    plot = add_highlited_bins(plot, highlighted_bins)

    # Add all added and important genes
    genes_to_plot = genes_fix + genes_sel
    gene_detail = detail[detail["Name"].isin(genes_to_plot)].copy()
    plot = add_genes(plot, gene_detail)

    # Draw the segments
    logger.info("Make CNV plot: segments...")
    plot = add_segments(plot, segments)
    return plot


class CNVPlot:
    """Creates an interactive plot for visualizing CNV data using Dash.

    This class sets up a Dash application to visualize copy number variation
    (CNV) data, including highlighting interactively genes.

    Attributes:
        cnv_dir (str): Path to the directory containing CNV data.
        cnv_file (str): Name of the CNV file in the directory.
        genes_fix (list): List of genes to highlight. Defaults to
            CONFIG["genes"]["default_genes_list"].
        host (str): Host address for running the Dash app. Defaults to
            "localhost".
        port (int): Port number for running the Dash app. Defaults to 8050.
        app (Dash): Dash application object created for the CNV plot.
    """

    def __init__(
        self,
        cnv_dir,
        cnv_file,
        genes=CONFIG["genes"]["default_genes_list"],
        host="localhost",
        port=8050,
    ):
        """Initializes a CNVPlot object.

        Args:
            cnv_dir (str): Path to the directory containing CNV data.
            cnv_file (str): Name of the CNV file in the directory.
            genes (list, optional): List of genes to highlight. Defaults to
                CONFIG["genes"]["default_genes_list"].
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
                    config={
                        "scrollZoom": False,
                        "doubleClick": "reset",
                        "modeBarButtonsToRemove": ["lasso2d", "select"],
                        "displaylogo": False,
                    },
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
        free_port = get_free_port(self.port)

        def open_browser_tab():
            webbrowser.open_new_tab(
                f"http://{self.host}:{free_port}/{init_sample_id}"
            )

        threading.Timer(1, open_browser_tab).start()
        self.app.run(
            debug=True, host=self.host, use_reloader=False, port=free_port
        )


def _cn_summary_per_chrom(df_seg, threshold=0.1):
    """Generates a CN summary on disjoint intervals for a single chromosome."""
    df_seg["Gain"] = df_seg["Median"] > threshold
    df_seg["Loss"] = df_seg["Median"] < -threshold
    df_seg["Balanced"] = ~(df_seg["Gain"] | df_seg["Loss"])
    chromosome = df_seg.iloc[0]["Chromosome"]
    # Determine boundaries for disjoint segments
    boundaries = sorted(set(df_seg["Start"]).union(set(df_seg["End"])))
    result_intervals = []
    for i in range(1, len(boundaries)):
        start = boundaries[i - 1]
        end = boundaries[i]
        involved_segments = df_seg[
            (df_seg["Start"] <= start) & (df_seg["End"] >= end)
        ]
        gain_ratio = involved_segments["Gain"].mean()
        loss_ratio = involved_segments["Loss"].mean()
        result_intervals.append(
            {
                "Chromosome": chromosome,
                "Start": start,
                "End": end,
                "Gain_ratio": gain_ratio,
                "Loss_ratio": -loss_ratio,
                "Balanced_ratio": 1 - gain_ratio - loss_ratio,
            }
        )
    return pd.DataFrame(result_intervals)


def get_cn_summary(cnv_dir, sample_ids):
    """Generate a CNV summary plot for a given set of samples.

    This function reads CNV segment data from disk for a list of sample IDs,
    computes disjoint intervals for each chromosome, and generates a visual
    plot of gain/loss ratios across chromosomes.

    Args:
        cnv_dir (str or Path): The directory path where CNV data files are
            stored.
        sample_ids (list of str): A list of sample IDs to process.

    Returns:
        plot (plotly.graph_objects.Figure): A Plotly figure object containing
            the gain/loss ratio plot.
        df_cn_summary (pd.DataFrame): A pandas DataFrame containing the
            data used for the plot.
    """
    segment_list = []
    for sample_id in sample_ids:
        segment = read_cnv_data_from_disk(
            cnv_dir, sample_id, extract="segments"
        )
        if segment is None:
            msg = f"CNV for {sample_id} does not contain 'segments'."
            raise ValueError(msg)
        segment["Id"] = sample_id
        segment_list.append(segment)
    segments = pd.concat(segment_list)
    df_cn_summary = pd.concat(
        [
            _cn_summary_per_chrom(segments_on_chrom)
            for _, segments_on_chrom in segments.groupby("Chromosome")
        ]
    )
    df_cn_summary["X_start"] = add_offset(df_cn_summary, "Chromosome", "Start")
    df_cn_summary["X_end"] = add_offset(df_cn_summary, "Chromosome", "End")
    loss_color = "#1F77B4"
    gain_color = "#D62728"
    contour_color = "black"
    plot = go.Figure(cnv_grid())
    for chrom in df_cn_summary["Chromosome"].unique():
        # If there are gaps in the genome, ratios may be nan
        df_cn_summary_chrom = df_cn_summary[
            df_cn_summary["Chromosome"] == chrom
        ].fillna(0)
        first_start = df_cn_summary_chrom.iloc[0]["X_start"]
        x_vals_gain = [first_start]
        y_vals_gain = [0]
        x_vals_loss = [first_start]
        y_vals_loss = [0]
        for _, row in df_cn_summary_chrom.iterrows():
            x_vals_gain.extend([row["X_start"], row["X_end"]])
            y_vals_gain.extend([row["Gain_ratio"], row["Gain_ratio"]])
            x_vals_loss.extend([row["X_start"], row["X_end"]])
            y_vals_loss.extend([row["Loss_ratio"], row["Loss_ratio"]])
        last_end = df_cn_summary_chrom.iloc[-1]["X_end"]
        x_vals_gain.append(last_end)
        y_vals_gain.append(0)
        x_vals_loss.append(last_end)
        y_vals_loss.append(0)
        plot.add_trace(
            go.Scatter(
                x=x_vals_gain,
                y=y_vals_gain,
                mode="lines",
                fill="tozeroy",
                fillcolor=gain_color,
                line={"color": contour_color, "width": 1.5},
                hovertemplate="<b>Value:</b> %{y}<br><extra></extra>",
                showlegend=False,
            )
        )
        plot.add_trace(
            go.Scatter(
                x=x_vals_loss,
                y=y_vals_loss,
                mode="lines",
                fill="tozeroy",
                fillcolor=loss_color,
                line={"color": contour_color, "width": 1.5},
                hovertemplate="<b>Value:</b> %{y}<br><extra></extra>",
                showlegend=False,
            )
        )
    plot.update_layout(
        yaxis_range=[-1, 1],
    )
    return plot, df_cn_summary
