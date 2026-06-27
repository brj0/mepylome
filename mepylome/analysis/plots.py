"""Contains plotting functions for visualizing methylation analysis results."""

import colorsys
import dataclasses
import hashlib
import logging
import math
import traceback
from collections.abc import Sequence
from functools import lru_cache, partial
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.colors
import plotly.express as px
import plotly.graph_objects as go
import psutil
from plotly.subplots import make_subplots
from tqdm import tqdm

from mepylome.dtypes import (
    CNV,
    Annotation,
    ArrayType,
    Chromosome,
    MethylData,
    PrepType,
    ReferenceMethylData,
    cnv_plot_from_data,
    read_cnv_data_from_disk,
)
from mepylome.utils import CONFIG

PLOTLY_RENDER_MODE = "webgl"
ERROR_ENDING = CONFIG["suffixes"]["cnv_error"]

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
ZIP_ENDING = CONFIG["suffixes"]["cnv_zip"]

logger = logging.getLogger(__name__)


def hash_from_str(string: str) -> int:
    """Calculates a pseudorandom int from a string."""
    hash_str = hashlib.blake2b(string.encode(), digest_size=16).hexdigest()
    return int(hash_str, 16)


def random_color(
    string: str,
    i: int,
    n_strings: int,
    rand: int,
) -> tuple[int, int, int]:
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
    r, g, b = (int(255 * x) for x in rgb_frac)
    return r, g, b


def discrete_colors(names: Sequence[str]) -> dict[str, str]:
    """Returns a colorscheme for all methylation classes."""
    sorted_names = sorted(names, key=hash_from_str)
    n_names = len(sorted_names)
    rand = hash_from_str("-".join(sorted_names))
    return {
        var: f"rgb{random_color(var, i, n_names, rand)}"
        for i, var in enumerate(sorted_names)
    }


def continuous_colors(names: Sequence[str]) -> dict[str, str]:
    """Returns a continuous colorscheme for all methylation classes."""
    n_names = len(names)
    color_scale = plotly.colors.get_colorscale("Plasma")
    colors = {}
    for i, name in enumerate(names):
        fraction = i / max(1, n_names - 1)
        color = plotly.colors.sample_colorscale(
            color_scale, fraction, colortype="rgb"
        )
        colors[name] = color[0]
    return colors


def _mixed_sort_key(s: str | float) -> tuple[int, float | str]:
    """Sorts numeric if input is a number, else alphanumeric."""
    try:
        return (0, float(s))
    except ValueError:
        return (1, s)


def umap_plot_from_data(
    umap_df: pd.DataFrame,
    use_discrete_colors: bool = True,
) -> go.Figure:
    """Create and return umap plot from UMAP data.

    Args:
        umap_df: pandas data frame containing UMAP matrix and
            attributes. First row,w corresponds to sample.
        use_discrete_colors: Wheather to use discrete or continuous colors.
            Defaults to True.

    Returns:
        UMAP plot as plotly object.
    """
    placeholder = "Not_Classified"
    umap_df["Umap_color"] = umap_df["Umap_color"].replace("", placeholder)
    methyl_classes = sorted(
        umap_df["Umap_color"].unique(), key=_mixed_sort_key
    )
    if use_discrete_colors:
        color_map = discrete_colors(methyl_classes)
    else:
        color_map = continuous_colors(methyl_classes)
    category_orders = {"Umap_color": methyl_classes}
    # If there are too many columns, they are not displayed correctly
    n_hover = 30
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
        hover_data=umap_df.columns[:n_hover],
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


def get_reference_methyl_data(
    reference_dir: str | Path,
    prep: PrepType,
) -> ReferenceMethylData:
    """Loads and caches CNV-neutral reference data."""
    try:
        reference = ReferenceMethylData(
            file=reference_dir, prep=prep, save_to_disk=True
        )
    except FileNotFoundError as exc:
        msg = f"File in reference dir {reference_dir} not found: {exc}"
        raise FileNotFoundError(msg) from exc
    return reference


def write_single_cnv_to_disk(
    idat_basepath: Path,
    reference_dir: str | Path,
    cnv_dir: str | Path,
    prep: PrepType,
    do_seg: bool,
) -> None:
    """Performs CNV analysis on a single sample and writes results to disk."""
    sample_id = idat_basepath.name
    try:
        sample_methyl = MethylData(file=idat_basepath, prep=prep)
        reference = get_reference_methyl_data(reference_dir, prep)
        cnv = CNV.set_all(sample_methyl, reference, do_seg=do_seg)
        cnv_filename = sample_id + ZIP_ENDING
        cnv.write(Path(cnv_dir, cnv_filename))
    except Exception as exc:
        cnv_filename = sample_id + ERROR_ENDING
        files_on_disk = [
            f"{x}, size={x.stat().st_size} B"
            for x in idat_basepath.parent.glob(f"{sample_id}*")
        ]
        full_traceback = traceback.format_exc()
        error_message = (
            "During processing '" + str(sample_id) + "' the following "
            "exception occurred:\n\n "
            + str(exc)
            + "\n\nFull traceback:\n"
            + str(full_traceback)
            + "\n\n Corresponding files on disk:\n "
            + "\n".join(files_on_disk)
            + "\n\n\nTo recalculate, delete this "
            "file."
        )
        logger.error(error_message)
        with Path(cnv_dir, cnv_filename).open("w") as f:
            f.write(error_message)


def get_optimal_core_count(reserve_mem_gb: float = 1.0) -> int:
    """Determine optimal core count based on CPU and memory constraints."""
    process = psutil.Process()
    current_proc_mem_gb = process.memory_info().rss / 1e9
    avail_mem_gb = psutil.virtual_memory().available / 1e9
    usable_mem_gb = max(0, avail_mem_gb - reserve_mem_gb)

    if current_proc_mem_gb <= 0:
        # Fallback: use 1 core if memory usage can't be estimated
        return 1

    mem_based_cores = int(usable_mem_gb // current_proc_mem_gb)

    return max(1, min(cpu_count() - 1, mem_based_cores))


def write_cnv_to_disk(
    sample_path: Sequence[Path],
    reference_dir: str | Path,
    cnv_dir: str | Path,
    prep: PrepType,
    do_seg: bool,
    pbar: Any | None = None,
    n_cores: int | None = None,
) -> None:
    """Generate and save CNV-analysis output files for given samples.

    Saves CNV data with a ZIP_ENDING extension. Processes unseen samples,
    avoiding existing CNV. Uses single-threading for one sample, and
    multi-threading for multiple samples.

    Args:
        sample_path (list): Paths to sample IDAT files.
        reference_dir (str): Directory with CNV neural reference data.
        cnv_dir (str): Directory to save CNV data.
        prep (str): Prepreparation method for MethylData.
        do_seg (bool): If segments should be calculated as well (slow)
        pbar (optional): Progress bar for tracking progress.
        n_cores (int, optional): Number of CPU cores to use. If None, a
            reasonable number of cores will be automatically chosen based on
            the system and workload.
    """
    new_idat_paths = [
        x
        for x in sample_path
        if not Path(cnv_dir, str(x.name) + ZIP_ENDING).exists()
    ]

    if len(new_idat_paths) == 0:
        return

    _write_single_cnv_to_disk = partial(
        write_single_cnv_to_disk,
        reference_dir=reference_dir,
        cnv_dir=cnv_dir,
        prep=prep,
        do_seg=do_seg,
    )

    if n_cores is None:
        n_cores = max(1, min(len(new_idat_paths), get_optimal_core_count()))
    else:
        n_cores = max(1, min(n_cores, len(new_idat_paths), cpu_count()))

    logger.info("Write CNV to disk using %s core(s).", n_cores)

    if n_cores == 1:
        # This is significantly faster than Pool(1)
        with tqdm(
            total=len(new_idat_paths), desc="Generating CNV files"
        ) as tqdm_bar:
            for sample in new_idat_paths:
                _write_single_cnv_to_disk(sample)
                if pbar is not None:
                    pbar.increment()
                _ = tqdm_bar.update(1)
    else:
        # Loading annotations and load References here prevents race conditions
        Annotation.load()
        _ = get_reference_methyl_data(reference_dir, prep)
        with (
            Pool(n_cores) as pool,
            tqdm(
                total=len(new_idat_paths), desc="Generating CNV files"
            ) as tqdm_bar,
        ):
            for _ in pool.imap(_write_single_cnv_to_disk, new_idat_paths):
                if pbar is not None:
                    pbar.increment()
                _ = tqdm_bar.update(1)


@lru_cache
def get_cnv_plot(
    sample_path: Path,
    reference_dir: str | Path,
    prep: PrepType,
    cnv_dir: str | Path,
    genes_sel: Sequence[str],
    do_seg: bool,
) -> go.Figure:
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
    write_cnv_to_disk(
        sample_path=[sample_path],
        reference_dir=reference_dir,
        cnv_dir=cnv_dir,
        prep=prep,
        do_seg=do_seg,
    )
    logger.info("Read CNV from disk....")
    bins, detail, segments = read_cnv_data_from_disk(cnv_dir, sample_id)
    assert bins is not None
    assert detail is not None
    plot = cnv_plot_from_data(
        sample_id,
        bins,
        detail,
        segments,
        CONFIG["genes"]["default_genes_list"],
        list(genes_sel),
    )
    return plot.update_layout(
        margin={"l": 0, "r": 0, "t": 30, "b": 0},
    )


@dataclasses.dataclass
class GenomicInfo:
    """Genomic information about a gene and its array CpG probes.

    Attributes:
        cpgs (np.ndarray): Illumina probe IDs located within the gene, ordered
            by ascending genomic start position.
        positions (pd.Series): Genomic start position for each CpG in ``cpgs``,
            indexed by CpG ID.
        chromosome (Chromosome): Chromosome of the gene.
        start (int): Genomic start position of the gene body.
        end (int): Genomic end position of the gene body.
        strand (str): Strand of the gene ("+" or "-").
    """

    cpgs: np.ndarray
    positions: pd.Series
    chromosome: Chromosome
    start: int
    end: int
    strand: str


def _region_cpgs(
    annotation: Annotation,
    chromosome: Chromosome,
    start: int,
    end: int,
) -> pd.DataFrame:
    """Returns manifest probes within a genomic region."""
    return annotation.manifest.data_frame[
        (annotation.manifest.data_frame.Chromosome == chromosome)
        & (annotation.manifest.data_frame.Start >= start)
        & (annotation.manifest.data_frame.Start <= end)
    ].sort_values("Start")


def get_gene_info(
    gene: str,
    array_type: str | ArrayType,
) -> GenomicInfo:
    """Returns genomic info and CpG probes for a gene, on a given array.

    Determines the genomic region of ``gene`` and intersects it with the CpG
    probes of the given array type.

    Args:
        gene (str): Gene symbol (e.g. "MLH1", "EGFR"). Must match a gene name
            in the bundled gene annotation (case-sensitive).
        array_type (str or ArrayType): The Illumina array type used to
            determine the corresponding manifest.

    Returns:
        ``GenomicInfo`` of the given gene.
    """
    annotation = Annotation(array_type=array_type)

    gene_rows = annotation.detail[annotation.detail.Name == gene]
    if gene_rows.empty:
        raise ValueError(f"Unknown gene: {gene}")

    gene_row = gene_rows.iloc[0]
    chromosome = Chromosome.from_string(gene_row.Chromosome)

    cpgs = _region_cpgs(
        annotation,
        chromosome,
        int(gene_row.Start),
        int(gene_row.End),
    )

    return GenomicInfo(
        cpgs=cpgs.IlmnID.to_numpy(),
        positions=cpgs.set_index("IlmnID")["Start"],
        chromosome=chromosome,
        start=int(gene_row.Start),
        end=int(gene_row.End),
        strand=str(gene_row.Strand),
    )


def get_region_info(
    chromosome: str | int | Chromosome,
    start: int,
    end: int,
    array_type: str | ArrayType,
) -> GenomicInfo:
    """Returns genomic info and CpG probes for an arbitrary genomic region.

    Args:
        chromosome: Chromosome of the region.
        start: Genomic start position.
        end: Genomic end position.
        array_type: Illumina array type.

    Returns:
        ``GenomicInfo`` describing the region.
    """
    annotation = Annotation(array_type=array_type)

    if isinstance(chromosome, str):
        chromosome = Chromosome.from_string(chromosome)
    elif isinstance(chromosome, int):
        chromosome = Chromosome(chromosome)

    if chromosome == Chromosome.INVALID:
        raise ValueError("'chromosome' is invalid")

    cpgs = _region_cpgs(annotation, chromosome, start, end)

    return GenomicInfo(
        cpgs=cpgs.IlmnID.to_numpy(),
        positions=cpgs.set_index("IlmnID")["Start"],
        chromosome=chromosome,
        start=start,
        end=end,
        strand="+",
    )


def region_methylation_plot(
    betas: pd.DataFrame,
    region_name: str,
    cpg_positions: pd.Series,
    region_start: int,
    region_end: int,
    strand: str,
    chromosome: Chromosome | None = None,
    max_width: int = 1500,
    max_tick_labels: int = 80,
) -> go.Figure:
    """Plots a methylation heatmap for a gene or arbitrary genomic region.

    Draws beta values as a heatmap (samples x CpGs, CpGs evenly spaced
    and ordered by genomic position) with a region track above it.
    Connector lines link each CpG's true genomic position on the track
    to its evenly-spaced column in the heatmap below.

    Figure width is capped at ``max_width`` regardless of CpG count
    (cell width shrinks down to a 2px floor instead of growing without
    bound), and x-axis tick labels are thinned out above
    ``max_tick_labels`` CpGs to keep rendering fast and readable; exact
    CpG IDs remain available via hover either way.

    Args:
        betas (pd.DataFrame): Beta values with samples as index and CpG
            probe IDs as columns, columns already ordered by genomic
            position.
        region_name (str): Name used in the title/label (e.g. a gene
            symbol), used when ``chromosome`` is not given.
        cpg_positions (pd.Series): Genomic start position for each CpG
            in ``betas.columns``.
        region_start (int): Genomic start position of the region/gene.
        region_end (int): Genomic end position of the region/gene.
        strand (str): Strand ("+" or "-"), used for the direction arrow.
        chromosome (Chromosome, optional): If given, the title/label
            shows "<chromosome>:<region_start>-<region_end>" instead of
            ``region_name``.
        max_width (int): Maximum heatmap width in pixels.
        max_tick_labels (int): Maximum number of CpG names drawn on the
            x-axis before labels are thinned out.

    Returns:
        go.Figure: Plotly figure with the region track stacked above the
        methylation heatmap.
    """
    cpgs = list(betas.columns)
    n = len(cpgs)
    n_samples = len(betas.index)

    positions = cpg_positions.reindex(cpgs).to_numpy()

    track_min = min(region_start, int(positions.min()))
    track_max = max(region_end, int(positions.max()))
    span = max(track_max - track_min, 1)

    top_x = (positions - track_min) / span * n
    bottom_x = np.arange(n) + 0.5

    # Fixed layout sizing -- but cell width now shrinks as n grows
    # instead of the figure growing without bound.
    cell_h = 20
    side_margin = 170
    ideal_cell_w = 22
    min_cell_w = 2
    cell_w = max(min_cell_w, min(ideal_cell_w, (max_width - side_margin) / n))

    top_track_px = 80
    top_margin_px = 55
    bottom_margin_px = 110 if n <= max_tick_labels else 90

    heatmap_w = cell_w * n
    heatmap_h = max(100, cell_h * n_samples)
    total_w = heatmap_w + side_margin
    total_h = top_margin_px + top_track_px + heatmap_h + bottom_margin_px
    top_frac = top_track_px / (top_track_px + heatmap_h)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[top_frac, 1 - top_frac],
        vertical_spacing=0.0,
    )

    # -------------------------
    # HEATMAP
    # -------------------------
    fig.add_trace(
        go.Heatmap(
            z=betas.to_numpy(),
            x=bottom_x,
            y=[str(s) for s in betas.index],
            zmin=0,
            zmax=1,
            colorscale=[
                [0.0, "#2166ac"],
                [0.5, "#f7f7f7"],
                [1.0, "#b2182b"],
            ],
            xgap=1 if n <= max_tick_labels else 0,
            ygap=1,
            colorbar={"title": "Beta", "len": 0.7, "thickness": 14},
            customdata=np.tile(cpgs, (n_samples, 1)),
            hovertemplate=(
                "Sample: %{y}<br>CpG: %{customdata}<br>"
                "Beta: %{z:.3f}<extra></extra>"
            ),
        ),
        row=2,
        col=1,
    )

    # -------------------------
    # REGION TRACK
    # -------------------------
    line_y = 0.5
    bar_h = 0.12

    bar_x0 = (region_start - track_min) / span * n
    bar_x1 = (region_end - track_min) / span * n

    fig.add_shape(
        type="line",
        x0=0,
        x1=n,
        y0=line_y,
        y1=line_y,
        line={"color": "rgba(60,60,60,0.35)", "width": 2},
        row=1,
        col=1,
    )

    fig.add_shape(
        type="rect",
        x0=bar_x0,
        x1=bar_x1,
        y0=line_y - bar_h,
        y1=line_y + bar_h,
        line={"width": 0},
        fillcolor="rgba(40,40,40,0.85)",
        row=1,
        col=1,
    )

    arrow_tail, arrow_head = (
        (bar_x0, bar_x1) if strand == "+" else (bar_x1, bar_x0)
    )

    fig.add_annotation(
        x=arrow_head,
        y=line_y,
        ax=arrow_tail,
        ay=line_y,
        xref="x",
        yref="y",
        axref="x",
        ayref="y",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=1.4,
        arrowcolor="black",
        text="",
        row=1,
        col=1,
    )

    label = (
        region_name
        if chromosome is None
        else f"chr{chromosome}:{region_start}-{region_end}"
    )

    fig.add_annotation(
        x=(bar_x0 + bar_x1) / 2,
        y=1.0,
        text=f"<b>{label}</b> ({strand})",
        showarrow=False,
        font={"size": 12},
        row=1,
        col=1,
    )

    # -------------------------
    # CONNECTORS
    # -------------------------
    mid_y = line_y - bar_h - 0.18

    cx, cy = [], []
    for tx, bx in zip(top_x, bottom_x, strict=True):
        cx += [bx, tx, tx, None]
        cy += [0, mid_y, line_y - bar_h, None]

    fig.add_trace(
        go.Scatter(
            x=cx,
            y=cy,
            mode="lines",
            line={"color": "rgba(150,150,150,0.6)", "width": 1},
            hoverinfo="skip",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # -------------------------
    # AXES
    # -------------------------
    fig.update_xaxes(visible=False, range=[0, n], row=1, col=1)
    fig.update_yaxes(visible=False, range=[0, 1], row=1, col=1)

    # Thin out tick labels once there are too many to render/read
    # individually; hover still gives the exact CpG ID either way.
    if n <= max_tick_labels:
        tickvals, ticktext = bottom_x, cpgs
    else:
        step = math.ceil(n / max_tick_labels)
        tickvals, ticktext = bottom_x[::step], cpgs[::step]

    fig.update_xaxes(
        tickmode="array",
        tickvals=tickvals,
        ticktext=ticktext,
        tickangle=-90,
        tickfont={"size": 9},
        row=2,
        col=1,
    )

    # For wide regions, let users zoom/pan within a width-capped figure
    # instead of scrolling a page that's tens of thousands of px wide.
    fig.update_xaxes(
        rangeslider={"visible": n > max_tick_labels, "thickness": 0.06},
        row=2,
        col=1,
    )

    fig.update_yaxes(autorange="reversed", row=2, col=1)

    fig.update_layout(
        title=f"Methylation of {label}",
        template="simple_white",
        width=total_w,
        height=total_h,
        margin={"t": top_margin_px, "b": bottom_margin_px, "l": 80, "r": 40},
    )

    return fig
