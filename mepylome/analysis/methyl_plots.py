"""Contains plotting functions for visualizing methylation analysis results."""

import colorsys
import hashlib
from functools import lru_cache, partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import plotly.colors
import plotly.express as px
import plotly.graph_objects as go
from tqdm import tqdm

from mepylome.dtypes import (
    CNV,
    IMPORTANT_GENES,
    ZIP_ENDING,
    Manifest,
    MethylData,
    ReferenceMethylData,
    cnv_plot_from_data,
    read_cnv_data_from_disk,
)
from mepylome.utils import (
    log,
)

PLOTLY_RENDER_MODE = "webgl"
ERROR_ENDING = "_error.txt"

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


def continuous_colors(names):
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


def _mixed_sort_key(s):
    """Sorts numeric if input is a number, else alphanumeric."""
    try:
        return (0, float(s))
    except ValueError:
        return (1, s)


def umap_plot_from_data(umap_df, use_discrete_colors=True):
    """Create and return umap plot from UMAP data.

    Args:
        umap_df: pandas data frame containing UMAP matrix and
            attributes. First row,w corresponds to sample.
        use_discrete_colors: Wheather to use discrete or continuous colors.
            Defaults to True.

    Returns:
        UMAP plot as plotly object.
    """
    methyl_classes = np.sort(umap_df["Umap_color"].unique())
    if use_discrete_colors:
        color_map = discrete_colors(methyl_classes)
    else:
        methyl_classes = sorted(methyl_classes, key=_mixed_sort_key)
        color_map = continuous_colors(methyl_classes)
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


@lru_cache(maxsize=None)
def get_reference_methyl_data(reference_dir, prep):
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
    idat_basepath, reference_dir, cnv_dir, prep, do_seg, verbose
):
    """Performs CNV analysis on a single sample and writes results to disk."""
    sample_id = idat_basepath.name
    try:
        sample_methyl = MethylData(file=idat_basepath)
        reference = get_reference_methyl_data(reference_dir, prep)
        cnv = CNV.set_all(
            sample_methyl, reference, do_seg=do_seg, verbose=verbose
        )
        cnv_filename = sample_id + ZIP_ENDING
        cnv.write(Path(cnv_dir, cnv_filename))
    except Exception as exc:
        cnv_filename = sample_id + ERROR_ENDING
        files_on_disk = [
            f"{x}, size={x.stat().st_size} B"
            for x in idat_basepath.parent.glob(f"{sample_id}*")
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
        log(error_message)
        with Path(cnv_dir, cnv_filename).open("w") as f:
            f.write(error_message)


def write_cnv_to_disk(
    sample_path, reference_dir, cnv_dir, prep, do_seg, pbar=None, verbose=False
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
        verbose (bool, optional): Whether to display verbose output.
    """
    new_idat_paths = [
        x
        for x in sample_path
        if not Path(cnv_dir, str(x.name) + ZIP_ENDING).exists()
        and not Path(cnv_dir, str(x.name) + ERROR_ENDING).exists()
    ]
    if len(new_idat_paths) == 0:
        return
    if verbose:
        log("[CNV-Plot] Write CNV to disk...")
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
        verbose=verbose,
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
    sample_path, reference_dir, prep, cnv_dir, genes_sel, do_seg, verbose=False
):
    """Generate and return a CNV plot for a given sample.

    Args:
        sample_path (Path): Path to the sample IDAT file.
        reference_dir (str): Directory with reference data.
        prep (str): Prepreparation method for MethylData.
        cnv_dir (str): Directory to save CNV data.
        genes_sel (list): List of genes to highlight in the plot.
        do_seg (bool): If segments should be calculated as well (slow)
        verbose (bool, optional): Whether to display verbose output.

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
        verbose=verbose,
    )
    if verbose:
        log("[CNV-Plot] Read CNV from disk....")
    bins, detail, segments = read_cnv_data_from_disk(cnv_dir, sample_id)
    plot = cnv_plot_from_data(
        sample_id,
        bins,
        detail,
        segments,
        IMPORTANT_GENES,
        list(genes_sel),
        verbose=verbose,
    )
    return plot.update_layout(
        margin={"l": 0, "r": 0, "t": 30, "b": 0},
    )
