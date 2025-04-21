"""Contains plotting functions for visualizing methylation analysis results."""

import colorsys
import hashlib
import logging
import traceback
from functools import lru_cache, partial
from multiprocessing import Pool, cpu_count
from pathlib import Path

import plotly.colors
import plotly.express as px
import plotly.graph_objects as go
import psutil
from tqdm import tqdm

from mepylome.dtypes import (
    CNV,
    MethylData,
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


def hash_from_str(string):
    """Calculates a pseudorandom int from a string."""
    hash_str = hashlib.blake2b(string.encode(), digest_size=16).hexdigest()
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
    idat_basepath, reference_dir, cnv_dir, prep, do_seg
):
    """Performs CNV analysis on a single sample and writes results to disk."""
    sample_id = idat_basepath.name
    try:
        sample_methyl = MethylData(file=idat_basepath)
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


def get_optimal_core_count(reserve_mem_gb=1.0):
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
    sample_path, reference_dir, cnv_dir, prep, do_seg, pbar=None, n_cores=None
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
        n_cores (int, optional): Number of CPU cores to use. If None, a
            reasonable number of cores will be automatically chosen based on
            the system and workload.
    """
    new_idat_paths = [
        x
        for x in sample_path
        if not Path(cnv_dir, str(x.name) + ZIP_ENDING).exists()
        and not Path(cnv_dir, str(x.name) + ERROR_ENDING).exists()
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
        # Line breaks are not supported in context managers in Python 3.8.
        # To avoid errors, both `Pool` and `tqdm` are used on the same line.
        # fmt: off
        with Pool(n_cores) as pool, tqdm(total=len(new_idat_paths), desc="Generating CNV files") as tqdm_bar: # noqa: E501
            # fmt: on
            for _ in pool.imap(_write_single_cnv_to_disk, new_idat_paths):
                if pbar is not None:
                    pbar.increment()
                _ = tqdm_bar.update(1)


@lru_cache
def get_cnv_plot(sample_path, reference_dir, prep, cnv_dir, genes_sel, do_seg):
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
