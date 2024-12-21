"""Command-line interface for running mepylome.

Usage:
    $ mepylome                     # Run mepylome with default settings
    $ mepylome -a /path/to/idats   # Specify analysis IDAT files directory
               -r /path/to/ref     # Specify reference IDAT directory
               -c 450k             # Specify CpG's to use
               -s                  # Add horizontal segmentation lines
    $ mepylome --help              # Show all parameters

"""

import argparse
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def get_app_version():
    """Retrieve the app version from the package metadata."""
    try:
        return version("mepylome")
    except PackageNotFoundError:
        return "unknown"


def print_welcome_message():
    """Prints ASCII art welcome message."""
    app_version = get_app_version()
    welcome_message = rf"""
                             _
                            | |
  _ __ ___   ___ _ __  _   _| | ___  _ __ ___   ___
 | '_ ` _ \ / _ \ '_ \| | | | |/ _ \| '_ ` _ \ / _ \
 | | | | | |  __/ |_) | |_| | | (_) | | | | | |  __/
 |_| |_| |_|\___| .__/ \__, |_|\___/|_| |_| |_|\___|
                | |     __/ |
                |_|    |___/


Version: {app_version}
Starting Methylation Analysis...
"""
    print(welcome_message)


def absolute_path(path):
    """Converts a relative path to an absolute path."""
    return Path(path).absolute()


def parse_args():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(
        description="Command-line interface for running mepylome."
    )
    parser.add_argument(
        "-a",
        "--analysis_dir",
        type=absolute_path,
        help="Directory that contains the IDAT files to be analyzed.",
    )
    parser.add_argument(
        "-t",
        "--test_dir",
        type=absolute_path,
        help=(
            "Directory for test files, including new cases for analysis or "
            "validation."
        ),
    )
    parser.add_argument(
        "-A", "--annotation", type=str, help="Path to the annotation file."
    )
    parser.add_argument(
        "-r",
        "--reference_dir",
        type=absolute_path,
        help="Directory that contains CNV neutral reference cases.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=absolute_path,
        help="Directory for output data.",
    )
    parser.add_argument(
        "-p",
        "--prep",
        type=str,
        help=(
            "Preparation method. Possible values are 'illumina', 'swan', "
            "and 'noob'."
        ),
    )
    parser.add_argument(
        "-c",
        "--cpgs",
        type=str,
        nargs="+",
        help=(
            "CpGs to use for UMAP. Possible values are 'all', '450k', "
            "'epic', or 'epicv2'."
        ),
    )
    parser.add_argument(
        "-n", "--n_cpgs", type=int, help="Number of CpGs to use for UMAP."
    )
    parser.add_argument(
        "-C",
        "--precalculate_cnv",
        action="store_true",
        help=(
            "Precalculate CNV data in advance. This will take 1-2 seconds "
            "per case and improve performance during run time."
        ),
    )
    parser.add_argument(
        "-l",
        "--no_load_full_betas",
        action="store_false",
        dest="load_full_betas",
        help=(
            "Prevent loading betas for all CpG's into memory. By default, "
            "betas are loaded for all CpG's to improve the speed of "
            "generating "
            "multiple UMAP plots. Use this option to avoid "
            "potential memory overflow if insufficient memory is available "
            "(3-4 MB per sample needed)."
        ),
    )
    parser.add_argument(
        "-O",
        "--overlap",
        action="store_true",
        help=(
            "Only select IDAT files in the analysis directory that are also "
            "present in the annotation."
        ),
    )
    parser.add_argument(
        "-S",
        "--cpg_selection",
        type=str,
        default="top",
        help=(
            "Method for selecting the number of CpGs. Either 'random' or "
            "'top', where 'top' selects the CpGs with the highest variation "
            "within the analysis directory."
        ),
    )
    parser.add_argument(
        "-H", "--host", type=str, help="Host for the web server."
    )
    parser.add_argument(
        "-P", "--port", type=int, help="Port for the web server."
    )
    parser.add_argument(
        "-d", "--debug", action="store_true", help="Use debug mode in dash."
    )
    parser.add_argument(
        "-s",
        "--do_seg",
        action="store_true",
        help=(
            "Add horizontal segmentation lines in CNV plot "
            "(adds 1-2 seconds per sample)."
        ),
    )
    parser.add_argument(
        "--tutorial",
        action="store_true",
        help=(
            "Downloads test IDAT files used in the tutorial and then launches "
            "the mepylome GUI session. Use this for a quick demonstration of "
            "how this package works."
        ),
    )

    return parser.parse_args()


def start_mepylome():
    """Entry point to start mepylome methylation analysis from command line."""
    args = parse_args()
    cli_args = {k: v for k, v in vars(args).items() if v is not None}

    print_welcome_message()

    if not cli_args["load_full_betas"] and cli_args["cpg_selection"] == "top":
        cli_args["cpg_selection"] = "random"
        msg = (
            "Warning: Since '-no_load_full_betas' is set, 'cpg_selection' "
            "has been automatically changed to 'random'.\n"
        )
        print(msg)

    from .methyl import MethylAnalysis

    if cli_args["tutorial"]:
        from mepylome.utils import setup_tutorial_files

        tutorial_dir = Path.home() / "mepylome" / "tutorial"
        cli_args["analysis_dir"] = tutorial_dir / "tutorial_analysis"
        cli_args["reference_dir"] = tutorial_dir / "tutorial_reference"
        if (
            not cli_args["analysis_dir"].exists()
            and not cli_args["reference_dir"].exists()
        ):
            "Download Tutorial IDAT files"
            setup_tutorial_files(
                cli_args["analysis_dir"], cli_args["reference_dir"]
            )
        cli_args["load_full_betas"] = True
        cli_args["cpgs"] = "epic"

    cli_args.pop("tutorial", None)

    methyl_analysis = MethylAnalysis(**cli_args)
    methyl_analysis.run_app(open_tab=True)
