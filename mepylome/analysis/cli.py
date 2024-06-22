"""Command-line interface for running mepylome.

Usage:
    $ mepylome                     # Run mepylome with default settings
    $ mepylome -a /path/to/idats   # Specify anylsis IDAT files directory
               -r /path/to/ref     # Specify reference IDAT directory
               -c 450k             # Specify CpG's to use
               -s                  # Improve UMAP speed by saving betas to disk
    $ mepylome --help              # Show all parameters

"""

import argparse


def print_welcome_message():
    """Prints ASCII art welcome message."""
    welcome_message = r"""
                             _
                            | |
  _ __ ___   ___ _ __  _   _| | ___  _ __ ___   ___
 | '_ ` _ \ / _ \ '_ \| | | | |/ _ \| '_ ` _ \ / _ \
 | | | | | |  __/ |_) | |_| | | (_) | | | | | |  __/
 |_| |_| |_|\___| .__/ \__, |_|\___/|_| |_| |_|\___|
                | |     __/ |
                |_|    |___/


Starting Methylation Analysis...
"""
    print(welcome_message)


def parse_args():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(
        description="Command-line interface for running mepylome."
    )
    parser.add_argument(
        "-a",
        "--analysis_dir",
        type=str,
        help="Directory that contains the IDAT files to be analyzed.",
    )
    parser.add_argument(
        "-A", "--annotation", type=str, help="Path to the annotation file."
    )
    parser.add_argument(
        "-r",
        "--reference_dir",
        type=str,
        help="Directory that contains CNV neutral reference cases.",
    )
    parser.add_argument(
        "-o", "--output_dir", type=str, help="Directory for output data."
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
        "-s",
        "--save_betas",
        action="store_true",
        help=(
            "Save betas to disk to improve the speed of generating multiple "
            "UMAP plots."
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
        "-D",
        "--do_seg",
        action="store_true",
        help="Add horizontal segmentation lines in CNV plot (slow).",
    )

    return parser.parse_args()


def start_mepylome():
    """Entry point to start mepylome methylation analysis from command line."""
    args = parse_args()
    methyl_analysis_args = {
        k: v for k, v in vars(args).items() if v is not None
    }

    print_welcome_message()
    from .methyl import MethylAnalysis

    methyl_analysis = MethylAnalysis(**methyl_analysis_args)
    methyl_analysis.run_app(open_tab=True)
