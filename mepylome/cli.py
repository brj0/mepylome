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
import re
import textwrap
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


class SmartFormatter(argparse.ArgumentDefaultsHelpFormatter):
    """Keeps new lines and doesn't break words, but still wraps lines.

    Source: https://gist.github.com/panzi/b4a51b3968f67b9ff4c99459fb9c5b3d
    """

    def _split_lines(self, text, width):
        lines = []
        for line in textwrap.dedent(text).strip().split("\n"):
            ident = re.match(r"^\s*", line).group(0)
            curr_line = [ident] if ident else []
            curr_len = len(ident)
            for word in line.split():
                if curr_line and curr_len + len(word) + 1 > width:
                    lines.append(" ".join(curr_line))
                    curr_line = [ident] if ident else []
                    curr_len = len(ident)
                curr_line.append(word)
                curr_len += len(word) + (1 if curr_line else 0)
            lines.append(" ".join(curr_line))
        return lines

    def _fill_text(self, text, width, indent):
        return "\n".join(
            indent + line
            for line in self._split_lines(text, width - len(indent))
        )

    def _format_action(self, action):
        """Adds an extra newline after each option."""
        return super()._format_action(action) + "\n"


def parse_args():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            """
            Mepylome: Methylation Array Analysis Toolkit
            --------------------------------------------

            Command-line interface (CLI) for running the Mepylome methylation analysis GUI interface. This tool allows you to analyze methylation microarray data, visualize results, and run various analyses, including CNV (Copy Number Variation) and UMAP (Uniform Manifold Approximation and Projection) through a Dash-based GUI application.
            """
        ),
        epilog=(
            """
            Example usage:
            --------------

            1. Minimal Setup:
            Starts the GUI with only essential components. All required directories and settings must be provided within the GUI:

                mepylome

            2. Tutorial:
            Downloads a small IDAT dataset and launches the GUI with a preconfigured example analysis:

                mepylome --tutorial

            3. Standard Options (Recommended):
            Launches the GUI with the specified analysis and reference directories, with segmentation enabled:

                mepylome -a /path/to/analysis -r /path/to/references -s

            4. Multiple Options:
            Starts the GUI with additional parameters:

                python mepylome.py \\
                    --analysis_dir /path/to/analysis \\
                    --test_dir /path/to/test \\
                    --annotation /path/to/annotation.xlsx \\
                    --reference_dir /path/to/reference \\
                    --output_dir /path/to/output \\
                    --prep swan \\
                    --cpgs epic \\
                    --n_cpgs 10000 \\
                    --precalculate_cnv \\
                    --host 127.0.0.1 \\
                    --port 8050 \\
                    --debug
            """
        ),
        formatter_class=SmartFormatter,
    )
    parser.add_argument(
        "-a",
        "--analysis_dir",
        type=absolute_path,
        help="Path to the directory containing IDAT files for analysis.",
    )
    parser.add_argument(
        "-t",
        "--test_dir",
        type=absolute_path,
        help=(
            "Directory for test files, including new cases for analysis or "
            "validation. Files uploaded via the GUI will be placed here. If "
            "set to None, the application will automatically use a temporary "
            "directory."
        ),
    )
    parser.add_argument(
        "-A",
        "--annotation",
        type=str,
        help=(
            "Path to an annotation spreadsheet used to map sample files "
            "located in both `analysis_dir` and `test_dir`. One of the "
            "columns must contain the ID corresponding to the IDAT files "
            "(such as SentrixID or ID from files downloaded from GEO). If not "
            "provided, the system will attempt to identify the correct column "
            "automatically. If the annotation file is missing, it will search "
            "for a spreadsheet within the `analysis_dir` if available."
        ),
    )
    parser.add_argument(
        "-r",
        "--reference_dir",
        type=absolute_path,
        help=(
            "Directory containing CNV neutral reference IDAT files. Must be "
            "provided if you wanna generate CNV plots."
        ),
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=absolute_path,
        help=(
            "Path to the directory where output files will be saved. If not "
            "provided, the default directory '/tmp/mepylome/analysis' will be "
            "used."
        ),
    )
    parser.add_argument(
        "-p",
        "--prep",
        type=str,
        choices=["illumina", "swan", "noob"],
        default="illumina",
        help=(
            "Prepreparation method used for methylation microarrays: "
            "'illumina', 'swan', or 'noob'."
        ),
    )
    parser.add_argument(
        "-c",
        "--cpgs",
        type=str,
        default="auto",
        help=(
            """
            Specifies the CpG sites to analyze. Possible values:

                1. A path to a CSV file containing the CpG sites.

                2. A string specifying a predefined array type:

                    - '450k'   : The CpG sites from the Illumina 450k array.
                    - 'epic'   : The CpG sites from the Illumina EPIC array.
                    - 'epicv2' : The CpG sites from the Illumina EPIC v2 array.
                    - 'msa48'  : The CpG sites from the Illumina MSA array.

                3. A '+'-joined string of the options above combining multiple array types, returning the intersection of their CpG sites. For example:

                    - '450k+epic'  : CpG sites both in the 450k and EPIC arrays.
                    - 'epic+epicv2': CpG sites both in the EPIC and EPICv2 arrays.

                4. 'auto': Automatically detects all array types from IDAT files in analysis_dir and returns the intersection of CpG sites. This process may take longer as all files need to be read and, if necessary, decompressed.
            """
        ),
    )
    parser.add_argument(
        "-n",
        "--n_cpgs",
        default=25000,
        type=int,
        help="Number of CpG sites to select for UMAP.",
    )
    parser.add_argument(
        "-C",
        "--precalculate_cnv",
        action="store_true",
        help=(
            "If set, CNV data will be precalculated before the main analysis. "
            "This process takes approximately 2-5 seconds per case initially, "
            "but it will improve performance during runtime by reducing "
            "computation time."
        ),
    )
    parser.add_argument(
        "-l",
        "--no_load_full_betas",
        action="store_false",
        dest="load_full_betas",
        help=(
            "Prevent loading betas for all CpGs into memory setting "
            "`load_full_betas` to False. By default, betas are loaded to "
            "speed up the generation of sequential UMAP plots. Use this "
            "option only if you encounter memory overflow due to insufficient "
            "available memory (3-4 MB per sample)."
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
        choices=["top", "random"],
        default="top",
        help=(
            "Method to select CpG sites ('top' or 'random'). For 'top', CpG "
            "sites are selected based on their variation, taking the most "
            "varying ones. For 'random', CpG sites are randomly selected."
        ),
    )
    parser.add_argument(
        "-H",
        "--host",
        type=str,
        default="localhost",
        help="Host address for the Dash application.",
    )
    parser.add_argument(
        "-P",
        "--port",
        type=int,
        default=8050,
        help="Port number for the Dash application.",
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Flag to enable debug mode for the Dash application.",
    )
    parser.add_argument(
        "-s",
        "--do_seg",
        action="store_true",
        help=(
            "If set, enables segmentation analysis on CNV data and adds "
            "horizontal segmentation lines to the CNV plot. This will take an "
            "additional 2-5 seconds per sample."
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
    parser.add_argument(
        "--version",
        action="version",
        version=f"{get_app_version()}",
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
            "Warning: Since '-no_load_full_betas' is set, 'cpg_selection' has "
            "been automatically changed to 'random'.\n"
        )
        print(msg)

    from mepylome.analysis.methyl import MethylAnalysis

    if cli_args["tutorial"]:
        from mepylome.utils import setup_tutorial_files

        tutorial_dir = Path.home() / "mepylome" / "tutorial"
        cli_args["analysis_dir"] = tutorial_dir / "tutorial_analysis"
        cli_args["reference_dir"] = tutorial_dir / "tutorial_reference"
        if (
            not cli_args["analysis_dir"].exists()
            and not cli_args["reference_dir"].exists()
        ):
            # Download Tutorial IDAT files
            setup_tutorial_files(
                cli_args["analysis_dir"], cli_args["reference_dir"]
            )
        cli_args["load_full_betas"] = True
        cli_args["cpgs"] = "epic"

    cli_args.pop("tutorial", None)

    methyl_analysis = MethylAnalysis(**cli_args)
    methyl_analysis.run_app(open_tab=True)
