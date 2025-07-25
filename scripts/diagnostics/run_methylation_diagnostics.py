"""Pipeline that runs diagnostics on new cases from config and saves reports.

This script executes a diagnostic pipeline defined in a JSON/YAML config file.
It processes IDAT files, applies supervised and unsupervised (UMAP)
classifiers, and saves the resulting reports in the directories containing the
new cases. The pipeline is highly customizable through the config file.
"""

import argparse
import io
import json
import logging
import zipfile
from pathlib import Path

import pandas as pd
import requests
import tomllib

import mepylome
from mepylome import ArrayType, Manifest
from mepylome.analysis import MethylAnalysis
from mepylome.analysis.methyl_clf import make_classifier_report_page
from mepylome.cli import get_app_version
from mepylome.dtypes.manifests import (
    DOWNLOAD_DIR,
    MANIFEST_URL,
    REMOTE_FILENAME,
)

IMG_HEIGHT = 2000
FONTSIZE = 23
IMG_WIDTH = 1000


def load_config(config_path):
    """Load a JSON or YAML file and return its contents as a dictionary."""
    with config_path.open(
        "rb" if config_path.suffix == ".toml" else "r"
    ) as file:
        if config_path.suffix == ".json":
            return json.load(file)
        if config_path.suffix in [".yaml", ".yml"]:
            import yaml

            return yaml.safe_load(file)
        if config_path.suffix == ".toml":
            return tomllib.load(file)
    raise ValueError("Unsupported config format")


def sex_chromosome_cpgs():
    """Returns a set of CpGs on sex chromosomes for all arrays."""
    array_types = ["450k", "epic", "epicv2"]
    sex_cpgs = set()
    for array_type in array_types:
        manifest = Manifest(array_type)
        sex_cpgs.update(
            manifest.data_frame.loc[
                manifest.data_frame.Chromosome.isin([23, 24]), "IlmnID"
            ]
        )
    return sex_cpgs


def generate_default_blacklist(blacklist_path):
    """Returns and caches CpG sites that should be blacklisted."""
    blacklist_path = Path(blacklist_path).expanduser()
    if not blacklist_path.exists():
        print("Generating blacklist. Can take some time...")
        manifest_url = MANIFEST_URL[ArrayType.ILLUMINA_EPIC]
        DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
        response = requests.get(manifest_url)
        html_sucess_ok_code = 200
        if response.status_code == html_sucess_ok_code:
            with zipfile.ZipFile(io.BytesIO(response.content)) as thezip:
                thezip.extractall(DOWNLOAD_DIR)
        else:
            msg = f"Failed to download the file: {response.status_code}"
            raise RuntimeError(msg)
        csv_path = DOWNLOAD_DIR / REMOTE_FILENAME[ArrayType.ILLUMINA_EPIC]
        manifest_df = pd.read_csv(csv_path, skiprows=7, low_memory=False)
        flagged_cpgs = manifest_df[
            manifest_df["MFG_Change_Flagged"].fillna(False)
        ]["IlmnID"]
        flagged_cpgs = list(set(flagged_cpgs) | sex_chromosome_cpgs())
        pd.DataFrame(flagged_cpgs).to_csv(
            blacklist_path, index=False, header=False
        )
        csv_path.unlink()
    blacklist_df = pd.read_csv(blacklist_path, header=None)
    return set(blacklist_df.iloc[:, 0])


def unsupervised_classifier(analysis, dataset_name, dataset_config):
    """Generate and save UMAP plots for all 'test_ids'."""
    # Collect only those IDs where outputs are missing
    umap_jobs = []
    for id_ in analysis.idat_handler.test_ids:
        basepath = analysis.idat_handler.test_id_to_path[id_].parent
        base_filename = f"{id_}_{dataset_name}"

        jpg_path = basepath / (base_filename + "_umap.jpg")
        html_path = basepath / (base_filename + "_umap.html")

        if not (jpg_path.exists() and html_path.exists()):
            umap_jobs.append((id_, html_path, jpg_path))

    def generate_and_save_umap(id_, html_path, jpg_path):
        """Generate and save UMAP plots for a given sample ID."""
        analysis.cnv_id = id_
        analysis.test_ids = [id_]

        # Save UMAP to disk
        analysis.make_umap()
        analysis.umap_plot.write_image(
            jpg_path,
            format="jpg",
            width=IMG_HEIGHT,
            height=IMG_WIDTH,
            scale=1,
        )
        analysis.umap_plot.write_html(html_path)

    for job in umap_jobs:
        generate_and_save_umap(*job)


def supervised_classifier(analysis, dataset_name, dataset_config):
    """Run supervised classifiers on test samples and save as HTML reports."""
    ids = analysis.idat_handler.test_ids

    # Determine which IDs do not yet have reports
    id_to_path = {}
    for id_ in ids:
        filename = f"{id_}_{dataset_name}_classifiers.html"
        path = analysis.idat_handler.test_id_to_path[id_].parent / filename
        if not path.exists():
            id_to_path[id_] = path

    if not id_to_path:
        return

    uncalculated_ids = list(id_to_path.keys())

    clf_list = dataset_config["classifier_list"]
    clf_results = analysis.classify(
        ids=uncalculated_ids,
        clf_list=clf_list,
    )
    id_to_report = {id_: [] for id_ in uncalculated_ids}

    for clf_result in clf_results:
        for id_, report in zip(uncalculated_ids, clf_result.reports["html"]):
            id_to_report[id_].append(report)

    app_version = get_app_version()
    clf_version = dataset_config.get("version")
    version_str = f"v{clf_version} " if clf_version else ""
    version_str += f"(Mepylome v{app_version})"
    clf_name = dataset_config.get("name", dataset_name)

    for id_, path in id_to_path.items():
        html_str = make_classifier_report_page(
            reports=id_to_report[id_],
            title=f"{clf_name} Classifier {version_str}",
        )
        path.write_text(html_str)


def make_cnv(analysis, dataset_name, dataset_config):
    """Generates CNV Plot and saves it to corresponding directory."""
    ids = analysis.idat_handler.test_ids
    analysis.precompute_cnvs(ids)
    for id_ in ids:
        save_dir = analysis.idat_handler.test_id_to_path[id_].parent
        jpg_path = save_dir / f"{id_}_{dataset_name}_cnv.jpg"
        html_path = save_dir / f"{id_}_{dataset_name}_cnv.html"
        if jpg_path.exists() and html_path.exists():
            continue
        analysis.make_cnv_plot(id_)
        cnv_plot = analysis.cnv_plot
        cnv_plot.update_layout(
            yaxis_range=[-2.0, 2.0],
            font={"size": FONTSIZE},
            margin={"t": 50},
        )
        cnv_plot.write_image(
            jpg_path,
            format="jpg",
            width=IMG_HEIGHT,
            height=IMG_WIDTH,
            scale=1,
        )
        cnv_plot.write_html(html_path)


def run_single_diagnostics(dataset_name, dataset_config, defaults):
    """Run methylation analysis for a single dataset."""
    print()
    logging.info("-- Starting diagnostics for %s --", dataset_name)

    dataset_config = {**defaults, **dataset_config}
    dataset_config["methyl_analysis"] = {
        **defaults["methyl_analysis"],
        **dataset_config["methyl_analysis"],
    }

    init_params = dataset_config.get("methyl_analysis", {})

    analysis = MethylAnalysis(**init_params)

    annotation_column = dataset_config.get("annotation_diagnosis_column")
    if annotation_column:
        analysis.idat_handler.selected_columns = annotation_column

    if dataset_config.get("do_classify"):
        print()
        logging.info("Starting supervised classifier for %s", dataset_name)
        supervised_classifier(analysis, dataset_name, dataset_config)

    if dataset_config.get("do_cnv"):
        print()
        logging.info("Starting CNV for %s", dataset_name)
        make_cnv(analysis, dataset_name, dataset_config)

    if dataset_config.get("do_umap"):
        print()
        logging.info("Starting UMAP for %s", dataset_name)
        unsupervised_classifier(analysis, dataset_name, dataset_config)

    del analysis
    mepylome.clear_cache()


def run_diagnostics(config_path, test_dir=None):
    """Run diagnostics for each dataset based on the provided config."""
    config_dict = load_config(Path(config_path))

    defaults = config_dict.get("defaults", {})

    # Inject test_dir if provided
    if test_dir:
        defaults.setdefault("methyl_analysis", {})["test_dir"] = test_dir

    # Choose default CpGs  that should be blacklisted
    blacklist_path = defaults.get("cpg_blacklist_path")
    if blacklist_path:
        defaults["methyl_analysis"]["cpg_blacklist"] = (
            generate_default_blacklist(blacklist_path)
        )
    defaults.pop("cpg_blacklist_path", None)

    for dataset_name, dataset_config in config_dict[
        "diagnostic_config"
    ].items():
        run_single_diagnostics(dataset_name, dataset_config, defaults)


def main():
    """Start diagnostics."""
    parser = argparse.ArgumentParser(
        description="Run methylation diagnostics."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "config.toml",
        help=(
            "Path to the config file (default: 'config.toml' in the current "
            "directory)",
        ),
    )
    parser.add_argument(
        "--test_dir",
        type=Path,
        required=True,
        help="Path to the directory with the new IDAT cases",
    )
    args = parser.parse_args()

    try:
        config_path = args.config
        test_dir = args.test_dir
        logging.info("Starting diagnostics with config: %s", config_path)
        run_diagnostics(config_path, test_dir)
        logging.info("Diagnostics completed successfully.")
    except Exception as e:
        logging.error("An error occurred: %s", e)
        raise


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(module)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    main()
