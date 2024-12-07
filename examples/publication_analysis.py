# %% [markdown]
"""# Script for Mepylome Publication Analysis.

This notebook/script performs the analysis presented in the Mepylome
publication. It automates the download of required datasets and performs the
corresponding data analysis.

### Usage
- All downloaded datasets and outputs will be saved in the
  `~/Documents/mepylome` directory.
- Run the script step-by-step for a clear understanding of the workflow and
  outputs.

### Recommended Environment
- **Operating System**: Ubuntu 20.04.6
- **Python Version**: 3.12

### Publication Title
**Mepylome: A User-Friendly Open-Source Toolkit for DNA-Methylation Analysis in
Tumor Diagnostics**

**Script Author**: Jon Brugger
"""

# %%
import io
import itertools
import tarfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from PIL import Image

from mepylome import ArrayType, Manifest
from mepylome.analysis import MethylAnalysis
from mepylome.dtypes.manifests import (
    DOWNLOAD_DIR,
    MANIFEST_URL,
    REMOTE_FILENAME,
)
from mepylome.utils import ensure_directory_exists
from mepylome.utils.files import download_file, download_geo_probes

FONTSIZE = 23

file_urls = {
    "salivary_gland_tumors": {
        "xlsx": "https://ars.els-cdn.com/content/image/1-s2.0-S0893395224002059-mmc4.xlsx",
        "xlsx_name": "mmc4.xlsx",
        "idat": "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE243075&format=file",
        "idat_name": "GSE243075",
    },
    "soft_tissue_tumors": {
        "xlsx": "https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-020-20603-4/MediaObjects/41467_2020_20603_MOESM4_ESM.xlsx",
        "xlsx_name": "41467_2020_20603_MOESM4_ESM.xlsx",
        "idat": "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE140686&format=file",
        "idat_name": "GSE140686",
    },
    "sinonasal_tumors": {
        "xlsx": "https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-022-34815-3/MediaObjects/41467_2022_34815_MOESM6_ESM.xlsx",
        "xlsx_name": "41467_2022_34815_MOESM6_ESM.xlsx",
        "idat": "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE196228&format=file",
        "idat_name": "GSE196228",
    },
    "head_and_neck_scc": {
        "xlsx": "https://www.science.org/doi/suppl/10.1126/scitranslmed.aaw8513/suppl_file/aaw8513_data_file_s1.xlsx",
        "xlsx_name": "aaw8513_data_file_s1.xlsx",
        "idat": "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE124633&format=file",
        "idat_name": "GSE124633",
    },
}

cn_neutral_probes = [
    "GSM4180453_201904410008_R06C01",
    "GSM4180454_201904410008_R05C01",
    "GSM4180455_201904410008_R04C01",
    "GSM4180456_201904410008_R03C01",
    "GSM4180457_201904410008_R02C01",
    "GSM4180458_201904410008_R01C01",
    "GSM4180459_201904410007_R08C01",
    "GSM4180460_201904410007_R07C01",
    "GSM4180741_201247480004_R05C01",
    "GSM4180742_201247480004_R04C01",
    "GSM4180743_201247480004_R03C01",
    "GSM4180751_201194010006_R01C01",
    "GSM4180909_200394870074_R04C02",
    "GSM4180910_200394870074_R03C02",
    "GSM4180911_200394870074_R02C02",
    "GSM4180912_200394870074_R01C02",
    "GSM4180913_200394870074_R05C01",
    "GSM4180914_200394870074_R04C01",
    "GSM4181456_203049640041_R03C01",
    "GSM4181509_203049640040_R07C01",
    "GSM4181510_203049640040_R08C01",
    "GSM4181511_203049640041_R01C01",
    "GSM4181512_203049640041_R02C01",
    "GSM4181513_203049640041_R04C01",
    "GSM4181514_203049640041_R05C01",
    "GSM4181515_203049640041_R06C01",
    "GSM4181516_203049640041_R07C01",
    "GSM4181517_203049640041_R08C01",
]

mepylome_dir = Path("~/Documents/mepylome").expanduser()
data_dir = mepylome_dir / "data"
output_dir = mepylome_dir / "out"
tests_dir = mepylome_dir / "tests"
reference_dir = mepylome_dir / "cn_neutral_idats"

ensure_directory_exists(data_dir)
ensure_directory_exists(tests_dir)
ensure_directory_exists(output_dir)
ensure_directory_exists(reference_dir)


def generate_blacklist_cpgs():
    """Returns and caches CpG sites that should be blacklisted."""
    print("Generate blacklist. Can take some time...")
    blacklist_path = data_dir / "cpg_blacklist.csv"
    if not blacklist_path.exists():
        manifest_url = MANIFEST_URL[ArrayType.ILLUMINA_EPIC]
        ensure_directory_exists(DOWNLOAD_DIR)
        response = requests.get(manifest_url)
        html_sucess_ok_code = 200
        if response.status_code == html_sucess_ok_code:
            with zipfile.ZipFile(io.BytesIO(response.content)) as thezip:
                thezip.extractall(DOWNLOAD_DIR)
        else:
            raise Exception(
                f"Failed to download the file: {response.status_code}"
            )
        csv_path = DOWNLOAD_DIR / REMOTE_FILENAME[ArrayType.ILLUMINA_EPIC]
        manifest_df = pd.read_csv(csv_path, skiprows=7)
        flagged_cpgs = manifest_df[
            manifest_df["MFG_Change_Flagged"].fillna(False)
        ]["IlmnID"]
        flagged_cpgs.to_csv(blacklist_path, index=False, header=False)
        csv_path.unlink()
    blacklist = pd.read_csv(blacklist_path, header=None)
    return set(blacklist.iloc[:, 0])


def sex_chromosome_cpgs():
    """Returns CpGs on sex chromosomes for EPIC and 450k arrays."""
    manifest = Manifest("epic")
    sex_cpgs_epic = manifest.data_frame[
        manifest.data_frame.Chromosome.isin([23, 24])
    ].IlmnID
    manifest = Manifest("450k")
    sex_cpgs_450k = manifest.data_frame[
        manifest.data_frame.Chromosome.isin([23, 24])
    ].IlmnID
    return set(sex_cpgs_epic) | set(sex_cpgs_450k)


def extract_tar(tar_path, output_directory):
    """Extracts tar file under 'tar_path' to 'output_directory'."""
    output_directory.mkdir(exist_ok=True)
    with tarfile.open(tar_path, "r") as tar:
        tar.extractall(path=output_directory, filter=None)
        print(f"Extracted {tar_path} to {output_directory}")


def calculate_cn_summary(class_):
    """Calculates and saves CN summary plots."""
    df_class = analysis.idat_handler.samples_annotated[class_]
    plot_list = []
    all_classes = sorted(df_class.unique())
    for methyl_class in all_classes:
        df_index = df_class == methyl_class
        sample_ids = df_class.index[df_index]
        plot, df_cn_summary = analysis.cn_summary(sample_ids)
        plot.update_layout(
            title=f"{methyl_class}",
            title_x=0.5,
            yaxis_title="Proportion of CNV gains/losses",
        )
        plot.update_layout(
            title_font_size=FONTSIZE + 3,
            yaxis_title_font_size=FONTSIZE - 2,
        )
        plot_list.append(plot)
    png_paths = [
        output_dir / f"{analysis_dir.name}-cn_summary-{x}.png"
        for x in all_classes
    ]
    for path, fig in zip(png_paths, plot_list):
        fig.write_image(path)
    images = [Image.open(path) for path in png_paths]
    width, height = images[0].size
    n_columns = 4
    n_images = len(images)
    n_rows = (n_images + n_columns - 1) // n_columns
    total_width = width * n_columns
    total_height = height * n_rows
    new_image = Image.new("RGB", (total_width, total_height), (255, 255, 255))
    for index, img in enumerate(images):
        row = index // n_columns
        col = index % n_columns
        x = col * width
        y = row * height
        new_image.paste(img, (x, y))
    new_image.save(output_dir / f"{analysis_dir.name}-cn_summary.png")


# Chose CpG list that should be blacklisted
blacklist = generate_blacklist_cpgs() | sex_chromosome_cpgs()


# %% [markdown]
# We use the control probes of Koelsche (2021;
# https://doi.org/10.1038/s41467-020-20603-4) as copy neutral reference set.
# This should contain both fresh-frozen tissue as well as FFPE samples.


# %%
download_geo_probes(reference_dir, cn_neutral_probes)


# %% [markdown]
# Salavary Gland Tumors
# =====================

# %%
tumor_site = "salivary_gland_tumors"
analysis_dir = data_dir / tumor_site
test_dir = tests_dir / tumor_site
ensure_directory_exists(test_dir)
idat_dir = analysis_dir / file_urls[tumor_site]["idat_name"]
if not idat_dir.exists():
    excel_path = analysis_dir / file_urls[tumor_site]["xlsx_name"]
    download_file(file_urls[tumor_site]["xlsx"], excel_path)
    # Deletes the first 2 (useless) rows from the excel file.
    pd.read_excel(excel_path, skiprows=2).to_excel(excel_path, index=False)
    idat_tar_path = analysis_dir / "tmp_idats.tar"
    download_file(file_urls[tumor_site]["idat"], idat_tar_path)
    extract_tar(idat_tar_path, idat_dir)
    idat_tar_path.unlink()


analysis = MethylAnalysis(
    analysis_dir=analysis_dir,
    reference_dir=reference_dir,
    output_dir=output_dir,
    test_dir=test_dir,
    n_cpgs=25000,
    load_full_betas=True,
    overlap=False,
    cpg_blacklist=blacklist,
    debug=True,
    do_seg=True,
    umap_parms={
        "n_neighbors": 8,
        "metric": "manhattan",
        "min_dist": 0.3,
    },
)

analysis.set_betas()
analysis.idat_handler.selected_columns = ["Methylation class"]


# Start GUI
analysis.make_umap()
analysis.run_app(open_tab=True)

# Save CNV example
analysis.make_cnv_plot("206842050057_R06C01")
cnv_plot = analysis.cnv_plot
cnv_plot.update_layout(
    yaxis_range=[-1.1, 1.1],
    font={"size": FONTSIZE},
    margin={"t": 50},
)
cnv_plot.write_image(
    output_dir / f"{analysis_dir.name}-cnv_plot.jpg",
    format="jpg",
    width=2000,
    height=1000,
    scale=2,
)

# TODO para tuning

# Make CN summary plots
analysis.precompute_cnvs()
calculate_cn_summary("Methylation class")


# %% [markdown]
# Sinonasal Tumors
# ================

# %%
tumor_site = "sinonasal_tumors"
analysis_dir = data_dir / tumor_site
test_dir = tests_dir / tumor_site
ensure_directory_exists(test_dir)
idat_dir = analysis_dir / file_urls[tumor_site]["idat_name"]
if not idat_dir.exists():
    excel_path = analysis_dir / file_urls[tumor_site]["xlsx_name"]
    download_file(file_urls[tumor_site]["xlsx"], excel_path)
    idat_tar_path = analysis_dir / "tmp_idats.tar"
    download_file(file_urls[tumor_site]["idat"], idat_tar_path)
    extract_tar(idat_tar_path, idat_dir)
    idat_tar_path.unlink()


analysis = MethylAnalysis(
    analysis_dir=analysis_dir,
    reference_dir=reference_dir,
    output_dir=output_dir,
    n_cpgs=25000,
    load_full_betas=True,
    overlap=False,
    cpg_blacklist=blacklist,
    debug=True,
    do_seg=True,
    umap_parms={
        "n_neighbors": 8,
        "metric": "manhattan",
        "min_dist": 0.3,
    },
)

# Start GUI
analysis.idat_handler.selected_columns = ["Methylation class"]
analysis.make_umap()
analysis.run_app(open_tab=True)

# Save CNV example
analysis.make_cnv_plot("9406921039_R01C02")
cnv_plot = analysis.cnv_plot
cnv_plot.update_layout(
    yaxis_range=[-1.1, 1.1],
    font={"size": FONTSIZE},
    margin={"t": 50},
)
cnv_plot.write_image(
    output_dir / f"{analysis_dir.name}-cnv_plot.jpg",
    format="jpg",
    width=2000,
    height=1000,
    scale=2,
)

# Make CN summary plots
analysis.precompute_cnvs()
calculate_cn_summary("Methylation class")

# Validate GUI superviseder lerner
ids = analysis.idat_handler.ids
clf_out = analysis.classify(
    ids=ids,
    clf_list=[
        "none-kbest-et",
        "none-kbest-lr",
        "none-kbest-rf",
        "none-kbest-svc_rbf",
        "none-pca-lr",
        "none-pca-et",
        "none-none-knn",
    ],
)

# Prinf reports of all clasifiers
for clf_result in clf_out:
    print(clf_result.reports[0])

best_clf = max(
    clf_out, key=lambda result: np.mean(result.metrics["accuracy_scores"])
)
print("Most accurate classifier:")
print(best_clf.reports[0])


clf_out = analysis.classify(ids=ids, clf_list="none-lda-et")
print(clf_out.reports[0])

scalers = [
    "minmax",
    "power",
    "quantile",
    "quantile_normal",
    "robust",
    "std",
]

selectors = [
    "lasso",
    "lda",
    "mutual_info",
    "none",
    "pca",
]

classifiers = [
    "ada",
    "bag",
    "dt",
    "gb",
    "gp",
    "hgb",
    "knn",
    "lda",
    "mlp",
    "nb",
    "none",
    "perceptron",
    "qda",
    "rf",
    "ridge",
    "sgd",
    "stacking",
    "svc_linear",
    "svc_rbf",
]
combinations = itertools.product(scalers, selectors, classifiers)


# %% [markdown]
# Soft Tissue Tumors
# ==================

# %%
tumor_site = "soft_tissue_tumors"
analysis_dir = data_dir / tumor_site
test_dir = tests_dir / tumor_site
ensure_directory_exists(test_dir)
idat_dir = analysis_dir / file_urls[tumor_site]["idat_name"]
if not idat_dir.exists():
    excel_path = analysis_dir / file_urls[tumor_site]["xlsx_name"]
    download_file(file_urls[tumor_site]["xlsx"], excel_path)
    idat_tar_path = analysis_dir / "tmp_idats.tar"
    download_file(file_urls[tumor_site]["idat"], idat_tar_path)
    extract_tar(idat_tar_path, idat_dir)
    idat_tar_path.unlink()


analysis = MethylAnalysis(
    analysis_dir=analysis_dir,
    reference_dir=reference_dir,
    output_dir=output_dir,
    n_cpgs=25000,
    load_full_betas=True,
    overlap=False,
    cpg_blacklist=blacklist,
    debug=True,
    do_seg=True,
    umap_parms={
        "n_neighbors": 8,
        "metric": "manhattan",
        "min_dist": 0.3,
    },
)

# Start GUI
analysis.idat_handler.selected_columns = ["Methylation Class Name"]
analysis.make_umap()
analysis.run_app(open_tab=True)

# Save CNV example
analysis.make_cnv_plot("3999112131_R05C01")
cnv_plot = analysis.cnv_plot
cnv_plot.update_layout(
    yaxis_range=[-1.1, 1.1],
    font={"size": FONTSIZE},
    margin={"t": 50},
)
cnv_plot.write_image(
    output_dir / f"{analysis_dir.name}-cnv_plot.jpg",
    format="jpg",
    width=2000,
    height=1000,
    scale=2,
)

# Make CN summary plots
analysis.precompute_cnvs()
calculate_cn_summary("Methylation class")


# %% [markdown]
# Squamous Cell Carcinoma
# =======================


# %%
tumor_site = "head_and_neck_scc"
analysis_dir = data_dir / tumor_site
test_dir = tests_dir / tumor_site
ensure_directory_exists(test_dir)
idat_dir = analysis_dir / file_urls[tumor_site]["idat_name"]
if not idat_dir.exists():
    idat_tar_path = analysis_dir / "tmp_idats.tar"
    download_file(file_urls[tumor_site]["idat"], idat_tar_path)
    extract_tar(idat_tar_path, idat_dir)
    idat_tar_path.unlink()


analysis = MethylAnalysis(
    analysis_dir=analysis_dir,
    reference_dir=reference_dir,
    output_dir=output_dir,
    test_dir=test_dir,
    n_cpgs=25000,
    load_full_betas=True,
    overlap=False,
    cpg_blacklist=blacklist,
    debug=True,
    do_seg=True,
    umap_parms={
        "n_neighbors": 8,
        "metric": "manhattan",
        "min_dist": 0.3,
    },
)

analysis.set_betas()
analysis.idat_handler.selected_columns = ["Methylation class"]
