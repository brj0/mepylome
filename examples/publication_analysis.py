# %% [markdown]
#
# <img alt="Mepylome Logo" src="https://raw.githubusercontent.com/brj0/mepylome/main/mepylome/data/assets/mepylome.svg" width="300">
#
# Mepylome: A Toolkit for DNA-Methylation Analysis in Tumor Diagnostics
# =====================================================================
#
# This notebook automates the analysis outlined in the Mepylome publication,
# performing all necessary steps from downloading datasets to executing the
# analyses.
#
#
# ### Usage
#
# - All datasets and outputs are saved in `~/mepylome`.
# - Follow the notebook/script step-by-step for an in-depth understanding of
#   the workflow and results.
#
#
# ### System Requirements
#
# - *Operating System*: Ubuntu 20.04.6
# - *Python Version*: 3.12
#
#
# ### Reference Publication (will follow)
#
# - *Title*: Mepylome: A User-Friendly Open-Source Toolkit for DNA-Methylation
#   Analysis in Tumor Diagnostics
# - *Author*: Jon Brugger et al.
#
#
# ### Run This Notebook in Google Colab
#
# You can quickly open and run this notebook in Google Colab without any setup
# by clicking the link below.
#
# **Note**: The graphical user interface (GUI) features won't run in Colab, but
# the rest of the analysis will work as expected.
#
# [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/brj0/mepylome/blob/main/examples/publication_analysis.ipynb)
#
#
# This notebook was automatically generated from the corresponding py-file
# with:
#
# ```bash
# jupytext --to ipynb publication_analysis.py
# ```


# -----------------------------------------------------------------------------

# %% [markdown]
# ## Contents
# 0. **[Initialization](#0.-Initialization)**
# 1. **[Salivary Gland Tumors](#1.-Salivary-Gland-Tumors)**
# 2. **[Soft Tissue Tumors](#2.-Soft-Tissue-Tumors)**
# 3. **[Squamous Cell Carcinoma](#3.-Squamous-Cell-Carcinoma)**


# %% [markdown]
# -----------------------------------------------------------------------------
# <a name="0.-Initialization"></a>
# ## 0. Initialization
#
# ### Install Required Packages
#
# To run the analysis, install the following Python packages:
# - `mepylome` – the main toolkit for DNA-methylation analysis.
# - `linear_segment` – used for segmentation calculations in CNV plots.
# - `kaleido` for saving plots.
#
# Install these packages using the commands below:

# %% language="bash"
# pip install mepylome
# pip install linear_segment
# pip install -U kaleido

# %% [markdown]
# ### Core Imports, Configuration and main Functions

# %%
import io
import os
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
from mepylome.utils.files import (
    download_file,
    download_geo_probes,
    get_resource_path,
)

# Define output font size for plots
FONTSIZE = 23
GEO_URL = "https://www.ncbi.nlm.nih.gov/geo/download/?acc={acc}&format=file"

# Define dataset URLs and filenames
datasets = {
    "salivary_gland_tumors": {
        "xlsx": "https://ars.els-cdn.com/content/image/1-s2.0-S0893395224002059-mmc4.xlsx",
        "geo_id": ["GSE243075"],
    },
    "soft_tissue_tumors": {
        "xlsx": "https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-020-20603-4/MediaObjects/41467_2020_20603_MOESM4_ESM.xlsx",
        "geo_id": ["GSE140686"],
    },
    "sinonasal_tumors": {
        "xlsx": "https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-022-34815-3/MediaObjects/41467_2022_34815_MOESM6_ESM.xlsx",
        "geo_id": ["GSE196228"],
    },
    "scc": {
        "xlsx": "https://www.science.org/doi/suppl/10.1126/scitranslmed.aaw8513/suppl_file/aaw8513_data_file_s1.xlsx",
        "geo_id": [
            "GSE124052",
            "GSE66836",
            "GSE79556",
            "GSE85566",
            "GSE87053",
            "GSE95036",
        ],
    },
}

# Determine basic storage directory depending on platform
if "COLAB_GPU" in os.environ:
    # Google Colab
    mepylome_dir = Path("/content/mepylome")
elif os.path.exists("/mnt/bender"):
    # Bender-specific path
    mepylome_dir = Path("/mnt/bender/mepylome")
else:
    # Default for local Linux or other environments
    mepylome_dir = Path.home() / "mepylome"

# Ensure the directory exists
os.makedirs(mepylome_dir, exist_ok=True)
os.environ["MEPYLOME_DIR"] = str(mepylome_dir)
print(f"Data will be stored in: {mepylome_dir}")

data_dir = mepylome_dir / "data"
output_dir = mepylome_dir / "out"
reference_dir = mepylome_dir / "cn_neutral_idats"
validation_dir = mepylome_dir / "validation_data"

ensure_directory_exists(data_dir)
ensure_directory_exists(output_dir)
ensure_directory_exists(reference_dir)
ensure_directory_exists(validation_dir)

# Main Funktions


def extract_tar(tar_path, output_directory):
    """Extracts tar file under 'tar_path' to 'output_directory'."""
    output_directory.mkdir(exist_ok=True)
    with tarfile.open(tar_path, "r") as tar:
        tar.extractall(path=output_directory, filter=None)
        print(f"Extracted {tar_path} to {output_directory}")


def download_from_geo_and_untar(analysis_dir, geo_ids):
    """Downloads all missing GEO files and untars them."""
    for geo_id in geo_ids:
        idat_dir = analysis_dir / geo_id
        if idat_dir.exists():
            print(f"Data for GEO ID {geo_id} already exists, skipping.")
            continue
        try:
            tar_path = analysis_dir / f"{geo_id}.tar"
            geo_url = GEO_URL.format(acc=geo_id)
            download_file(geo_url, tar_path)
            extract_tar(tar_path, idat_dir)
            tar_path.unlink()
        except Exception as e:
            print(f"Error processing GEO ID {geo_id}: {e}")


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


# %% [markdown]
# ### Blacklist Generation for CpG Sites
#
# Some CpG sites should be excluded from the analysis. Here we choose probes
# flagged with `MFG_Change_Flagged` that should be excluded according to the
# manifest and those that are on sex chromosomes.


# %%
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


# Choose CpG list that should be blacklisted
blacklist = generate_blacklist_cpgs() | sex_chromosome_cpgs()

# %% [markdown]
# ### Copy-Neutral Reference Probes
#
# To ensure accurate analysis, we utilize control probes from the [Koelsche2021
# study](https://doi.org/10.1038/s41467-020-20603-4). These probes are stored
# in the designated reference directory `reference_dir`.
#
# **Best Practices**:
# - Include both fresh-frozen and FFPE (formalin-fixed paraffin-embedded)
#   samples in the copy-neutral reference set for optimal results.

# %%
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

download_geo_probes(reference_dir, cn_neutral_probes)


# %% [markdown]
# -----------------------------------------------------------------------------
# <a name="1.-Salivary-Gland-Tumors"></a>
# ## 1. Salivary Gland Tumors
#
# This section replicates the methylation analysis performed in the study by
# [Jurmeister2024](https://doi.org/10.1016/j.modpat.2024.100625). To begin, we
# download the required data and organize it within the designated directories.

# %%
# Initialize directories.
tumor_site = "salivary_gland_tumors"
analysis_dir = data_dir / tumor_site
test_dir = validation_dir / tumor_site

ensure_directory_exists(test_dir)
ensure_directory_exists(analysis_dir)

# Download the the annotation spreadsheet.
if not (excel_path := analysis_dir / f"{tumor_site}-annotation.xlsx").exists():
    download_file(datasets[tumor_site]["xlsx"], excel_path)
    # Deletes the first 2 rows (useless description).
    pd.read_excel(excel_path, skiprows=2).to_excel(excel_path, index=False)

# Download the IDAT files.
download_from_geo_and_untar(analysis_dir, datasets[tumor_site]["geo_id"])


# %% [markdown]
# ### Create the Methylation Analysis Object
#
# The `MethylAnalysis` object serves as the main interface for performing DNA
# methylation analysis. Key parameters such as the directory structure, number
# of CpG sites, and UMAP settings are configured here.

# %%
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


# %% [markdown]
# ### Load Beta Values
#
# Reads and processes beta values from the provided dataset. This step can also
# be performed interactively within the GUI.

# %%
analysis.set_betas()


# %% [markdown]
# ### Generate UMAP Plot
#
# Set the columns used for coloring the UMAP plot before initiating the
# dimensionality reduction process. The UMAP algorithm produces a visual
# representation of the sample clusters, which is stored as a Plotly object in
# `analysis.umap_plot`.

# %%
analysis.idat_handler.selected_columns = ["Methylation class"]
analysis.make_umap()

# Show the results
print(analysis.umap_df)
analysis.umap_plot.show()

# %% [markdown]
# ### Launch the Analysis GUI
#
# Initializes an interactive GUI for further exploration of the methylation
# data.
#
# **Note**: This step is only supported in local environments (not in
# cloud-based platforms like Google Colab or Binder).

# %%
analysis.run_app(open_tab=True)


# %% [markdown]
# ### Generate and Save CNV Plot
#
# Creates a copy number variation (CNV) plot for a specified sample and saves
# the output as a high-resolution image.

# %%
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

# %% [markdown]
# ### Generate CNV Summary Plots
#
# In addition to individual CNV plots, this step computes summary plots to
# highlight genomic alterations across multiple samples.
#
# **Note**:
# Generating all copy number variation (CNV) plots can be resource-intensive.
# The process may take up to 30 minutes, depending on the computational
# resources available.

# %%
analysis.precompute_cnvs()
calculate_cn_summary("Methylation class")

# %% [markdown]
# ### Supervised Classifier Validation
#
# The next step involves validating various supervised classification
# algorithms to evaluate their performance on the dataset. This process helps
# identify the most accurate model for methylation-based classification.
#
# **Note**:
# Training can be resource-intensive. The process may take up to 10 minutes,
# depending on the computational resources available.

# %%
# Validate supervised classifiers
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

# Print reports for all classifier for the first sample
for clf_result in clf_out:
    print(clf_result.reports[0])

# Identify and display the best classifier
best_clf = max(
    clf_out, key=lambda result: np.mean(result.metrics["accuracy_scores"])
)
print("Most accurate classifier:")
print(best_clf.reports[0])


# %% [markdown]
# -----------------------------------------------------------------------------
# <a name="2.-Soft-Tissue-Tumors"></a>
# ## 2. Soft Tissue Tumors
#
# This section replicates the methylation analysis performed in the study by
# [Koelsche2021 study](https://doi.org/10.1038/s41467-020-20603-4). To begin,
# we download the required data and organize it within the designated
# directories.

# %%
# Initialize directories.
tumor_site = "soft_tissue_tumors"
analysis_dir = data_dir / tumor_site
test_dir = validation_dir / tumor_site

ensure_directory_exists(test_dir)
ensure_directory_exists(analysis_dir)

# Download the the annotation spreadsheet.
if not (excel_path := analysis_dir / f"{tumor_site}-annotation.xlsx").exists():
    download_file(datasets[tumor_site]["xlsx"], excel_path)

# Download the IDAT files.
download_from_geo_and_untar(analysis_dir, datasets[tumor_site]["geo_id"])


# %% [markdown]
# ### Create the Methylation Analysis Object
#
# The `MethylAnalysis` object serves as the main interface for performing DNA
# methylation analysis. Key parameters such as the directory structure, number
# of CpG sites, and UMAP settings are configured here.

# %%
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

# %% [markdown]
# ### Load Beta Values
#
# Reads and processes beta values from the provided dataset. This step can also
# be performed interactively within the GUI.

# %%
analysis.set_betas()


# %% [markdown]
# ### Generate UMAP Plot
#
# Set the columns used for coloring the UMAP plot before initiating the
# dimensionality reduction process. The UMAP algorithm produces a visual
# representation of the sample clusters, which is stored as a Plotly object in
# `analysis.umap_plot`.

# %%
analysis.idat_handler.selected_columns = ["Methylation Class Name"]
analysis.make_umap()

# Show the results
print(analysis.umap_df)
analysis.umap_plot.show()

# %% [markdown]
# ### Launch the Analysis GUI
#
# Initializes an interactive GUI for further exploration of the methylation
# data.
#
# **Note**: This step is only supported in local environments (not in
# cloud-based platforms like Google Colab or Binder).

# %%
analysis.run_app(open_tab=True)


# %% [markdown]
# ### Generate and Save CNV Plot
#
# Creates a copy number variation (CNV) plot for a specified sample and saves
# the output as a high-resolution image.

# %%
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

# %% [markdown]
# ### Generate CNV Summary Plots
#
# In addition to individual CNV plots, this step computes summary plots to
# highlight genomic alterations across multiple samples.
#
# **Note**:
# Generating all copy number variation (CNV) plots can be resource-intensive.
# The process may take up to 30 minutes, depending on the computational
# resources available.

# %%
analysis.precompute_cnvs()
calculate_cn_summary("Methylation class")

# %% [markdown]
# ### Supervised Classifier Validation
#
# The next step involves validating various supervised classification
# algorithms to evaluate their performance on the dataset. This process helps
# identify the most accurate model for methylation-based classification.
#
# **Note**:
# Training can be resource-intensive. The process may take up to 10 minutes,
# depending on the computational resources available.

# %%
# Validate supervised classifiers
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

# Print reports for all classifier for the first sample
for clf_result in clf_out:
    print(clf_result.reports[0])

# Identify and display the best classifier
best_clf = max(
    clf_out, key=lambda result: np.mean(result.metrics["accuracy_scores"])
)
print("Most accurate classifier:")
print(best_clf.reports[0])


# %% [markdown]
# -----------------------------------------------------------------------------
# <a name="3.-Squamous-Cell-Carcinoma"></a>
# ## 3. Squamous Cell Carcinoma

# %% [markdown]
# Step 1: Download and setup GDC client

# %% language="bash"
# # Define the variables
# GDC_CLIENT_URL="https://gdc.cancer.gov/system/files/public/file/gdc-client_2.3_Ubuntu_x64-py3.8-ubuntu-20.04.zip"
# GDC_CLIENT_DIR="$HOME/mepylome"
# GDC_CLIENT_BIN="$GDC_CLIENT_DIR/gdc-client"
#
# # Download and set up the GDC client
# if [ ! -f "$GDC_CLIENT_BIN" ]; then
#     mkdir -p "$GDC_CLIENT_DIR"
#     cd "$GDC_CLIENT_DIR"
#     echo "Downloading GDC client..."
#     wget -q "$GDC_CLIENT_URL" -P "$GDC_CLIENT_DIR"
#     unzip -qo "$GDC_CLIENT_DIR/*.zip" -d "$GDC_CLIENT_DIR"
#     unzip -qo "$GDC_CLIENT_DIR/*.zip" -d "$GDC_CLIENT_DIR" > /dev/null 2>&1
#     rm -f "$GDC_CLIENT_DIR"/gdc-client*.zip
#     echo "GDC client binary downloaded and set up at $GDC_CLIENT_BIN"
# else
#     echo "GDC client already exists at $GDC_CLIENT_BIN"
# fi

# %% language="bash"
# # Define the variables
# TCGA_SCC_DIR="$HOME/mepylome/data/tcga_scc"
# DOWNLOAD_COMPLETE="$TCGA_SCC_DIR/.download_complete"
# GDC_CLIENT_BIN="$HOME/mepylome/gdc-client"
# TAR_FILE_PATH="$HOME/mepylome/data/tcga_scc_anno.tar.gz"
#
# # Check if the data directory exists
# mkdir -p "$TCGA_SCC_DIR"
# if [ -f "$TAR_FILE_PATH" ]; then
#     echo "Setting up TCGA SCC directory..."
#     cd "$TCGA_SCC_DIR" || exit 1
#     cp -f "$TAR_FILE_PATH" .
#     TAR_FILE_NAME=$(basename "$TAR_FILE_PATH")
#     tar xvzf "$TAR_FILE_NAME"
#     echo "Setting up TCGA SCC directory done."
# else
#     echo "Error: TAR file not found at $TAR_FILE_PATH"
#     exit 1
# fi
#
# if [ ! -f "$DOWNLOAD_COMPLETE" ]; then
#     echo "Download has not been completed yet."
#     if [ -x "$GDC_CLIENT_BIN" ]; then
#         echo "Downloading TCGA files. This may take some time!"
#         "$GDC_CLIENT_BIN" download -m gdc_manifest* -d "$TCGA_SCC_DIR" \
#             || { echo "Download failed"; exit 1; }
#         echo "Download finished."
#         touch "$DOWNLOAD_COMPLETE"
#     else
#         echo "Error: GDC client not found at $GDC_CLIENT_BIN"
#         exit 1
#     fi
# else
#     echo "Data directory already exists. Skipping download and setup."
# fi
#

# %% [markdown]
# Next we are going to download all the IDAT files mentioned in
# [Jurmeister2019](https://doi.org/10.1126/scitranslmed.aaw8513).

# %%
# Initialize directories.
tumor_site = "scc"
analysis_dir = data_dir / tumor_site
test_dir = validation_dir / tumor_site

ensure_directory_exists(test_dir)
ensure_directory_exists(analysis_dir)

# Download the the annotation spreadsheet.
def merge_csv_from_tar(tar_path):
    """Reads all CSV files from a tar.gz archive and merges them."""
    merged_df = pd.DataFrame()
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar.getmembers():
            if member.name.endswith(".csv"):
                print(f"Reading {member.name}")
                file = tar.extractfile(member)
                if file is not None:
                    df = pd.read_csv(file)
                    merged_df = pd.concat([merged_df, df], ignore_index=True)
    return merged_df

if not (excel_path := analysis_dir / f"{tumor_site}-annotation.xlsx").exists():
    pass
# download_file(datasets[tumor_site]["xlsx"], excel_path)

tar_path = "geo_annotations.tar.gz"
tar_path = Path("~/mepylome/data/scc/geo_annotations.tar.gz").expanduser()
merged_df = merge_csv_from_tar(tar_path)


# Download the IDAT files.
download_from_geo_and_untar(analysis_dir, datasets[tumor_site]["geo_id"])


# %% [markdown]
# ### Create the Methylation Analysis Object
#
# The `MethylAnalysis` object serves as the main interface for performing DNA
# methylation analysis. Key parameters such as the directory structure, number
# of CpG sites, and UMAP settings are configured here.

# %%
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
