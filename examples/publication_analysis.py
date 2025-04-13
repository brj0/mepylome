# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

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
# - Follow the notebook/script step-by-step.
# - If you only intend to run a specific section (e.g., 1, 2, or 3), ensure
#   that you first execute the setup section (0). This approach is essential if
#   memory is limited.
#
#
# ### System Tested
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
# **Note**: The graphical user interface (GUI) features are limited in Google
# Colab. If using the free version, memory constraints may arise, so it is
# recommended to run sections 1, 2, and 3 separately with a kernel restart
# between each part. Additionally, long download operations (e.g., for the
# squamous cell carcinoma section) may face timeouts or interruptions.
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

# %% [markdown]
# -----------------------------------------------------------------------------
# ## Contents
# 0. **[Initialization](#0.-Initialization)**
# 1. **[Salivary Gland Tumors](#1.-Salivary-Gland-Tumors)**
# 2. **[Soft Tissue Tumors](#2.-Soft-Tissue-Tumors)**
# 3. **[Squamous Cell Carcinoma](#3.-Squamous-Cell-Carcinoma)**
# 4. **[Appendix](#4.-Appendix)**


# %% [markdown]
# -----------------------------------------------------------------------------
# <a name="0.-Initialization"></a>
# ## 0. Initialization
#
# ### Install Required Packages
#
# To run the analysis, install the following Python packages:
# - `mepylome` - the main toolkit for DNA-methylation analysis.
# - `ruptures` - used for segmentation calculations in CNV plots.
# - `kaleido` for saving plots.
# - `ipython` and `pillow` - supporting libraries for interactive and graphical
#   functionality.
#
#
# Install these packages (may take 1 to 2 minutes) using the command below:

# %% language="bash"
#
# pip install mepylome ipython pillow ruptures ipywidgets
# pip install -U kaleido


# %% [markdown]
# ### Core Imports, Configuration and main Functions

# %%
import gc
import io
import json
import multiprocessing
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from IPython.display import Image as IPImage
from PIL import Image

from mepylome import ArrayType, Manifest, idat_basepaths
from mepylome.analysis import MethylAnalysis
from mepylome.dtypes.manifests import (
    DOWNLOAD_DIR,
    MANIFEST_URL,
    REMOTE_FILENAME,
)
from mepylome.utils.files import (
    download_file,
    download_geo_probes,
)

# Define output font size for plots
FONTSIZE = 23
IMG_HEIGHT = 2000
IMG_WIDTH = 1000
GEO_URL = "https://www.ncbi.nlm.nih.gov/geo/download/?acc={acc}&format=file"

# Define dataset URLs and filenames
datasets = {
    "salivary_gland_tumors": {
        "xlsx": "https://ars.els-cdn.com/content/image/1-s2.0-S0893395224002059-mmc4.xlsx",
        "geo_ids": ["GSE243075"],
    },
    "soft_tissue_tumors": {
        "xlsx": "https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-020-20603-4/MediaObjects/41467_2020_20603_MOESM4_ESM.xlsx",
        "geo_ids": ["GSE140686"],
    },
    "sinonasal_tumors": {
        "xlsx": "https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-022-34815-3/MediaObjects/41467_2022_34815_MOESM6_ESM.xlsx",
        "geo_ids": ["GSE196228"],
    },
    "scc": {
        "xlsx": "https://www.science.org/doi/suppl/10.1126/scitranslmed.aaw8513/suppl_file/aaw8513_data_file_s1.xlsx",
        "geo_ids": [],
    },
    "scc_test": {
        "geo_ids": [
            "GSE124052",  # HNSQ_CA, NSCLC_SC
            "GSE66836",  # NSCLC_AD, CONTR_LUNG
            "GSE79556",  # HNSQ_CA (oral tongue)
            "GSE87053",  # HNSQ_CA, CONTR_OC
            "GSE95036",  # HNSQ_CA
            "GSE124052",  # HNSQ_CA, NSCLC_SC
        ],
    },
}

# Determine basic storage directory depending on platform
if "COLAB_GPU" in os.environ:
    # Google Colab
    mepylome_dir = Path("/content/mepylome")
elif Path("/mnt/bender").exists():
    # Bender-specific path
    mepylome_dir = Path("/mnt/bender/mepylome")
else:
    # Default for local Linux or other environments
    mepylome_dir = Path.home() / "mepylome"


data_dir = mepylome_dir / "data"
output_dir = mepylome_dir / "outputs"
reference_dir = mepylome_dir / "cnv_references"
validation_dir = mepylome_dir / "validation_data"

# Ensure the directory exists
mepylome_dir.mkdir(parents=True, exist_ok=True)
data_dir.mkdir(parents=True, exist_ok=True)
output_dir.mkdir(parents=True, exist_ok=True)
reference_dir.mkdir(parents=True, exist_ok=True)
validation_dir.mkdir(parents=True, exist_ok=True)


print("=== System Information ===")
print(f"Python Version: {sys.version.split()[0]}")
print(f"Platform: {platform.system()} {platform.release()}")
print(f"Processor: {platform.processor()}")
print(f"Number of CPUs: {multiprocessing.cpu_count()}")
print(f"Data will be stored in: {mepylome_dir}")


# Main Functions


def extract_tar(tar_path, output_directory):
    """Extracts tar file under 'tar_path' to 'output_directory'."""
    output_directory.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r") as tar:
        tar.extractall(path=output_directory, filter="data")
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


def clean_filename(name):
    """Replace invalid characters with a single underscore."""
    return re.sub(r"[^\w\-]+", "_", name)


def calculate_cn_summary(analysis, class_):
    """Calculates and saves CN summary plots."""
    df_class = analysis.idat_handler.samples_annotated[class_]
    plot_list = []
    analysis_dir = analysis.analysis_dir
    all_classes = sorted(df_class.unique())
    for methyl_class in all_classes:
        df_index = df_class == methyl_class
        sample_ids = df_class.index[df_index]
        plot, _ = analysis.cn_summary(sample_ids)
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
        output_dir / f"{analysis_dir.name}_cn_summary_{clean_filename(x)}.png"
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
    output_path = output_dir / f"{analysis_dir.name}_cn_summary.png"
    new_image.save(output_path)
    return output_path


# %% [markdown]
# ### Blacklist Generation for CpG Sites
#
# Some CpG sites should be excluded from the analysis. Here we choose probes
# flagged with `MFG_Change_Flagged` that should be excluded according to the
# manifest and those that are on sex chromosomes.


# %%
def generate_blacklist_cpgs():
    """Returns and caches CpG sites that should be blacklisted."""
    print("Generating blacklist. Can take some time...")
    blacklist_path = data_dir / "cpg_blacklist.csv"
    if not blacklist_path.exists():
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
        flagged_cpgs.to_csv(blacklist_path, index=False, header=False)
        csv_path.unlink()
    blacklist_df = pd.read_csv(blacklist_path, header=None)
    print("Generating blacklist done.")
    return set(blacklist_df.iloc[:, 0])


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
# To ensure accurate analysis, we utilize control probes from [Koelsche et
# al. (2021)](https://doi.org/10.1038/s41467-020-20603-4). These probes
# are stored in the designated reference directory `reference_dir`.
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
# [Jurmeister et al. (2024)](https://doi.org/10.1016/j.modpat.2024.100625). To
# begin, we download the required data and organize it within the designated
# directories.

# %%
# Initialize directories.
tumor_site = "salivary_gland_tumors"
analysis_dir_sg = data_dir / tumor_site
test_dir_sg = validation_dir / tumor_site

test_dir_sg.mkdir(parents=True, exist_ok=True)
analysis_dir_sg.mkdir(parents=True, exist_ok=True)

# Download the annotation spreadsheet.
if not (excel_path := analysis_dir_sg / f"{tumor_site}.xlsx").exists():
    download_file(datasets[tumor_site]["xlsx"], excel_path)
    # Deletes the first 2 rows (useless description).
    pd.read_excel(excel_path, skiprows=2).to_excel(excel_path, index=False)

# Download the IDAT files.
download_from_geo_and_untar(analysis_dir_sg, datasets[tumor_site]["geo_ids"])


# %% [markdown]
# ### Create the Methylation Analysis Object
#
# The `MethylAnalysis` object serves as the main interface for performing DNA
# methylation analysis. Key parameters such as the directory structure, number
# of CpG sites, and UMAP settings are configured here.

# %%
analysis_sg = MethylAnalysis(
    analysis_dir=analysis_dir_sg,
    reference_dir=reference_dir,
    output_dir=output_dir,
    test_dir=test_dir_sg,
    n_cpgs=25000,
    load_full_betas=True,
    overlap=False,
    cpg_blacklist=blacklist,
    debug=False,
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
analysis_sg.set_betas()


# %% [markdown]
# ### Generate UMAP Plot
#
# Set the columns used for coloring the UMAP plot before initiating the
# dimensionality reduction process. The UMAP algorithm produces a visual
# representation of the sample clusters, which is stored as a Plotly object in
# `analysis_sg.umap_plot`.

# %%
# Calculate UMAP
analysis_sg.idat_handler.selected_columns = ["Methylation class"]
analysis_sg.make_umap()

# %%
# Show the results
print(analysis_sg.umap_df)

# %%
# Generate and show image
output_path = output_dir / f"{analysis_dir_sg.name}_umap_plot.jpg"
analysis_sg.umap_plot.write_image(
    output_path,
    format="jpg",
    width=IMG_HEIGHT,
    height=IMG_WIDTH,
    scale=1,
)
IPImage(filename=output_path)

# %% [markdown]
# ### Launch the Analysis GUI
#
# Initializes an interactive GUI for further exploration of the methylation
# data.
#
# **Note:** This step works best in local environments and may have limitations
# on platforms like Google Colab or Binder.

# %%
analysis_sg.run_app(open_tab=True)


# %% [markdown]
# ### Generate and Save CNV Plot
#
# Creates a copy number variation (CNV) plot for a specified sample and saves
# the output as an image.

# %%
# Save CNV example
analysis_sg.make_cnv_plot("206842050057_R06C01")
cnv_plot = analysis_sg.cnv_plot
cnv_plot.update_layout(
    yaxis_range=[-1.1, 1.1],
    font={"size": FONTSIZE},
    margin={"t": 50},
)
output_path = output_dir / f"{analysis_dir_sg.name}_cnv_plot.jpg"
cnv_plot.write_image(
    output_path,
    format="jpg",
    width=IMG_HEIGHT,
    height=IMG_WIDTH,
    scale=1,
)
IPImage(filename=output_path)

# %% [markdown]
# ### Generate CNV Summary Plots
#
# In addition to individual CNV plots, this step computes summary plots to
# highlight genomic alterations across multiple samples.
#
# **Note**:
# Generating all copy number variation (CNV) plots is resource- and
# time-intensive. The process can take a significant amount of time, depending
# on the computational resources available.

# %%
analysis_sg.precompute_cnvs()
cn_summary_path_sg = calculate_cn_summary(analysis_sg, "Methylation class")

# %%
IPImage(filename=cn_summary_path_sg)

# %% [markdown]
# ### Supervised Classifier Validation
#
# The next step involves validating various supervised classification
# algorithms to evaluate their performance on the dataset. This process helps
# identify the most accurate model for methylation-based classification.
#
# **Note**:
# Training is resource- and time-intensive. The process may take up to 10
# minutes, depending on the computational resources available.

# %%
# Train supervised classifiers
ids = analysis_sg.idat_handler.ids
clf_out_sg = analysis_sg.classify(
    ids=ids,
    clf_list=[
        "none-kbest-et",
        "none-kbest-lr",
        "none-kbest-rf",
        # "none-kbest-svc_rbf",
        # "none-pca-lr",
        # "none-pca-et",
    ],
)

# %%
# Print reports for all classifier for the first sample
for clf_result in clf_out_sg:
    print(clf_result.reports["txt"][0])
    print()

# %%
# Identify and display the best classifier
best_clf_sg = max(
    clf_out_sg, key=lambda result: np.mean(result.metrics["accuracy_scores"])
)
print("Most accurate classifier:")
print(best_clf_sg.reports["txt"][0])

# %%
# Free memory
del analysis_sg
gc.collect()

# %% [markdown]
# -----------------------------------------------------------------------------
# <a name="2.-Soft-Tissue-Tumors"></a>
# ## 2. Soft Tissue Tumors
#
# This section replicates the methylation analysis performed in [Koelsche et
# al. (2021)](https://doi.org/10.1038/s41467-020-20603-4). To begin, we
# download the required data and organize it within the designated directories.

# %%
# Initialize directories.
tumor_site = "soft_tissue_tumors"
analysis_dir_st = data_dir / tumor_site
test_dir_st = validation_dir / tumor_site

test_dir_st.mkdir(parents=True, exist_ok=True)
analysis_dir_st.mkdir(parents=True, exist_ok=True)

# Download the annotation spreadsheet.
if not (excel_path := analysis_dir_st / f"{tumor_site}.xlsx").exists():
    download_file(datasets[tumor_site]["xlsx"], excel_path)

# Download the IDAT files.
download_from_geo_and_untar(analysis_dir_st, datasets[tumor_site]["geo_ids"])


# %% [markdown]
# ### Create the Methylation Analysis Object
#
# The `MethylAnalysis` object serves as the main interface for performing DNA
# methylation analysis. Key parameters such as the directory structure, number
# of CpG sites, and UMAP settings are configured here.

# %%
analysis_st = MethylAnalysis(
    analysis_dir=analysis_dir_st,
    reference_dir=reference_dir,
    output_dir=output_dir,
    n_cpgs=25000,
    load_full_betas=True,
    overlap=False,
    cpgs="450k+epic+epicv2",
    cpg_blacklist=blacklist,
    debug=False,
    do_seg=True,
    umap_parms={
        "metric": "manhattan",
    },
)

# %% [markdown]
# ### Load Beta Values
#
# Reads and processes beta values from the provided dataset. This step can also
# be performed interactively within the GUI.

# %%
analysis_st.set_betas()


# %% [markdown]
# ### Generate UMAP Plot
#
# Set the columns used for coloring the UMAP plot before initiating the
# dimensionality reduction process. The UMAP algorithm produces a visual
# representation of the sample clusters, which is stored as a Plotly object in
# `analysis_st.umap_plot`.

# %%
# Calculate UMAP
analysis_st.idat_handler.selected_columns = ["Methylation Class Name"]
analysis_st.make_umap()

# %%
# Show the results
print(analysis_st.umap_df)

# %%
# Generate and show image
output_path = output_dir / f"{analysis_dir_st.name}_umap_plot.jpg"
analysis_st.umap_plot.write_image(
    output_path,
    format="jpg",
    width=IMG_HEIGHT,
    height=IMG_WIDTH,
    scale=1,
)
IPImage(filename=output_path)


# %% [markdown]
# ### Launch the Analysis GUI
#
# Initializes an interactive GUI for further exploration of the methylation
# data.
#
# **Note:** This step works best in local environments and may have limitations
# on platforms like Google Colab or Binder.

# %%
analysis_st.run_app(open_tab=True)


# %% [markdown]
# ### Generate and Save CNV Plot
#
# Creates a copy number variation (CNV) plot for a specified sample and saves
# the output as an image.

# %%
# Save CNV example
analysis_st.make_cnv_plot("3999112131_R05C01")
cnv_plot = analysis_st.cnv_plot
cnv_plot.update_layout(
    yaxis_range=[-1.1, 1.1],
    font={"size": FONTSIZE},
    margin={"t": 50},
)
output_path = output_dir / f"{analysis_dir_st.name}_cnv_plot.jpg"
cnv_plot.write_image(
    output_path,
    format="jpg",
    width=IMG_HEIGHT,
    height=IMG_WIDTH,
    scale=1,
)
IPImage(filename=output_path)

# %% [markdown]
# ### Generate CNV Summary Plots
#
# In addition to individual CNV plots, this step computes summary plots to
# highlight genomic alterations across multiple samples.
#
# **Note**:
# Generating all copy number variation (CNV) plots is resource- and
# time-intensive. The process can take a significant amount of time, depending
# on the computational resources available.

# %%
analysis_st.precompute_cnvs()
cn_summary_path_st = calculate_cn_summary(
    analysis_st, "Methylation Class Name"
)

# %%
IPImage(filename=cn_summary_path_st)

# %% [markdown]
# ### Supervised Classifier Validation
#
# The next step involves validating various supervised classification
# algorithms to evaluate their performance on the dataset. This process helps
# identify the most accurate model for methylation-based classification.
#
# **Note**:
# Training is resource- and time-intensive. The process may take up to 10
# minutes, depending on the computational resources available.

# %%
# Train supervised classifiers
ids = analysis_st.idat_handler.ids
clf_out_st = analysis_st.classify(
    ids=ids,
    clf_list=[
        "none-kbest-et",
        "none-kbest-lr",
        "none-kbest-rf",
        # "none-kbest-svc_rbf",
        # "none-pca-lr",
        # "none-pca-et",
    ],
)

# %%
# Print reports for all classifier for the first sample
for clf_result in clf_out_st:
    print(clf_result.reports["txt"][0])
    print()

# %%
# Identify and display the best classifier
best_clf_st = max(
    clf_out_st, key=lambda result: np.mean(result.metrics["accuracy_scores"])
)
print("Most accurate classifier:")
print(best_clf_st.reports["txt"][0])


# %%
# Free memory
del analysis_st
gc.collect()


# %% [markdown]
# -----------------------------------------------------------------------------
# <a name="3.-Squamous-Cell-Carcinoma"></a>
# ## 3. Squamous Cell Carcinoma

# %% In this example, we aim to reproduce the pan-SCC classifier [markdown]
# presented in the study by [Jurmeister et al.
# (2019)](https://doi.org/10.1126/scitranslmed.aaw8513). Our goal is to gather
# data for Squamous Cell Carcinoma (SCC) from multiple sources, as outlined in
# the publication, including datasets from The Cancer Genome Atlas (TCGA) and
# from the Gene Expression Omnibus (GEO) repository. Datasets without IDAT
# files are omitted from the collection process.

# %%
# Initialize directories.
tumor_site = "scc"
analysis_dir_scc = data_dir / tumor_site
test_dir_scc = validation_dir / tumor_site

test_dir_scc.mkdir(parents=True, exist_ok=True)
analysis_dir_scc.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ### Step 1: Download TCGA Data
# First we download the GDC client that is used for downloading data from TCGA.

# %%
gdc_client_url = "https://gdc.cancer.gov/system/files/public/file/gdc-client_2.3_Ubuntu_x64-py3.8-ubuntu-20.04.zip"
gdc_client_bin = analysis_dir_scc / "gdc-client"

# Download and set up the GDC client
if not gdc_client_bin.exists():
    zip_path_0 = analysis_dir_scc / "gdc-client.zip"
    zip_path_1 = analysis_dir_scc / "gdc-client_2.3_Ubuntu_x64.zip"
    download_file(gdc_client_url, zip_path_0)
    with zipfile.ZipFile(zip_path_0, "r") as zip_file:
        zip_file.extractall(analysis_dir_scc)
    with zipfile.ZipFile(zip_path_1, "r") as zip_file:
        zip_file.extractall(analysis_dir_scc)
    zip_path_0.unlink()
    zip_path_1.unlink()
    gdc_client_bin.chmod(0o755)
    print(f"GDC client binary downloaded and set up at {gdc_client_bin}")
else:
    print(f"GDC client already exists at {gdc_client_bin}")


# %% [markdown]
# Now we download the complete TCGA data. **This may take several hours** due
# to slow server connection. It is recommended to run this process overnight
# and not to abort the process to ensure it completes successfully.

# %%
tcga_dir = analysis_dir_scc / "tcga_scc"
tcga_downloaded_tag = tcga_dir / ".download_complete"
tcga_metadata_dir_tar = analysis_dir_scc / "tcga_metadata.tar.gz"
tcga_metadata_dir = tcga_metadata_dir_tar.with_suffix("").with_suffix("")

tcga_dir.mkdir(parents=True, exist_ok=True)

geo_metadata_url = "https://raw.githubusercontent.com/brj0/mepylome-data/main/examples/geo_metadata.tar.gz"
tcga_metadata_url = "https://raw.githubusercontent.com/brj0/mepylome-data/main/examples/tcga_metadata.tar.gz"

# Check if the TCGA annotation tar file exists and extract
if not tcga_metadata_dir.exists():
    print("Setting up TCGA annotation directory...")
    download_file(tcga_metadata_url, tcga_metadata_dir_tar)
    extract_tar(tcga_metadata_dir_tar, analysis_dir_scc)
    print("Setting up TCGA annotation directory done.")

# Check if the download is complete
if not tcga_downloaded_tag.exists():
    print("Download has not been completed yet.")
    if not gdc_client_bin.exists():
        msg = f"Error: GDC client not found at {gdc_client_bin}"
        raise FileNotFoundError(msg)
    print("Downloading TCGA files. This may take some time!")
    manifest_file = next(tcga_metadata_dir.glob("gdc_manifest.*txt"))
    if not manifest_file.exists():
        msg = "No TCGA manifest file found."
        raise FileNotFoundError(msg)
    print(f"Downloading TCGA data from manifest file: {manifest_file}")
    subprocess.run(
        [
            str(gdc_client_bin),
            "download",
            "--latest",
            "--manifest",
            manifest_file,
            "--dir",
            str(tcga_dir),
        ],
        check=True,
    )
    print("Download finished.")
    tcga_downloaded_tag.touch()
else:
    print("TCGA data already completely downloaded.")


# %% [markdown]
# Clean up by moving all IDAT files into one directory and removing array types
# other than `450k` and `epic`.


# %%
def move_idat_files_and_cleanup(root_dir):
    """Move all idat files one dir up and delete empty subdirectories."""
    root_dir = Path(root_dir)
    for sub_dir in root_dir.iterdir():
        if sub_dir.is_dir():
            idat_files = list(sub_dir.glob("*.idat"))
            for file in idat_files:
                destination = root_dir / file.name
                shutil.move(str(file), str(destination))
            shutil.rmtree(sub_dir)


def remove_invalid_array_types(root_dir):
    """Removes all IDAT files that are not of type 450k or epicv1."""
    idat_files = root_dir.glob("*idat")
    valid_array_types = {ArrayType.ILLUMINA_450K, ArrayType.ILLUMINA_EPIC}
    for idat_file in idat_files:
        array_type = ArrayType.from_idat(idat_file)
        if array_type not in valid_array_types:
            print(f"Removing {idat_file.name} (Type: {array_type})")
            idat_file.unlink()


move_idat_files_and_cleanup(tcga_dir)
remove_invalid_array_types(tcga_dir)


# %% [markdown]
# Next we extract the TCGA annotation.


# %%
def extract_tcga_case_id_dict(json_path):
    """Extracts a dictionary mapping from IDAT IDs to case IDs."""
    with json_path.open() as f:
        data = json.load(f)
    case_id_mapping = {}
    n_suffix = len("_Grn.idat")
    for item in data:
        file_name = item.get("file_name", "")[:-n_suffix]
        case_id = item.get("associated_entities", [{}])[0].get("case_id", "")
        if case_id and file_name:
            case_id_mapping[case_id] = file_name
    return case_id_mapping


json_metadata = next(tcga_metadata_dir.glob("metadata.cart.*json"))
case_id_to_sample_id = extract_tcga_case_id_dict(json_metadata)

# Load the clinical data and map the case_id to the IDAT
tcga_annotation = pd.read_csv(
    tcga_metadata_dir / "clinical.tsv", delimiter="\t"
)
tcga_annotation["Sample_ID"] = tcga_annotation["case_id"].map(
    case_id_to_sample_id
)
tcga_annotation = tcga_annotation.drop(columns="case_id")

# Rename columns restrict to the useful ones
columns_dict = {
    "gender": "Sex",
    "age_at_index": "Age",
    "tissue_or_organ_of_origin": "Tumor_site",
    "site_of_resection_or_biopsy": "Site_of_resection_or_biopsy",
    "tumor_grade": "Tumor_grade",
    "morphology": "Morphology",
    "primary_diagnosis": "Primary_diagnosis",
    "Sample_ID": "Sample_ID",
}
tcga_annotation = tcga_annotation.rename(columns=columns_dict)
tcga_annotation = tcga_annotation[columns_dict.values()]

# Standardize the 'Sex' column and convert 'Age' to numeric
tcga_annotation["Sex"] = tcga_annotation["Sex"].replace(
    {"female": "Female", "male": "Male"}
)
tcga_annotation["Age"] = pd.to_numeric(tcga_annotation["Age"], errors="coerce")

# Mark the samples that to be censored
diag_to_censor_stat = {
    "Adenocarcinoma with mixed subtypes": 0,
    "Adenocarcinoma, NOS": 0,
    "Adenosquamous carcinoma": 1,
    "Basaloid squamous cell carcinoma": 1,
    "Lymphoepithelial carcinoma": 1,
    "Papillary carcinoma, NOS": 1,
    "Papillary squamous cell carcinoma": 1,
    "Squamous cell carcinoma, NOS": 0,
    "Squamous cell carcinoma, keratinizing, NOS": 0,
    "Squamous cell carcinoma, large cell, nonkeratinizing, NOS": 1,
    "Squamous cell carcinoma, nonkeratinizing, NOS": 0,
    "Squamous cell carcinoma, small cell, nonkeratinizing": 1,
    "Squamous cell carcinoma, spindle cell": 1,
    "Warty carcinoma": 1,
}

tcga_annotation["Censor"] = tcga_annotation["Primary_diagnosis"].map(
    diag_to_censor_stat
)

# Condense the primary tumor site.
nsclc_sites = {
    "Lower lobe, lung",
    "Lung, NOS",
    "Main bronchus",
    "Middle lobe, lung",
    "Overlapping lesion of lung",
    "Upper lobe, lung",
}
hnsq_sites = {
    "Anterior floor of mouth",
    "Base of tongue, NOS",
    "Border of tongue",
    "Cheek mucosa",
    "Floor of mouth, NOS",
    "Gum, NOS",
    "Hard palate",
    "Head, face or neck, NOS",
    "Hypopharynx, NOS",
    "Larynx, NOS",
    "Lip, NOS",
    "Lower gum",
    "Mandible",
    "Mouth, NOS",
    "Nasal cavity",
    "Oropharynx, NOS",
    "Overlapping lesion of lip, oral cavity and pharynx",
    "Palate, NOS",
    "Pharynx, NOS",
    "Posterior wall of oropharynx",
    "Retromolar area",
    "Supraglottis",
    "Tongue, NOS",
    "Tonsil, NOS",
    "Upper Gum",
    "Ventral surface of tongue, NOS",
}
cervix_sites = {"Cervix uteri"}
eso_sites = {
    "Cardia, NOS",
    "Esophagus, NOS",
    "Lower third of esophagus",
    "Middle third of esophagus",
    "Thoracic esophagus",
    "Upper third of esophagus",
}
censor_sites = {
    "Breast, NOS",
    "Bladder, NOS",
}

tcga_annotation["Diagnosis"] = None

# Classify each row based on the tumor site
for index, row in tcga_annotation.iterrows():
    site = str(row["Tumor_site"]).strip()
    diagnosis = row["Primary_diagnosis"]
    if diagnosis.startswith("Adenocarcinoma") and site in nsclc_sites:
        tcga_annotation.loc[index, "Diagnosis"] = "NSCLC_AD"
    elif site in cervix_sites:
        tcga_annotation.loc[index, "Diagnosis"] = "CERSQ_CA"
    elif site in nsclc_sites:
        tcga_annotation.loc[index, "Diagnosis"] = "NSCLC_SC"
    elif site in hnsq_sites:
        tcga_annotation.loc[index, "Diagnosis"] = "HNSQ_CA"
    elif site in eso_sites:
        tcga_annotation.loc[index, "Diagnosis"] = "ESO_CA_SQ"
    else:
        tcga_annotation.loc[index, "Censor"] = 1
        print(f"Unmatched tumor site: {site} (index {index} - censored)")

# Removed censored samples
tcga_annotation = tcga_annotation[tcga_annotation["Censor"] == 0]

# %% [markdown]
# ### Step 2: Download and unzip the GEO data

# %%
geo_metadata_dir_tar = analysis_dir_scc / "geo_metadata.tar.gz"
geo_metadata_dir = geo_metadata_dir_tar.with_suffix("").with_suffix("")

# Check if the GEO annotation tar file exists and extract
if not geo_metadata_dir.exists():
    print("Setting up GEO annotation directory...")
    download_file(geo_metadata_url, geo_metadata_dir_tar)
    extract_tar(geo_metadata_dir_tar, analysis_dir_scc)
    print("Setting up GEO annotation directory done.")

# Download the IDAT files.
download_from_geo_and_untar(analysis_dir_scc, datasets[tumor_site]["geo_ids"])
download_from_geo_and_untar(
    test_dir_scc, datasets[tumor_site + "_test"]["geo_ids"]
)


# Download the annotation spreadsheet.
def merge_csv(dir_path):
    """Reads all CSV files merges them."""
    dir_path = Path(dir_path)
    merged_df = pd.DataFrame()
    for csv_file in dir_path.glob("*.csv"):
        print(f"Reading {csv_file}")
        data_frame = pd.read_csv(csv_file)
        merged_df = pd.concat([merged_df, data_frame], ignore_index=True)
    return merged_df


geo_annotation = merge_csv(geo_metadata_dir)

# %% [markdown]
# ### Step 3: Construct the annotation file of all data.
# Join the TCGA and GEO annotation files

# %%
if (csv_path := analysis_dir_scc / f"{tumor_site}.csv").exists():
    anno_df = pd.read_csv(csv_path)
    print("Merged annotation file allready exists.")
else:
    anno_df = pd.concat([geo_annotation, tcga_annotation], ignore_index=True)
    # Remove Adenocarcinoma and normal oral cavity samples
    scc_types = {"HNSQ_CA", "NSCLC_SC", "ESO_CA_SQ", "CERSQ_CA"}
    anno_df = anno_df[anno_df["Diagnosis"].isin(scc_types)]
    anno_df.to_csv(csv_path, index=False)
    print("Merged annotation file created.")


# %% [markdown]
# ### Create the Methylation Analysis Object
#
# The `MethylAnalysis` object serves as the main interface for performing DNA
# methylation analysis. Key parameters such as the directory structure, number
# of CpG sites, and UMAP settings are configured here.

# %%
# Only consider test files with annotation.
test_ids_scc = set(anno_df.Sample_ID).intersection(
    x.name for x in idat_basepaths(test_dir_scc)
)

analysis_scc = MethylAnalysis(
    analysis_dir=analysis_dir_scc,
    reference_dir=reference_dir,
    output_dir=output_dir,
    test_dir=test_dir_scc,
    test_ids=test_ids_scc,
    n_cpgs=25000,
    load_full_betas=True,
    overlap=True,
    cpg_blacklist=blacklist,
    cpgs="450k+epic+epicv2",
    debug=False,
    do_seg=True,
    umap_parms={
        "n_neighbors": 5,
        "metric": "manhattan",
        "min_dist": 0.1,
    },
)

# %% [markdown]
# ### Load Beta Values
#
# Reads and processes beta values from the provided dataset. This step can also
# be performed interactively within the GUI.

# %%
analysis_scc.set_betas()


# %% [markdown]
# ### Generate UMAP Plot
#
# Set the columns used for coloring the UMAP plot before initiating the
# dimensionality reduction process. The UMAP algorithm produces a visual
# representation of the sample clusters, which is stored as a Plotly object in
# `analysis_scc.umap_plot`.

# %%
# Calculate UMAP
analysis_scc.idat_handler.selected_columns = ["Diagnosis"]
analysis_scc.make_umap()

# %%
# Show the results
print(analysis_scc.umap_df)

# %%
# Generate and show image
output_path = output_dir / f"{analysis_dir_scc.name}_umap_plot.jpg"
analysis_scc.umap_plot.write_image(
    output_path,
    format="jpg",
    width=IMG_HEIGHT,
    height=IMG_WIDTH,
    scale=1,
)
IPImage(filename=output_path)


# %% [markdown]
# ### Launch the Analysis GUI
#
# Initializes an interactive GUI for further exploration of the methylation
# data.
#
# **Note:** This step works best in local environments and may have limitations
# on platforms like Google Colab or Binder.

# %%
analysis_scc.run_app(open_tab=True)


# %% [markdown]
# ### Generate and Save CNV Plot
#
# Creates a copy number variation (CNV) plot for a specified sample and saves
# the output as an image.

# %%
# Save CNV example
analysis_scc.make_cnv_plot("364f7953-d0af-4929-8491-7b5e94d488aa_noid")
cnv_plot = analysis_scc.cnv_plot
cnv_plot.update_layout(
    yaxis_range=[-1.1, 1.1],
    font={"size": FONTSIZE},
    margin={"t": 50},
)
output_path = output_dir / f"{analysis_dir_scc.name}_cnv_plot.jpg"
cnv_plot.write_image(
    output_path,
    format="jpg",
    width=IMG_HEIGHT,
    height=IMG_WIDTH,
    scale=1,
)
IPImage(filename=output_path)

# %% [markdown]
# ### Generate CNV Summary Plots
#
# In addition to individual CNV plots, this step computes summary plots to
# highlight genomic alterations across multiple samples.
#
# **Note**:
# Generating all copy number variation (CNV) plots is resource- and
# time-intensive. The process can take a significant amount of time, depending
# on the computational resources available.

# %%
analysis_scc.precompute_cnvs()
cn_summary_path_scc = calculate_cn_summary(analysis_scc, "Diagnosis")

# %%
IPImage(filename=cn_summary_path_scc)

# %% [markdown]
# ### Supervised Classifier Validation
#
# The next step involves validating various supervised classification
# algorithms to evaluate their performance on the dataset. This process helps
# identify the most accurate model for methylation-based classification.
#
# **Note**:
# Training is resource- and time-intensive. The process may take up to 10
# minutes, depending on the computational resources available.

# %%
# Train supervised classifiers
ids = analysis_scc.idat_handler.ids
clf_out_scc = analysis_scc.classify(
    ids=ids,
    clf_list=[
        "none-kbest-et",
        "none-kbest-lr",
        "none-kbest-rf",
        # "none-kbest-svc_rbf",
        # "none-pca-lr",
        # "none-pca-et",
    ],
)

# %%
# Print reports for all classifier for the first sample
for clf_result in clf_out_scc:
    print(clf_result.reports["txt"][0])
    print()

# %%
# Identify and display the best classifier
best_clf_scc = max(
    clf_out_scc, key=lambda result: np.mean(result.metrics["accuracy_scores"])
)
print("Most accurate classifier:")
print(best_clf_scc.reports["txt"][0])

# %% [markdown]
# Now we apply the best classifier on the independent validation samples:

# %%
test_ids = analysis_scc.idat_handler.test_ids
# Ignore all files that are not in annotation file
test_ids = list(
    set(test_ids).intersection(analysis_scc.idat_handler.annotation_df.index)
)
clf_out_pred = analysis_scc.classify(ids=test_ids, clf_list=best_clf_scc.model)
pred = clf_out_pred[0].prediction.idxmax(axis=1)
true_values = analysis_scc.idat_handler.samples_annotated.loc[test_ids][
    "Diagnosis"
]
correct = np.sum(pred == true_values)
total = len(pred)
accuracy_scc = correct / total
print(f"Classifier Accuracy: {100 * accuracy_scc:.2f} % ({correct}/{total})")

# Analyze misclassified samples
misclassified = pred != true_values
misclassified_samples = clf_out_pred[0].prediction[misclassified].copy()
misclassified_samples["Pred"] = pred[misclassified]
misclassified_samples["True"] = true_values[misclassified]
print("Missclassified samples:\n", misclassified_samples)

# %%
# Free memory
del analysis_scc
gc.collect()

# %% [markdown]
# # 4. Appendix

# %% [markdown]
# The GEO annotation files were downloaded and manually curated. Below is the
# code used to download the GEO datasets and save their associated metadata:

# %% language="bash"
# pip install geoparse


# %%
import GEOparse

geo_numbers = datasets["scc"]["geo_ids"] + datasets["scc_test"]["geo_ids"]

# Download and save metadata for each GEO series
for geo_nr in geo_numbers:
    gse = GEOparse.get_GEO(geo=geo_nr, destdir=geo_metadata_dir)
    metadata = gse.phenotype_data
    print(metadata.head())
    metadata.to_csv(
        geo_metadata_dir / f"{geo_nr}_raw_metadata.csv", index=True
    )

for file in geo_metadata_dir.glob("*.soft.gz"):
    print(f"Removing file: {file}")
    file.unlink()
