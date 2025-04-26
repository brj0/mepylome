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
# - *Authors*: Jon Brugger et al.
#
#
# ### Run This Notebook in Google Colab
#
# You can quickly open and run this notebook in Google Colab without any setup
# by clicking the link below.
#
# **Note**: The graphical user interface (GUI) features are limited in Google
# Colab. If using the free version, memory constraints may arise. Additionally,
# long download operations  may face timeouts or interruptions.
#
# [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/brj0/mepylome/blob/main/examples/publication/soft_tissue_tumors.ipynb)
#
#
# This notebook was automatically generated from the corresponding py-file
# with:
#
# ```bash
# jupytext --to ipynb *.py
# ```

# %% [markdown]
# -----------------------------------------------------------------------------
# ## Contents
# 0. **[Initialization](#0.-Initialization)**
# 1. **[Data Loading](#1.-Data-Loading)**
# 2. **[UMAP Calculation](#2.-UMAP-Calculation)**
# 3. **[Supervised Classifier Training](#3.-Supervised-Classifier-Training)**
# 4. **[CNV Analysis](#4.-CNV-Analysis)**


# %% [markdown]
# -----------------------------------------------------------------------------
# <a name="0.-Initialization"></a>
# ## 0. Initialization
#
# ### Install Required Packages
#
# To run the analysis, install the following Python packages:
# - `mepylome` for DNA-methylation analysis
# - `ruptures` for segmentation in CNV plots
# - `kaleido` for saving plots
# - `ipython`, `pillow`, and `ipywidgets` for interactive and graphical
#   functionality
#
# Install them (1-2 minutes) using:

# %% language="bash"
#
# pip install mepylome ipython pillow ruptures ipywidgets kaleido


# %% [markdown]
# ### Core Imports, Configuration and main Functions

# %%
import io
import multiprocessing
import os
import platform
import re
import sys
import tarfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from IPython.display import Image as IPImage
from PIL import Image

from mepylome import ArrayType, Manifest, clear_cache
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
    "soft_tissue_tumors": {
        "xlsx": "https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-020-20603-4/MediaObjects/41467_2020_20603_MOESM4_ESM.xlsx",
        "geo_ids": ["GSE140686"],
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
# ### CNV-Neutral Reference Samples
#
# For generating copy number variation (CNV) plots, a sufficiently large set of
# CNV-neutral reference probes is required. Here, we use control probes from
# [Koelsche et al. (2021)](https://doi.org/10.1038/s41467-020-20603-4). These
# probes are stored in the designated reference_dir.
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
# <a name="1.-Data-Loading"></a>
# ## 1. Data Loading
#
# This section replicates the methylation analysis performed in [Koelsche et
# al. (2021)](https://doi.org/10.1038/s41467-020-20603-4). To begin, we
# download the required data and organize it within the designated directories.

# %%
# Initialize directories.
tumor_site = "soft_tissue_tumors"
analysis_dir = data_dir / tumor_site
test_dir = validation_dir / tumor_site

test_dir.mkdir(parents=True, exist_ok=True)
analysis_dir.mkdir(parents=True, exist_ok=True)

# Download the annotation spreadsheet.
if not (excel_path := analysis_dir / f"{tumor_site}.xlsx").exists():
    download_file(datasets[tumor_site]["xlsx"], excel_path)

# Download the IDAT files.
download_from_geo_and_untar(analysis_dir, datasets[tumor_site]["geo_ids"])


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
# Reads and processes beta values from the provided dataset. This step is
# optional and primarily demonstrates the time required for processing. If not
# performed here, it will be automatically executed in the background when
# needed.

# %%
analysis.set_betas()


# %% [markdown]
# -----------------------------------------------------------------------------
# <a name="2.-UMAP-Calculation"></a>
# ## 2. UMAP Calculation
#
# ### Generate UMAP Plot
#
# Set the columns used for coloring the UMAP plot before initiating the
# dimensionality reduction process. The UMAP algorithm produces a visual
# representation of the sample clusters, which is stored as a Plotly object in
# `analysis.umap_plot`.

# %%
# Calculate UMAP
analysis.idat_handler.selected_columns = ["Methylation Class Name"]
analysis.make_umap()

# %%
# Show the results
print(analysis.umap_df)

# %%
# Generate and show image
output_path = output_dir / f"{analysis_dir.name}_umap_plot.jpg"
analysis.umap_plot.write_image(
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
analysis.run_app(open_tab=True)


# %% [markdown]
# On memory-limited platforms such as Google Colab, we need to manually free up
# memory between operations to avoid crashes.

# %%
# Free memory
clear_cache()


# %% [markdown]
# -----------------------------------------------------------------------------
# <a name="3.-Supervised-Classifier-Training"></a>
# ## 3. Supervised Classifier Training
#
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
analysis.idat_handler.selected_columns = ["Methylation Class Name"]
ids = analysis.idat_handler.ids
clf_out = analysis.classify(
    ids=ids,
    clf_list=[
        # Classifiers optimized for low-memory platforms (e.g. Google Colab)
        "top-kbest(k=10000)-et",
        "top-kbest(k=10000)-lr(max_iter=10000)",
        "top-kbest(k=10000)-rf",
        # "vtl-kbest(k=10000)-et",
        # "vtl-kbest(k=10000)-lr(max_iter=10000)",
        # "vtl-kbest(k=10000)-rf",
    ],
)

# %%
# Print reports for all classifier for the first sample
for clf_result in clf_out:
    print(clf_result.reports["txt"][0])
    print()

# %%
# Identify and display the best classifier
best_clf = max(
    clf_out, key=lambda result: np.mean(result.metrics["accuracy_scores"])
)
print("Most accurate classifier:")
print(best_clf.reports["txt"][0])

# %%
# Free memory
clear_cache()


# %% [markdown]
# -----------------------------------------------------------------------------
# <a name="4.-CNV-Analysis"></a>
# ## 4. CNV Analysis
#
# ### Generate and Save CNV Plot
#
# Creates a copy number variation (CNV) plot for a specified sample and saves
# the output as an image.

# %%
# Save CNV example
analysis.make_cnv_plot("3999112131_R05C01")
cnv_plot = analysis.cnv_plot
cnv_plot.update_layout(
    yaxis_range=[-1.1, 1.1],
    font={"size": FONTSIZE},
    margin={"t": 50},
)
output_path = output_dir / f"{analysis_dir.name}_cnv_plot.jpg"
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
analysis.precompute_cnvs()
cn_summary_path = calculate_cn_summary(analysis, "Methylation Class Name")

# %%
IPImage(filename=cn_summary_path)
