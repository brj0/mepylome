# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
#
# <img alt="Mepylome Logo" src="https://raw.githubusercontent.com/brj0/mepylome/main/mepylome/data/assets/mepylome.svg" width="300">
#
# Mepylome Tutorial
# =================
#
# ### Run This Notebook in Google Colab
#
# You can quickly open and run this notebook in Google Colab without any setup
# by clicking the link below.
#
# **Note**: The graphical user interface (GUI) features are limited in Google
# Colab.
#
# [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/brj0/mepylome/blob/main/examples/tutorial.ipynb)
#
#
# This notebook was automatically generated from the corresponding py-file
# with:
#
# ```bash
# jupytext --to ipynb tutorial.py
# ```

# %% [markdown]
# -----------------------------------------------------------------------------
# ### Install Required Packages
#
# To run the tutorial, install the following Python packages:
# - `mepylome` - the main toolkit for DNA-methylation analysis.
# - `ruptures` - used for segmentation calculations in CNV plots.
#
#
# Install these packages (may take 1 to 2 minutes) using the command below:

# %% language="bash"
#
# pip install -q mepylome ruptures


# %% [markdown]
# ### Run Tutorial

# %%
from pathlib import Path

from mepylome.analysis import MethylAnalysis
from mepylome.utils import setup_tutorial_files

DIR = Path.home() / "mepylome" / "tutorial"
ANALYSIS_DIR = DIR / "tutorial_analysis"
REFERENCE_DIR = DIR / "tutorial_reference"

setup_tutorial_files(ANALYSIS_DIR, REFERENCE_DIR)

analysis = MethylAnalysis(
    analysis_dir=ANALYSIS_DIR,
    reference_dir=REFERENCE_DIR,
    do_seg=True,
)
analysis.make_umap()

# %% [markdown]
# ### Graphical User Interface

# %%
analysis.run_app(open_tab=True)
