# Tutorial (library)
# ==================


# This tutorial will guide you through the steps to analyze methylation data
# using the ``mepylome`` package. We'll cover setting up the environment,
# parsing IDAT files, working with manifest files, processing raw data,
# performing copy number variation (CNV) analysis, and conducting methylation
# analysis using UMAP plots with a GUI. You can find the code of this tutorial
# as a script in
# https://github.com/brj0/mepylome/blob/main/examples/rtd_tutorial.py.


# .. contents:: Contents of Tutorial
#    :depth: 3


# Setup
# -----


# First, import the required modules and set up the necessary directories:

from pathlib import Path

from mepylome import (
    CNV,
    IdatParser,
    Manifest,
    MethylData,
    RawData,
    ReferenceMethylData,
)

# Setup directories for the tutorial:
DIR = Path.home() / "mepylome" / "tutorial"
ANALYSIS_DIR = DIR / "tutorial_analysis"
REFERENCE_DIR = DIR / "tutorial_reference"

# Download and setup the necessary data files (approximately 653 MB):
from mepylome.utils import setup_tutorial_files

setup_tutorial_files(ANALYSIS_DIR, REFERENCE_DIR)


# Parsing IDAT files
# ------------------


# Define the IDAT file path:
idat_file = ANALYSIS_DIR / "200925700125_R07C01_Grn.idat"

# Parse the IDAT file:
idat_data = IdatParser(idat_file)

# Print overview of parsed data:
print(idat_data)
"""
IdatParser(
    file_size: 13686991
    num_fields: 19
    illumina_ids: array([ 1600101,  1600111, ..., 99810990, 99810992], dtype=int32)
    probe_means: array([15629,  8469, ...,  7971,   943], dtype=uint16)
    std_dev: array([1377,  408, ...,  702,  312], dtype=uint16)
    n_beads: array([16,  7, ...,  6, 10], dtype=uint8)
    mid_block: array([ 1600101,  1600111, ..., 99810990, 99810992], dtype=int32)
    red_green: 0
    mostly_null:
    barcode: 200925700125
    chip_type: BeadChip 8x5
    mostly_a: R07C01
    unknown_1:
    unknown_2:
    unknown_3:
    unknown_4:
    unknown_5:
    unknown_6:
    unknown_7:
)
"""

# The parsed data is available as attributes of the ``IdatParser`` object. For
# example the  Illumina IDs (probes IDs) can be accessed by:
ids = idat_data.illumina_ids

print(ids)
"""
[ 1600101  1600111  1600115 ... 99810978 99810990 99810992]
"""


# C++ Parser
# ~~~~~~~~~~


# If you installed mepylome with C++ support (see `installation
# <installation.html>`_) you can also use the C++ parser (input must be a
# string, not a Path object)
try:
    from mepylome import _IdatParser

    _idat_data = _IdatParser(str(idat_file))
    print("C++ parser available")

except ImportError:
    print("C++ parser NOT available")


# Manifest files
# --------------


# The mepylome package includes a ``Manifest`` class that provides
# functionality to download, process, and save Illumina manifest files
# internally in a efficient format (stored in ~/.mepylome). These manifest
# files contain information about the CpG sites on the methylation array,
# including genetic coordinates, probe types, and more.

# Load the available manifest files for different array types.
manifest_450k = Manifest("450k")
manifest_epic = Manifest("epic")
manifest_epic_v2 = Manifest("epicv2")
manifest_msa48 = Manifest("msa48")

# .. note::
#
#     The first time you run this, the manifest files will be downloaded and
#     saved locally to ~/.mepylome. This initial download might take some time.

# Obtain values from attributes:
probes_df = manifest_450k.data_frame
controls_df = manifest_450k.control_data_frame

# Print overview:
print(probes_df)
"""
            IlmnID  AddressA_ID  AddressB_ID  ...  N_CpG    End  Probe_Type
0       cg13869341     62703328     16661461  ...      2  15865           1
1       cg14008030     27651330           -1  ...      2  18827           2
2       cg12045430     25703424     34666387  ...      7  29407           1
3       cg20826792     61731400     14693326  ...      7  29425           1
4       cg00381604     26752380     50693408  ...      6  29435           1
...            ...          ...          ...  ...    ...    ...         ...
485572   rs1416770     28667385           -1  ...      0     -1           4
485573   rs1941955     33709340           -1  ...      0     -1           4
485574   rs2125573     25698376           -1  ...      0     -1           4
485575   rs2521373     12625304           -1  ...      0     -1           4
485576   rs4331560     10654345           -1  ...      0     -1           4

[485577 rows x 12 columns]
"""


# RawData
# -------


# The ``RawData`` class extracts both raw green and raw red signal intensity
# data from a IDAT file pair. You can initialize it using a base path to the
# IDAT files (without the _Grn.idat / _Red.idat suffix), or by providing the
# full path to either the Grn or Red IDAT file.
idat_file = ANALYSIS_DIR / "200925700125_R07C01_Red.idat"
## or
idat_file = ANALYSIS_DIR / "200925700125_R07C01_Grn.idat"
## or
idat_file = ANALYSIS_DIR / "200925700125_R07C01"
raw_data = RawData(idat_file)

# The data is saved within the following attributes:

## Intensity signals
raw_data.grn
raw_data.red

## Type of the array_type (e.g., 450k, EPIC)
raw_data.array_type

## Corresponding manifest file
raw_data.manifest

## IDs on the bead
raw_data.ids

# Print an overview of the raw data
print(raw_data)
"""
RawData():
**********

array_type: epic

manifest: epic

probes:
['200925700125_R07C01']

ids:
[ 1600101  1600111  1600115 ... 99810978 99810990 99810992]

_grn:
[[15629  8469  7015 ... 10228  7971   943]]

_red:
[[ 4429  1575 24955 ...  6594 15010  5336]]

grn:
          200925700125_R07C01
1600101                 15629
1600111                  8469
1600115                  7015
1600123                  7975
1600131                   938
...                       ...
99810958                 6292
99810970                  318
99810978                10228
99810990                 7971
99810992                  943

[1052641 rows x 1 columns]

red:
          200925700125_R07C01
1600101                  4429
1600111                  1575
1600115                 24955
1600123                 17707
1600131                  8967
...                       ...
99810958                 1881
99810970                 1936
99810978                 6594
99810990                15010
99810992                 5336

[1052641 rows x 1 columns]
"""

# RawData can also read multiple files of the same array type (used for
# reference files):
idat_file0 = ANALYSIS_DIR / "200925700125_R07C01_Grn.idat"
idat_file1 = ANALYSIS_DIR / "200925700133_R02C01"

raw_data_2 = RawData([idat_file0, idat_file1])

# Alternatively, read all IDAT files in a directory (supports recursive
# search):
raw_data_all = RawData(REFERENCE_DIR)


# MethylData
# ----------

# The ``MethylData`` class allows for processing raw intensity data and can
# calculate methylation signals as well as beta values. The raw data can be
# preprocessed using one of the following methods: 'illumina' (default),
# 'swan', or 'noob'. Initialize MethylData with raw data using the default
# 'illumina' preprocessing method.
methyl_data = MethylData(raw_data)

methyl_data_all = MethylData(raw_data_all)

# Alternatively, you can explicitly specify the 'illumina' preprocessing
# method.
methyl_data = MethylData(raw_data, prep="illumina")

# You can also initialize MethylData directly from an IDAT file path, without
# using ``RawData``. This is the preferred method if you want to obtain
# methylation signals or beta values.
methyl_data = MethylData(file=idat_file)

# Obtain various values via the attributes of the MethylData object:

## The methylation signals for the green and red channels.
methylated_signals = methyl_data.methylated
unmethylated_signals = methyl_data.unmethylated

## The corrected color signals.
corrected_green_signals = methyl_data.grn
corrected_red_signals = methyl_data.red

## The type of the array used (e.g., 450k, EPIC, EPICv2).
array_type = methyl_data.array_type

## The corresponding manifest file.
corresponding_manifest = methyl_data.manifest

# Print an overview of the methylation data.
print(methyl_data)
"""
MethylData():
*************

array_type: epic

manifest: epic

probes:
['200925700125_R07C01']

_grn:
[[16785.56811897  9044.70326442  7472.74551323 ... 10946.40456039
   8506.302329     908.14615612]]

_red:
[[ 3957.99684771  1303.20883282 23051.26201716 ...  5971.87773497
  13800.43272211  4801.6873626 ]]

grn:
          200925700125_R07C01
1600101          16785.568359
1600111           9044.703125
1600115           7472.745605
1600123           8510.626953
1600131            902.740540
...                       ...
99810958          6691.091309
99810970           232.442169
99810978         10946.404297
99810990          8506.302734
99810992           908.146179

[1052641 rows x 1 columns]

red:
          200925700125_R07C01
1600101           3957.996826
1600111           1303.208862
1600115          23051.261719
1600123          16309.179688
1600131           8179.240234
...                       ...
99810958          1587.849731
99810970          1639.010620
99810978          5971.877930
99810990         13800.432617
99810992          4801.687500

[1052641 rows x 1 columns]

methylated:
            200925700125_R07C01
IlmnID
cg14817997          2510.375488
cg26928153         10454.492188
cg16269199          7020.834473
cg13869341         30160.773438
cg14008030         19805.154297
...                         ...
cg10488260          2282.708496
cg14273923         12481.604492
cg09748881          6418.647461
cg07587934          9533.372070
cg16855331          5837.001465

[865859 rows x 1 columns]

unmethylated:
            200925700125_R07C01
IlmnID
cg14817997           855.783081
cg26928153           926.525330
cg16269199          3892.054932
cg13869341          5986.760742
cg14008030          8202.495117
...                         ...
cg10488260          8199.704102
cg14273923          2489.212646
cg09748881           950.663391
cg07587934          5264.926270
cg16855331         15618.041992

[865859 rows x 1 columns]
"""

# Beta values are a indicator wheather a CpG is methylated or not. They can be
# calculated for all sites of the corresponding array:
betas = methyl_data.betas

# You can also access the beta values at specific CpG sites (here at all the
# sites of the EPICv2 manifest). Missing data will be replaced with `fill`.
epicv2_cpgs = manifest_epic_v2.methylation_probes
beta_specific = methyl_data.betas_at(epicv2_cpgs, fill=0.5)


# Using Alternative Preprocessing Methods
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# Preprocess the raw data using the SWAN method.
methyl_data_swan = MethylData(raw_data, prep="swan")

# Preprocess the raw data using the NOOB method.
methyl_data_noob = MethylData(raw_data, prep="noob")


# See `api <api.html>`_ for more information about SWAN and NOOB.


# Copy number variation (CNV)
# ---------------------------


# Copy number variations (CNV) are significant alterations in the genome
# involving the loss or gain of large DNA segments, often encompassing multiple
# genes. These variations are frequently linked to cancer development and can
# aid in tumor classification. The CNV profile can be calculated from signal
# intensity using methylation arrays. With the mepylome package, CNV can be
# efficiently calculated and visualized.


# **1. Set up analysis file**

# First, initialize your sample data for analysis:
sample = methyl_data


# **2. Set Up Reference Data**

# Within the reference directory there must be multiple CNV-neutral IDAT
# pairs of the **same array type** as `sample`.
reference = MethylData(file=REFERENCE_DIR)

# Alternatively, if the reference directory contains IDAT files of multiple
# array types, you can use ``ReferenceMethylData`` to load all files into
# memory. This way, the reference object can be used for multiple array types.
# The CNV class will automatically extract the files for the needed array type.
reference_all = ReferenceMethylData(REFERENCE_DIR)


# **3. Initialize CNV Analysis**

# Create an instance of the CNV class for the analysis, and fit the data (this
# is basically a linear regression model comparing `sample` signal with the
# `reference` signals at each CpG site):
cnv = CNV(sample, reference)

## Alternative with ReferenceMethylData
cnv = CNV(sample, reference_all)


# **4. Calculate CNV for Bins and Genes**

# Compute CNV values for genomic bins and gene regions:
cnv.set_bins()
cnv.set_detail()


# **5. Calculate CNV Segments**

# Use the binary circular segmentation algorithm for genome segmentation:
cnv.set_segments()

# .. note::
#
#     For this step, additional packages must be installed (see `installation
#     <installation.html>`_).


# **6. Streamlined Analysis**

# Alternatively, perform all CNV computations in a single call:
cnv = CNV.set_all(sample, reference)

## or
cnv = CNV.set_all(sample, reference_all)


# **7. Visualize CNV Data**

# Display an interactive plot using Plotly, where genes can be highlighted:
cnv.plot()


# Methylation Analysis
# --------------------


# **1. Set up analysis object and run GUI in browser**

# For methylation analysis, ensure you have the setup described in
# :ref:`general_setup`

# First, import the `MethylAnalysis` class from the `mepylome.analysis` module.
from mepylome.analysis import MethylAnalysis

# Create an instance of `MethylAnalysis` with the specified analysis and
# reference directories.
methyl_analysis = MethylAnalysis(
    analysis_dir=ANALYSIS_DIR,
    reference_dir=REFERENCE_DIR,
)

# You can print an overview of the parameters of the object:
print(methyl_analysis)
"""
MethylAnalysis():
*****************

analysis_dir:
/home/username/mepylome/tutorial/tutorial_analysis

annotation:
/home/username/mepylome/tutorial/tutorial_analysis/annotation.csv

app:
None

betas_sel:
None

betas_all:
None

betas_dir:
/tmp/mepylome/analysis/betas-tutorial_analysis-illumina-3b616e0e24a8b0e2d443b777b8ad8b61

cnv_dir:
/tmp/mepylome/analysis/cnv-tutorial_analysis-tutorial_reference-illumina-False-f77099f7bcb04262a0456d122215ed4d

cnv_id:
None

cnv_plot:
Figure({
    'data': [], 'layout': {'template': '...', 'yaxis': {'range': [-2, 2...

cpg_selection:
top

cpgs:
['cg00000029' 'cg00000103' 'cg00000109' ... 'ch.X.97651759F'
 'ch.X.97737721F' 'ch.X.98007042R']

debug:
False

do_seg:
False

dropdown_id:
[]

host:
localhost

ids:
Index(['200925700133_R04C01', '201530470054_R05C01', '201869690203_R03C01',
       '201904410008_R05C01', '201869690168_R08C01', '201530470054_R01C01',
       '201869690203_R06C01', '200925700133_R03C01', '201904410008_R02C01',
       '201904410008_R04C01', '201530470054_R02C01', '201530470054_R03C01',
       '201904410008_R03C01', '201870610040_R03C01', '200925700133_R02C01',
       '200925700125_R07C01', '201530470054_R04C01', '201870610040_R04C01',
       '200925700133_R05C01', '201904410008_R06C01'],
      dtype='object')

ids_to_highlight:
None

load_full_betas:
False

n_cpgs:
25000

output_dir:
/tmp/mepylome/analysis

overlap:
False

port:
8050

precalculate_cnv:
False

prep:
illumina

raw_umap_plot:
Figure({
    'data': [{'customdata': array([['Chondrosarcoma', 'methylation clas...

reference_dir:
/home/username/mepylome/tutorial/tutorial_reference

selected_columns:
['Diagnosis']

umap_cpgs:
None

umap_df:
                       Umap_x    Umap_y  ...  Colour                 Umap_color
200925700133_R04C01  7.089545  1.715319  ...  2E3092  Osteosarcoma (high-grade)
201530470054_R05C01  8.902815  1.723695  ...  6B66AE              Osteoblastoma
201869690203_R03C01  3.801413  5.398326  ...  6282C2             Chondrosarcoma
201904410008_R05C01  6.164794  5.332215  ...  7C7E82    Control (muscle tissue)
201869690168_R08C01  3.486633  4.991620  ...  6282C2             Chondrosarcoma
201530470054_R01C01  8.228397  1.753262  ...  6B66AE              Osteoblastoma
201869690203_R06C01  3.823109  4.597386  ...  6282C2             Chondrosarcoma
200925700133_R03C01  7.502202  1.312939  ...  2E3092  Osteosarcoma (high-grade)
201904410008_R02C01  6.569446  5.207128  ...  7C7E82    Control (muscle tissue)
201904410008_R04C01  6.959955  4.955368  ...  7C7E82    Control (muscle tissue)
201530470054_R02C01  8.313415  2.253792  ...  6B66AE              Osteoblastoma
201530470054_R03C01  8.599755  1.344881  ...  6B66AE              Osteoblastoma
201904410008_R03C01  6.801136  5.651977  ...  7C7E82    Control (muscle tissue)
201870610040_R03C01  4.220203  5.398547  ...  6282C2             Chondrosarcoma
200925700133_R02C01  6.599361  2.232107  ...  2E3092  Osteosarcoma (high-grade)
200925700125_R07C01  6.635824  1.511574  ...  2E3092  Osteosarcoma (high-grade)
201530470054_R04C01  8.751877  2.230259  ...  6B66AE              Osteoblastoma
201870610040_R04C01  4.270664  4.871753  ...  6282C2             Chondrosarcoma
200925700133_R05C01  7.345413  2.310878  ...  2E3092  Osteosarcoma (high-grade)
201904410008_R06C01  6.281984  4.778615  ...  7C7E82    Control (muscle tissue)

[20 rows x 13 columns]

umap_dir:
/tmp/mepylome/analysis/umap-tutorial_analysis-illumina-25000-top-6ff1e8e7ec2b3a2f8d856ee634404085

umap_plot_path:
/tmp/mepylome/analysis/umap-tutorial_analysis-illumina-25000-top-6ff1e8e7ec2b3a2f8d856ee634404085/umap_plot.csv

upload_dir:
/tmp/mepylome/analysis/upload-tutorial_analysis-c8e2f8ac691e9c3deba2e880aa7c5251
"""

# To enable interactive analysis, you can launch a GUI. This will open a new
# tab in your web browser and you can start a GUI-based methylation analysis.
methyl_analysis.run_app(open_tab=True)


# `MethylAnalysis` has multiple parameters. For example, you can provide a
# custom set of CpG sites. For instance, you can set:
cpgs = Manifest("epic").methylation_probes[:10000]
methyl_analysis = MethylAnalysis(
    analysis_dir=ANALYSIS_DIR,
    reference_dir=REFERENCE_DIR,
    cpgs=cpgs,
)

# Mepylome saves all results in a temporary directory ('/tmp/mepylome'). You
# can provide a custom directory for output:
OUTPUT_DIR = DIR / "output_dir"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
methyl_analysis = MethylAnalysis(
    analysis_dir=ANALYSIS_DIR,
    reference_dir=REFERENCE_DIR,
    output_dir=OUTPUT_DIR,
)

# Here is a more comprehensive example with multiple custom parameters:
TEST_DIR = DIR / "test_dir"
methyl_analysis = MethylAnalysis(
    analysis_dir=ANALYSIS_DIR,
    reference_dir=REFERENCE_DIR,
    output_dir=OUTPUT_DIR,
    ## New cases for validation, excluded from classifier training.
    test_dir=TEST_DIR,
    ## Load beta values for all CpG sites into memory
    load_full_betas=True,
    ## Use SWAN preprocessing method
    prep="swan",
    ## Provide annotation file (if not already in analysis_dir)
    annotation=ANALYSIS_DIR / "annotation.csv",
    ## Number of CpGs for UMAP analysis
    n_cpgs=5000,
    ## Analyze the 'best' CpG sites
    cpg_selection="top",
    ## Show segmentation intervals in CNV plot
    do_seg=True,
)

# Many parameters can be modified within the GUI application after
# initialization, but not all.


# **2. Set up beta values and generate UMAP**

# All calculations that can be performed within the GUI can also be done
# manually. For example, to extract the beta values:
methyl_analysis.set_betas()

# The beta values are then stored in:

methyl_analysis.betas_sel
"""
                     cg15836656  cg12823387  cg25563772  ...  cg22115994  cg27601809  cg00444740
200925700133_R04C01    0.117822    0.058700    0.058626  ...    0.295462    0.738677    0.474659
201530470054_R05C01    0.087871    0.050942    0.065696  ...    0.119668    0.305578    0.105583
201869690203_R03C01    0.810276    0.735967    0.721423  ...    0.751743    0.200545    0.757986
201904410008_R05C01    0.892641    0.777778    0.840475  ...    0.731927    0.636772    0.737969
201869690168_R08C01    0.811684    0.886028    0.888461  ...    0.725186    0.186773    0.812097
201530470054_R01C01    0.065221    0.050916    0.024315  ...    0.069606    0.408356    0.102764
201869690203_R06C01    0.948498    0.892334    0.859396  ...    0.776809    0.080326    0.875806
200925700133_R03C01    0.123428    0.193491    0.173582  ...    0.406198    0.754906    0.231158
201904410008_R02C01    0.868791    0.838671    0.922514  ...    0.704358    0.789430    0.617703
201904410008_R04C01    0.791002    0.818421    0.770333  ...    0.707137    0.851243    0.643282
201530470054_R02C01    0.058704    0.092474    0.122352  ...    0.173067    0.606287    0.340654
201530470054_R03C01    0.046851    0.043868    0.106606  ...    0.050913    0.281679    0.151072
201904410008_R03C01    0.718867    0.819853    0.899481  ...    0.701382    0.788357    0.659056
201870610040_R03C01    0.783809    0.905159    0.851122  ...    0.822050    0.104641    0.862166
200925700133_R02C01    0.054682    0.076460    0.089566  ...    0.087220    0.665932    0.159535
200925700125_R07C01    0.048604    0.093311    0.129750  ...    0.457459    0.867225    0.323574
201530470054_R04C01    0.048784    0.036592    0.048904  ...    0.105748    0.214249    0.056179
201870610040_R04C01    0.870172    0.896742    0.842515  ...    0.659235    0.134309    0.786843
200925700133_R05C01    0.048411    0.153956    0.164447  ...    0.462702    0.662564    0.290133
201904410008_R06C01    0.825491    0.828089    0.876774  ...    0.752596    0.833333    0.645860

[20 rows x 5000 columns]
"""

# To perform the UMAP algorithm:
methyl_analysis.compute_umap()

# The result of the UMAP algorithm is then stored in:
methyl_analysis.umap_df
"""
                       Umap_x    Umap_y  ...                    ID  Colour
200925700133_R04C01  6.644683 -7.024573  ...  REFERENCE_SAMPLE 367  2E3092
201530470054_R05C01  8.254219 -8.493701  ...  REFERENCE_SAMPLE 109  6B66AE
201869690203_R03C01  5.185830 -5.596291  ...   REFERENCE_SAMPLE 78  6282C2
201904410008_R05C01  7.058737 -5.482265  ...    REFERENCE_SAMPLE 2  7C7E82
201869690168_R08C01  4.727582 -5.662395  ...   REFERENCE_SAMPLE 80  6282C2
201530470054_R01C01  8.136439 -7.440626  ...  REFERENCE_SAMPLE 113  6B66AE
201869690203_R06C01  4.729284 -5.012035  ...   REFERENCE_SAMPLE 76  6282C2
200925700133_R03C01  6.980153 -7.641444  ...  REFERENCE_SAMPLE 368  2E3092
201904410008_R02C01  7.460121 -5.769195  ...    REFERENCE_SAMPLE 5  7C7E82
201904410008_R04C01  8.094703 -5.589474  ...    REFERENCE_SAMPLE 3  7C7E82
201530470054_R02C01  7.892371 -7.894600  ...  REFERENCE_SAMPLE 112  6B66AE
201530470054_R03C01  7.611932 -8.378901  ...  REFERENCE_SAMPLE 111  6B66AE
201904410008_R03C01  7.850667 -6.202358  ...    REFERENCE_SAMPLE 4  7C7E82
201870610040_R03C01  5.266904 -4.993761  ...   REFERENCE_SAMPLE 79  6282C2
200925700133_R02C01  6.583021 -7.879497  ...  REFERENCE_SAMPLE 369  2E3092
200925700125_R07C01  6.138983 -7.349401  ...  REFERENCE_SAMPLE 370  2E3092
201530470054_R04C01  8.566432 -7.849686  ...  REFERENCE_SAMPLE 110  6B66AE
201870610040_R04C01  5.656568 -5.504724  ...   REFERENCE_SAMPLE 77  6282C2
200925700133_R05C01  7.385612 -7.223170  ...  REFERENCE_SAMPLE 366  2E3092
201904410008_R06C01  7.607561 -5.099674  ...    REFERENCE_SAMPLE 1  7C7E82

[20 rows x 12 columns]
"""

# To generate the UMAP plot:
methyl_analysis.make_umap_plot()

# The Plotly object for the UMAP plot is then available in
# `methyl_analysis.umap_plot`.


# **3. CN-summary plots**

# Mepylome has the ability to calculate CN-summary plots, which provide an
# overview of copy number variations (CNV) across samples. To calculate these
# plots, you need to activate segmentation by setting 'do_seg' to True when
# initializing the analysis.
methyl_analysis = MethylAnalysis(
    analysis_dir=ANALYSIS_DIR,
    reference_dir=REFERENCE_DIR,
    do_seg=True,
)

# The annotation dataframe contains metadata for each sample, including
# diagnosis information.
annotation_df = methyl_analysis.idat_handler.samples_annotated

# Loop through each unique diagnosis in the dataset to generate CN-summary
# plots for all diagnosis categories.
for diagnosis in annotation_df["Diagnosis"].unique():
    ## Filter sample IDs for the current diagnosis group.
    sample_ids = annotation_df[annotation_df["Diagnosis"] == diagnosis].index
    ## Generate CN-summary plot and retrieve data (if needed for further
    ## analysis).
    cn_plot, df_cn_summary = methyl_analysis.cn_summary(sample_ids)
    ## Update the plot layout with a title for the diagnosis group and label
    ## the y-axis.
    cn_plot.update_layout(
        title=f"CN-summary for {diagnosis}",
        yaxis_title="Proportion of CNV gains/losses",
    )
    ## Display the CN-summary plot in the browser.
    cn_plot.show()


# Supervised Classification
# -------------------------


# **1. Preimplemented Classifier**

# Methylome allows the use of preimplemented classifiers to predict methylation
# classes for samples. Below are examples of performing predictions on sample
# IDs and randomly generated values. Do setup as before:
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
)
from sklearn.feature_selection import (
    SelectKBest,
)
from sklearn.pipeline import Pipeline

from mepylome.analysis import MethylAnalysis

methyl_analysis = MethylAnalysis(
    analysis_dir=ANALYSIS_DIR,
    reference_dir=REFERENCE_DIR,
)

# Perform prediction on sample IDs:

## Retrieve all sample IDs
ids = methyl_analysis.idat_handler.ids
## Set the annotation column for classification
methyl_analysis.idat_handler.selected_columns = ["Diagnosis"]
## Perform classification using a preimplemented classifier (pipeline with no
## scaler, SelectKBest as selector and RandomForestClassifier for
## classification) and cross-validation
clf_out = methyl_analysis.classify(ids=ids, clf_list="none-kbest-rf")

## Output accuracy scores:
print("Accuracy Scores:", clf_out[0].metrics["accuracy_scores"])
"""
Accuracy Scores: [1.0, 1.0, 1.0, 1.0, 1.0]
"""
## Detailed classifier report of first sample:
print(clf_out[0].reports["txt"][0])
"""
201530470054_R05C01
===================

Pipeline Structure:
- Scaler            : passthrough
- Feature_selection : SelectKBest
    k: 10000
- Classifier        : RandomForestClassifier

Metrics:
- Method    : 5-fold cross validation
- Samples   : 20
- Features  : 865859
- Accuracy  : 1.0000 (SD 0.0000)
- AUC       : 1.0000 (SD 0.0000)
- F1-Score  : 1.0000 (SD 0.0000)
- Precision : 1.0000 (SD 0.0000)
- Recall    : 1.0000 (SD 0.0000)

Classification Probability:
--------------------------------------
- Osteoblastoma             :  74.00 %
- Osteosarcoma (high-grade) :  23.00 %
- Control (muscle tissue)   :   2.00 %
- Chondrosarcoma            :   1.00 %
--------------------------------------
"""

# Perform prediction on beta values:

## Number of CpGs in the dataset
methyl_analysis.set_betas()
n_cpgs = methyl_analysis.betas_all.shape[1]
## Generate random beta values for 10 artificial samples
random_beta_values = pd.DataFrame(
    np.random.rand(10, n_cpgs), columns=methyl_analysis.betas_all.columns
)
## Perform classification on random values:
clf_out = methyl_analysis.classify(
    values=random_beta_values, clf_list="none-kbest-rf"
)

# **2. Train Your Own Classifiers**

# You can train custom classifiers using the scikit-learn API and incorporate
# them into the analysis. Below is an example using a Random Forest classifier.

## Set the annotation column for classification
methyl_analysis.idat_handler.selected_columns = ["Diagnosis"]

## Extract features (X) and target labels (y)
X = methyl_analysis.betas_all
y = methyl_analysis.idat_handler.features()

## Exclude test indices and invalid samples
valid_indices = [
    i
    for i, (idx, label) in enumerate(zip(X.index, y))
    if label and idx not in methyl_analysis.idat_handler.test_ids
]
X = X.iloc[valid_indices]
y = [y[i] for i in valid_indices]

## Choose classifier (maybe with tuned parameters)
rf_clf = RandomForestClassifier(
    n_estimators=500,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    bootstrap=True,
    class_weight="balanced",
    random_state=42,
)

## Train the classifier
rf_clf.fit(X, y)

## If the classifier is already trained, mepylome will not repeat the training
## (and cross-validation) process
ids = methyl_analysis.idat_handler.ids
clf_out = methyl_analysis.classify(ids=ids, clf_list=rf_clf)

print(clf_out[0].reports["txt"][0])
"""
201530470054_R05C01
===================

Pipeline Structure:
- Classifier: RandomForestClassifier
    class_weight: balanced
    max_depth: 20
    min_samples_leaf: 2
    min_samples_split: 5
    n_estimators: 500
    random_state: 42

Metrics:
- Samples  : 20
- Features : 865859

Classification Probability:
--------------------------------------
- Osteoblastoma             :  83.19 %
- Osteosarcoma (high-grade) :   9.87 %
- Control (muscle tissue)   :   4.49 %
- Chondrosarcoma            :   2.45 %
--------------------------------------
"""

# **3. Use Untrained Scikit-Learn Classifiers**

# Untrained classifiers can also be passed directly for classification.
# Mepylome will then execute the training and cross-validation process. Below
# is an example using an Extra Trees classifier.

## Define an untrained Extra Trees classifier
et_clf = ExtraTreesClassifier(n_estimators=300, random_state=0)

## Perform classification
ids = methyl_analysis.idat_handler.ids
clf_out = methyl_analysis.classify(ids=ids, clf_list=et_clf)

print(clf_out[0].reports["txt"][0])
"""
201530470054_R05C01
===================

Pipeline Structure:
- Classifier: ExtraTreesClassifier
    n_estimators: 300
    random_state: 0

Metrics:
- Method    : 5-fold cross validation
- Samples   : 20
- Features  : 865859
- Accuracy  : 1.0000 (SD 0.0000)
- AUC       : 1.0000 (SD 0.0000)
- F1-Score  : 1.0000 (SD 0.0000)
- Precision : 1.0000 (SD 0.0000)
- Recall    : 1.0000 (SD 0.0000)

Classification Probability:
--------------------------------------
- Osteoblastoma             :  77.00 %
- Osteosarcoma (high-grade) :  13.00 %
- Control (muscle tissue)   :   6.00 %
- Chondrosarcoma            :   4.00 %
--------------------------------------
"""


# **4. Create a Custom Classifier**

# You can define a custom trained classifier by implementing a class that
# inherits from `TrainedClassifier`. This allows you to use non-standard models
# or apply additional functionality.

## Define a Custom Classifier
from mepylome.analysis.methyl_clf import TrainedClassifier


class CustomClassifier(TrainedClassifier):
    def __init__(self, clf):
        self.clf = clf
        self._classes = clf.classes_

    def predict_proba(self, betas, id_=None):
        return self.clf.predict_proba(betas)

    def classes(self):
        return self._classes

    def info(self, output_format="txt"):
        return "This text will be printed in reports."

    def metrics(self):
        return {"Key0": "Value0", "Key1": "Value1"}


## Initialize the custom classifier
custom_clf = CustomClassifier(rf_clf)

## Perform classification
clf_out = methyl_analysis.classify(ids=ids, clf_list=custom_clf)

print(clf_out[0].reports["txt"][0])
"""
201530470054_R05C01
===================

This text will be printed in reports.

Classification Probability:
--------------------------------------
- Osteoblastoma             :  83.19 %
- Osteosarcoma (high-grade) :   9.87 %
- Control (muscle tissue)   :   4.49 %
- Chondrosarcoma            :   2.45 %
--------------------------------------
"""


# **5. Add a Classifier into the MethylAnalysis Object**

# Finally, you can integrate a classifier directly into the `MethylAnalysis`
# object for seamless use in workflows and the interactive GUI app.

## Add a custom classifier into the MethylAnalysis object
pipeline = Pipeline(
    [
        ("feature_selection", SelectKBest(k=20000)),
        ("classifier", RandomForestClassifier(n_estimators=500)),
    ]
)
methyl_analysis.classifiers = {"model": pipeline, "name": "Custom RF"}

## Launch the interactive app with the integrated classifier. Now the classifier
## is available in the GUI under the name 'Custom RF'.
methyl_analysis.run_app(open_tab=True)


# For further details and advanced usage, refer to the mepylome documentation.
