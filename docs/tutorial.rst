Tutorial
========



This tutorial will guide you through the steps to analyze methylation data
using the ``mepylome`` package. We'll cover setting up the environment,
parsing IDAT files, working with manifest files, processing raw data,
performing copy number variation (CNV) analysis, and conducting methylation
analysis using UMAP plots with a GUI. You can find the code of this tutorial
as a script in
https://github.com/brj0/mepylome/blob/main/tests/rtd_tutorial.py.


.. contents:: Contents of Tutorial
   :depth: 3



Setup
-----


First, import the required modules and set up the necessary directories:


.. code-block:: python

    >>> from pathlib import Path

    >>> from mepylome import (
    >>>     CNV,
    >>>     IdatParser,
    >>>     Manifest,
    >>>     MethylData,
    >>>     RawData,
    >>>     ReferenceMethylData,
    >>> )


Setup directories for the tutorial:

.. code-block:: python

    >>> DIR = Path.home() / "Documents" / "mepylome" / "tutorial"
    >>> ANALYSIS_DIR = DIR / "tutorial_analysis_idat"
    >>> REFERENCE_DIR = DIR / "tutorial_reference_idat"


Download and setup the necessary data files (approximately 653 MB):

.. code-block:: python

    >>> from mepylome.utils import setup_tutorial_files

    >>> setup_tutorial_files(ANALYSIS_DIR, REFERENCE_DIR)




Parsing IDAT files
------------------


Define the IDAT file path:

.. code-block:: python

    >>> idat_file = ANALYSIS_DIR / "200925700125_R07C01_Grn.idat"


Parse the IDAT file:

.. code-block:: python

    >>> idat_data = IdatParser(idat_file)


Print overview of parsed data:

.. code-block:: python

    >>> print(idat_data)
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



The parsed data is available as attributes of the ``IdatParser`` object. For
example the  Illumina IDs (probes IDs) can be accessed by:

.. code-block:: python

    >>> ids = idat_data.illumina_ids

    >>> print(ids)
    [ 1600101  1600111  1600115 ... 99810978 99810990 99810992]





C++ Parser
~~~~~~~~~~


If you installed mepylome with C++ support (see `installation
<installation.html>`_) you can also use the C++ parser (input must be a
string, not a Path object)

.. code-block:: python

    >>> try:
    >>>     from mepylome import _IdatParser

    >>>     _idat_data = _IdatParser(str(idat_file))
    >>>     print("C++ parser available")

    >>> except ImportError:
    >>>     print("C++ parser NOT available")




Manifest files
--------------


The mepylome package includes a ``Manifest`` class that provides
functionality to download, process, and save Illumina manifest files
internally in a efficient format (stored in ~/.mepylome). These manifest
files contain information about the CpG sites on the methylation array,
including genetic coordinates, probe types, and more.

Load the available manifest files for different array types.

.. code-block:: python

    >>> manifest_450k = Manifest("450k")
    >>> manifest_epic = Manifest("epic")
    >>> manifest_epic_v2 = Manifest("epicv2")


.. note::

    The first time you run this, the manifest files will be downloaded and
    saved locally to ~/.mepylome. This initial download might take some time.

Obtain values from attributes:

.. code-block:: python

    >>> probes_df = manifest_450k.data_frame
    >>> controls_df = manifest_450k.control_data_frame


Print overview:

.. code-block:: python

    >>> print(probes_df)
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





RawData
-------


The ``RawData`` class extracts both raw green and raw red signal intensity
data from a IDAT file pair. You can initialize it using a base path to the
IDAT files (without the _Grn.idat / _Red.idat suffix), or by providing the
full path to either the Grn or Red IDAT file.

.. code-block:: python

    >>> idat_file = ANALYSIS_DIR / "200925700125_R07C01_Red.idat"
    >>> # or
    >>> idat_file = ANALYSIS_DIR / "200925700125_R07C01_Grn.idat"
    >>> # or
    >>> idat_file = ANALYSIS_DIR / "200925700125_R07C01"
    >>> raw_data = RawData(idat_file)


The data is saved within the following attributes:


.. code-block:: python

    >>> # Intensity signals
    >>> raw_data.grn
    >>> raw_data.red

    >>> # Type of the array_type (e.g., 450k, EPIC)
    >>> raw_data.array_type

    >>> # Corresponding manifest file
    >>> raw_data.manifest

    >>> # IDs on the bead
    >>> raw_data.ids


Print an overview of the raw data

.. code-block:: python

    >>> print(raw_data)
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



RawData can also read multiple files of the same array type (used for
reference files):

.. code-block:: python

    >>> idat_file0 = ANALYSIS_DIR / "200925700125_R07C01_Grn.idat"
    >>> idat_file1 = ANALYSIS_DIR / "200925700133_R02C01"

    >>> raw_data_2 = RawData([idat_file0, idat_file1])


Alternatively, read all IDAT files in a directory (supports recursive
search):

.. code-block:: python

    >>> raw_data_all = RawData(REFERENCE_DIR)




MethylData
----------

The ``MethylData`` class allows for processing raw intensity data and can
calculate methylation signals as well as beta values. The raw data can be
preprocessed using one of the following methods: 'illumina' (default),
'swan', or 'noob'. Initialize MethylData with raw data using the default
'illumina' preprocessing method.

.. code-block:: python

    >>> methyl_data = MethylData(raw_data)

    >>> methyl_data_all = MethylData(raw_data_all)


Alternatively, you can explicitly specify the 'illumina' preprocessing
method.

.. code-block:: python

    >>> methyl_data = MethylData(raw_data, prep="illumina")


You can also initialize MethylData directly from an IDAT file path, without
using ``RawData``. This is the preferred method if you want to obtain
methylation signals or beta values.

.. code-block:: python

    >>> methyl_data = MethylData(file=idat_file)


Obtain various values via the attributes of the MethylData object:


.. code-block:: python

    >>> # The methylation signals for the green and red channels.
    >>> methylated_signals = methyl_data.methylated
    >>> unmethylated_signals = methyl_data.unmethylated

    >>> # The corrected color signals.
    >>> corrected_green_signals = methyl_data.grn
    >>> corrected_red_signals = methyl_data.red

    >>> # The type of the array used (e.g., 450k, EPIC, EPICv2).
    >>> array_type = methyl_data.array_type

    >>> # The corresponding manifest file.
    >>> corresponding_manifest = methyl_data.manifest


Print an overview of the methylation data.

.. code-block:: python

    >>> print(methyl_data)
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



Beta values are a indicator wheather a CpG is methylated or not. They can be
calculated for all sites of the corresponding array:

.. code-block:: python

    >>> betas = methyl_data.betas


You can also access the beta values at specific CpG sites (here at all the
sites of the EPICv2 manifest). Missing data will be replaced with `fill`.

.. code-block:: python

    >>> epicv2_cpgs = manifest_epic_v2.methylation_probes
    >>> beta_specific = methyl_data.betas_at(epicv2_cpgs, fill=0.5)




Using Alternative Preprocessing Methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Preprocess the raw data using the SWAN method.

.. code-block:: python

    >>> methyl_data_swan = MethylData(raw_data, prep="swan")


Preprocess the raw data using the NOOB method.

.. code-block:: python

    >>> methyl_data_noob = MethylData(raw_data, prep="noob")



See `api <api.html>`_ for more information about SWAN and NOOB.



Copy number variation (CNV)
---------------------------


Copy number variations (CNV) are significant alterations in the genome
involving the loss or gain of large DNA segments, often encompassing multiple
genes. These variations are frequently linked to cancer development and can
aid in tumor classification. The CNV profile can be calculated from signal
intensity using methylation arrays. With the mepylome package, CNV can be
efficiently calculated and visualized.


**1. Set up analysis file**

First, initialize your sample data for analysis:

.. code-block:: python

    >>> sample = methyl_data



**2. Set Up Reference Data**

Within the reference directory there must be multiple CNV-neutral IDAT
pairs of the **same array type** as `sample`.

.. code-block:: python

    >>> reference = MethylData(file=REFERENCE_DIR)


Alternatively, if the reference directory contains IDAT files of multiple
array types, you can use ``ReferenceMethylData`` to load all files into
memory. This way, the reference object can be used for multiple array types.
The CNV class will automatically extract the files for the needed array type.

.. code-block:: python

    >>> reference_all = ReferenceMethylData(REFERENCE_DIR)



**3. Initialize CNV Analysis**

Create an instance of the CNV class for the analysis, and fit the data (this
is basically a linear regression model comparing `sample` signal with the
`reference` signals at each CpG site):

.. code-block:: python

    >>> cnv = CNV(sample, reference)

    >>> # Alternative with ReferenceMethylData
    >>> cnv = CNV(sample, reference_all)



**4. Calculate CNV for Bins and Genes**

Compute CNV values for genomic bins and gene regions:

.. code-block:: python

    >>> cnv.set_bins()
    >>> cnv.set_detail()



**5. Calculate CNV Segments**

Use the binary circular segmentation algorithm for genome segmentation:

.. code-block:: python

    >>> cnv.set_segments()


.. note::

    For this step, additional packages must be installed (see `installation
    <installation.html>`_).


**6. Streamlined Analysis**

Alternatively, perform all CNV computations in a single call:

.. code-block:: python

    >>> cnv = CNV.set_all(sample, reference)

    >>> # or
    >>> cnv = CNV.set_all(sample, reference_all)



**7. Visualize CNV Data**

Display an interactive plot using Plotly, where genes can be highlighted:

.. code-block:: python

    >>> cnv.plot()



Methylation Analysis
--------------------

under construction...
