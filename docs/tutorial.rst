Tutorial
========


This tutorial will guide you through the steps to analyze methylation data
using the mepylome package. We'll cover setting up the environment, parsing
IDAT files, working with manifest files, processing raw data, and performing
copy number variation (CNV) analysis. You can find the code of this tutorial
as a script in ``mepylome/tests/tutorial_basic.py``.


Prerequisites
-------------

Before starting, ensure you have the mepylome package installed. You can
install it using pip:

.. code:: bash

   pip install mepylome


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


Setup directories for the tutorial

.. code-block:: python

    >>> DIR = Path.home() / "Documents" / "mepylome" / "projects" / "basic"
    >>> ANALYSIS_DIR = DIR / "analysis"
    >>> REFERENCE_DIR = DIR / "reference"


Download and setup the necessary data files (approximately 653 MB)

.. code-block:: python

    >>> from mepylome.tutorial import setup_tutorial_files

    >>> setup_tutorial_files(DIR)



Parsing IDAT files
------------------


Define the IDAT file path

.. code-block:: python

    >>> idat_file = ANALYSIS_DIR / "200925700125_R07C01_Grn.idat"


Parse the IDAT file

.. code-block:: python

    >>> idat_data = IdatParser(idat_file)



If you installed mepylome with C++ support (see installation) you can
also use the C++ parser (input must be a string, not a Path object)

.. code-block:: python

    >>> from mepylome import _IdatParser

    >>> _idat_data = _IdatParser(str(idat_file))


Display parsed data

.. code-block:: python

    >>> print(idat_data)


You should see the following output:

.. code-block:: python

    >>> idat_data
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

The parsed data is available as attributes of the ``IdatParser`` object.

Access the Illumina IDs (probes IDs) from the parsed IDAT file

.. code-block:: python

    >>> ids = idat_data.illumina_ids

    >>> print(ids)



Manifest files
--------------


Load the available manifest files for different array types.

.. code-block:: python

    >>> manifest_450k = Manifest("450k")
    >>> manifest_epic = Manifest("epic")
    >>> manifest_epic_v2 = Manifest("epicv2")


.. note::

    The first time you run this, the manifest files will be downloaded and
    saved locally to ~/.mepylome. This initial download might take some time.

Obtain values from attributes

.. code-block:: python

    >>> probes_df = manifest_450k.data_frame
    >>> controls_df = manifest_450k.control_data_frame


Print overview

.. code-block:: python

    >>> print(manifest_450k)



RawData
-------


The ``RawData`` class handles both green and red signal intensity data from
IDAT files. You can initialize it using a base path to the IDAT files
(without the _Grn.idat / _Red.idat suffix), or by providing the full path to
either the Grn or Red IDAT file.

.. code-block:: python

    >>> idat_file = ANALYSIS_DIR / "200925700125_R07C01"
    >>> raw_data = RawData(idat_file)


The data is saved within the following attributes:
  - raw_data.grn: Green signal intensities
  - raw_data.red: Red signal intensities
  - raw_data.array_type: Type of the array (e.g., 450k, EPIC)
  - raw_data.manifest: Corresponding manifest file
  - raw_data.ids: IDs on the bead

Print an overview of the raw data

.. code-block:: python

    >>> print(raw_data)



This function also accepts basepaths (without the _Grn.idat / _Red.idat
ending:

.. code-block:: python

    >>> idat_file = ANALYSIS_DIR / "200925700125_R07C01"
    >>> raw_data = RawData(idat_file)



RawData can also read multiple files, such as reference files:

.. code-block:: python

    >>> idat_file0 = ANALYSIS_DIR / "200925700125_R07C01"
    >>> idat_file1 = ANALYSIS_DIR / "200925700133_R02C01"

    >>> raw_data_2 = RawData([idat_file0, idat_file1])


Alternatively, read all IDAT files in a directory (supports recursive
search):

.. code-block:: python

    >>> raw_data_all = RawData(ANALYSIS_DIR)

    >>> print(raw_data)



MethylData
----------


The raw data can be preprocessed using one of the following methods:
'illumina' (default), 'swan', or 'noob'. Initialize MethylData with raw data
using the default 'illumina' preprocessing method.


.. code-block:: python

    >>> methyl_data = MethylData(raw_data)

    >>> methyl_data_all = MethylData(raw_data_all)


Alternatively, you can explicitly specify the 'illumina' preprocessing
method.

.. code-block:: python

    >>> methyl_data = MethylData(raw_data, prep="illumina")


You can also initialize MethylData directly from an IDAT file path, without
using ``RawData``.

.. code-block:: python

    >>> methyl_data = MethylData(file=idat_file)


Obtain various values via the attributes of the MethylData object:

Access the beta values.

.. code-block:: python

    >>> beta = methyl_data.beta


Access the beta values at specific CpG sites (here at all the sites of the
EPICv2 manifest. You can fill missing values with a specified value (e.g.,
0.5).

.. code-block:: python

    >>> epicv2_cpgs = manifest_epic_v2.methylation_probes
    >>> beta_specific = methyl_data.betas_for_cpgs(epicv2_cpgs, fill=0.5)


Access the methylation signals for the green and red channels.

.. code-block:: python

    >>> methylated_signals = methyl_data.methylated
    >>> unmethylated_signals = methyl_data.unmethylated


Access the corrected color signals.

.. code-block:: python

    >>> corrected_green_signals = methyl_data.grn
    >>> corrected_red_signals = methyl_data.red


Access the type of the array used (e.g., 450k, EPIC).

.. code-block:: python

    >>> array_type = methyl_data.array_type


Access the corresponding manifest file.

.. code-block:: python

    >>> corresponding_manifest = methyl_data.manifest


Print an overview of the methylation data.

.. code-block:: python

    >>> print(methyl_data)


Preprocess the raw data using the SWAN or NOOB method.

.. code-block:: python

    >>> methyl_data_swan = MethylData(raw_data, prep="swan")
    >>> methyl_data_noob = MethylData(raw_data, prep="noob")



Print overview

.. code-block:: python

    >>> print(methyl_data)



Using Alternative Preprocessing Methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Preprocess the raw data using the SWAN method.

.. code-block:: python

    >>> methyl_data_swan = MethylData(raw_data, prep="swan")


Preprocess the raw data using the NOOB method.

.. code-block:: python

    >>> methyl_data_noob = MethylData(raw_data, prep="noob")



Copy number variation (CNV)
---------------------------


**1. Set up analysis file**

First, initialize your sample data for analysis:

.. code-block:: python

    >>> sample = methyl_data



**2. Set Up Reference Data**

Ensure the reference data matches the sample array type:

.. code-block:: python

    >>> reference = MethylData(file=REFERENCE_DIR)

    >>> # Confirm array types match
    >>> assert sample.array_type == reference.array_type


Alternatively if the reference directory contains IDAT files of multiple
array types you can use ReferenceMethylData to load all files into memory.
The CNV class will then extract the files of the needed array.

.. code-block:: python

    >>> reference_all = ReferenceMethylData(REFERENCE_DIR)


**3. Initialize CNV Analysis**

Create an instance of the CNV class for analysis, and fit data (linear
regression model):


.. code-block:: python

    >>> cnv = CNV(sample, reference)


Alternative with ReferenceMethylData

.. code-block:: python

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

    For this step, additional packages must be installed (see
    installation).

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
