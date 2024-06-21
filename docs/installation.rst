Installation
============


From PyPI
---------

You can install `mepylome` directly from PyPI using pip:

.. code-block:: sh

    pip install mepylome

From Source
-----------

If you want the latest version, you can download `mepylome` directly from the source:

.. code-block:: sh

    git clone https://github.com/brj0/mepylome.git && cd mepylome && pip install .

CNV Segments
------------

If you want to perform segmentation on the CNV plot (horizontal lines identifying significant changes), additional packages are needed. These packages require a C compiler and can be installed with the following command:

.. code-block:: sh

    pip install numpy==1.26.4 cython ailist==1.0.4 cbseg
