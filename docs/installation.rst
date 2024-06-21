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

To perform segmentation on the CNV plot (horizontal lines identifying
significant changes), additional packages are required. These packages depend
on a C compiler. Follow the instructions below to install them based on your
Python version.

**For Python < 3.10**, install the necessary packages using the following
command:

.. code-block:: sh

    pip install numpy==1.26.4 cython ailist==1.0.4 cbseg


**For Python 3.10 and Later**, you can install the linear_segment package
instead. Use the following command:

.. code-block:: sh

    pip install linear_segment

Make sure you have a C compiler installed on your system to build these
packages.

