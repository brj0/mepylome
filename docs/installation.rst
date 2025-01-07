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

    pip install cython
    pip install ailist==1.0.4
    pip install cbseg

If you encounter any compatibility issues, you may need to downgrade numpy to
version 1.26.4.


**For Python 3.10 and Later**, you can install the linear_segment package
instead. Use the following command:

.. code-block:: sh

    pip install linear_segment

Make sure you have a C compiler installed on your system to build these
packages.


C++ parser
----------

Mepylome also includes a C++ parser (``_IdatParser``) with Python bindings. Due
to no significant speed gain, it is currently not included by default. To
enable it, install from source after you execute the following command:


.. code-block:: sh

    export MEPYLOME_CPP=1


Uninstallation
--------------

To uninstall Mepylome:

1. Run:

   .. code-block:: sh

      pip uninstall mepylome

2. Delete the following directories:

   - ~/.mepylome  (Contains manifest files)
   - ~/mepylome  (Contains tutorial/example files)
   - /tmp/mepylome  (Default output directory, if not changed)


Platform Compatibility
----------------------

This package is written for Linux and has been tested under Ubuntu.

.. warning::
    Windows users may encounter issues with the `pyranges` package, which
    affects CNV calculation, making it not possible to perform CNV calculations
    on Windows directly. However, apart from CNV calculations, other
    functionalities of the package should work on Windows.

To overcome this limitation, Windows users are advised to use Windows Subsystem
for Linux (WSL).

