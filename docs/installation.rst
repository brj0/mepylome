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


Python Versions Below 3.10
~~~~~~~~~~~~~~~~~~~~~~~~~~

For Python < 3.10, install the necessary packages using the following
command:

.. code-block:: sh

    pip install cython
    pip install ailist==1.0.4
    pip install cbseg

If you encounter any compatibility issues, you may need to downgrade numpy to
version 1.26.4. Make sure you have a C compiler installed on your system to
build this package.


Python 3.10 and Later
~~~~~~~~~~~~~~~~~~~~~

For Python 3.10 and Later, you can install the `linear_segment` package
instead. Use the following command:

.. code-block:: sh

    pip install linear_segment

Make sure you have a C compiler installed on your system to build this
package.


Alternative Package
~~~~~~~~~~~~~~~~~~~

As an alternative, you can try installing the `ruptures` package, which
provides similar functionality. Its probably the fastes package. To install,
use:

.. code-block:: sh

    pip install ruptures


Important Note
~~~~~~~~~~~~~~

Only **one** of the above packages should be installed at a time to avoid conflicts.


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

