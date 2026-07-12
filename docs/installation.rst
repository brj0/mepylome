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


Uninstallation
--------------

To uninstall Mepylome:

1. Run:

   .. code-block:: sh

      pip uninstall mepylome

2. Delete the following directories:

   - ~/.cache/mepylome (Contains cached files)
   - ~/mepylome  (Contains tutorial/example files)


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

