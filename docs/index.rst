.. image:: ../mepylome/data/assets/mepylome.svg
   :align: center
   :width: 250

----



Mepylome: Methylation Array Analysis Toolkit
============================================


Mepylome is an efficient Python toolkit tailored for parsing, processing, and
analyzing methylation array IDAT files. Serving as a versatile library,
Mepylome supports a wide range of methylation analysis tasks. It also includes
an interactive GUI that enables users to generate UMAP plots and CNV plots
(Copy Number Variation) directly from collections of IDAT files.

Mepylome is open source, and hosted at github: https://github.com/brj0/mepylome


Features
~~~~~~~~

- Parsing of IDAT files
- Extraction of methylation signals
- Calculation of Copy Number Variations (CNV) with visualization using
  `plotly <https://github.com/plotly/plotly.py>`_
- Support for the following Illumina array types: 450k, EPIC, EPICv2
- Significantly faster compared to `minfi <https://github.com/hansenlab/minfi>`_
  and `conumee2 <https://github.com/hovestadtlab/conumee2>`_
- Methylation analysis tool with a graphical browser interface for UMAP
  analysis and CNV plots

  - Can be run from the command line with minimal setup or customized through a
    Python script
- CN-summary plots




Documentation outline
~~~~~~~~~~~~~~~~~~~~~

#. :doc:`Installation instructions <./installation>`
#. :doc:`The GUI <./gui_cli>`,  recommended for all new users
#. :doc:`The tutorial <./tutorial>`,  recommended for all new users


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   gui_cli
   tutorial
   performance
   api



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
