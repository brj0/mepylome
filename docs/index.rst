.. image:: ../mepylome/data/assets/mepylome.svg
   :align: center
   :width: 250

----



Mepylome: Ultra-Fast Methylation Array Analysis Toolkit
=======================================================


Mepylome is a high-performance Python toolkit tailored for parsing, processing,
and analyzing methylation array IDAT files. Serving as a versatile library,
Mepylome supports a wide range of methylation analysis tasks. It also includes
an interactive GUI that enables users to generate UMAP plots and CNV plots
(Copy Number Variation) directly from collections of IDAT files.
Mepylome is designed for **large-scale cohort processing and delivers
substantial speed improvements over existing methylation analysis pipelines.**

Mepylome is open source, and hosted at github: https://github.com/brj0/mepylome



✨ Key Highlights
~~~~~~~~~~~~~~~~~

- ⚡ **Very fast processing engine**
- 🧬 Supports Illumina **450k, EPIC, EPICv2, 27k, MSA48, and MM285 arrays**
- 📊 CNV analysis with interactive Plotly visualizations
- 🧹 Clean API for large-scale cohort processing
- 🖥️ Optional GUI for UMAP + CNV exploration
- 🧪 Built-in QC including **pOOBAH detection and quality metrics**



Features
~~~~~~~~

- Parsing of IDAT files
- Extraction of methylation signals
- Calculation of Copy Number Variations (CNV) with visualization using
  `plotly <https://github.com/plotly/plotly.py>`_
- Interactive CNV visualization in the browser
- CN-summary plots
- Supported Array Types

   - Illumina EPIC / EPICv2
   - Illumina 450k
   - Illumina 27k
   - Illumina MSA48
   - Illumina MM285 (Mouse)
- Significantly **faster** compared to:

  - `minfi <https://github.com/hansenlab/minfi>`_
  - `conumee2 <https://github.com/hovestadtlab/conumee2>`_
  - `sesame <https://github.com/zwdzwd/sesame>`_
- pOOBAH-based probe filtering
- Sample- and probe-level quality metrics
- Methylation analysis tool with a graphical browser interface for:

  - UMAP analysis
  - CNV plots
  - Supervised classification

  Can be run from the command line with minimal setup or customized through a
  Python script.



Publication
~~~~~~~~~~~

This library is described in the following peer-reviewed publication:

https://doi.org/10.1002/aisy.202500778



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
