<img alt="Mepylome Logo" src="https://raw.githubusercontent.com/brj0/mepylome/main/mepylome/data/assets/mepylome.svg" width="300">


[![PyPI version](https://badge.fury.io/py/mepylome.svg)](https://badge.fury.io/py/mepylome)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/mepylome.svg)
[![Documentation Status](https://readthedocs.org/projects/mepylome/badge/?version=latest)](https://mepylome.readthedocs.io/en/latest/?badge=latest)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/brj0/mepylome/blob/main/examples/tutorial.ipynb)

-----------------


# Mepylome: Ultra-Fast Methylation Array Analysis Toolkit

Mepylome is a high-performance Python toolkit tailored for parsing, processing,
and analyzing methylation array IDAT files. Serving as a versatile library,
Mepylome supports a wide range of methylation analysis tasks. It also includes
an interactive GUI that enables users to generate UMAP plots and CNV plots
(Copy Number Variation) directly from collections of IDAT files.
Mepylome is designed for **large-scale cohort processing and delivers
substantial speed improvements over existing methylation analysis pipelines.**


## ✨ Key Highlights

- ⚡ **Very fast processing engine**
- 🧬 Supports Illumina **450k, EPIC, EPICv2, 27k, MSA48, and Mouse arrays**
- 📊 CNV analysis with interactive Plotly visualizations
- 🧹 Clean API for large-scale cohort processing
- 🖥️ Optional GUI for UMAP + CNV exploration
- 🧪 Built-in QC including **pOOBAH detection and quality metrics**


## Features

### Data Processing
- Parsing of IDAT files
- Extraction of methylation signals

### Copy Number Analysis
- Calculation of Copy Number Variations (CNV) with visualization using
  [plotly](https://github.com/plotly/plotly.py).
- Interactive visualization using Plotly
- CN-summary plots

### Supported Array Types
- Illumina EPIC / EPICv2
- Illumina 450k
- Illumina 27k
- Illumina MSA48
- Illumina Mouse

### Performance
- Significantly **faster** compared to
  - [minfi](https://github.com/hansenlab/minfi),
  - [conumee2](https://github.com/hovestadtlab/conumee2) and
  - [sesame](https://github.com/zwdzwd/sesame).

### Quality Control
- pOOBAH-based probe calculation
- Quality metrics for samples and probes

### GUI
- Methylation analysis tool with a graphical browser interface for UMAP
  analysis, CNV plots and supervised classification
  - Can be run from the command line with minimal setup or customized through a
    Python script


## Documentation

The mepylome documentation, including installation instructions, tutorial and API, is available at <https://mepylome.readthedocs.io/>


## Usage

### Methylation extraction and copy number variation plots

```python
from pathlib import Path

from mepylome import CNV, MethylData

# Sample
analysis_dir = Path("/path/to/idat/directory")
sample_file = analysis_dir / "200925700125_R07C01"

# CNV neutral reference files
reference_dir = Path("/path/to/reference/directory")

# Get methylation data
sample_methyl = MethylData(file=sample_file)
reference_methyl = MethylData(file=reference_dir)

# Beta value
betas = sample_methyl.betas

# Print overview of processed data
print(sample_methyl)

# CNV anylsis
cnv = CNV.set_all(sample_methyl, reference_methyl)

# Visualize CNV in the browser
cnv.plot()
```

### Methylation analysis: Command-line interface and GUI

<img alt="Mepylome Logo" src="https://raw.githubusercontent.com/brj0/mepylome/main/docs/images/screenshot.png">

#### Basic usage:

Mepylome provides a command-line interface for launching a GUI and performing
methylation analysis. Ensure you have an analysis directory, a CNV reference
directory, and an annotation file (located within the analysis directory). Use
the following command to initiate the analysis:

```sh
mepylome --analysis_dir /path/to/idats --reference_dir /path/to/ref
```

If you want to perform a **quick test**, use:

```sh
mepylome --tutorial
```

This command downloads sample IDAT files and provides a demonstration of the
package's functionality.

You can try the tutorial directly in **Google Colab**-without downloading or
installing anything-by clicking the link below. Please note that GUI support is
limited in Colab.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/brj0/mepylome/blob/main/examples/tutorial.ipynb)


See <https://mepylome.readthedocs.io/> for details.


## Publication

This library is described in the following peer-reviewed publication:

https://doi.org/10.1002/aisy.202500778


## Contributing

Contributions are welcome! If you have any bug reports, feature requests, or suggestions, please open an issue or submit a pull request.


## License

This project is licensed under the [MIT license](LICENSE).


## Acknowledgements

Mepylome is strongly influenced by [minfi](https://github.com/hansenlab/minfi) and [conumee2](https://github.com/hovestadtlab/conumee2). Some functionalities, such as the manifest handler and parser, are adapted from [methylprep](https://github.com/FoxoTech/methylprep).
