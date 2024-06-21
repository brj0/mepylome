<picture>
  <img alt="Mepylome Logo" src="https://raw.githubusercontent.com/brj0/mepylome/main/mepylome/data/assets/mepylome.svg" width="300">
</picture>

-----------------


# Mepylome: Methylation Array Analysis Toolkit


Mepylome is an efficient Python toolkit tailored for parsing, processing, and
analyzing methylation array IDAT files. Serving as a versatile library,
Mepylome supports a wide range of methylation analysis tasks. It also includes
an interactive GUI that enables users to generate UMAP plots and CNV plots
(Copy Number Variation) directly from collections of IDAT files.

**Note: Mepylome is still under construction.**


## Features

- Support for Illumina array types: 450k, EPIC, EPICv2
- Parsing of IDAT files
- Extraction of methylation signals
- Calculation of copy number variations (CNV) with Plotly plots visualization
- Methylation analysis tool with a graphical browser interface for UMAP analysis and CNV plots
  - Can be run from the command line with minimal setup or customized through a Python script


## Installation


### From PyPI

You can install mepylome directly from PyPI using pip:

```sh
pip install mepylome
```

### From Source

If you want the latest version, you can download mepylome directly from the source:

```sh
git clone https://github.com/brj0/mepylome.git && cd mepylome && pip install .
```


### CNV Segments

If you want to perform segmentation on the CNV plot (horizontal lines identifying significant changes), additional packages are needed. These packages require a C compiler and can be installed with the following command:


```sh
pip install numpy==1.26.4 cython ailist==1.0.4 cbseg
```


## Usage

### Parsing and methylation extraction


```python
from pathlib import Path
from mepylome import CNV, MethylData, RawData

DIR = Path("/path/to/idat/directory")

# Sample
sample = DIR / "samples" / "203049640041_R04C01_Grn.idat"

# CNV neutral reference files
reference_dir = DIR / "references"

# Extract the signals from the idat files
raw_sample = RawData(sample)
raw_reference = RawData(reference_dir)

# Get methylation information
sample_methyl = MethylData(raw_sample)
reference_methyl = MethylData(raw_reference)

# Beta value
beta = sample_methyl.beta

# CNV anylsis
cnv = CNV.set_all(sample_methyl, reference_methyl)

# Visualize CNV in the browser
cnv.plot()
```


### Methylation analysis: Command-line interface

To perform the analysis, you must define an analysis directory that contains the IDAT files you want to analyze. Additionally, you need an annotation file (preferably in CSV format rather than XLSX) with a header where the first column is the Sentrix ID. It is best to place this annotation file within the analysis directory. Furthermore, you should have a directory with CNV-neutral reference cases for CNV analysis.

#### Basic usage:

To start the interface, run the following command (you'll need to manually copy directories into the interface):

```sh
mepylome
```

#### Prefered usage:

For a more streamlined experience, specify the analysis IDAT files directory, reference IDAT directory, and CpG array type. This command also improves UMAP speed by saving betas to disk:

```sh
mepylome -a /path/to/idats -r /path/to/ref -c 450k -s
```

#### Show All Parameters
To display all available command-line parameters, use:

```sh
mepylome --help
```


## C++ parser
Mepylome also includes a C++ parser (`_IdatParser`) with Python bindings. Due to no significant speed gain, it is currently not included by default. To enable it, install from source after you execute the following command:


```sh
export MEPYLOME_CPP=1
```

## Contributing

Contributions are welcome! If you have any bug reports, feature requests, or suggestions, please open an issue or submit a pull request.


## License

This project is licensed under the [GPL-3.0 license](LICENSE).


## Acknowledgements

Mepylome is strongly influenced by [minfi](https://github.com/hansenlab/minfi) and [conumee2](https://github.com/hovestadtlab/conumee2). Some functionalities, such as the manifest handler and parser, are adapted from [methylprep](https://github.com/FoxoTech/methylprep).
