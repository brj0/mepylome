<picture>
  <img alt="Mepylome Logo" src="/mepylome/data/assets/mepylome.svg" width="300">
</picture>

-----------------


# mepylome: Methylation Array Analysis Toolkit

Mepylome is a Python toolkit for parsing, processing, and analyzing methylation array IDAT files.

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
pip install mepylome[cgb]
```

Ensure you install the 'cgb' variant after (!) the regular package (cgb is used for CNV segmentation).


### From Source

1. Clone the repository:

```sh
git clone https://github.com/brj0/mepylome.git
cd mepylome
```

2. Build and install the package:

```sh
pip install .
pip install .[cgb]
```


## Usage

### Parsing and methylation extraction
Refer to tests/basic_usage.py for a detailed example.

### Methylation analysis: Command-line interface

#### Basic usage:

To start the interface, run the following command (you'll need to manually copy directories into the interface):

```sh
mepylome
```

#### Prefered usage:

For a more streamlined experience, specify the analysis IDAT files directory, reference IDAT directory, and CpG array type. This command also improves UMAP speed by saving betas to disk:

sh
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

Mepylome is strongly influenced by [minfi] (https://github.com/hansenlab/minfi) and [conumee2] (https://github.com/hovestadtlab/conumee2). Some functionalities, such as the manifest handler and parser, are adapted from [methylprep](https://github.com/FoxoTech/methylprep).
