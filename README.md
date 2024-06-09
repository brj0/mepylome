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


```python
# 1) IDAT files

# Idat file
sample = "/data/epidip_IDAT/6042324058_R03C02_Grn.idat"

# Reference files (either basename like below or full pathname of at least
# one of the Red/Grn pairs).
reference0 = "/data/ref_IDAT/cnvrefidat_450k/3999997083_R02C02"
reference1 = "/data/ref_IDAT/cnvrefidat_450k/5775446049_R06C01"

# Alternatively a directory can be given. All files within will be used.
reference_dir = "/data/ref_IDAT/cnvrefidat_450k"


# 2) Parsing IDAT

# The idat file must contain the full path with _Red.idat/_Grn.idat ending
idat_data = IdatParser(sample)


# 3) Manifest data

manifest_450k = ManifestLoader.get_manifest("450k")
manifest_epic = ManifestLoader.get_manifest("epic")
manifest_epicv2 = ManifestLoader.get_manifest("epicv2")


# 4) The raw methylation data

raw_reference = RawData([reference0, reference1])

# Alternative using reference directory
raw_reference = RawData(reference_dir)

# Sample data must include 1 idat-pair
raw_sample = RawData(sample)


# 5) The preprocessed methylation data
sample_methyl = MethylData(raw_sample)
ref_methyl_data = MethylData(raw_reference)


# 6) Beta value
beta = sample_methyl.beta


# 7) Annotation object
annotation = Annotation(manifest, gap=gap, detail=genes)


# 8) CNV
cnv = CNV(sample_methyl, ref_methyl_data)

# Apply linear regression model
cnv.fit()

# Get CNV for all bins
cnv.set_bins()

# Get CNV for all genes
cnv.set_detail()

# Segment Genome using binary circular segmentation algorithm
cnv.set_segments()

# Show CNV plot
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
