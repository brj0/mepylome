"""Setup script for package installation."""

import os
from pathlib import Path

import numpy as np
import pybind11
from setuptools import Extension, find_packages, setup

include_dirs = [
    pybind11.get_include(),
    np.get_include(),
]

# If C++ parser should be added, do: export MEPYLOME_CPP=1
add_cpp = os.getenv("MEPYLOME_CPP") == "1"

# If debug compiler flags should be used, do: export MEPYLOME_DEBUG=1
debug = os.getenv("MEPYLOME_DEBUG") == "1"

# Set the compiler flags based on the debug flag
if debug:
    print("\n*** Debugging compiler flags are used. ***\n")
    compile_args = [
        "-O0",
        "-Wall",
        "-Wextra",
        "-fno-stack-protector",
        "-g",
        "-pg",
    ]
else:
    compile_args = [
        "-O3",
        "-march=native",
    ]

sources = list(map(str, Path("pybindings").glob("*.cpp"))) + list(
    map(str, Path("src").glob("*.cpp"))
)
cpp_extension = Extension(
    name="_mepylome",
    sources=sources,
    include_dirs=include_dirs,
    extra_compile_args=compile_args,
    extra_link_args=["-fopenmp"],
    language="c++",
)

if add_cpp:
    ext_modules = [cpp_extension] if add_cpp else []
    print("\n*** Adding cpp extension. ***\n")
else:
    ext_modules = []

with open("README.md") as f:
    long_description = f.read()

setup(
    name="mepylome",
    version="0.8.2",
    description="Python package for processing Infinum DNA methylation arrays",
    packages=find_packages(),
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        "numpy",
        "odfpy",
        "pandas",
        "openpyxl",
        "tqdm",
        "pyranges!=0.1.3",  # BUG: pyranges bug with version 0.1.3
        "psutil",
        "xlrd",
        "scikit-learn",
        "dash>=2.16.0",
        "plotly!=6.0.0",  # BUG: plotly 6.0.0 go.Scattergl cannot display text
        "umap-learn",
        "dash_bootstrap_components",
        "importlib_resources; python_version < '3.9'",
        "xxhash",
    ],
    ext_modules=ext_modules,
    keywords="Illumina, Methylation, Infinum, Microarray, BeadChip",
    license="MIT license",
    author="Jon Brugger",
    url="https://github.com/brj0/mepylome",
    # Include package data such as csv-Files, images, ...
    include_package_data=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX",
        "Programming Language :: C++",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development",
    ],
    entry_points={
        "console_scripts": [
            "mepylome=mepylome.cli:start_mepylome",
        ]
    },
)
