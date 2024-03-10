from setuptools import Extension, setup, find_packages
import glob
import numpy as np
import os
import re
import pybind11


include_dirs = [
    pybind11.get_include(),
    np.get_include(),
]

# Get the value of the MEPYLOME_DEBUG environment variable
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
        # "-Ofast",
        # "-flto",
        # "-fno-math-errno",
        # "-fopenmp",
        "-march=native",
        # "-mtune=native",
    ]

module = Extension(
    name="_mepylome",
    sources=glob.glob("pybindings/*.cpp") + glob.glob("src/*.cpp"),
    include_dirs=include_dirs,
    extra_compile_args=compile_args,
    extra_link_args=["-fopenmp"],
    language="c++",
)

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="mepylome",
    version="0.0.0",
    description="Python package for processing Infinum DNA methylation arrays",
    packages=find_packages(),
    package_data={"mepylome": ["data/*.gz"]},
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        "cbseg",
        "cython",
        "ailist==1.0.4",
        "numpy",
        "pandas",
        "pyranges",
        "scikit-learn",
    ],
    extras_require={
        "full": [
        ],
    },
    ext_modules=[module],
    keywords="Illumina, Methylation, Infinum, Microarray, BeadChip",
    license="GPL-3.0 license",
    author="Jon Brugger",
    url="https://github.com/brj0/mepylome",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: POSIX",
        "Programming Language :: C++",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development",
    ],
)
