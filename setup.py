"""Setup script for package installation."""

import os
from pathlib import Path

import numpy as np
import pybind11
from setuptools import Extension, setup

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

setup(
    ext_modules=ext_modules,
)
