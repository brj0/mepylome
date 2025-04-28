Performance
===========

Overview
--------

`mepylome` is designed to be a highly efficient tool for methylation data
analysis, significantly outperforming other tools like `minfi` and `conumee2.0` in
terms of speed and memory consumption. This section provides an overview of its
performance metrics and explains the underlying mechanisms that contribute to
its efficiency.

Performance Comparison
-----------------------

`mepylome` vs. `minfi` and `conumee2.0`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Import Time**: While the import time for `minfi` and `conumee2.0` is a
  couple of seconds (up to a minute), `mepylome` has an import time of just a
  fraction of a second.
- **Data Extraction and CNV Analysis**: `mepylome` performs data extraction and
  CNV (Copy Number Variation) analysis multiple times faster than `minfi` and
  `conumee2.0`.
- **Overall Speed**: On average, `mepylome` is >60 times faster than `minfi`
  and >10 times faster than `conumee2.0`.
- **Memory Consumption**: `mepylome` uses >3 times less memory compared to
  `minfi` and `conumee2.0` together.


Performance Test
~~~~~~~~~~~~~~~~

| Tested on: 12th Gen Intel Core i5-12500, 12 cores at 3.0 GHz, 2 threads per
| core, 250 GB SSD, 96 GB RAM , Ubuntu 22.04 LTS
| Prepreparation method: `illumina`
| No. of Cases: 250 cases for data extraction, 20 reference cases for CNV analysis

| 1. **Import Time**:
|    - `mepylome`: **0.73** seconds
|    - `minfi`: 10.51 seconds
|    - `conumee2.0`: 39.48 seconds

| 2. **Methylation Data Extraction Per Case**:
|    - `mepylome`: **0.06** seconds
|    - `minfi`: 3.73 seconds

| 3. **CNV Analysis Per Case**:
|    - `mepylome`: **2.08** seconds
|    - `conumee2.0`: 20.44 seconds

| 4. **Memory Consumption Data Extraction**:
|    - `mepylome`: **0.39** GB
|    - `minfi`: 1.49 GB

| 5. **Memory Consumption CNV Analysis**:
|    - `mepylome`: **2.29** GB
|    - `conumee2.0`: 7.78 GB


How It Works
------------

Caching Mechanism
~~~~~~~~~~~~~~~~~

`mepylome` employs an efficient caching mechanism to enhance performance. The
first operation in a session might take slightly longer due to the initial
setup and data loading. However, subsequent operations are significantly faster
as the data is cached in memory.

Utilization of the `numpy` Package
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`mepylome` extensively uses the `numpy` package for numerical computations. `numpy` is known for its speed and efficiency, particularly with large arrays and matrices. This contributes significantly to the performance gains observed in `mepylome`.

Lazy Loading
~~~~~~~~~~~~

The package employs lazy loading techniques to delay the loading of data until it is actually needed. This reduces the initial load time and memory usage.

Parallel Processing
~~~~~~~~~~~~~~~~~~~

`mepylome` leverages parallel processing where applicable, distributing tasks
across multiple CPU cores to speed up computation.


