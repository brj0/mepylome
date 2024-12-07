"""Basic usage of the package."""

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
