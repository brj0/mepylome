"""Generates a gene annotation file using the Illumina manifest file."""

from pathlib import Path

import pandas as pd

from mepylome import Manifest
from mepylome.dtypes.chromosome import Chromosome
from mepylome.utils.files import (
    get_csv_file,
    get_resource_path,
)
from mepylome.utils.varia import CONFIG, MEPYLOME_TMP_DIR

# Set up download directory and file paths
download_dir = MEPYLOME_TMP_DIR / "manifests"
manifest_url = Path(CONFIG["urls"]["manifest"]["epicv2"])
manifest_csv_filename = CONFIG["files"]["remote"]["epicv2"]

# Download the manifest
Manifest("epicv2")._download_manifest()

manifest_path = download_dir / manifest_url.name

# Columns to read from the manifest
columns_to_load = [
    "CHR",
    "IlmnID",
    "MAPINFO",
    "Name",
    "UCSC_RefGene_Name",
]


# Load raw manifest data
with get_csv_file(manifest_path, manifest_csv_filename) as manifest_file:
    Manifest._seek_to_start(manifest_file)
    manifest_df = pd.read_csv(
        manifest_file,
        low_memory=False,
        usecols=columns_to_load,
    )
    n_probes = manifest_df[manifest_df.IlmnID.str.startswith("[")].index[0]
    manifest_df = manifest_df[:n_probes]

manifest_df["Chromosome"] = Chromosome.pd_to_string(
    Chromosome.pd_from_string(manifest_df["CHR"])
)

manifest_df["UCSC_RefGene_Name"] = manifest_df["UCSC_RefGene_Name"].str.split(
    ";"
)
manifest_df = manifest_df.explode("UCSC_RefGene_Name")
manifest_df["UCSC_RefGene_Name"] = manifest_df["UCSC_RefGene_Name"].str.strip()

grouped = manifest_df.groupby(["UCSC_RefGene_Name", "Chromosome"])

genes_df = grouped["MAPINFO"].agg(Start="min", End="max").reset_index()
genes_df.columns = ["Name", "Chromosome", "Start", "End"]

genes_df["Strand"] = "+"
genes_df = genes_df[["Start", "End", "Strand", "Name", "Chromosome"]]

genes_df["Start"] = genes_df["Start"].astype(int)
genes_df["End"] = genes_df["End"].astype(int)
genes_df = genes_df.sort_values(by="Name").reset_index(drop=True)

data_dir = get_resource_path("mepylome", "data")
genes_df.to_csv(
    data_dir / "gene_loci_epicv2.tsv.gz",
    sep="\t",
    index=False,
    compression="gzip",
)
