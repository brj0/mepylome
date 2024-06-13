"""This module contains genetic data."""

import pkg_resources

__all__ = ["CHROMOSOME_DATA", "IMPORTANT_GENES", "GAPS", "GENES"]

# Data copied from conumee
GAPS = pkg_resources.resource_filename("mepylome", "data/gaps.csv.gz")

# HG19 Gene data downloaded from:
# https://grch37.ensembl.org/biomart/martview
GENES = pkg_resources.resource_filename("mepylome", "data/hg19_genes.tsv.gz")

CHROMOSOME_DATA = {
    "name": [
        "chr1",
        "chr2",
        "chr3",
        "chr4",
        "chr5",
        "chr6",
        "chr7",
        "chr8",
        "chr9",
        "chr10",
        "chr11",
        "chr12",
        "chr13",
        "chr14",
        "chr15",
        "chr16",
        "chr17",
        "chr18",
        "chr19",
        "chr20",
        "chr21",
        "chr22",
        "chrX",
        "chrY",
    ],
    "len": [
        249250621,
        243199373,
        198022430,
        191154276,
        180915260,
        171115067,
        159138663,
        146364022,
        141213431,
        135534747,
        135006516,
        133851895,
        115169878,
        107349540,
        102531392,
        90354753,
        81195210,
        78077248,
        59128983,
        63025520,
        48129895,
        51304566,
        155270560,
        59373566,
    ],
    "centromere_start": [
        121535434,
        92326171,
        90504854,
        49660117,
        46405641,
        58830166,
        58054331,
        43838887,
        47367679,
        39254935,
        51644205,
        34856694,
        16000000,
        16000000,
        17000000,
        35335801,
        22263006,
        15460898,
        24681782,
        26369569,
        11288129,
        13000000,
        58632012,
        10104553,
    ],
    "centromere_end": [
        124535434,
        95326171,
        93504854,
        52660117,
        49405641,
        61830166,
        61054331,
        46838887,
        50367679,
        42254935,
        54644205,
        37856694,
        19000000,
        19000000,
        20000000,
        38335801,
        25263006,
        18460898,
        27681782,
        29369569,
        14288129,
        16000000,
        61632012,
        13104553,
    ],
}

IMPORTANT_GENES = [
    "BRD4",
    "CDK4",
    "CDKN2A",
    "CDKN2B",
    "EGFR",
    "ERBB2",
    "IL13RA2",
    "KIT",
    "MDM4",
    "MET",
    "NF1",
    "NF2",
    "NTRK3",
    "NUTM1",
    "PDGFRA",
    "PTEN",
    "RB1",
    "SOX2",
]
