# Performs CNV analysis for performance tests.
#
# This script reads IDAT files from a specified subdirectory of
# ~/mepylome/tests/, preprocesses them using a specified method
# (illumina, noob, raw, swan), and performs CNV analysis on one of them. All the
# files must have the same array type.
#
# Usage:
#     Rscript benchmark_cnv.R <preprocessing_method> <subdir_containing_idat>
#     Rscript benchmark_cnv.R illumina idat_epic
#
#     # To perform profiling
#     /usr/bin/time -v Rscript benchmark_cnv.R illumina idat_450k


args <- commandArgs(trailingOnly = TRUE)
prep <- args[1]
preps <- c("illumina", "noob", "raw", "swan")

if (!prep %in% preps) {
    cat("First command line argument must be in",
        paste(preps, collapse = ", "), "\n")
    cat("Received:", prep, "\n")
    quit(status = 1)
}

HOME_DIR <- Sys.getenv("HOME")
TEST_DIR <- file.path(HOME_DIR, "mepylome", "tests")
GENES <- "../../mepylome/data/gene_loci_epicv2.tsv.gz"
ARRAY_TYPE_MAP <- c("IlluminaHumanMethylation450k"   = "450k",
                    "IlluminaHumanMethylationEPIC"   = "EPIC",
                    "IlluminaHumanMethylationEPICv2" = "EPICv2")
subdir <- file.path(TEST_DIR, args[2])

if (!dir.exists(subdir)) {
    cat(paste0(
        "Second command line argument must be a subdir of ", TEST_DIR, "\n"))
    cat("Received:", subdir, "\n")
    quit(status = 1)
}


# Load minfi
time0 <- Sys.time()
suppressMessages(suppressWarnings(library(minfi)))
time1 <- Sys.time()

minfi_import_time <- difftime(time1, time0, units = "secs")

cat(paste("minfi import time:", minfi_import_time, "s\n"))

time0 <- Sys.time()
suppressMessages(suppressWarnings(library(conumee2.0)))
time1 <- Sys.time()

conumee_import_time <- difftime(time1, time0, units = "secs")

cat(paste("conumee2.0 import time:", conumee_import_time, "s\n"))


# Get all *_Grn.idat files
grn_idat_files <- list.files(
    subdir,
    pattern = "_Grn.idat",
    recursive = TRUE,
    full.names = TRUE)

# Get all IDAT basepaths
basepaths <- sort(sub("_Grn.idat$", "", grn_idat_files))

get_mset <- function(file) {
    rgSet <- read.metharray(file, force = TRUE)
    if (prep == "illumina") {
        methyl_data <- preprocessIllumina(rgSet)
    } else if (prep == "swan") {
        methyl_data <- preprocessSWAN(rgSet)
    } else if (prep == "noob") {
        methyl_data <- preprocessNoob(rgSet)
    } else if (prep == "raw") {
        methyl_data <- preprocessRaw(rgSet)
    }
    return(methyl_data)
}

get_annotation <- function(array_type) {
    # CNV is calculated for all genes
    genes_df <- read.csv(GENES, sep = "\t", header = TRUE)
    all_genes <- GRanges(genes_df$Chromosome,
                         ranges = IRanges(genes_df$Start, genes_df$End),
                         strand = genes_df$Strand,
                         name = genes_df$Name)
    genome(all_genes) <- "hg19"

    anno <- CNV.create_anno(
        array_type = ARRAY_TYPE_MAP[[array_type]],
        chrXY = TRUE,
        detail_regions = all_genes)
    return(anno)
}

num_samples <- 10
total_time <- minfi_import_time + conumee_import_time
reference_cnv_data <- NULL
anno <- NULL

for (i in 1:num_samples) {
    time0 <- Sys.time()

    sample_path <- basepaths[i]
    sample_mset <- get_mset(sample_path)
    array_type <- sample_mset@annotation["array"]

    if (is.null(reference_cnv_data)) {
        reference_mset <- get_mset(tail(basepaths, 20))
        reference_cnv_data <- CNV.load(reference_mset)
        anno <- get_annotation(array_type)
    }

    sample_cnv_data <- CNV.load(sample_mset)
    cnv <- CNV.fit(sample_cnv_data, reference_cnv_data, anno)
    cnv <- CNV.segment(CNV.detail(CNV.bin(cnv)))

    time1 <- Sys.time()

    elapsed_time <- as.numeric(difftime(time1, time0, units = "secs"))
    total_time <- total_time + elapsed_time

    cat(paste0("Time for CNV analysis (Sample ", i, "): ", elapsed_time, " s\n"))
}

average_time <- total_time / num_samples
cat(paste0("Average time for CNV analysis: ", average_time, " s\n"))
