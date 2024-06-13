# Used to tests the performance of the minfi package.
#
# This script performs extraction of methylation information from all IDAT files
# located in ~/Documents/mepylome/tests. It requires a preprocessing method as a
# command-line argument (e.g., "illumina", "noob", "raw", "swan").
#
# If '--save' is added it saves the extracted methylated and unmethylated
# information to disk, enabling further comparison tests with the mepylome
# package using the ./test_idat_extraction.py script and validating correctness
# with the ./correctness.py script.
#
# Usage:
#     Rscript test_idat_extraction.R illumina --save
#
#     # To measure time and memory consumption
#     /usr/bin/time -v Rscript test_idat_extraction.R noob


# Parse command line arguments
args <- commandArgs(trailingOnly = TRUE)
prep <- args[1]
preps <- c("illumina", "noob", "raw", "swan")

if (!prep %in% preps) {
    cat("First command line argument must be in",
        paste(preps, collapse = ", "), "\n")
    cat("Received:", prep, "\n")
    quit(status = 1)
}

# Load minfi
time0 <- Sys.time()
suppressMessages(suppressWarnings(library(minfi)))
time1 <- Sys.time()

time_diff <- difftime(time1, time0, units = "secs")

cat(paste("Time for importing minfi:", time_diff, "s\n"))

ensure_directory_exists <- function(dir) {
    if (!dir.exists(dir)) {
        dir.create(dir, recursive = TRUE)
    }
}

HOME_DIR <- Sys.getenv("HOME")
TEST_DIR <- file.path(HOME_DIR, "Documents", "mepylome", "tests")
TEST_OUTPUT_DIR <- file.path(HOME_DIR, "Documents", "mepylome", "output_tests")
ensure_directory_exists(TEST_DIR)
ensure_directory_exists(TEST_OUTPUT_DIR)

# Get all *_Grn.idat files
grn_idat_files <- list.files(
    TEST_DIR,
    pattern = "_Grn.idat",
    recursive = TRUE,
    full.names = TRUE)

# Get all IDAT basepaths
basepaths <- sort(sub("_Grn.idat$", "", grn_idat_files))

# Read IDAT files and preprocess
time0 <- Sys.time()
for (idat_file in basepaths) {
    rgSet <- read.metharray(idat_file)
    if (prep == "illumina") {
        methyl_data <- preprocessIllumina(rgSet)
    } else if (prep == "swan") {
        methyl_data <- preprocessSWAN(rgSet)
    } else if (prep == "noob") {
        methyl_data <- preprocessNoob(rgSet)
    } else if (prep == "raw") {
        methyl_data <- preprocessRaw(rgSet)
    }
}
time1 <- Sys.time()

N <- length(basepaths)
time_diff <- difftime(time1, time0, units = "secs")
tpc <- time_diff / N


cat(paste0(
    "Time for analysis (", prep, "): ", time_diff, " s (", N, " cases)\n"))
cat(paste0("   Time per case: ", tpc, " s \n\n"))


if (toString(args[2]) != "--save") {
    cat(paste0(
        "Exit script.\nIf you want to save output to disk for comparison ",
        "with mepylome, rerun this script by adding '--save'"))
    quit(status = 1)
}

# Save processed data
save_data <- function(data, file_path) {
  write.csv(data, file_path, row.names = TRUE)
}

for (idat_file in basepaths) {
    probe <- basename(idat_file)
    prefix <- paste0(probe, "-minfi-", prep)
    methyl_path_r <- file.path(
        TEST_OUTPUT_DIR, paste0(prefix, "-methylated.csv"))
    unmethyl_path_r <- file.path(
        TEST_OUTPUT_DIR, paste0(prefix, "-unmethylated.csv"))

    rgSet <- read.metharray(idat_file)
    if (prep == "illumina") {
        methyl_data <- preprocessIllumina(rgSet)
    } else if (prep == "swan") {
        methyl_data <- preprocessSWAN(rgSet)
    } else if (prep == "noob") {
        methyl_data <- preprocessNoob(rgSet)
    } else if (prep == "raw") {
        methyl_data <- preprocessRaw(rgSet)
    }
    methylated <- getMeth(methyl_data)
    unmethylated <- getUnmeth(methyl_data)
    save_data(methylated, methyl_path_r)
    save_data(unmethylated, unmethyl_path_r)
}
