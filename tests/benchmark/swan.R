# Extends test_idat_extraction.py by averaging SWAN.
#
# As SWAN is stochastic, a direct comparison between outputs of mepylome and
# minfi is not straightforward. To better compare the two outputs, we apply
# SWAN preprocessing multiple times and then average the results. This process
# leverages the central limit theorem to obtain more stable and reliable
# methylation estimates.
#
# This script processes IDAT files located in ~/Documents/mepylome/tests using
# the mepylome package. It performs SWAN preprocessing on the data and averages
# the methylation and unmethylation levels obtained over multiple iterations.
# The averaged data is then saved to disk for further comparison tests.
#
# Usage:
#     Rscript mepylome_vs_minfi_swan.R


library(minfi)

HOME_DIR <- Sys.getenv("HOME")
TEST_DIR <- file.path(HOME_DIR, "Documents", "mepylome", "tests")
TEST_OUTPUT_DIR <- file.path(HOME_DIR, "Documents", "mepylome", "output_tests")
N_LOOPS <- 1000

if (!dir.exists(TEST_DIR)) {
  dir.create(TEST_DIR, recursive = TRUE)
}
if (!dir.exists(TEST_OUTPUT_DIR)) {
  dir.create(TEST_OUTPUT_DIR, recursive = TRUE)
}

# Get all *_Grn.idat files
grn_idat_files <- list.files(
    TEST_DIR,
    pattern = "_Grn.idat",
    recursive = TRUE,
    full.names = TRUE)

# Get all IDAT basepaths
basepaths <- sub("_Grn.idat$", "", grn_idat_files)

save_data <- function(data, file_path) {
  write.csv(data, file_path, row.names = TRUE)
}

# Process each IDAT file
for (idat_file in basepaths) {
    probe <- basename(idat_file)
    cat(paste0(probe, ":...\n"))
    rgSet <- read.metharray(idat_file)
    mSet_raw <- preprocessRaw(rgSet)
    mset <- preprocessSWAN(rgSet, mSet = mSet_raw)

    methyl_acc <- getMeth(mset)
    unmethyl_acc <- getUnmeth(mset)

    for (i in seq_len(N_LOOPS - 1)) {
        mset <- preprocessSWAN(rgSet, mSet = mSet_raw)
        methyl_acc <- methyl_acc + getMeth(mset)
        unmethyl_acc <- unmethyl_acc + getUnmeth(mset)
    }

    methyl_mean <- methyl_acc / N_LOOPS
    unmethyl_mean <- unmethyl_acc / N_LOOPS

    prefix <- paste0(probe, "-minfi-swan")
    methyl_path_r <- file.path(
        TEST_OUTPUT_DIR, paste0(prefix, "-methylated.csv"))
    unmethyl_path_r <- file.path(
        TEST_OUTPUT_DIR, paste0(prefix, "-unmethylated.csv"))

    save_data(methyl_mean, methyl_path_r)
    save_data(unmethyl_mean, unmethyl_path_r)
}
