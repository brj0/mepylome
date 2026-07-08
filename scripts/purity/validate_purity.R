# Test the original R implementation of RFpurify.
#
# Put test IDAT files into:
#   ~/mepylome/tests/purity_tests/
#
# The generated TSV is used as a reference to compare against the Python
# RFpurify implementation.
#
# Install if needed:
# BiocManager::install("minfi")
# install.packages("RFpurify")

library(minfi)
library(RFpurify)

TEST_DIR <- "~/mepylome/tests/purity_tests/"
out_file <- file.path(TEST_DIR, "rfpurify_purity_results_r.tsv")

# Get all Grn IDAT files
grn_idat_files <- list.files(
    TEST_DIR,
    pattern = "_Grn.idat$",
    recursive = TRUE,
    full.names = TRUE
)

# Get all IDAT basepaths, sorted
basepaths <- sort(sub("_Grn.idat$", "", grn_idat_files))

results <- data.frame(
    sample_id = basename(basepaths),
    absolute = NA_real_,
    estimate = NA_real_,
    stringsAsFactors = FALSE
)

for (i in seq_along(basepaths)) {

    basepath <- basepaths[i]
    sample_id <- basename(basepath)

    message("Processing: ", sample_id)

    # Read only this IDAT pair
    rgSet <- read.metharray(basepath)

    # Normalize
    mSet <- preprocessIllumina(rgSet)

    # Predict purity
    results$absolute[i] <- as.numeric(
        predict_purity(mSet, method = "ABSOLUTE")
    )

    results$estimate[i] <- as.numeric(
        predict_purity(mSet, method = "ESTIMATE")
    )
}

# Save TSV
write.table(
    results,
    file = out_file,
    sep = "\t",
    quote = FALSE,
    row.names = FALSE
)

print(results)
message("Saved: ", out_file)
