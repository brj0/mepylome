time0 <- Sys.time()

library(minfi)
library(tidyverse)

time1 <- Sys.time()
print(paste("Import time:", time1 - time0, "s"))

ensure_directory_exists <- function(dir) {
    if (!dir.exists(dir)) {
        dir.create(dir, recursive = TRUE)
    }
}

args <- commandArgs(trailingOnly = TRUE)
prep <- args[1]
preps <- c("illumina", "swan", "noob")

if (!prep %in% preps) {
    cat("First command line argument must be in",
        paste(preps, collapse = ", "), "\n")
    cat("Received:", prep, "\n")
    quit(status = 1)
}



HOME_DIR <- Sys.getenv("HOME")
DEFAULT_TEST_DIR <- file.path(HOME_DIR, "Documents", "mepylome", "tests")
ensure_directory_exists(DEFAULT_TEST_DIR)

grn_idat_files <- list.files(
    DEFAULT_TEST_DIR,
    pattern = "_Grn.idat",
    full.names = TRUE)

basepaths <- sub("_Grn.idat$", "", grn_idat_files)


time1 <- Sys.time()

# Read IDAT files and preprocess
for (idat_file in basepaths) {
    rgSet <- read.metharray(idat_file)
    if (prep == "illumina") {
        methyl_data <- preprocessIllumina(rgSet)
    } else if (prep == "swan") {
        methyl_data <- preprocessSWAN(rgSet)
    } else if (prep == "noob") {
        methyl_data <- preprocessNoob(rgSet)
    }
}

time2 <- Sys.time()
N <- length(basepaths)
tpc <- (time2 - time1) / N
print(paste0("Extraction time (", prep, "): ", time2 - time1, " s"))
print(paste0("   Time per case: ", tpc, " s (No. of cases:", N, ")"))

# Save processed data
save_data <- function(data, file_path) {
  write.csv(data, file_path, row.names = TRUE)
}

rgSet <- read.metharray(idat_files[1])
if (prep == "illumina") {
  methyl_data <- preprocessIllumina(rgSet)
} else if (prep == "swan") {
  methyl_data <- preprocessSWAN(rgSet)
} else if (prep == "noob") {
  methyl_data <- preprocessNoob(rgSet)
}

grn <- getGreen(methyl_data) %>% as.data.frame() %>% arrange(row.names(.))
red <- getRed(methyl_data) %>% as.data.frame() %>% arrange(row.names(.))
methylated <- getMeth(methyl_data) %>% as.data.frame() %>% arrange(row.names(.))
unmethylated <- getUnmeth(methyl_data) %>% as.data.frame() %>% arrange(row.names(.))

save_data(
    methylated,
    file.path(DEFAULT_TEST_DIR, paste0("minfi-", prep, "-methylated.csv")))

save_data(
    unmethylated,
    file.path(DEFAULT_TEST_DIR, paste0("minfi-", prep, "-unmethylated.csv")))


