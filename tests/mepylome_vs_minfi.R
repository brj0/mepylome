library("pryr")

# Parse command line arguments
args <- commandArgs(trailingOnly = TRUE)
prep <- args[1]
preps <- c("illumina", "swan", "noob")

if (!prep %in% preps) {
    cat("First command line argument must be in",
        paste(preps, collapse = ", "), "\n")
    cat("Received:", prep, "\n")
    quit(status = 1)
}


# Measure initila time and memory usage
memory0 <- mem_used()
time0 <- Sys.time()

# Load minfi
memory1 <- mem_used()
time1 <- Sys.time()
suppressMessages(suppressWarnings(library(minfi)))
time2 <- Sys.time()
memory2 <- mem_used()

time_diff <- difftime(time2, time1, units = "secs")
memory_diff <- (memory2 - memory1) / 2^20

cat("minfi\n")
cat(paste("    Import time:", time_diff, "s\n"))
cat(paste("    Memory usage:", memory_diff, "MB\n\n"))


# Load conumee2.0
memory1 <- mem_used()
time1 <- Sys.time()
suppressMessages(suppressWarnings(library(conumee2.0)))
time2 <- Sys.time()
memory2 <- mem_used()

time_diff <- difftime(time2, time1, units = "secs")
memory_diff <- (memory2 - memory1) / 2^20

cat("conumee2.0\n")
cat(paste("    Import time:", time_diff, "s\n"))
cat(paste("    Memory usage:", memory_diff, "MB\n\n"))


ensure_directory_exists <- function(dir) {
    if (!dir.exists(dir)) {
        dir.create(dir, recursive = TRUE)
    }
}


HOME_DIR <- Sys.getenv("HOME")
DEFAULT_TEST_DIR <- file.path(HOME_DIR, "Documents", "mepylome", "tests")
ensure_directory_exists(DEFAULT_TEST_DIR)

# Get all *_Grn.idat files
grn_idat_files <- list.files(
    DEFAULT_TEST_DIR,
    pattern = "_Grn.idat",
    full.names = TRUE)

# Get all IDAT basepaths
basepaths <- sub("_Grn.idat$", "", grn_idat_files)


# Read IDAT files and preprocess
time1 <- Sys.time()
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
time_diff <- difftime(time2, time1, units = "secs")
tpc <- time_diff / N
memory1 <- mem_used()


cat(paste0("Extraction time (", prep, "): ", time_diff, " s\n"))
cat(paste0("   Time per case: ", tpc, " s (No. of cases:", N, ")\n\n"))

time_diff <- difftime(time2, time0, units = "secs")
memory_diff <- (memory1 - memory0) / 2^20

cat("Total:\n")
cat(paste0("    Time (includes benchmarking utils): ", time_diff, " s\n"))
cat(paste0("    Memory usage: ", memory_diff, " MB\n"))


# Save processed data
save_data <- function(data, file_path) {
  write.csv(data, file_path, row.names = TRUE)
}

rgSet <- read.metharray(basepaths[1])
if (prep == "illumina") {
  methyl_data <- preprocessIllumina(rgSet)
} else if (prep == "swan") {
  methyl_data <- preprocessSWAN(rgSet)
} else if (prep == "noob") {
  methyl_data <- preprocessNoob(rgSet)
}

methylated <- getMeth(methyl_data)
unmethylated <- getMeth(methyl_data)

save_data(
    methylated,
    file.path(DEFAULT_TEST_DIR, paste0("minfi-", prep, "-methylated.csv")))

save_data(
    unmethylated,
    file.path(DEFAULT_TEST_DIR, paste0("minfi-", prep, "-unmethylated.csv")))


