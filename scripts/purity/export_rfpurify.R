# Export random forest trees from RFpurify to JSON
#
# Run ONCE in R to export the RFpurify models to JSON.
#
# Requires:
#
# install.packages(c("randomForest","jsonlite","remotes"))
# BiocManager::install(c(
#   "minfi",
#   "IlluminaHumanMethylation450kmanifest",
#   "IlluminaHumanMethylationEPICmanifest"
#   "IlluminaHumanMethylationEPICv2manifest"
# ))
# remotes::install_github("mwsill/RFpurify")
#
#
# Usage:
#   Rscript export_rfpurify.R
#
# Output:
#   rfpurify_ABSOLUTE.json
#   rfpurify_ESTIMATE.json

library(RFpurify)
library(randomForest)
library(jsonlite)


export_rf <- function(model, output_path) {
  features <- rownames(model$importance)
  n_trees  <- model$ntree
 
  cat(sprintf("Exporting %d trees, %d features -> %s\n",
              n_trees, length(features), output_path))
 
  trees <- vector("list", n_trees)
  for (i in seq_len(n_trees)) {
    t <- getTree(model, k = i, labelVar = FALSE)
    trees[[i]] <- list(
      left      = as.integer(t[, "left daughter"]),
      right     = as.integer(t[, "right daughter"]),
      feature   = as.integer(t[, "split var"]),
      threshold = as.numeric(t[, "split point"]),
      is_leaf   = t[, "status"] == -1L,
      value     = as.numeric(t[, "prediction"])
    )
  }
 
  payload <- list(features = features, trees = trees)
  write(toJSON(payload, auto_unbox = FALSE, digits = 15), output_path)
  cat("Done.\n")
}

export_rf(RFpurify_ABSOLUTE, "rfpurify_ABSOLUTE.json")
export_rf(RFpurify_ESTIMATE, "rfpurify_ESTIMATE.json")
