library(illuminaio)
library(minfi)

file0 <- path.expand("/data/ref_IDAT/450k/3999997083_R02C02_Grn.idat")
file1 <- path.expand("/data/ref_IDAT/450k/5775446049_R01C02_Grn.idat")

idat <- readIDAT(file0)


timing <- system.time({
    idat <- readIDAT(file0)
})
cat("Elapsed time:", timing[["elapsed"]], "seconds\n")




rgSet <- read.metharray(c(file0,file1), force = TRUE)
rgSet <- read.metharray(file0, force = TRUE)
mset <- preprocessIllumina(rgSet)

