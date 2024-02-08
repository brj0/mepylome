library(illuminaio)

file_path <- path.expand("~/MEGA/work/programming/pyidat/101130760092_R05C02_Grn.idat")


idat <- readIDAT(file_path)


timing <- system.time({
    idat <- readIDAT(file_path)
})
cat("Elapsed time:", timing[["elapsed"]], "seconds\n")

