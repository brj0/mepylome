library(illuminaio)
library(minfi)
library(conumee2.0)

file0 <- "/data/epidip_IDAT/7970368088_R01C01_Grn.idat"
file1 <- "/data/epidip_IDAT/7970368088_R01C02_Grn.idat"
file2 <- "/data/ref_IDAT/cnvrefidat_450k/5775446049_R06C01_Red.idat"
file3 <- "/data/ref_IDAT/cnvrefidat_450k/5775446051_R02C01"
file4 <- "/data/epidip_IDAT/206171430049_R08C01"
file5 <- "/data/epidip_IDAT/6042324058_R03C02"


# file1 <- path.expand("/data/epidip_IDAT/101130760092_R05C02_Red.idat")

idat <- readIDAT(file0)


timing <- system.time({
    idat <- readIDAT(file0)
})
cat("Elapsed time:", timing[["elapsed"]], "seconds\n")



# rgSet <- read.metharray(file0, force = TRUE)
rgSet <- read.metharray(c(file0,file1), force = TRUE)
mset <- preprocessIllumina(rgSet)


GENES <- "/applications/nanodip_cache/reference_data/hg19_cnv/hg19_genes.tsv"
genes_df <- read.csv(GENES, sep = "\t", header = TRUE)
all_genes <- GRanges(genes_df$seqname,
                        ranges = IRanges(genes_df$start, genes_df$end),
                        strand = genes_df$strand,
                        name = genes_df$name)
genome(all_genes) <- "hg19"

anno <- CNV.create_anno(array_type = "450k", chrXY = TRUE,
                                detail_regions = all_genes)


ref <- CNV.load(mset)

smp_rgSet <- read.metharray(file5)
sample_mset <- preprocessIllumina(read.metharray(file5))



query <- CNV.load(sample_mset)

timing <- system.time({
    cnv <- CNV.fit(query, ref, anno)
})
cat("Elapsed time: (fit)", timing[["elapsed"]], "seconds\n")

timing <- system.time({
    cnv <- CNV.bin(cnv)
})
cat("Elapsed time: (bin)", timing[["elapsed"]], "seconds\n")

timing <- system.time({
    cnv <- CNV.detail(cnv)
})
cat("Elapsed time: (detail)", timing[["elapsed"]], "seconds\n")

timing <- system.time({
    cnv <- CNV.segment(cnv)
})
cat("Elapsed time: (segment)", timing[["elapsed"]], "seconds\n")


# cnv <- CNV.segment(CNV.detail(CNV.bin(cnv)))

result <- list()
WHAT <- c("bins", "detail", "gistic", "overview", "segments", "probes")

for (what in WHAT) {
    cnv_write <- CNV.write(cnv, what = what)
    if (what %in% c("detail", "segments")) {
        cnv_write <- cnv_write[[1]]
    }
    result[[what]] <- cnv_write
}


