library(illuminaio)
library(minfi)
library(conumee2.0)

ref_dir = "/data/ref_IDAT/cnvrefidat_450k"
smp0 = "/data/epidip_IDAT/6042324058_R03C02_Grn.idat"
smp1 = "/data/epidip_IDAT/6042324058_R04C01_Red.idat"
smp2 = "/data/epidip_IDAT/6042324058_R04C02_Red.idat"
smp3 = "/data/epidip_IDAT/6042324058_R05C01_Grn.idat"
smp4 = "/data/epidip_IDAT/6042324058_R05C02_Grn.idat"
smp5 = "/data/epidip_IDAT/6042324058_R06C01_Red.idat"
smp6 = "/data/epidip_IDAT/6042324058_R06C02_Grn.idat"
smp7 = "/data/epidip_IDAT/7970368088_R01C01_Grn.idat"
smp8 = "/data/epidip_IDAT/7970368088_R01C02_Grn.idat"
ref0 = "/data/ref_IDAT/cnvrefidat_450k/3999997083_R02C02_Grn.idat"
ref1 = "/data/ref_IDAT/cnvrefidat_450k/5775446049_R06C01_Red.idat"
ref2 = "/data/ref_IDAT/cnvrefidat_450k/5775446051_R02C01"
ref3 = "/data/ref_IDAT/cnvrefidat_450k/3999997083_R02C02_Grn.idat"
ref4 = "/data/ref_IDAT/cnvrefidat_450k/5775446049_R01C02_Grn.idat"




idat <- readIDAT(smp0)


timing <- system.time({
    idat <- readIDAT(smp0)
})
cat("Elapsed time:", timing[["elapsed"]], "seconds\n")



# rgSet <- read.metharray(ref0, force = TRUE)
rgSet <- read.metharray(c(ref0,ref1), force = TRUE)

mSet <- preprocessIllumina(rgSet)

mSet_raw <- preprocessRaw(rgSet)

mset <- preprocessSWAN(rgSet, mSet=mSet_raw)
mset <- preprocessSWAN(rgSet)

mset <- preprocessNoob(rgSet)


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

# smp_rgSet <- read.metharray(c(smp0,smp1))
smp_rgSet <- read.metharray(smp0)
sample_mset <- preprocessIllumina(smp_rgSet)



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


# test beta values

s = preprocessIllumina(read.metharray(c(smp0, smp1, smp2)))
b = getBeta(s)



getSubset <- function(counts, subset){
    x <- integer(0)
    for (i in 1:3) {
        x <- c(x, seq.int(1, length(counts))[counts == i][1:subset])
    }
    seq.int(1, length(counts)) %in% x
}



mset <- preprocessSWAN(rgSet, mSet=mSet_raw)
M_s = getMeth(mset)
U_s = getUnmeth(mset)
for (i in 1:999) {
    print(i)
    mset <- preprocessSWAN(rgSet, mSet=mSet_raw)
    M=getMeth(mset)
    U=getUnmeth(mset)
    M_s = M_s + M
    U_s = U_s + U
}

M = M_s / 1000
U = U_s / 1000


