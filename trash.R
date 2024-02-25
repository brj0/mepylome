library(illuminaio)
library(minfi)
library(conumee2.0)

file0 <- "/data/epidip_IDAT/7970368088_R01C01_Grn.idat"
file1 <- "/data/epidip_IDAT/7970368088_R01C02_Grn.idat"
file2 <- "/data/ref_IDAT/450k/5775446049_R06C01_Red.idat"
file3 <- "/data/ref_IDAT/450k/5775446051_R02C01"

# file1 <- path.expand("/data/epidip_IDAT/101130760092_R05C02_Red.idat")

idat <- readIDAT(file0)


timing <- system.time({
    idat <- readIDAT(file0)
})
cat("Elapsed time:", timing[["elapsed"]], "seconds\n")



# rgSet <- read.metharray(file0, force = TRUE)
rgSet <- read.metharray(c(file0,file1), force = TRUE)
mset <- preprocessIllumina(rgSet)

sample_idat = file3 

GENES <- "/applications/nanodip_cache/reference_data/hg19_cnv/hg19_genes.tsv"
genes_df <- read.csv(GENES, sep = "\t", header = TRUE)
all_genes <- GRanges(genes_df$seqname,
                        ranges = IRanges(genes_df$start, genes_df$end),
                        strand = genes_df$strand,
                        name = genes_df$name)
genome(all_genes) <- "hg19"

anno <- CNV.create_anno(array_type = "450k", chrXY = TRUE,
                                detail_regions = all_genes)


ref_cnv <- CNV.load(mset)

smp_rgSet <- read.metharray(sample_idat)
sample_mset <- preprocessIllumina(read.metharray(sample_idat))



sample_cnv_data <- CNV.load(sample_mset)
cnv <- CNV.fit(sample_cnv_data, ref_cnv, anno)
cnv <- CNV.segment(CNV.detail(CNV.bin(cnv)))

result <- list()
WHAT <- c("bins", "detail", "gistic", "overview", "segments", "probes")

for (what in WHAT) {
    cnv_write <- CNV.write(cnv, what = what)
    if (what %in% c("detail", "segments")) {
        cnv_write <- cnv_write[[1]]
    }
    result[[what]] <- cnv_write
}


