import random
from multiprocessing import Pool

import numpy as np
from nanodip import Reference
from tqdm import tqdm

from mepylome import MethylData, idat_basepaths

IDAT_DIR = "/data/epidip_IDAT"
IDAT_DIR = "/mnt/ws528695/data/epidip_IDAT"

INDEX_FILE = "/applications/reference_data/betaEPIC450Kmix_bin/index.csv"

with open(INDEX_FILE) as f:
    cpg_index_450k = np.array(f.read().splitlines())


idat_files = idat_basepaths(IDAT_DIR)[:10000]
len(idat_files)

timer.start()
m = MethylData(file=smp7)
m.beta
timer.stop("3")

reference = Reference("AllIDATv2_20210804_HPAP_Sarc")
NR_CPGS = 9000
random_cpg_sample = random.sample(reference.cpg_sites, NR_CPGS)

cpg_mask = np.isin(cpg_index_450k, random_cpg_sample)


def extract_beta(idat_file):
    try:
        methyl = MethylData(file=idat_file)
        betas_450k_df = methyl.converted_beta(cpgs=cpg_index_450k, fill=0.49)
        betas = betas_450k_df.values.ravel()
        return betas[cpg_mask]
    except ValueError as e:
        return (idat_file, e)


with Pool() as pool:
    betas_450k_results = list(
        tqdm(
            pool.imap(extract_beta, idat_files),
            total=len(idat_files),
        )
    )

valid_betas = [x for x in betas_450k_results if len(x) == NR_CPGS]

methyl_mtx = np.vstack(valid_betas)
