import logging
import subprocess
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from tqdm import tqdm

from mepylome import ManifestLoader, MethylData, RawData
from pathlib import Path


IDAT_DIR = "/data/epidip_IDAT"



timer.start()
m = MethylData(file=smp7)
m.beta
timer.stop("3")
