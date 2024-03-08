import mepylome
from pathlib import Path
import time
import os

class Timer:
    """Measures the time elapsed in milliseconds."""
    def __init__(self):
        self.time0 = time.time()
    def start(self):
        """Resets timer."""
        self.time0 = time.time()
    def stop(self, text=None):
        """Resets timer and return elapsed time."""
        delta_time = 1000 * (time.time() - self.time0)
        comment = "" if text is None else "(" + text + ")"
        print("Time passed:", delta_time, "ms", comment)
        self.time0 = time.time()
        return delta_time


filepath = "/data/ref_IDAT/cnvrefidat_450k/3999997083_R02C02_Grn.idat"
filepath = "/data/epidip_IDAT/101130760092_R05C02_Red.idat"

timer = Timer()
idat_data = mepylome.IdatData(filepath)
timer.stop("Parsing IDAT")


file0 = "/data/ref_IDAT/cnvrefidat_450k/3999997083_R02C02_Grn.idat"
file1 = "/data/ref_IDAT/cnvrefidat_450k/5775446049_R01C02_Grn.idat"
idat_data = mepylome.IdatData(file0)
idat_data = mepylome.IdatData(file1)

