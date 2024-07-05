import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
from tqdm import tqdm

from mepylome.dtypes.arrays import ArrayType
from mepylome.dtypes.beads import _overlap_indices
from mepylome.dtypes.manifests import Manifest
from mepylome.utils.files import MEPYLOME_TMP_DIR, ensure_directory_exists

NEUTRAL_BETA = 0.49
DTYPE = np.float32

def check_memory(nrows, ncols, dtype):
    available_memory = psutil.virtual_memory().available
    dtype_size = np.dtype(dtype).itemsize
    required_memory = nrows * ncols * dtype_size
    if required_memory > available_memory:
        msg = (
            f"Not enout free memory available. For the given dimension "
            f"({nrows} samples, {ncols} CpG's), "
            f"{required_memory / (1024 ** 3):.1f} GB is required."
        )
        raise MemoryError(msg)

def get_array_cpgs():
    path = MEPYLOME_TMP_DIR / "all_cpgs.pkl"
    if not path.exists():
        with path.open("wb") as f:
            array_cpgs = {
                ArrayType("450k"): Manifest("450k").methylation_probes,
                ArrayType("epic"):  Manifest("epic").methylation_probes,
                ArrayType("epicv2"):  Manifest("epicv2").methylation_probes,
            }
            pickle.dump(array_cpgs, f)
    with path.open("rb") as f:
        return pickle.load(f)


class BetasHandler:
    def __init__(self, directory, array_cpgs=None):
        self.basedir = Path(directory).expanduser()
        self.array_cpgs = array_cpgs
        if self.array_cpgs is None:
            self.array_cpgs = get_array_cpgs()
        dirname = self.basedir.name
        self.dir = {}
        for key in self.array_cpgs:
            self.dir[key] = self.basedir / f"{dirname}-betas_{key}"
            ensure_directory_exists(self.dir[key])
        self.dir["error"] = self.basedir / "error"
        ensure_directory_exists(self.dir["error"])
        self.array_type_from_dir = {
            item: key for key, item in self.dir.items()
        }
        self.paths = None
        self.update()

    @property
    def ids(self):
        return list(self.paths.keys())

    @property
    def invalid_ids(self):
        return list(self.invalid_paths.keys())

    def add(self, betas, id_, array_type):
        betas.astype(DTYPE).tofile(self.dir[array_type] / id_)

    def add_error(self, id_, msg):
        with (self.dir["error"] / id_).open("w") as f:
            f.write(str(msg))

    def get(self, ids, cpgs, fill=NEUTRAL_BETA, parallel=False):
        check_memory(len(ids), len(cpgs), DTYPE)
        result = np.full((len(ids), len(cpgs)), fill, dtype=DTYPE)
        left_idx = {}
        right_idx = {}
        for key, item in self.array_cpgs.items():
            left_idx[key], right_idx[key] = _overlap_indices(cpgs, item)

        def process_id(i, id_):
            path = self.paths.get(id_, None)
            if path is not None:
                key = self.array_type_from_dir[path.parent]
                betas = np.fromfile(path, dtype=DTYPE)
                result[i, left_idx[key]] = betas[right_idx[key]]

        if parallel:
            with ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(process_id, i, id_)
                    for i, id_ in enumerate(ids)
                ]
                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="Reading beta values from disk (parallel)",
                ):
                    future.result()
        else:
            for i, id_ in tqdm(
                enumerate(ids),
                desc="Reading beta values from disk (serial)",
                total=len(ids),
            ):
                process_id(i, id_)
        return pd.DataFrame(result, columns=cpgs, index=ids)

    def update(self):
        self.paths = {
            path.name: path
            for path in self.basedir.rglob("*")
            if not path.is_dir()
        }
        self.invalid_paths = {
            path.name: path for path in self.dir["error"].rglob("*")
        }
