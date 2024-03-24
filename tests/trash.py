# print("__file__=", __file__); quit()

from methylprep import Manifest as Manifest_old
from pathlib import Path
from pathlib import Path, PurePath
from mepylome.dtypes.manifests import Manifest
from urllib.error import URLError
from urllib.parse import urljoin
from urllib.request import urlopen
import gzip
import io
import logging
import methylprep
import numpy as np
import pandas as pd
import pickle
import requests
import shutil
import ssl
import time

# App
from mepylome.dtypes import ArrayType  # , Channel, ProbeType
from mepylome.utils import (
    download_file,
    get_file_object,
    is_file_like,
    reset_file,
    ensure_directory_exists,
)

LOGGER = logging.getLogger(__name__)

from mepylome.dtypes.manifests import (
    MANIFEST_DIR,
    MANIFEST_DOWNLOAD_DIR,
    ENDING_CONTROL_PROBES,
    ARRAY_FILENAME,
    ARRAY_URL,
    ARRAY_TYPE_MANIFEST_FILENAMES,
    PROBES_COLUMNS,
    MANIFEST_COLUMNS,
    MOUSE_MANIFEST_COLUMNS,
    CONTROL_COLUMNS,
)

pdp = lambda x: print(x.to_string())


logging.basicConfig(level=logging.INFO)
# self = Manifest_old("450k")


quit()

self = Manifest("450k")

epic = Manifest("epic")

epicv2 = Manifest("epicv2")




array_type = ArrayType("450k")
p0, c0 = download_and_process_manifest(array_type)

array_type = ArrayType("epic")
p1, c1 = download_and_process_manifest(array_type)

array_type = ArrayType("epicv2")
p2, c2 = download_and_process_manifest(array_type)


probes_file = (
    Path(MANIFEST_DIR, manifest_name).expanduser().with_suffix(".csv.gz")
)
probes_pickle_file = probes_file.with_suffix(".pkl")

t = time.time()
read_data_frame = pd.read_csv(
    filepath,
    dtype=self.get_data_types(),
)
print(time.time() - t)

t = time.time()
read_data_frame = pd.read_pickle(probes_pickle_file)
print(time.time() - t)



# >> timer.start()
# >>> methylated = pd.DataFrame(
# ...     NONE,
# ...     index=locus_names,
# ...     columns=[Channel.RED.value, Channel.GREEN.value],
# ...     dtype="int32",
# ... )
# >>> timer.stop("loc")
# Time passed: 4.3354034423828125 ms (loc)
# 4.3354034423828125
# >>>
# >>> timer.start()
# >>> methylated.iloc[type_i_red.index] = red.loc[type_i_red.AddressB_ID.values].values
# >>> methylated.iloc[type_i_grn.index] = grn.loc[type_i_grn.AddressB_ID.values].values
# >>> methylated.iloc[type_ii.index] = grn.loc[type_ii.AddressA_ID.values].values
# >>> timer.stop("iloc")
# Time passed: 111.45830154418945 ms (iloc)
# 111.45830154418945
# >>>
# >>> methylated


timer.start()
methylated = pd.DataFrame(
    NONE,
    index=locus_names,
    columns=[Channel.RED.value, Channel.GREEN.value],
    dtype="int32",
)
timer.stop("loc")

timer.start()
methylated.loc[type_i_red.Name.values] = red.loc[type_i_red.AddressB_ID.values].values
methylated.loc[type_i_grn.Name.values] = grn.loc[type_i_grn.AddressB_ID.values].values
methylated.loc[type_ii.Name.values] = grn.loc[type_ii.AddressA_ID.values].values
timer.stop("loc")


timer.start()
methylated.loc[type_i_red.index, 1:] = red.loc[type_i_red.AddressB_ID.values].values
methylated.loc[type_i_grn.index, 1:] = grn.loc[type_i_grn.AddressB_ID.values].values
methylated.loc[type_ii.index, 1:] = grn.loc[type_ii.AddressA_ID.values].values
timer.stop("iloc")

i0 = methylated.index.get_indexer(type_i_red.Name.values)
j0 = red.index.get_indexer(type_i_red.AddressB_ID)
i1 = methylated.index.get_indexer(type_i_grn.Name.values)
j1 = grn.index.get_indexer(type_i_grn.AddressB_ID)
i2 = methylated.index.get_indexer(type_ii.Name.values)
j2 = grn.index.get_indexer(type_ii.AddressB_ID)

timer.start()
methylated.iloc[i0] = red.iloc[j0].values
methylated.iloc[i1] = grn.iloc[j1].values
methylated.iloc[i2] = grn.iloc[j2].values
timer.stop("iloc")

red_array = np.array(self.red.T, dtype=np.int64)
grn_array = np.array(self.grn.T, dtype=np.int64)
methylated = pd.DataFrame(
    index=locus_names,
    columns=[Channel.RED.value, Channel.GREEN.value],
    dtype=np.int64,
)
methylated.loc[type_i_red.Name.values] = red_array[type_i_red.AddressB_ID]
methylated.loc[type_i_grn.Name.values] = grn_array[type_i_grn.AddressB_ID]
methylated.loc[type_ii.Name.values] = grn_array[type_ii.AddressA_ID]


locus_names = np.concatenate(
    [
        type_i.Name.values,
        type_ii.Name.values,
    ]
)

red.sort_index(inplace=True)
grn.sort_index(inplace=True)
timer.stop("sort")

# overlap = bins.join(anno.ranges[["Name"]], how="left", preserve_order=True)
# cnv.ratio.bins_index = None
# cnv.ratio.loc[overlap.Name, "bins_index"] = overlap.bins_index.values.astype(int)
# overlap.ratio = None
# idx = overlap.Name != "-1"
# overlap.ratio[idx] = cnv.ratio.loc[overlap.Name[idx]].ratio
# result = overlap.df.groupby("bins_index", dropna=False).mean()
from numba import njit
@njit
def numba_merge_bins(matrix, min_probes_per_bin, verbose=False):
    I_START = 0
    I_END = 1
    I_N_PROBES = 2
    INVALID = np.iinfo(np.int64).max
    while np.any(matrix[:, I_N_PROBES] < min_probes_per_bin):
        i_min = np.argmin(matrix[:, I_N_PROBES])
        n_probes_left = INVALID
        n_probes_right = INVALID
        # Left
        if i_min > 0:
            delta_left = np.argmax(
                matrix[i_min - 1 :: -1, I_N_PROBES] != INVALID
            )
            i_left = i_min - delta_left - 1
            if (
                matrix[i_left, I_N_PROBES] != INVALID
                and matrix[i_min, I_START] == matrix[i_left, I_END]
            ):
                n_probes_left = matrix[i_left, I_N_PROBES]
        # Right
        if i_min < len(matrix) - 1:
            delta_right = np.argmax(matrix[i_min + 1 :, I_N_PROBES] != INVALID)
            i_right = i_min + delta_right + 1
            if (
                matrix[i_right, I_N_PROBES] != INVALID
                and matrix[i_min, I_END] == matrix[i_right, I_START]
            ):
                n_probes_right = matrix[i_right, I_N_PROBES]
        # Invalid
        if n_probes_left == INVALID and n_probes_right == INVALID:
            matrix[i_min, I_N_PROBES] = INVALID
            continue
        elif n_probes_right == INVALID or n_probes_left <= n_probes_right:
            i_merge = i_left
        else:
            i_merge = i_right
        matrix[i_merge, I_N_PROBES] += matrix[i_min, I_N_PROBES]
        matrix[i_merge, I_START] = min(
            matrix[i_merge, I_START], matrix[i_min, I_START]
        )
        matrix[i_merge, I_END] = max(
            matrix[i_merge, I_END], matrix[i_min, I_END]
        )
        matrix[i_min, I_N_PROBES] = INVALID
    return matrix

# z = numba_merge_bins(
#     df[["Start", "End", "n_probes"]].values.astype(np.int64),
#     min_probes_per_bin,
#     verbose=True,
# )






#

self = sample_methyl
fill = 0.49
cpgs = cpg_index_450k

if cpgs is None:
    cpgs = self.manifest.methylation_probes

timer.start()
beta = self.get_beta(self.methylated, self.unmethylated)
timer.stop("1")
converted = pd.DataFrame(fill, columns=beta.columns, index=cpgs)
timer.stop("2")
common_indices = converted.index.intersection(beta.index)
converted.loc[common_indices] = beta.loc[common_indices].values
timer.stop("3")



timer.stop("1")
timer.stop("2")

np.intersect1d(converted.index.values, beta.index.values)

converted_index = np.array([probe_to_index[p] for p in converted.index])
beta_index = np.array([probe_to_index[p] for p in beta.index])

timer.start()
common_indices, converted_positions, beta_positions = np.intersect1d(
    converted_index, beta_index, return_indices=True
)
converted.iloc[converted_positions] = beta.iloc[beta_positions].values
timer.stop("inter")


import functools

np.all(left_arr == converted.index)

left_arr = converted.index
right_arr = beta.index


def overlap_indices(left_arr, right_arr):
    key = (left_arr.values[:5].tostring(), right_arr.values[:5].tostring())
    if key in cache:
        print("CACHE")
        return cache[key]
    common_indices = left_arr.intersection(right_arr)
    left_index = left_arr.get_indexer(common_indices)
    right_index = right_arr.get_indexer(common_indices)
    cache[key] = left_index, right_index
    return left_index, right_index




def overlap_indices(left_arr, right_arr):
    left_arr_tuple = tuple(left_arr)
    right_arr_tuple = tuple(right_arr)
    return hashable_overlap_indices(left_arr_tuple, right_arr_tuple)


@functools.lru_cache(maxsize=None)
def hashable_overlap_indices(left_arr_tuple, right_arr_tuple):
    left_arr = pd.Index(left_arr_tuple)
    right_arr = pd.Index(right_arr_tuple)
    common_indices = left_arr.intersection(right_arr)
    left_index = left_arr.get_indexer(common_indices)
    right_index = right_arr.get_indexer(common_indices)
    return left_index, right_index


left_idx, right_idx = _overlap_indices(converted.index, beta.index)

timer.start()
left_idx, right_idx = overlap_indices(converted.index, beta.index)
timer.stop("inter")

timer.start()
# x=converted.index.values.tobytes()
# x=tuple(converted.index)
# x = str(converted.index.values)
# x = converted.index.values.tostring()
x = "".join(converted.index)
# x = converted.index.values
timer.stop("inter")

np.intersect1d(left_cpgs, right_cpgs, assume_unique=True)

timer.start()
set1 = set(left_cpgs)
set2 = set(right_cpgs)
overlap = set1.intersection(set2)
indices_arr1 = [i for i, x in enumerate(left_cpgs) if x in overlap]
indices_arr2 = [i for i, x in enumerate(right_cpgs) if x in overlap]
timer.stop("inter")
