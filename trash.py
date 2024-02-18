# print("__file__=", __file__); quit()

from methylprep import Manifest as Manifest_old
from pathlib import Path
from pathlib import Path, PurePath
from pyllumina.dtypes.manifests import Manifest
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
from pyllumina.dtypes import ArrayType  # , Channel, ProbeType
from pyllumina.utils import (
    download_file,
    get_file_object,
    is_file_like,
    reset_file,
    ensure_directory_exists,
)

LOGGER = logging.getLogger(__name__)

from pyllumina.dtypes.manifests import (
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




# filepath=downloaded_filepath
# manifest_name=manifest_name
# dest_probes=manifest_filepath
# dest_control=control_filepath


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
