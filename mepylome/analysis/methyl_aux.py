"""Auxiliary methods for the methylation analysis."""

import pickle
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
from tqdm import tqdm

from mepylome.dtypes.arrays import ArrayType
from mepylome.dtypes.beads import (
    MethylData,
    _overlap_indices,
    idat_basepaths,
    is_valid_idat_basepath,
)
from mepylome.dtypes.manifests import Manifest
from mepylome.utils.files import MEPYLOME_TMP_DIR, ensure_directory_exists
from mepylome.utils.varia import log

DTYPE = np.float32
INVALID_PATH = Path("None")
NEUTRAL_BETA = 0.49


class ProgressBar:
    """A thread-safe progress bar.

    Attributes:
        cur_value (int): The current value of the progress bar.
        max_value (int): The maximum value of the progress bar.
        text (str): Optional text to display alongside the progress.
        lock (threading.Lock): A lock to ensure thread safety.
    """

    def __init__(self, max_value=100, text=""):
        self.cur_value = 0
        self.max_value = int(max_value)
        self.text = str(text)
        self.lock = threading.Lock()

    def reset(self, max_value=100, cur_value=0, text=""):
        with self.lock:
            self.cur_value = cur_value
            self.max_value = int(max_value)
            self.text = str(text)

    def increment(self, n=1):
        with self.lock:
            self.cur_value = min(self.cur_value + n, self.max_value)

    def get_progress(self):
        with self.lock:
            if self.max_value == 0:
                return 100
            return self.cur_value * 100 // self.max_value

    def get_text(self):
        with self.lock:
            if self.cur_value == self.max_value:
                out_str = "100 %"
            else:
                out_str = (
                    f"{self.cur_value}/{self.max_value} {self.text}".rstrip()
                )
            return out_str

    def __str__(self):
        lines = [
            "ProgressBar(",
            f"    cur_value: {self.cur_value}",
            f"    max_value: {self.max_value}",
            f"    progress: {self.get_progress()}",
            ")",
        ]
        return "\n".join(lines)

    def __repr__(self):
        return str(self)


def read_dataframe(path, **kwargs):
    """Reads a DataFrame from the specified file path.

    Supports ods, xlsx, xls, csv (comma-separated), csv (column-separated), and
    tsv formats.

    Args:
        path (str): The file path to read the DataFrame from.
        **kwargs: Additional keyword arguments to pass to the underlying pandas
            read function.

    Returns:
        pd.DataFrame: The loaded DataFrame.

    Raises:
        ValueError: If the file format is not supported.
    """
    path = Path(path)
    if path.suffix in [".xlsx", ".xls"]:
        return pd.read_excel(path, **kwargs)
    if path.suffix == ".ods":
        return pd.read_excel(path, engine="odf", **kwargs)
    if path.suffix in [".csv", ".tsv"]:
        return pd.read_csv(path, sep=None, **kwargs, engine="python")
    raise ValueError(
        "Unsupported file format. Supported: ods, xlsx, xls, csv, tsv."
    )


def guess_annotation_file(directory, verbose=False):
    """Returns the first spreadsheat file in the given directory."""
    if verbose:
        log("[MethylAnalysis] Try to read annotation file...")
    supported_extensions = [".csv", ".tsv", ".ods", ".xls", ".xlsx"]
    for file in directory.glob("*"):
        if file.suffix in supported_extensions:
            return file
    return INVALID_PATH


def extract_sentrix_id(text):
    """Extracts the sentrix ID if found."""
    matches = re.findall(r"\d+_R\d{2}C\d{2}", str(text))
    return matches[-1] if matches else text


class IdatHandler:
    """A class for handling IDAT files with annotation.

    Includes reading annotation from various file formats and provides
    description lookups for methylation classes.

    Args:
        idat_dir (str or Path): The directory where the IDAT files are located.
        annotation_file (str or Path, optional): The path to the annotation
            file. Defaults to None.
        upload_dir (str or Path, optional): The directory where uploaded IDAT
            files are stored. Defaults to None.
        overlap (bool, optional): If True, restricts the sample paths to only
            those present in both the IDAT files and the annotation file.
            Defaults to False.
        sample_ids (list, optional): A list of sample IDs. If provided, the
            analysis will be restricted to these samples only. If `None`, the
            analysis will include all available samples.

    Attributes:
        idat_dir (Path): The directory path where the IDAT files are located.
        upload_dir (Path or None): The directory path for uploaded IDAT files,
            if provided. If not provided or the directory does not exist, this
            is set to None.
        overlap (bool): A flag indicating whether to restrict sample paths to
            only those present in both the IDAT files and the annotation file.
        uploaded_sample_ids (list): A sorted list of valid uploaded IDAT sample
            IDs.
        id_to_path (dict): A dictionary where the keys are sample IDs and the
            values are the file paths of IDAT files (from both `idat_dir` and
            `upload_dir`).
        annotation_file (Path): The path to the annotation file.
        annotation_df (pandas.DataFrame or None): A DataFrame containing the
            annotation data, if loaded.
        samples_annotated (pandas.DataFrame or None): A DataFrame containing
            the samples as index and the annotation in the columns.
        selected_columns (list): A list of selected columns from the annotated
            samples, initialized with the first column.
        sample_ids (list, optional): A list of sample IDs. If provided, the
            analysis will be restricted to these samples only. If `None`, the
            analysis will include all available samples.
    """

    def __init__(
        self,
        idat_dir,
        annotation_file=None,
        *,
        upload_dir=None,
        overlap=False,
        sample_ids=None,
    ):
        self._annotation_samples_mismatch = False
        self.idat_dir = Path(idat_dir)
        self.sample_ids = sample_ids
        self.overlap = overlap

        if upload_dir is not None and Path(upload_dir).exists():
            self.upload_dir = Path(upload_dir)
            uploaded_sample_paths = idat_basepaths(self.upload_dir)
        else:
            self.upload_dir = None
            uploaded_sample_paths = []

        if annotation_file is None:
            self.annotation_file = guess_annotation_file(
                self.idat_dir, verbose=True
            )
        else:
            self.annotation_file = Path(annotation_file)

        # Sort uploaded IDAT's for consistent hash
        self.uploaded_sample_ids = sorted(
            x.name for x in uploaded_sample_paths if is_valid_idat_basepath(x)
        )

        self.id_to_path = {
            x.name: x
            for x in idat_basepaths(self.idat_dir) + uploaded_sample_paths
            if is_valid_idat_basepath(x)
        }

        self.annotation_df = self._get_annotation()
        self.samples_annotated = self._get_samples_annotated()

        if self.overlap:
            valid_ids = self.annotation_df.index.intersection(
                self.id_to_path.keys()
            )
            self.id_to_path = {x: self.id_to_path[x] for x in valid_ids}
            self.samples_annotated = self.samples_annotated.loc[valid_ids]

        # Need to extract sentrix ID's if necessary
        if self._annotation_samples_mismatch and self.sample_ids is not None:
            _sample_ids = [extract_sentrix_id(x) for x in self.sample_ids]
        else:
            _sample_ids = self.sample_ids

        if _sample_ids is not None:
            self.id_to_path = {x: self.id_to_path[x] for x in _sample_ids}
            self.samples_annotated = self.samples_annotated.loc[_sample_ids]

        self.selected_columns = [self.samples_annotated.columns[0]]
        self.idat_basename_to_id = {
            v.name: k for k, v in self.id_to_path.items()
        }

    def __len__(self):
        return len(self.id_to_path)

    def _get_annotation(self):
        try:
            return read_dataframe(self.annotation_file)
        except (FileNotFoundError, ValueError):
            log(
                "[IdatHandler] Annotation file is missing, invalid or could "
                "not be read."
            )
            return pd.DataFrame()

    def _get_samples_annotated(self):
        # Check if any column in annotation_df contains existing IDAT samples
        if any(
            self._extract_sentrix_ids(col)
            for col in self.annotation_df.columns
        ):
            result_df = pd.DataFrame(index=self.id_to_path.keys())
            result_df = result_df.join(self.annotation_df).fillna("")
        else:
            result_df = pd.DataFrame(index=self.id_to_path.keys())
            log(
                "[IdatHandler] No matching IDAT files on disk found in "
                "the annotation file."
            )

        if result_df.empty:
            result_df["Methylation_Class"] = "NO_ANNOTATION"
        result_df.loc[self.uploaded_sample_ids] = "UPLOADED"
        return result_df

    def _extract_sentrix_ids(self, col):
        if set(self.id_to_path.keys()).intersection(self.annotation_df[col]):
            log(f"[IdatHandler] Setting column '{col}' as annotation index.")
            self.annotation_df = self.annotation_df.set_index(col)
            return True

        new_paths = {
            extract_sentrix_id(k): v for k, v in self.id_to_path.items()
        }
        new_index = [extract_sentrix_id(x) for x in self.annotation_df[col]]

        if (
            len(new_paths) == len(self.id_to_path)
            and len(set(new_index)) == len(set(self.annotation_df[col]))
            and set(new_paths.keys()).intersection(new_index)
        ):
            log(f"[IdatHandler] Extracted Sentrix IDs from column '{col}'.")
            self.id_to_path = new_paths
            self.annotation_df.index = new_index
            self._annotation_samples_mismatch = True
            return True

        return False

    @property
    def ids(self):
        return list(self.id_to_path.keys())

    @property
    def idat_basenames(self):
        return list(self.idat_basename_to_id.keys())

    @property
    def paths(self):
        return list(self.id_to_path.values())

    @property
    def properties(self):
        return self.samples_annotated.columns.tolist()

    def compound_class(self, columns=None):
        if columns is None:
            return self.samples_annotated.iloc[:, 0].tolist()
        if not isinstance(columns, list):
            columns = [columns]
        return (
            self.samples_annotated[columns]
            .apply(lambda row: "|".join(row.values.astype(str)), axis=1)
            .tolist()
        )

    def __str__(self):
        lines = [
            "IdatHandler():",
            f"annotation_file: '{self.annotation_file}'",
            f"samples_annotated:\n{self.samples_annotated}",
            f"selected_columns:\n{self.selected_columns}",
        ]
        return "\n".join(lines)

    def __repr__(self):
        return str(self)


def check_memory(nrows, ncols, dtype):
    """Checks if sufficient free memory is available for a given numpy array.

    Raises:
        MemoryError: If the required memory exceeds the available memory.

    Example:
        >>> check_memory(1000, 500000, np.float32)
    """
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
    """Returns all CpG sites for all array types."""
    path = MEPYLOME_TMP_DIR / "all_cpgs.pkl"
    if not path.exists():
        with path.open("wb") as f:
            array_cpgs = {
                ArrayType("450k"): Manifest("450k").methylation_probes,
                ArrayType("epic"): Manifest("epic").methylation_probes,
                ArrayType("epicv2"): Manifest("epicv2").methylation_probes,
            }
            pickle.dump(array_cpgs, f)
    with path.open("rb") as f:
        return pickle.load(f)


class BetasHandler:
    """Manages storage and retrieval of beta values.

    Args:
        directory (str or Path): Directory path where beta files ara stored.
        array_cpgs (dict, optional): Dictionary mapping ArrayType names to CpG
            arrays. If not provided, CpG arrays are fetched using default
            method.

    Attributes:
        basedir (Path): Path to the directory containing beta files.
        array_cpgs (dict): Dictionary mapping ArrayType names to CpG arrays.
        dir (dict): Dictionary mapping ArrayType and 'error' to subdirectories
            of basedir.
        array_type_from_dir (dict): Reverse mapping of directory paths to
            ArrayType names.
        paths (dict): Dictionary of all beta file paths.
        invalid_paths (dict): Dictionary of invalid beta file paths.

    """

    def __init__(self, directory, array_cpgs=None):
        self.basedir = Path(directory).expanduser()
        self.array_cpgs = array_cpgs
        if self.array_cpgs is None:
            self.array_cpgs = get_array_cpgs()
        self.dir = {}
        for key in self.array_cpgs:
            self.dir[key] = self.basedir / f"{key}"
            ensure_directory_exists(self.dir[key])
        self.dir["error"] = self.basedir / "error"
        ensure_directory_exists(self.dir["error"])
        self.array_type_from_dir = {
            item: key for key, item in self.dir.items()
        }
        self.paths = None
        self.update()

    @property
    def filenames(self):
        """Returns all idat basenames."""
        return list(self.paths.keys())

    @property
    def invalid_filenames(self):
        """Returns all invalid invalid basenames."""
        return list(self.invalid_paths.keys())

    def add(self, betas, filename, array_type):
        """Adds beta values to the file system on disk."""
        betas.astype(DTYPE).tofile(self.dir[array_type] / filename)

    def add_error(self, filename, msg):
        """Adds error message to the file system on disk."""
        with (self.dir["error"] / filename).open("w") as f:
            f.write(str(msg))

    def get(self, filenames, cpgs, fill=NEUTRAL_BETA, parallel=False):
        """Retrieves beta values for specified IDs and CpGs."""
        check_memory(len(filenames), len(cpgs), DTYPE)
        beta_matrix = np.full((len(filenames), len(cpgs)), fill, dtype=DTYPE)
        left_idx = {}
        right_idx = {}
        for key, item in self.array_cpgs.items():
            left_idx[key], right_idx[key] = _overlap_indices(cpgs, item)

        def process_beta_value(i, filename):
            path = self.paths.get(filename, None)
            if path is not None:
                key = self.array_type_from_dir[path.parent]
                betas = np.fromfile(path, dtype=DTYPE)
                beta_matrix[i, left_idx[key]] = betas[right_idx[key]]

        if parallel:
            with ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(process_beta_value, i, filename)
                    for i, filename in enumerate(filenames)
                ]
                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="Reading beta values from disk (parallel)",
                ):
                    future.beta_matrix()
        else:
            for i, filename in tqdm(
                enumerate(filenames),
                desc="Reading beta values from disk (serial)",
                total=len(filenames),
            ):
                process_beta_value(i, filename)
        return pd.DataFrame(beta_matrix, columns=cpgs, index=filenames)

    def update(self):
        """Updates the paths if beta vales were added after initialization."""
        self.paths = {
            path.name: path
            for path in self.basedir.rglob("*")
            if not path.is_dir()
        }
        self.invalid_paths = {
            path.name: path for path in self.dir["error"].rglob("*")
        }


def extract_beta(data):
    """Extracts and saves beta values for specified CpGs from an IDAT file."""
    idat_file, prep, betas_handler = data
    try:
        methyl = MethylData(file=idat_file, prep=prep)
        betas = methyl.betas_at().values.ravel()
        array_type = methyl.array_type
        betas_handler.add(betas, idat_file.name, array_type)
    except Exception as error:
        betas_handler.add_error(idat_file.name, error)
        print(f"The following error occured for {idat_file.name}: {error}")


def get_betas(idat_handler, cpgs, prep, betas_path, pbar=None):
    """Extracts and processes beta values from IDAT files.

    This function processes IDAT files to extract beta values for specified
    CpG sites (CpGs). The processed beta values are saved in a temporary
    folder to facilitate quicker subsequent extractions.

    Args:
        idat_handler (IdatHandler): Handler for IDAT file paths and metadata.
        cpgs (list): List of CpGs to include in the output matrix.
        prep (str): Prepreparation method for the MethylData.
        betas_path (Path): Path to save/load the betas.
        pbar (ProgressBar): Progress bar for tracking progress.

    Returns:
        pd.DataFrame: DataFrame containing the beta values for the specified
            CpGs.
    """
    betas_handler = BetasHandler(betas_path)
    ids_found = {
        idat_handler.idat_basename_to_id.get(x)
        for x in betas_handler.filenames
    }
    missing_ids = list(set(idat_handler.ids) - ids_found)
    if len(missing_ids) > 0:
        args_list = [
            (idat_handler.id_to_path[id_], prep, betas_handler)
            for id_ in missing_ids
        ]
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(extract_beta, args) for args in args_list
            ]
            with tqdm(
                total=len(missing_ids), desc="Reading IDAT files"
            ) as tqdm_bar:
                for future in as_completed(futures):
                    future.result()
                    tqdm_bar.update(1)
                    if pbar is not None:
                        pbar.increment()
        betas_handler.update()
    valid_beta_value_files = [
        filename
        for filename in idat_handler.idat_basenames
        if filename not in betas_handler.invalid_filenames
    ]
    return betas_handler.get(valid_beta_value_files, cpgs)
