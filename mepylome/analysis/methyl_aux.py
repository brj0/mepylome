"""Auxiliary methods for the methylation analysis."""

import pickle
import re
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
from tqdm import tqdm

from mepylome.dtypes.arrays import ArrayType
from mepylome.dtypes.beads import (
    NEUTRAL_BETA,
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
UPLOADED = "Uploaded"
TEST_CASE = "Test_Case"
METHYLATION_CLASS = "Methylation_Class"


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
    msg = "Unsupported file format. Supported: ods, xlsx, xls, csv, tsv."
    raise ValueError(msg)


def guess_annotation_file(directory, verbose=False):
    """Returns the first spreadsheat file recursively found."""
    if verbose:
        log("[guess_annotation_file] Searching for annotation file...")
    supported_extensions = [".csv", ".tsv", ".ods", ".xls", ".xlsx"]
    for file in directory.rglob("*"):
        if file.suffix.lower() in supported_extensions:
            if verbose:
                log(f"[guess_annotation_file] Found annotation file: {file}")
            return file
    if verbose:
        log("[guess_annotation_file] No annotation file found.")
    return INVALID_PATH


def extract_sentrix_id(text):
    """Extracts the Sentrix ID from a given text if found."""
    matches = re.findall(r"\d+_R\d{2}C\d{2}", str(text))
    return matches[-1] if matches else text


def convert_to_sentrix_ids(data):
    """Tries to convert every ID in 'data' to a Sentrix ID."""
    if data is None:
        return None
    if isinstance(data, dict):
        return {extract_sentrix_id(key): value for key, value in data.items()}
    if isinstance(data, set):
        return {extract_sentrix_id(item) for item in data}
    return [extract_sentrix_id(item) for item in data]


class IdatHandler:
    """A class for handling IDAT files with annotation.

    Includes reading annotation from various file formats and provides
    description lookups for methylation classes.

    Args:
        idat_dir (str or Path): The directory where the IDAT files are located.
        annotation_file (str or Path, optional): The path to the annotation
            file. Defaults to None.
        test_dir (Path or None, optional): Directory for test files, including
            new cases or validation IDAT files or other test cases. Defaults to
            `None`.
        overlap (bool, optional): If True, restricts the sample paths to only
            those present in both the IDAT files and the annotation file.
            Defaults to False.
        sample_ids (list, optional): A list of sample IDs. If provided, the
            analysis will be restricted to these samples only. If `None`, the
            analysis will include all available samples.

    Attributes:
        idat_dir (Path): The directory path where the IDAT files are located.
        test_dir (Path or None, optional): Directory for test files, including
            new cases or validation IDAT files or other test cases. Defaults to
            `None`.
        overlap (bool): A flag indicating whether to restrict sample paths to
            only those present in both the IDAT files and the annotation file.
        test_ids (list): A list of valid test IDAT sample
            IDs.
        id_to_path (dict): A dictionary where the keys are sample IDs and the
            values are the file paths of IDAT files (from both `idat_dir` and
            `test_dir`).
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
        test_dir=None,
        overlap=False,
        sample_ids=None,
    ):
        # Initialize paths and attributes
        self.idat_dir = Path(idat_dir)
        self.annotation_file = (
            Path(annotation_file)
            if annotation_file and Path(annotation_file).exists()
            else guess_annotation_file(self.idat_dir, verbose=True)
        )
        self.test_dir = Path(test_dir) if test_dir else None
        self.overlap = overlap
        self.sample_ids = sample_ids

        # Load IDAT paths and annotation data
        self.analysis_id_to_path = self._get_id_to_path(self.idat_dir)
        self.test_id_to_path = self._get_id_to_path(self.test_dir)
        self.annotation_df = self._read_annotation_file()

        # Find ID column in annotation, set as index and filter cases
        id_mismatch, matched_column = self._identify_annotation_index()
        self._set_annotation_index_and_convert_ids(id_mismatch, matched_column)
        self._restrict_sample_ids(id_mismatch)
        self._apply_overlap_filter()

        # Derived attributes
        self.test_ids = list(self.test_id_to_path.keys())
        self.id_to_path = {**self.analysis_id_to_path, **self.test_id_to_path}
        self.idat_basename_to_id = {
            v.name: k for k, v in self.id_to_path.items()
        }
        self.id_to_basename = {k: v.name for k, v in self.id_to_path.items()}

        # Set available annotation for all IDAT files
        self.samples_annotated = self._get_samples_annotated()
        self.selected_columns = [self.samples_annotated.columns[0]]

        # Validation
        self._warn_on_sample_overlap()

    def _get_id_to_path(self, directory):
        """Retrieve valid IDAT sample IDs and paths from a directory."""
        if not directory or not Path(directory).exists():
            return {}
        return {
            path.name: path
            for path in idat_basepaths(directory)
            if is_valid_idat_basepath(path)
        }

    def _read_annotation_file(self):
        try:
            return read_dataframe(self.annotation_file)
        except (FileNotFoundError, ValueError):
            log(
                "[IdatHandler] Annotation file is missing, invalid or could "
                "not be read."
            )
            return pd.DataFrame()

    def _identify_annotation_index(self):
        """Identify and set the appropriate annotation column."""
        analysis_samples = set(self.analysis_id_to_path.keys())
        sentrix_analysis_samples = convert_to_sentrix_ids(analysis_samples)

        for col in self.annotation_df.columns:
            column_samples = set(self.annotation_df[col])

            if analysis_samples & column_samples:
                return False, col

            sentrix_column_samples = convert_to_sentrix_ids(column_samples)

            if (
                len(analysis_samples) == len(sentrix_analysis_samples)
                and len(column_samples) == len(sentrix_column_samples)
                and sentrix_analysis_samples & sentrix_column_samples
            ):
                return True, col

        return False, None

    def _set_annotation_index_and_convert_ids(self, id_missmatch, col_name):
        """Set annotation index and convert IDs to Sentrix format if needed."""
        if col_name is None:
            log(
                "[IdatHandler] No IDAT files found that are both on disk and "
                "in the annotation file."
            )
            return
        if not id_missmatch:
            log(f"[IdatHandler] Setting '{col_name}' as annotation index.")
            self.annotation_df = self.annotation_df.set_index(col_name)
            return

        log(f"[IdatHandler] Extracted Sentrix IDs from column '{col_name}'.")
        self.analysis_id_to_path = convert_to_sentrix_ids(
            self.analysis_id_to_path
        )
        self.test_id_to_path = convert_to_sentrix_ids(self.test_id_to_path)
        self.annotation_df.index = convert_to_sentrix_ids(
            self.annotation_df[col_name]
        )
        self.sample_ids = convert_to_sentrix_ids(self.sample_ids)

    def _get_samples_annotated(self):
        result_df = pd.DataFrame(index=self.id_to_path.keys())

        # Remove duplicate rows
        unique_annotation_df = self.annotation_df.loc[
            ~self.annotation_df.index.duplicated(keep="first")
        ]
        result_df = result_df.join(unique_annotation_df).fillna("")

        if result_df.empty:
            result_df[METHYLATION_CLASS] = ""

        if self.test_ids:
            result_df[TEST_CASE] = False
            result_df.loc[self.test_ids, TEST_CASE] = True

        return result_df

    def _restrict_sample_ids(self, id_mismatch):
        """Restricts samples to the ones in 'sample_ids'."""
        if not self.sample_ids:
            return

        restricted_ids = (
            convert_to_sentrix_ids(self.sample_ids)
            if id_mismatch
            else self.sample_ids
        )
        self.analysis_id_to_path = {
            id_: path
            for id_, path in self.analysis_id_to_path.items()
            if id_ in restricted_ids
        }

    def _apply_overlap_filter(self):
        """Filter samples to include only those IDATs present in annotation."""
        if not self.overlap:
            return

        valid_ids = set(self.annotation_df.index).intersection(
            self.analysis_id_to_path.keys()
        )
        self.analysis_id_to_path = {
            x: self.analysis_id_to_path[x] for x in valid_ids
        }

    def _warn_on_sample_overlap(self):
        """Warn about overlapping samples between analysis and test samples."""
        n_inters = len(
            set(self.analysis_id_to_path.keys()).intersection(self.test_ids)
        )
        if n_inters:
            warnings.warn(
                f"WARNING: 'test_dir' and 'analysis_dir' share {n_inters} "
                f"sample(s). 'idat_handler' may not work as expected. "
                "Verify inputs.",
                stacklevel=2,
            )

    def __len__(self):
        return len(self.id_to_path)

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
    def columns(self):
        return self.samples_annotated.columns.tolist()

    def features(self, columns=None, separator="|"):
        """Combines specified columns into a single label per sample.

        If `columns` is not provided, it defaults to the first column in
        `samples_annotated` or `selected_columns` if they are available. The
        function joins the values from the specified columns for each sample,
        converting them to strings and joining them with the specified
        separator.

        Args:
            columns (list, str, or None): List of column names (or a single
                column name) to use for creating the label. If None, defaults
                to the first column in `samples_annotated` or
                `selected_columns` if not None.
            separator (str): The separator used to join values from the
                columns. Default is "|".

        Returns:
            list: A list of combined labels, one per sample.

        Example:
            >>> idat_handler.features(columns=["GEO", "CNVs"])
            ['SGT_103|Balanced', 'SGT_056|Balanced', 'SGT_276|Balanced', ...]
        """
        if columns is None:
            if self.selected_columns is not None:
                columns = self.selected_columns
            else:
                columns = [self.samples_annotated.columns[0]]
        if not isinstance(columns, list):
            columns = [columns]
        return (
            self.samples_annotated[columns]
            .apply(lambda row: separator.join(row.values.astype(str)), axis=1)
            .tolist()
        )

    def __str__(self):
        title = f"{self.__class__.__name__}()"
        header = title + "\n" + "*" * len(title)
        lines = [header]

        def format_value(value):
            length_info = ""
            if isinstance(value, (pd.DataFrame, pd.Series, pd.Index)):
                display_value = str(value)
            elif isinstance(value, np.ndarray):
                display_value = str(value)
                length_info = f"\n\n[{len(value)} items]"
            elif hasattr(value, "__len__"):
                display_value = str(value)[:80] + (
                    "..." if len(str(value)) > 80 else ""
                )
                if len(str(value)) > 80:
                    length_info = f"\n\n[{len(value)} items]"
            else:
                display_value = str(value)[:80] + (
                    "..." if len(str(value)) > 80 else ""
                )
            return display_value, length_info

        for attr, value in sorted(self.__dict__.items()):
            display_value, length_info = format_value(value)
            lines.append(f"{attr}:\n{display_value}{length_info}")
        return "\n\n".join(lines)

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

    def get(self, idat_handler, cpgs, fill=NEUTRAL_BETA, parallel=True):
        """Retrieves beta values for specified IDs and CpGs."""
        filenames = [
            filename
            for filename in idat_handler.idat_basenames
            if filename not in self.invalid_filenames
        ]
        ids = [idat_handler.idat_basename_to_id[x] for x in filenames]

        check_memory(len(filenames), len(cpgs), DTYPE)
        beta_matrix = np.full((len(filenames), len(cpgs)), fill, dtype=DTYPE)

        left_idx = {}
        right_idx = {}

        for key, item in self.array_cpgs.items():
            left_idx[key], right_idx[key] = _overlap_indices(cpgs, item)

        def process_beta_value(i, filename):
            path = self.paths.get(filename)
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
                    future.result()
        else:
            for i, filename in tqdm(
                enumerate(filenames),
                desc="Reading beta values from disk (serial)",
                total=len(filenames),
            ):
                process_beta_value(i, filename)
        return pd.DataFrame(beta_matrix, columns=cpgs, index=ids)

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
    # Loading manifests here prevents race conditions
    Manifest.load()
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
    return betas_handler.get(idat_handler, cpgs)
