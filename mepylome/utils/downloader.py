"""Downloader for GEO, ArrayExpress, and TCGA datasets.

Provides `download_idats()` to fetch IDAT files and/or metadata from GEO
series/samples, ArrayExpress series, or TCGA datasets. Supports single strings,
dicts, or lists of datasets.

Examples:
    # Download a GEO or AE series
    download_idats("GSE123456", save_dir="~/mepylome/data")
    download_idats("E-MTAB-12346", save_dir="~/mepylome/data")

    # Download specific GEO samples
    download_idats(
        dataset={
            "source": "geo",
            "series": "GSE140686",
            "samples": [
                "GSM4180453_201904410008_R06C01",
                "GSM4180454_201904410008_R05C01",
                "GSM4180455_201904410008_R04C01",
            ],
        },
        save_dir="~/mepylome/data",
        idat=True,
        metadata=True,
    )

    # Download a TCGA dataset
    download_idats(
        dataset={
            "source": "tcga",
            "metadata_cart": "~/mepylome/data/metadata.cart.2025-01-01.json",
            "metadata_clinical": "~/mepylome/data/clinical.tsv",
        },
        save_dir="~/mepylome/data",
        idat=True,
        metadata=True,
    )
"""

import gzip
import hashlib
import json
import logging
import re
import shutil
import sys
import tarfile
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd
import requests

from mepylome.utils.files import (
    download_file,
    download_files,
    get_resource_path,
)
from mepylome.utils.varia import MEPYLOME_TMP_DIR

logger = logging.getLogger(__name__)


GEO_RAW_IDAT_URL = (
    "https://www.ncbi.nlm.nih.gov/geo/download/?acc={acc}&format=file"
)
GEO_SINGLE_IDAT_URL = (
    "https://www.ncbi.nlm.nih.gov/geo/download/?acc={acc}&format=file"
    "&file={filename}"
)
GEO_MINIML_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/{geo_group}/{acc}/miniml/"
    "{acc}_family.xml.tgz"
)
ARRAY_EXPRESS_URL = (
    "https://ftp.ebi.ac.uk/biostudies/fire/E-MTAB-/{ae_group}/{acc}/Files/"
)
TCGA_URL = "https://api.gdc.cancer.gov/data/{file_id}"


def _geo_group(geo_id: str) -> str:
    """Compute the GEO series group folder used on the FTP server.

    Example:
        >>> _geo_group('GSE12345')
        'GSE12nnn'
    """
    if len(geo_id) < 4:
        raise ValueError(f"geo_id seems too short: {geo_id}")
    return f"{geo_id[:-3]}nnn"


def _strip_ns(elem: ET.Element) -> None:
    """Remove namespace prefix in-place for an element tree."""
    for e in elem.iter():
        if "}" in e.tag:
            e.tag = e.tag.split("}", 1)[1]


def _text_of(el: ET.Element) -> str:
    """Return .text stripped or empty string."""
    return (el.text or "").strip()


def _first_attr_value(attrib: Optional[dict[str, Any]]) -> str:
    """Return attribute values as string joined with ';'."""
    vals = [str(v).strip() for v in attrib.values()] if attrib else []
    return ";".join(vals)


def _get_val(node: ET.Element) -> str:
    """Returns value of a xml node."""
    text = _text_of(node)
    return text or _first_attr_value(node.attrib)


def _unique_add(key: str, value: str, dictionary: dict[str, Any]) -> None:
    """Add a key–value pair, appending _1, _2... if the key already exists."""
    if key not in dictionary:
        dictionary[key] = value
        return
    counter = 1
    while f"{key}_{counter}" in dictionary:
        counter += 1
    dictionary[f"{key}_{counter}"] = value


def parse_miniml_to_df(
    miniml_path: Path,
    series_id: str,
    samples: Optional[Iterable[str]] = None,
    meta: Optional[str] = None,
) -> None:
    """Parse a GEO MINiML family XML and save as spreadsheet to disk."""
    tree = ET.parse(miniml_path)
    root = tree.getroot()
    _strip_ns(root)
    sample_elements = root.findall(".//Sample")
    if not sample_elements:
        raise ValueError("No <Sample> elements found in the MINiML file.")
    rows = []
    for s in sample_elements:
        row: dict = {}
        for child in list(s):
            tag = child.tag
            if tag == "Supplementary-Data" and "Sample_ID" not in row:
                idat_filename = _text_of(child).split("/")[-1]
                sample_id = (
                    idat_filename.removesuffix(".idat.gz")
                    .removesuffix("_Grn")
                    .removesuffix("_Red")
                )
                _unique_add("Sample_ID", sample_id, row)
            if list(child):
                for sub in child:
                    sub_tag = sub.tag
                    if sub_tag == "Characteristics":
                        sub_tag = sub.attrib.get("tag") or sub_tag
                    _unique_add(sub_tag, _get_val(sub), row)
            else:
                _unique_add(tag, _get_val(child), row)
        rows.append(row)
    annotation = pd.DataFrame(rows)

    # If files specified, restrict rows to those Sample_IDs
    if not samples or samples == "all":
        result_df = annotation
        logger.info(
            "Writing metadata for all %d samples for %s",
            len(result_df),
            series_id,
        )
    else:
        requested = set(samples)
        filtered = annotation[annotation["Sample_ID"].isin(requested)].copy()
        logger.info(
            "Filtered metadata: %d of %d samples retained for %s",
            len(filtered),
            len(annotation),
            series_id,
        )
        result_df = filtered

    annotation_name = meta or "annotation"
    csv_path = miniml_path.parent / f"{annotation_name}.csv"
    result_df.to_csv(csv_path, index=False)


def download_geo_metadata(
    series_id: str,
    save_dir: Path,
    show_progress: bool = True,
    samples: Optional[Iterable[str]] = None,
    subdir: Optional[str] = None,
    meta: Optional[str] = None,
) -> None:
    """Download and extract the MINiML (family XML) for a GEO series.

    Args:
        series_id (str): The GEO accession ID of the dataset to download
            (e.g., "GSE1234").
        save_dir (Path): Directory path where the metadata will be saved.
        show_progress (bool, optional): If True, displays logging messages and
            progress bar during download. Defaults to True.
        samples (Iterable[str]): Optional iterable of Sample_ID bases (e.g.
            "GSM4429896_201503470062_R02C01"). If provided, restrict the CSV to
            those Sample_IDs only.
        subdir (Optional[str]): Optional subdirectory name under `save_dir` for
            the dataset folder. Defaults to "series_id" if None.
        meta (Optional[str]): Optional base name for the output annotation file
            (without extension). Defaults to "annotation" if None.
    """
    subdir = subdir or series_id
    samples_dir = save_dir / subdir
    miniml_path = samples_dir / f"{series_id}.xml"
    miniml_tar_path = samples_dir / f"{series_id}_family.xml.tgz"
    samples_dir.mkdir(parents=True, exist_ok=True)

    # Download the miniml tarball
    geo_group = _geo_group(series_id)
    miniml_tar_url = GEO_MINIML_URL.format(geo_group=geo_group, acc=series_id)
    download_file(miniml_tar_url, miniml_tar_path, show_progress=show_progress)

    # Extract the XML inside the tarball.
    try:
        if not miniml_path.exists():
            with tarfile.open(miniml_tar_path, "r:gz") as tar:
                member_name = miniml_tar_path.stem
                if sys.version_info >= (3, 12):
                    # Python 3.12+ needs the `filter` argument
                    tar.extract(
                        member=member_name, path=samples_dir, filter="data"
                    )
                else:
                    # Older Python versions: don't pass `filter`
                    tar.extract(member=member_name, path=samples_dir)
                miniml_tar_path.with_suffix("").rename(miniml_path)
    except Exception as exc:
        logger.debug("Could not unzip %s: %s", miniml_tar_path, exc)

    parse_miniml_to_df(miniml_path, series_id, samples, meta)


def download_geo_idat_all_files(
    series_id: str,
    save_dir: Path,
    show_progress: bool = True,
    subdir: Optional[str] = None,
) -> None:
    """Download and extract the RAW IDAT archive for a GEO series.

    Args:
        series_id (str): The GEO accession ID of the dataset to download
            (e.g., "GSE1234").
        save_dir (Path): Directory path where the metadata will be saved.
        show_progress (bool, optional): If True, displays logging messages and
            progress bar during download. Defaults to True.
        subdir (Optional[str]): Optional subdirectory name under `save_dir` for
            the dataset folder. Defaults to "series_id" if None.
    """
    subdir = subdir or series_id
    samples_dir = Path(save_dir) / subdir
    idat_dir = samples_dir / "idat"
    if idat_dir.exists():
        logger.info(
            "IDAT directory already exists: %s. Skipping download.", idat_dir
        )
        return
    samples_dir.mkdir(parents=True, exist_ok=True)

    # Download the RAW tarball
    tar_idat_url = GEO_RAW_IDAT_URL.format(acc=series_id)
    tar_idat_path = samples_dir / f"{series_id}_RAW.tar"
    download_file(tar_idat_url, tar_idat_path, show_progress=show_progress)
    idat_dir.mkdir(parents=True, exist_ok=True)

    # Extract idat files
    try:
        with tarfile.open(tar_idat_path, "r:*") as tar:
            if sys.version_info >= (3, 12):
                # Python 3.12+ supports the `filter` kwarg
                tar.extractall(path=idat_dir, filter="data")
            else:
                # Older Python versions: no filter argument
                tar.extractall(path=idat_dir)

        # Remove unwanted GPL*csv.gz manifest files if present
        for file_path in idat_dir.rglob("*"):
            if file_path.is_file() and "idat" not in file_path.name.lower():
                try:
                    file_path.unlink()
                    logger.info("Deleted non-IDAT file: %s", file_path)
                except Exception as exc:
                    logger.debug(
                        "Could not delete non-IDAT file %s: %s", file_path, exc
                    )
    finally:
        # remove the RAW tar if extraction succeeded
        try:
            tar_idat_path.unlink()

        except Exception:
            logger.debug("Could not delete %s", tar_idat_path)

    logger.info("Extracted idat files to %s", idat_dir)


def download_geo_idat_single_files(
    series_id: str,
    save_dir: Path,
    samples: Iterable[str],
    show_progress: bool = True,
    subdir: Optional[str] = None,
) -> None:
    """Download individual IDAT files for the provided samples.

    Args:
        series_id: GEO series accession (used to build save path).
        save_dir: base directory where series folder will be created.
        samples: iterable of sample base names like
            "GSM4180454_201904410008_R05C01".
        show_progress (bool, optional): If True, displays logging messages and
            progress bar during download. Defaults to True.
        subdir (Optional[str]): Optional subdirectory name under `save_dir` for
            the dataset folder. Defaults to "series_id" if None.
    """
    subdir = subdir or series_id
    samples_dir = Path(save_dir) / subdir
    idat_dir = samples_dir / "idat"

    samples_dir.mkdir(parents=True, exist_ok=True)
    idat_dir.mkdir(parents=True, exist_ok=True)

    urls = []
    paths = []
    for file in samples:
        geo_acc = file.split("_", 1)[0]
        for color in ("Grn", "Red"):
            filename = f"{file}_{color}.idat.gz"
            encoded_filename = filename.replace("_", "%5F").replace(".", "%2E")
            url = GEO_SINGLE_IDAT_URL.format(
                acc=geo_acc, filename=encoded_filename
            )
            idat_path = idat_dir / filename
            urls.append(url)
            paths.append(idat_path)

    download_files(urls, paths, show_progress=show_progress)
    logger.info("Downloaded %d idat files to %s", len(paths), idat_dir)


def download_geo_idat(
    series_id: str,
    save_dir: Path,
    show_progress: bool = True,
    samples: Optional[Iterable[str]] = None,
    subdir: Optional[str] = None,
) -> None:
    """Downloads IDAT files from geo.

    Either downloads the complete RAW tarball if samples is None or 'all' or
    per-sample download.

    Args:
        series_id (str): The GEO accession ID of the dataset to download
            (e.g., "GSE1234").
        save_dir (Path): Directory path where the metadata will be saved.
        show_progress (bool, optional): If True, displays logging messages and
            progress bar during download. Defaults to True.
        samples (Iterable[str]): Optional iterable of Sample_ID bases (e.g.
            "GSM4429896_201503470062_R02C01"). If provided, restricts download
            to those Sample_IDs only (downloads per-sample Grn/Red idat .gz
            files).
        subdir (Optional[str]): Optional subdirectory name under `save_dir` for
            the dataset folder. Defaults to "series_id" if None.
    """
    if not samples or samples == "all":
        download_geo_idat_all_files(
            series_id=series_id,
            save_dir=save_dir,
            show_progress=show_progress,
            subdir=subdir,
        )
    else:
        download_geo_idat_single_files(
            series_id=series_id,
            save_dir=save_dir,
            samples=samples,
            show_progress=show_progress,
            subdir=subdir,
        )


def download_arrayexpress_metadata(
    series_id: str,
    save_dir: Path,
    samples: Optional[Iterable[str]] = None,
    show_progress: bool = True,
    subdir: Optional[str] = None,
    meta: Optional[str] = None,
) -> None:
    """Download the SDRF metadata file and save as simplified CSV.

    Args:
        series_id (str): The ArrayExpress accession ID of the dataset to
            download (e.g., "E-MTAB-1234").
        save_dir (Path): Directory path where the metadata will be saved.
        samples (Iterable[str]): Optional iterable of Sample_ID bases (e.g.
            "201503470062_R02C01"). If provided, restrict the CSV to those
            Sample_IDs only.
        show_progress (bool, optional): If True, displays logging messages and
            progress bar during download. Defaults to True.
        subdir (Optional[str]): Optional subdirectory name under `save_dir` for
            the dataset folder. Defaults to "series_id" if None.
        meta (Optional[str]): Optional base name for the output annotation file
            (without extension). Defaults to "annotation" if None.
    """
    subdir = subdir or series_id
    samples_dir = save_dir / subdir
    annotation_name = meta or "annotation"
    csv_path = samples_dir / f"{annotation_name}.csv"
    samples_dir.mkdir(parents=True, exist_ok=True)

    # Download SDRF file
    url = ARRAY_EXPRESS_URL.format(ae_group=series_id[-3:], acc=series_id)
    sdrf_filename = f"{series_id}.sdrf.txt"
    sdrf_url = f"{url}{sdrf_filename}"
    sdrf_path = samples_dir / sdrf_filename
    download_file(sdrf_url, sdrf_path, show_progress=show_progress)

    # Read SDRF and extract unique Sample_IDs
    annotation = pd.read_csv(sdrf_path, sep="\t")
    annotation["Sample_ID"] = (
        annotation["Array Data File"].str.split("_").str[:2].str.join("_")
    )
    annotation = annotation.drop_duplicates(subset=["Sample_ID"], keep="first")

    # If files specified, restrict rows to those Sample_IDs
    if not samples or samples == "all":
        result_df = annotation
        logger.info(
            "Writing metadata for all %d samples for %s",
            len(result_df),
            series_id,
        )
    else:
        requested = set(samples)
        filtered = annotation[annotation["Sample_ID"].isin(requested)].copy()
        logger.info(
            "Filtered metadata: %d of %d samples retained for %s",
            len(filtered),
            len(annotation),
            series_id,
        )
        result_df = filtered

    # Save simplified CSV
    result_df.to_csv(csv_path, index=False)


def download_arrayexpress_idat(
    series_id: str,
    save_dir: Path,
    samples: Optional[Iterable[str]] = None,
    show_progress: bool = True,
    subdir: Optional[str] = None,
) -> None:
    """Download all IDAT files for a given ArrayExpress ID.

    Args:
        series_id (str): The ArrayExpress accession ID of the dataset to
            download (e.g., "E-MTAB-1234").
        save_dir (Path): Directory path where the metadata will be saved.
        samples (Iterable[str]): Optional iterable of Sample_ID bases (e.g.
            "201503470062_R02C01"). If provided, restricts download to those
            Sample_IDs only.
        show_progress (bool, optional): If True, displays logging messages and
            progress bar during download. Defaults to True.
        subdir (Optional[str]): Optional subdirectory name under `save_dir` for
            the dataset folder. Defaults to "series_id" if None.
    """
    subdir = subdir or series_id
    samples_dir = save_dir / subdir
    idat_dir = samples_dir / "idat"
    samples_dir.mkdir(parents=True, exist_ok=True)
    idat_dir.mkdir(parents=True, exist_ok=True)

    # Fetch file listing from ArrayExpress
    url = ARRAY_EXPRESS_URL.format(ae_group=series_id[-3:], acc=series_id)
    response = requests.get(url, timeout=20)
    response.raise_for_status()

    # Find all hrefs that contain .idat
    remote_idats = re.findall(
        r'href=[\'"]?([^\'" >]+?\.idat[^\'" >]*)', response.text
    )

    if not samples or samples == "all":
        idat_urls = sorted(url + filename for filename in remote_idats)
    else:
        bases = list(samples)
        requested_idats = [f"{id_}_Grn.idat" for id_ in bases] + [
            f"{id_}_Red.idat" for id_ in bases
        ]
        missing = set(requested_idats) - set(remote_idats)
        if missing:
            missing_str = ", ".join(missing)
            raise ValueError(
                f"The following files are not found remotely: {missing_str}"
            )
        idat_urls = sorted(url + filename for filename in requested_idats)

    # Download IDAT files
    logger.info("Downloading %d IDAT files to %s", len(idat_urls), idat_dir)

    save_paths = [idat_dir / Path(url).name for url in idat_urls]
    download_files(
        urls=idat_urls,
        save_paths=save_paths,
        overwrite=False,
        show_progress=show_progress,
    )


def _get_tcga_series(path: Path) -> str:
    """Return an 8-byte BLAKE2b hex digest for the file at `path`."""
    path = Path(path).expanduser()
    with open(path, "rb") as f:
        data = f.read()
    hash_id = hashlib.blake2b(data, digest_size=8).hexdigest()
    return f"TCGA_{hash_id}"


def make_tcga_metadata(
    save_dir: Path,
    metadata_cart: Path,
    metadata_clinical: Path,
    subdir: Optional[str] = None,
    meta: Optional[str] = None,
) -> None:
    """Build merged TCGA annotation DataFrame and saves to disk.

    Args:
        save_dir (Path): directory where output CSV will be written (if
            write_csv True).
        metadata_cart (Path): path to metadata.cart JSON from GDC.
        metadata_clinical (Path): path to clinical TSV (tab-separated).
        subdir (Optional[str]): Optional subdirectory name under `save_dir` for
            the dataset folder. Defaults to "TCGA_<hash>" if None.
        meta (Optional[str]): Optional base name for the output annotation file
            (without extension). Defaults to "annotation" if None.
    """
    save_dir = Path(save_dir).expanduser()
    subdir = subdir or _get_tcga_series(metadata_cart)
    samples_dir = save_dir / subdir
    samples_dir.mkdir(parents=True, exist_ok=True)

    # local helper: extract dataframe from JSON content
    def _extract_case_file_df(json_path: Path) -> pd.DataFrame:
        """Extracts a dictionary mapping from IDAT IDs to case IDs."""
        with json_path.open() as f:
            data = json.load(f)
        rows = []
        n_suffix = len("_Grn.idat")
        for item in data:
            case_id = item.get("associated_entities", [{}])[0].get(
                "case_id", ""
            )
            row = {
                "file_id": item.get("file_id"),
                "file_name": item.get("file_name", ""),
                "Sample_ID": item.get("file_name", "")[:-n_suffix],
                "md5sum": item.get("md5sum"),
                "case_id": case_id,
            }
            rows.append(row)
        return pd.DataFrame(rows)

    download_df = _extract_case_file_df(metadata_cart).rename(
        columns={"file_id": "id", "file_name": "filename"}
    )

    # Deduplicate so we have one filename per case_id (keep first seen)
    case_sample_df = download_df.drop_duplicates(
        subset=["case_id"], keep="first"
    )[["case_id", "Sample_ID"]]
    clinical_df = pd.read_csv(metadata_clinical, sep="\t")

    annotation = (
        clinical_df.merge(case_sample_df, on="case_id", how="left")
        # Drop duplicates, replace '-- by NaN and drop empty entries
        .drop_duplicates(subset=["Sample_ID"], keep="first")
        .replace("'--", pd.NA)
        .dropna(axis=1, how="all")
    )
    cols = ["Sample_ID"] + [c for c in annotation.columns if c != "Sample_ID"]
    annotation = annotation[cols]

    download_csv_path = samples_dir / "manifest.csv"
    download_df.to_csv(download_csv_path, index=False)

    annotation_name = meta or "annotation"
    annotation_csv_path = samples_dir / f"{annotation_name}.csv"
    annotation.to_csv(annotation_csv_path, index=False)


def download_tcga_idat(
    save_dir: Path,
    metadata_cart: Path,
    show_progress: bool = True,
    subdir: Optional[str] = None,
) -> None:
    """Download missing TCGA IDAT files listed in the manifest.

    This function expects a `manifest.csv` file (generated by
    `make_tcga_metadata`) to be located in the dataset directory (e.g.,
    `<save_dir>/<subdir>/manifest.csv`). The manifest should list file IDs and
    filenames required for download.

    Args:
        save_dir (Path): Directory to store idat files.
        metadata_cart (Path): Path to CSV manifest listing 'id' and 'filename'.
        show_progress (bool): Whether to show download progress.
        subdir (Optional[str]): Optional subdirectory name under `save_dir` for
            the dataset folder. Defaults to "TCGA_<hash>" if None.
    """
    subdir = subdir or _get_tcga_series(metadata_cart)
    samples_dir = save_dir / subdir
    idat_dir = samples_dir / "idat"
    idat_dir.mkdir(parents=True, exist_ok=True)

    # Read manifest
    manifest_csv_path = samples_dir / "manifest.csv"
    if not manifest_csv_path.exists():
        raise FileNotFoundError(
            f"Manifest file {manifest_csv_path} not found. Run "
            "`make_tcga_metadata` first."
        )
    manifest = pd.read_csv(manifest_csv_path, sep=None, engine="python")

    # Determine which files are missing
    logger.info("Starting TCGA IDAT download to: %s", idat_dir)

    file_paths = idat_dir / manifest["filename"]
    pending_mask = ~file_paths.map(lambda p: p.exists())
    pending = manifest[pending_mask]

    if pending.empty:
        logger.info("All files already downloaded.")
        return

    # Prepare download URLs and local paths
    urls = [TCGA_URL.format(file_id=id_) for id_ in pending["id"]]
    paths = [idat_dir / fname for fname in pending["filename"]]

    # Download
    download_files(urls, paths, show_progress=show_progress)

    # final remaining
    file_paths = idat_dir / manifest["filename"]
    remaining_mask = ~file_paths.map(lambda p: p.exists())
    remaining = list(manifest[remaining_mask]["filename"])
    if remaining:
        logger.warning(
            "Reached maximal download attempts. %d files remain:\n%s",
            len(remaining),
            "\n".join(remaining),
        )
    else:
        logger.info(
            "Successfully downloaded all %d files to %s",
            len(file_paths),
            idat_dir,
        )


def make_dataset(
    dataset: Union[dict[str, Union[str, list[str]]], Iterable[str], str],
) -> list[dict[str, Union[str, list[str]]]]:
    """Normalize dataset input into a list of standardized dictionaries.

    Accepts E-MTAB*, GSE*, or GSM* identifiers.
    Groups all GSMs (including those in dicts) into one GEO dataset
    with series='GSE_MIXED'.

    Examples:
        >>> make_dataset("GSE1234")
        [{'source': 'geo', 'series': 'GSE1234', 'samples': 'all'}]

        >>> make_dataset(["E-MTAB-5678", "GSM1", "GSM2"])
        [
            {'source': 'ae', 'series': 'E-MTAB-5678', 'samples': 'all'},
            {'source': 'geo', 'series': 'GSE_MIXED', 'samples': ['GSM1',
            'GSM2']}
        ]
    """
    if isinstance(dataset, (dict, str)):
        items = [dataset]
    elif isinstance(dataset, Iterable):
        items = list(dataset)
    else:
        raise TypeError(f"Unsupported dataset type: {type(dataset)}")

    datasets = []
    geo_samples = []

    for item in items:
        if isinstance(item, dict):
            datasets.append(item.copy())
            continue
        if not isinstance(item, str):
            raise TypeError(
                f"Expected string items or dicts, got {type(item)}"
            )

        name = item.strip()

        if name.startswith("E-MTAB-"):
            datasets.append({"source": "ae", "series": name, "samples": "all"})
        elif name.startswith("GSE"):
            datasets.append(
                {"source": "geo", "series": name, "samples": "all"}
            )
        elif name.startswith("GSM"):
            geo_samples.append(name)
        else:
            raise ValueError(
                f"Unrecognized dataset prefix '{name}'. Must start with "
                "'E-MTAB-', 'GSE', or 'GSM'."
            )

    # Group all GSMs into a single dataset
    if geo_samples:
        datasets.insert(
            0,
            {"source": "geo", "series": "GSE_MIXED", "samples": geo_samples},
        )

    return datasets


def _download_single_dataset(
    dataset: dict[str, Any],
    save_dir: Union[str, Path],
    idat: bool = True,
    metadata: bool = True,
) -> None:
    """Helper: download IDAT/metadata for a single normalized dataset dict."""

    def to_path(p: Union[str, Path]) -> Path:
        return Path(p).expanduser()

    save_dir = to_path(save_dir)
    series_id = dataset.get("series")
    source = dataset["source"]
    samples = dataset.get("samples")
    subdir = dataset.get("subdir")
    meta = dataset.get("meta")
    if source == "ae":
        assert isinstance(series_id, str)
        if metadata:
            download_arrayexpress_metadata(
                series_id=series_id,
                save_dir=save_dir,
                samples=samples,
                subdir=subdir,
                meta=meta,
            )
        if idat:
            download_arrayexpress_idat(
                series_id=series_id,
                save_dir=save_dir,
                samples=samples,
                subdir=subdir,
            )
    elif source == "geo":
        assert isinstance(series_id, str)
        if metadata:
            if series_id == "GSE_MIXED":
                logger.info(
                    "For mixed GEO files, annotation cannot be downloaded"
                )
            else:
                download_geo_metadata(
                    series_id=series_id,
                    save_dir=save_dir,
                    samples=samples,
                    subdir=subdir,
                    meta=meta,
                )
        if idat:
            download_geo_idat(
                series_id=series_id,
                save_dir=save_dir,
                samples=samples,
                subdir=subdir,
            )
    elif source == "tcga":
        metadata_cart = to_path(dataset["metadata_cart"])
        metadata_clinical = to_path(dataset["metadata_clinical"])
        make_tcga_metadata(
            save_dir=save_dir,
            metadata_cart=metadata_cart,
            metadata_clinical=metadata_clinical,
            subdir=subdir,
            meta=meta,
        )
        if idat:
            download_tcga_idat(
                save_dir=save_dir,
                metadata_cart=metadata_cart,
                subdir=subdir,
            )
    else:
        raise ValueError(
            f"Invalid source: '{source}'. Expected 'ae', 'geo', or 'tcga'."
        )


def download_idats(
    dataset: Union[dict[str, Union[str, list[str]]], Iterable[str], str],
    save_dir: Union[str, Path],
    idat: bool = True,
    metadata: bool = True,
) -> None:
    """Download IDAT files and/or metadata from GEO, ArrayExpress, and TCGA.

    This function accepts single datasets or lists of datasets, with flexible
    formats:

    1. **Strings representing series IDs:**
        - GEO: `"GSE1234"`
        - ArrayExpress: `"E-MTAB-1234"`
    2. **Strings representing individual GEO samples:** `"GSM12345"`
    3. **Dictionaries describing a dataset**, which allow more control and
        optional overrides (including folder and annotation names):

       GEO / ArrayExpress dicts may include:
         - source: "geo" or "ae" (required)
         - series: series ID, e.g., "GSE1234" (required)
         - samples: "all" or list of sample IDs (optional, default "all")
         - subdir: output folder under save_dir (optional, default <series>)
         - meta: annotation/metadata filename (optional, default "annotation")

       TCGA dicts may include:
         - source: "tcga" (required)
         - metadata_cart: path to GDC metadata JSON (required)
         - metadata_clinical: path to clinical TSV (required)
         - subdir: output folder under save_dir (optional, default
           "TCGA_<hash>")
         - meta: annotation/metadata filename (optional, default "annotation")

    Notes:
        - All individual GEO sample IDs (`GSM*`) across strings or dicts are
          automatically grouped into a single GEO dataset with series
          `"GSE_MIXED"`.
        - Optional `subdir` and `meta` parameters allow the user to control the
          folder and annotation filename for each dataset.

    Args:
        dataset: Dataset(s) to download. Can be:
            - A single string (series or sample)
            - A dict describing a dataset
            - A list of strings and/or dicts
        save_dir (str or Path): Directory where downloaded files and metadata
            will be saved.
        idat (bool): If True, download IDAT files (default: True).
        metadata (bool): If True, download or generate metadata/annotation
            files (default: True).

    Examples:
        # Download a single GEO series
        >>> download_idats("GSE1234", "~/Downloads/geo_data")

        # Download a single TCGA dataset with custom folder and annotation
        # names
        >>> download_idats({
        ...     "source": "tcga",
        ...     "metadata_cart": "cart.json",
        ...     "metadata_clinical": "clinical.tsv",
        ...     "subdir": "TCGA_NSCLC",
        ...     "meta": "tcga_annotation"
        ... }, "./tcga")

        # Download mixed datasets: AE, GEO series, individual GSM samples, and
        # TCGA
        >>> download_idats([
        ...     "E-MTAB-8542",
        ...     "GSE147391",
        ...     "GSM4180453",
        ...     {"source": "tcga",
        ...      "metadata_cart": "cart.json",
        ...      "metadata_clinical": "clinical.tsv"},
        ...     "GSM4180454"
        ... ], "~/Downloads/mixed_data")
    """
    save_dir = Path(save_dir).expanduser()
    dataset_list: list[dict[str, Union[str, list[str]]]] = make_dataset(
        dataset
    )

    for ds in dataset_list:
        _download_single_dataset(
            dataset=ds, save_dir=save_dir, idat=idat, metadata=metadata
        )


def unzip_and_remove_gz_files(
    directory: Path,
    use_sentrix_id: bool = False,
) -> None:
    """Function to unzip .gz files and remove the original .gz files."""
    for gz in directory.glob("*.gz"):
        out = gz.with_suffix("")
        if use_sentrix_id:
            sentrix_name = out.name.split("_", 1)[1]
            out = out.with_name(sentrix_name)
        with gzip.open(gz, "rb") as fi, open(out, "wb") as fo:
            shutil.copyfileobj(fi, fo)
        gz.unlink()


def setup_tutorial_files(
    analysis_dir: Union[str, Path],
    reference_dir: Union[str, Path],
) -> None:
    """Prepare the directory structure and files for the tutorial.

    This function sets up the necessary directory structure, processes the
    tutorial data, and downloads required IDAT files for both analysis and
    reference.

    Args:
        analysis_dir (str or Path): Path to the directory for storing analysis
            files.
        reference_dir (str or Path): Path to the directory for storing
            reference files.
    """
    tutorial_csv_path = get_resource_path("mepylome", "data/tutorial.csv.gz")

    analysis_dir = Path(analysis_dir)
    reference_dir = Path(reference_dir)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    reference_dir.mkdir(parents=True, exist_ok=True)

    tutorial_df = pd.read_csv(tutorial_csv_path)
    tutorial_df.drop(columns=["Geo_File_ID"]).to_csv(
        analysis_dir / "annotation.csv", index=False
    )

    def _missing_files(dir_path: Path, geo_ids: Iterable[str]) -> list:
        return [
            gid
            for gid in geo_ids
            if not (dir_path / f"{gid.split('_', 1)[1]}_Grn.idat").exists()
            or not (dir_path / f"{gid.split('_', 1)[1]}_Red.idat").exists()
        ]

    def _fetch_and_place(missing_ids: Iterable, target_dir: Path) -> None:
        tmp_idat_dir = MEPYLOME_TMP_DIR / "tutorial"
        if not missing_ids:
            return
        download_idats(
            dataset=missing_ids,
            save_dir=tmp_idat_dir,
            idat=True,
            metadata=False,
        )
        matches = [
            f
            for f in tmp_idat_dir.rglob("*idat*")
            if any(mid in f.name for mid in missing_ids)
        ]
        for f in matches:
            shutil.copy2(f, target_dir / f.name)
        unzip_and_remove_gz_files(target_dir, use_sentrix_id=True)

    missing_analysis = _missing_files(analysis_dir, tutorial_df["Geo_File_ID"])
    _fetch_and_place(missing_analysis, analysis_dir)

    control = "Control (muscle tissue)"
    is_control = tutorial_df["Diagnosis"] == control
    missing_reference = _missing_files(
        reference_dir, tutorial_df[is_control]["Geo_File_ID"]
    )
    _fetch_and_place(missing_reference, reference_dir)
