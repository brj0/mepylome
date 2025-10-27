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

import hashlib
import json
import logging
import sys
import tarfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

import pandas as pd
import requests
from bs4 import BeautifulSoup

from mepylome.utils.files import download_file, download_files

logger = logging.getLogger("mepylome")


GEO_RAW_IDAT_URL = (
    "https://www.ncbi.nlm.nih.gov/geo/download/?acc={geo_id}&format=file"
)
GEO_SINGLE_IDAT_URL = (
    "https://www.ncbi.nlm.nih.gov/geo/download/?acc={acc}&format=file"
    "&file={filename}"
)
GEO_MINIML_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/{geo_group}/{geo_id}/miniml/"
    "{geo_id}_family.xml.tgz"
)
ARRAY_EXPRESS_URL = (
    "https://ftp.ebi.ac.uk/biostudies/fire/E-MTAB-/{ae_group}/{ae_id}/Files/"
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


def strip_ns(elem: ET.Element) -> None:
    """Remove namespace prefix in-place for an element tree."""
    for e in elem.iter():
        if "}" in e.tag:
            e.tag = e.tag.split("}", 1)[1]


def text_of(el: ET.Element) -> str:
    """Return .text stripped or empty string."""
    return (el.text or "").strip()


def first_attr_value(attrib: Optional[Dict[str, Any]]) -> str:
    """Return attribute values as string joined with ';'."""
    vals = [str(v).strip() for v in attrib.values()] if attrib else []
    return ";".join(vals)


def get_val(node: ET.Element) -> str:
    """Returns value of a xml node."""
    text = text_of(node)
    return text or first_attr_value(node.attrib)


def unique_add(key: str, value: Any, dictionary: Dict[str, Any]) -> None:
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
) -> None:
    """Parse a GEO MINiML family XML and save as spreadsheet to disk."""
    tree = ET.parse(miniml_path)
    root = tree.getroot()
    strip_ns(root)
    sample_elements = root.findall(".//Sample")
    if not sample_elements:
        raise ValueError("No <Sample> elements found in the MINiML file.")
    rows = []
    for s in sample_elements:
        row = {}
        for child in list(s):
            tag = child.tag
            if tag == "Supplementary-Data" and "Sample_ID" not in row:
                idat_filename = text_of(child).split("/")[-1]
                sample_id = (
                    idat_filename.removesuffix(".idat.gz")
                    .removesuffix("_Grn")
                    .removesuffix("_Red")
                )
                unique_add("Sample_ID", sample_id, row)
            if list(child):
                for sub in child:
                    sub_tag = sub.tag
                    if sub_tag == "Characteristics":
                        sub_tag = sub.attrib.get("tag") or sub_tag
                    unique_add(sub_tag, get_val(sub), row)
            else:
                unique_add(tag, get_val(child), row)
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

    csv_path = miniml_path.parent / f"{miniml_path.stem}.csv"
    result_df.to_csv(csv_path, index=False)


def download_geo_metadata(
    series_id: str,
    save_dir: Path,
    show_progress: bool = True,
    samples: Optional[Iterable[str]] = None,
):
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
    """
    samples_dir = save_dir / series_id
    miniml_path = samples_dir / f"{series_id}.xml"
    miniml_tar_path = samples_dir / f"{series_id}_family.xml.tgz"
    # if miniml_path.exists():
    #     logger.info(
    #         "Meta data already exists: %s. Skipping download.", miniml_path
    #     )
    #     return
    samples_dir.mkdir(parents=True, exist_ok=True)

    # Download the miniml tarball
    geo_group = _geo_group(series_id)
    miniml_tar_url = GEO_MINIML_URL.format(
        geo_group=geo_group, geo_id=series_id
    )
    download_file(miniml_tar_url, miniml_tar_path, show_progress=show_progress)

    # Extract the XML inside the tarball.
    try:
        with tarfile.open(miniml_tar_path, "r:gz") as tar:
            kwargs = {}
            if sys.version_info >= (3, 12):
                kwargs["filter"] = "data"  # only add for Python 3.12+
            tar.extract(
                member=f"{miniml_tar_path.stem}", path=samples_dir, **kwargs
            )
            miniml_tar_path.with_suffix("").rename(miniml_path)
    except Exception as exc:
        logger.debug("Could not unzip %s: %s", miniml_tar_path, exc)

    parse_miniml_to_df(miniml_path, series_id, samples)


def download_geo_idat_all_files(
    series_id: str,
    save_dir: Path,
    show_progress: bool = True,
) -> None:
    """Download and extract the RAW IDAT archive for a GEO series.

    Args:
        series_id (str): The GEO accession ID of the dataset to download
            (e.g., "GSE1234").
        save_dir (Path): Directory path where the metadata will be saved.
        show_progress (bool, optional): If True, displays logging messages and
            progress bar during download. Defaults to True.
    """
    samples_dir = save_dir / series_id
    idat_dir = samples_dir / "idat"
    if idat_dir.exists():
        logger.info(
            "IDAT directory already exists: %s. Skipping download.", idat_dir
        )
        return
    samples_dir.mkdir(parents=True, exist_ok=True)

    # Download the RAW tarball
    geo_group = _geo_group(series_id)
    tar_idat_url = GEO_RAW_IDAT_URL.format(
        geo_group=geo_group, geo_id=series_id
    )
    tar_idat_path = samples_dir / f"{series_id}_RAW.tar"
    download_file(tar_idat_url, tar_idat_path, show_progress=show_progress)
    idat_dir.mkdir(parents=True, exist_ok=True)

    # Extract idat files
    try:
        with tarfile.open(tar_idat_path, "r:*") as tar:
            kwargs = {}
            if sys.version_info >= (3, 12):
                kwargs["filter"] = "data"  # only add for Python 3.12+
            tar.extractall(path=idat_dir, **kwargs)

        # Remove unwanted GPL*csv.gz manifest files if present
        for file_path in idat_dir.rglob("GPL*csv.gz"):
            try:
                file_path.unlink()
                logger.info("Deleted metadata file: %s", file_path)
            except Exception as exc:
                logger.debug("Could not delete %s: %s", file_path, exc)
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
) -> None:
    """Download individual IDAT files for the provided samples.

    Args:
        series_id: GEO series accession (used to build save path).
        save_dir: base directory where series folder will be created.
        samples: iterable of sample base names like
            "GSM4180454_201904410008_R05C01".
        show_progress: whether to pass show_progress to download_files.
    """
    samples_dir = Path(save_dir) / series_id
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
) -> None:
    """Downloads IDAT files from geo.

    Either downloads the complete RAW tarball if samples is None or per-sample
    download.

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
    """
    if not samples or samples == "all":
        download_geo_idat_all_files(
            series_id=series_id,
            save_dir=save_dir,
            show_progress=show_progress,
        )
    else:
        download_geo_idat_single_files(
            series_id=series_id,
            save_dir=save_dir,
            samples=samples,
            show_progress=show_progress,
        )


def download_arrayexpress_metadata(
    series_id: str,
    save_dir: Path,
    samples: Optional[Iterable[str]] = None,
    show_progress: bool = True,
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
    """
    samples_dir = save_dir / series_id
    csv_path = samples_dir / f"{series_id}.csv"
    # if csv_path.exists():
    #     logger.info(
    #         "Meta data already exists: %s. Skipping download.", csv_path
    #     )
    #     return
    samples_dir.mkdir(parents=True, exist_ok=True)

    # Download SDRF file
    url = ARRAY_EXPRESS_URL.format(ae_group=series_id[-3:], ae_id=series_id)
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
    """
    samples_dir = save_dir / series_id
    idat_dir = samples_dir / "idat"
    samples_dir.mkdir(parents=True, exist_ok=True)
    idat_dir.mkdir(parents=True, exist_ok=True)

    # Fetch file listing from ArrayExpress
    url = ARRAY_EXPRESS_URL.format(ae_group=series_id[-3:], ae_id=series_id)
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    remote_idats = [
        node.get("href")
        for node in soup.find_all("a")
        if ".idat" in node.get("href", "")
    ]

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


def get_tcga_series(path: Path) -> str:
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
) -> None:
    """Build merged TCGA annotation DataFrame and saves to disk.

    Args:
        save_dir: directory where output CSV will be written (if write_csv
            True).
        metadata_cart: path to metadata.cart JSON from GDC.
        metadata_clinical: path to clinical TSV (tab-separated).
    """
    save_dir = Path(save_dir).expanduser()
    series_id = get_tcga_series(metadata_cart)
    samples_dir = save_dir / series_id
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

    download_csv_path = samples_dir / f"{series_id}_manifest.csv"
    download_df.to_csv(download_csv_path, index=False)
    annotation_csv_path = samples_dir / f"{series_id}_annotation.csv"
    annotation.to_csv(annotation_csv_path, index=False)


def download_tcga_idat(
    save_dir: Path,
    metadata_cart: Path,
    show_progress: bool = True,
    wait_seconds: int = 5,
) -> None:
    """Download missing TCGA IDAT files listed in the manifest.

    Args:
        idat_dir: Directory to store idat files.
        manifest_path: Path to CSV manifest listing 'id' and 'filename'.
        show_progress: Whether to show download progress.
        wait_seconds: Delay (in seconds) between download attempts.
    """
    series_id = get_tcga_series(metadata_cart)
    samples_dir = save_dir / series_id
    idat_dir = samples_dir / "idat"
    idat_dir.mkdir(parents=True, exist_ok=True)

    # Read manifest (auto-detect separator)
    manifest_csv_path = samples_dir / f"{series_id}_manifest.csv"
    manifest = pd.read_csv(manifest_csv_path, sep=None, engine="python")

    # Determine which files are missing
    logger.info("Starting TCGA IDAT download to: %s", idat_dir)

    file_paths = idat_dir / manifest["filename"]
    pending_mask = ~file_paths.map(lambda p: p.exists())
    pending = manifest[pending_mask]

    if pending.empty:
        logger.info("All files allready downloaded.")
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
            "Sucessfully downloaded all %d files to %s",
            len(file_paths),
            idat_dir,
        )


def make_dataset(
    dataset: Union[Dict[str, Union[str, List[str]]], Iterable[str], str],
) -> List[Dict[str, Union[str, List[str]]]]:
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
        datasets.append(
            {"source": "geo", "series": "GSE_MIXED", "samples": geo_samples}
        )

    return datasets


def _download_single_dataset(
    dataset: Dict[str, Union[str, List[str]]],
    save_dir: Union[str, Path],
    idat: bool = True,
    metadata: bool = True,
) -> None:
    """Helper: download IDAT/metadata for a single normalized dataset dict."""

    def to_path(p: Union[str, Path]):
        return Path(p).expanduser()

    save_dir = to_path(save_dir)
    source = dataset["source"]
    series_id = dataset.get("series")
    samples = dataset.get("samples")
    if source == "ae":
        if metadata:
            download_arrayexpress_metadata(
                save_dir=save_dir,
                series_id=series_id,
                samples=samples,
            )
        if idat:
            download_arrayexpress_idat(
                save_dir=save_dir,
                series_id=series_id,
                samples=samples,
            )
    elif source == "geo":
        if metadata:
            if series_id == "GSE_MIXED":
                logger.info(
                    "For mixed GEO files, annotation cannot be downloaded"
                )
            else:
                download_geo_metadata(
                    save_dir=save_dir,
                    series_id=series_id,
                    samples=samples,
                )
        if idat:
            download_geo_idat(
                save_dir=save_dir,
                series_id=series_id,
                samples=samples,
            )
    elif source == "tcga":
        metadata_cart = to_path(dataset["metadata_cart"])
        metadata_clinical = to_path(dataset["metadata_clinical"])
        if metadata:
            make_tcga_metadata(
                save_dir=save_dir,
                metadata_cart=metadata_cart,
                metadata_clinical=metadata_clinical,
            )
        if idat:
            download_tcga_idat(
                save_dir=save_dir,
                metadata_cart=metadata_cart,
            )
    else:
        raise ValueError(
            f"Invalid source: '{source}'. Expected 'ae', 'geo', or 'tcga'."
        )


def download_idats(
    dataset: Union[str, Dict, List[Union[str, Dict]]],
    save_dir: Union[str, Path],
    idat: bool = True,
    metadata: bool = True,
) -> None:
    """Download IDAT files and/or metadata for GEO, ArrayExpress, or TCGA.

    This function accepts single datasets or lists of datasets, in flexible
    formats:
    - Strings representing series IDs:
        - GEO: "GSE1234"
        - ArrayExpress: "E-MTAB-1234"
    - Strings representing individual GEO samples: "GSM12345"
    - Dictionaries for detailed datasets, including TCGA or specific subsets of
      GEO/AE samples:
        - GEO/AE dict: {"source": "geo"|"ae", "series": str, "samples":
          "all"|List[str]}
        - TCGA dict: {"source": "tcga", "metadata_cart": str,
          "metadata_clinical": str}

    All GSM sample IDs across strings or dicts are automatically grouped into a
    single GEO dataset with series='GSE_MIXED'.

    Args:
        datasets: Dataset(s) to download. Can be:
            - A single string (series or sample)
            - A dict describing a dataset
            - A list of strings and/or dicts
        save_dir: Directory where downloaded files and metadata will be saved.
        idat: If True, download IDAT files (default: True).
        metadata: If True, download or generate metadata (default: True).

    Raises:
        TypeError: If input types are invalid.
        ValueError: If a dataset has an unsupported source type.

    Examples:
        # Download a single GEO series
        >>> download_idats("GSE1234", "~/Downloads/geo_data")

        # Download a single TCGA dataset
        >>> download_idats({
        ...     "source": "tcga",
        ...     "metadata_cart": "cart.json",
        ...     "metadata_clinical": "clinical.tsv"
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
    dataset_list: List[Dict[str, Union[str, List[str]]]] = make_dataset(
        dataset
    )

    for ds in dataset_list:
        _download_single_dataset(
            dataset=ds, save_dir=save_dir, idat=idat, metadata=metadata
        )
