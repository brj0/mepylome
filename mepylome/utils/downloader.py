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

    # Download a whole TCGA project (IDATs + clinical metadata from GDC)
    download_idats("TCGA-LUAD", save_dir="~/mepylome/data")

    # Download only a subset of cases from a TCGA project
    download_idats(
        dataset={
            "source": "tcga",
            "project": "TCGA-LUAD",
            "samples": ["TCGA-05-4244", "TCGA-05-4245"],
        },
        save_dir="~/mepylome/data",
        idat=True,
        metadata=True,
    )

    # TCGA dataset from a manually pre-downloaded GDC cart +
    # clinical TSV
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
import json
import logging
import re
import shutil
import tarfile
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

from mepylome.utils.files import (
    download_file,
    download_files,
    get_resource_path,
)
from mepylome.utils.varia import MEPYLOME_TMP_DIR

logger = logging.getLogger(__name__)


GEO_RAW_IDAT_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/{geo_group}/{acc}/suppl/"
    "{acc}_RAW.tar"
)
GEO_SINGLE_IDAT_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/samples/{geo_group}/{acc}/suppl/"
    "{filename}"
)
GEO_MINIML_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/{geo_group}/{acc}/miniml/"
    "{acc}_family.xml.tgz"
)

ARRAY_EXPRESS_URL = (
    "https://ftp.ebi.ac.uk/biostudies/fire/E-MTAB-/{ae_group}/{acc}/Files/"
)

TCGA_DATA_URL = "https://api.gdc.cancer.gov/data/{file_id}"
TCGA_FILES_URL = "https://api.gdc.cancer.gov/files"
TCGA_CASES_URL = "https://api.gdc.cancer.gov/cases"
TCGA_CLINICAL_FIELDS = [
    "case_id",
    "submitter_id",
    "project.project_id",
    "demographic.gender",
    "demographic.race",
    "demographic.ethnicity",
    "demographic.vital_status",
    "demographic.age_at_index",
    "demographic.days_to_death",
    "diagnoses.primary_diagnosis",
    "diagnoses.classification_of_tumor",
    "diagnoses.morphology",
    "diagnoses.tissue_or_organ_of_origin",
    "diagnoses.site_of_resection_or_biopsy",
    "diagnoses.tumor_grade",
    "diagnoses.ajcc_pathologic_stage",
    "diagnoses.ajcc_pathologic_t",
    "diagnoses.ajcc_pathologic_n",
    "diagnoses.ajcc_pathologic_m",
    "diagnoses.residual_disease",
    "diagnoses.prior_malignancy",
    "diagnoses.days_to_last_follow_up",
    "diagnoses.days_to_recurrence",
    "diagnoses.progression_or_recurrence",
    "diagnoses.last_known_disease_status",
]


# -------------------------------------
# GEO
# -------------------------------------


def _geo_group(geo_id: str) -> str:
    """Compute the GEO series/sample group folder used on the FTP server.

    Example:
        >>> _geo_group('GSE12345')
        'GSE12nnn'
        >>> _geo_group('GSM4180454')
        'GSM4180nnn'
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


def _first_attr_value(attrib: dict[str, Any] | None) -> str:
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
    samples: Iterable[str] | None = None,
    meta: str | None = None,
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
    samples: Iterable[str] | None = None,
    subdir: str | None = None,
    meta: str | None = None,
) -> None:
    """Download and extract the MINiML (family XML) for a GEO series.

    Args:
        series_id: The GEO accession ID of the dataset to download (e.g.,
            "GSE1234").

        save_dir: Directory path where the metadata will be saved.

        show_progress: If True, displays logging messages and progress bar
            during download.

        samples: Optional iterable of Sample_ID bases (e.g.
            "GSM4429896_201503470062_R02C01"). If provided, restrict the CSV to
            those Sample_IDs only.

        subdir: Optional subdirectory name under `save_dir` for the dataset
            folder. Defaults to "series_id" if None.

        meta: Optional base name for the output annotation file (without
            extension). Defaults to "annotation" if None.
    """
    subdir = subdir or series_id
    samples_dir = save_dir / subdir
    miniml_path = samples_dir / f"{series_id}.xml"
    miniml_tar_path = samples_dir / f"{series_id}_family.xml.tgz"
    samples_dir.mkdir(parents=True, exist_ok=True)

    # Download the miniml tarball
    geo_group = _geo_group(series_id)
    # BUG: If user inputs GSE1234/ instead of GSE1234 error
    miniml_tar_url = GEO_MINIML_URL.format(geo_group=geo_group, acc=series_id)
    download_file(miniml_tar_url, miniml_tar_path, show_progress=show_progress)

    # Extract the XML inside the tarball.
    try:
        if not miniml_path.exists():
            with tarfile.open(miniml_tar_path, "r:gz") as tar:
                member_name = miniml_tar_path.stem
                tar.extract(
                    member=member_name, path=samples_dir, filter="data"
                )
                miniml_tar_path.with_suffix("").rename(miniml_path)
    except Exception as exc:
        logger.debug("Could not unzip %s: %s", miniml_tar_path, exc)

    parse_miniml_to_df(miniml_path, series_id, samples, meta)


def download_geo_idat_all_files(
    series_id: str,
    save_dir: Path,
    show_progress: bool = True,
    subdir: str | None = None,
) -> None:
    """Download and extract the RAW IDAT archive for a GEO series.

    Args:
        series_id: The GEO accession ID of the dataset to download (e.g.,
            "GSE1234").

        save_dir: Directory path where the metadata will be saved.

        show_progress: If True, displays logging messages and progress bar
            during download.

        subdir: Optional subdirectory name under `save_dir` for the dataset
            folder. Defaults to "series_id" if None.
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

    # Download the RAW tarball (via GEO's FTP mirror, not the web/CGI
    # endpoint, which is gated behind a reCAPTCHA challenge for non-browser
    # clients).
    geo_group = _geo_group(series_id)
    tar_idat_url = GEO_RAW_IDAT_URL.format(geo_group=geo_group, acc=series_id)
    tar_idat_path = samples_dir / f"{series_id}_RAW.tar"
    download_file(tar_idat_url, tar_idat_path, show_progress=show_progress)
    idat_dir.mkdir(parents=True, exist_ok=True)

    # Extract idat files
    try:
        with tarfile.open(tar_idat_path, "r:*") as tar:
            tar.extractall(path=idat_dir, filter="data")

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
    subdir: str | None = None,
) -> None:
    """Download individual IDAT files for the provided samples.

    Args:
        series_id: GEO series accession (used to build save path).

        save_dir: base directory where series folder will be created.

        samples: iterable of sample base names like
            "GSM4180454_201904410008_R05C01".

        show_progress: If True, displays logging messages and
            progress bar during download.

        subdir: Optional subdirectory name under `save_dir` for the dataset
            folder. Defaults to "series_id" if None.
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
        geo_group = _geo_group(geo_acc)
        for color in ("Grn", "Red"):
            filename = f"{file}_{color}.idat.gz"
            url = GEO_SINGLE_IDAT_URL.format(
                geo_group=geo_group, acc=geo_acc, filename=filename
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
    samples: Iterable[str] | None = None,
    subdir: str | None = None,
) -> None:
    """Downloads IDAT files from geo.

    Either downloads the complete RAW tarball if samples is None or 'all' or
    per-sample download.

    Args:
        series_id: The GEO accession ID of the dataset to download (e.g.,
            "GSE1234").

        save_dir: Directory path where the metadata will be saved.

        show_progress: If True, displays logging messages and progress bar
            during download.

        samples: Optional iterable of Sample_ID bases (e.g.
            "GSM4429896_201503470062_R02C01"). If provided, restricts download
            to those Sample_IDs only (downloads per-sample Grn/Red idat .gz
            files).

        subdir: Optional subdirectory name under `save_dir` for the dataset
            folder. Defaults to "series_id" if None.
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


# -------------------------------------
# ArrayExpress
# -------------------------------------


def download_arrayexpress_metadata(
    series_id: str,
    save_dir: Path,
    samples: Iterable[str] | None = None,
    show_progress: bool = True,
    subdir: str | None = None,
    meta: str | None = None,
) -> None:
    """Download the SDRF metadata file and save as simplified CSV.

    Args:
        series_id: The ArrayExpress accession ID of the dataset to download
            (e.g., "E-MTAB-1234").

        save_dir: Directory path where the metadata will be saved.

        samples: Optional iterable of Sample_ID bases (e.g.
            "201503470062_R02C01"). If provided, restrict the CSV to those
            Sample_IDs only.

        show_progress: If True, displays logging messages and progress bar
            during download.

        subdir: Optional subdirectory name under `save_dir` for the dataset
            folder. Defaults to "series_id" if None.

        meta: Optional base name for the output annotation file (without
            extension). Defaults to "annotation" if None.
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
    samples: Iterable[str] | None = None,
    show_progress: bool = True,
    subdir: str | None = None,
) -> None:
    """Download all IDAT files for a given ArrayExpress ID.

    Args:
        series_id: The ArrayExpress accession ID of the dataset to download
            (e.g., "E-MTAB-1234").

        save_dir: Directory path where the metadata will be saved.

        samples: Optional iterable of Sample_ID bases (e.g.
            "201503470062_R02C01"). If provided, restricts download to those
            Sample_IDs only.

        show_progress: If True, displays logging messages and progress bar
            during download.

        subdir: Optional subdirectory name under `save_dir` for the dataset
            folder. Defaults to "series_id" if None.
    """
    import requests

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


# -------------------------------------
# TCGA
# -------------------------------------


def _gdc_post(
    url: str,
    filters: dict[str, Any],
    fields: list[str],
    expand: list[str] | None = None,
    size: int = 50000,
) -> list[dict[str, Any]]:
    """POST a query to the GDC API and return the list of result hits."""
    import requests

    payload: dict[str, Any] = {
        "filters": filters,
        "fields": ",".join(fields),
        "format": "JSON",
        "size": size,
    }
    if expand:
        payload["expand"] = ",".join(expand)
    response = requests.post(url, json=payload, timeout=60)
    response.raise_for_status()
    return response.json()["data"]["hits"]


def query_tcga_project_files(
    project_id: str,
    samples: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Query for all Illumina methylation IDAT files of a TCGA project.

    Args:
        project_id: TCGA project ID, e.g. "TCGA-LUAD".

        samples: Optional iterable of TCGA barcodes to restrict the query
            to. If None or "all", all cases with IDAT files in the project
            are returned. Accepts either case-level barcodes (e.g.
            "TCGA-05-4244", returns *all* aliquots/samples of that case) or
            sample-level barcodes (e.g. "TCGA-05-4244-01A", returns exactly
            that one aliquot) — both can be mixed in the same list. Use
            sample-level barcodes for precise, per-aliquot partial
            downloads.

    Returns:
        DataFrame with columns 'id' (GDC file_id), 'filename', 'Sample_ID'
        (Sentrix ID or GDC file UUID), 'md5sum', 'case_id', and
        'sample_submitter_id' (the TCGA sample barcode, e.g.
        "TCGA-05-4244-01A").
    """
    filters: dict[str, Any] = {
        "op": "and",
        "content": [
            {
                "op": "in",
                "content": {
                    "field": "cases.project.project_id",
                    "value": [project_id],
                },
            },
            {
                "op": "in",
                "content": {"field": "data_format", "value": ["IDAT"]},
            },
            {
                "op": "in",
                "content": {
                    "field": "experimental_strategy",
                    "value": ["Methylation Array"],
                },
            },
        ],
    }
    if samples and samples != "all":
        # A given barcode may be case-level (matches cases.submitter_id) or
        # sample-level (matches cases.samples.submitter_id) — match either,
        # so callers can mix case- and sample-level barcodes freely.
        filters["content"].append(
            {
                "op": "or",
                "content": [
                    {
                        "op": "in",
                        "content": {
                            "field": "cases.submitter_id",
                            "value": list(samples),
                        },
                    },
                    {
                        "op": "in",
                        "content": {
                            "field": "cases.samples.submitter_id",
                            "value": list(samples),
                        },
                    },
                ],
            }
        )

    hits = _gdc_post(
        TCGA_FILES_URL,
        filters=filters,
        fields=[
            "file_id",
            "file_name",
            "md5sum",
            "cases.case_id",
            "cases.samples.submitter_id",
        ],
        expand=["cases", "cases.samples"],
    )

    n_suffix = len("_Grn.idat")
    rows = []
    for hit in hits:
        case = (hit.get("cases") or [{}])[0]
        sample = (case.get("samples") or [{}])[0]
        rows.append(
            {
                "id": hit.get("file_id"),
                "filename": hit.get("file_name", ""),
                "Sample_ID": hit.get("file_name", "")[:-n_suffix],
                "md5sum": hit.get("md5sum"),
                "case_id": case.get("case_id", ""),
                "sample_submitter_id": sample.get("submitter_id", ""),
            }
        )
    if not rows:
        raise ValueError(
            f"No methylation IDAT files found on GDC for project "
            f"'{project_id}' (samples={samples!r})."
        )
    return pd.DataFrame(rows)


def _get_nested(hit: dict, dotted_field: str) -> Any:
    """Resolve a dotted GDC field path against one case hit."""
    value: Any = hit
    for part in dotted_field.split("."):
        if isinstance(value, list):
            value = value[0] if value else {}
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def query_tcga_clinical(project_id: str) -> pd.DataFrame:
    """Query the GDC API for clinical metadata of a TCGA project.

    Returns one row per case with the most important fields for methylation
    research (tumor type, age, sex, tumor location, survival, staging), as
    defined in `TCGA_CLINICAL_FIELDS`.
    """
    filters = {
        "op": "in",
        "content": {"field": "project.project_id", "value": [project_id]},
    }
    expand = sorted(
        {field.split(".")[0] for field in TCGA_CLINICAL_FIELDS if "." in field}
    )
    hits = _gdc_post(
        TCGA_CASES_URL,
        filters=filters,
        fields=TCGA_CLINICAL_FIELDS,
        expand=expand,
    )
    rows = [
        {
            field.split(".")[-1]: _get_nested(hit, field)
            for field in TCGA_CLINICAL_FIELDS
        }
        for hit in hits
    ]
    return pd.DataFrame(rows)


def _extract_case_file_df(json_path: Path) -> pd.DataFrame:
    """Extract a file/case mapping from a legacy GDC metadata.cart JSON."""
    with json_path.open(encoding="utf-8") as f:
        data = json.load(f)
    rows = []
    n_suffix = len("_Grn.idat")
    for item in data:
        case_id = item.get("associated_entities", [{}])[0].get("case_id", "")
        rows.append(
            {
                "id": item.get("file_id"),
                "filename": item.get("file_name", ""),
                "Sample_ID": item.get("file_name", "")[:-n_suffix],
                "md5sum": item.get("md5sum"),
                "case_id": case_id,
            }
        )
    return pd.DataFrame(rows)


def make_tcga_metadata(
    save_dir: Path,
    project: str | None = None,
    samples: Iterable[str] | None = None,
    metadata_cart: Path | None = None,
    metadata_clinical: Path | None = None,
    subdir: str | None = None,
    meta: str | None = None,
    include_clinical: bool = True,
) -> None:
    """Build the file manifest and clinical annotation and save both to disk.

    Two modes are supported:

    1. **Project mode** (recommended): pass `project` (e.g. "TCGA-LUAD").
       The list of IDAT files and the clinical metadata are both fetched
       live from the GDC API. Use `samples` to restrict to specific cases
       (partial project download).

    2. **Legacy mode**: pass a pre-downloaded `metadata_cart` (GDC cart
       JSON) and `metadata_clinical` (clinical TSV) instead.

    Args:
        save_dir: directory where output CSVs will be written.

        project: TCGA project ID, e.g. "TCGA-LUAD". Triggers project mode.

        samples: Optional iterable of case submitter IDs to restrict a
            project-mode download to (partial download).

        metadata_cart: (legacy) path to metadata.cart JSON from GDC.

        metadata_clinical: (legacy) path to clinical TSV (tab-separated).

        subdir: Optional subdirectory name under `save_dir` for the dataset
            folder. Defaults to `project` (or "TCGA" in legacy mode).

        meta: Optional base name for the output annotation file (without
            extension). Defaults to "annotation" if None.

        include_clinical: If True, also fetch/write the clinical annotation
            CSV. The file manifest (needed for IDAT download) is always
            written.
    """
    save_dir = Path(save_dir).expanduser()
    subdir = subdir or project or "TCGA"
    samples_dir = save_dir / subdir
    samples_dir.mkdir(parents=True, exist_ok=True)

    if project:
        download_df = query_tcga_project_files(project, samples=samples)
    elif metadata_cart:
        download_df = _extract_case_file_df(Path(metadata_cart).expanduser())
    else:
        raise ValueError(
            "Either 'project' (e.g. 'TCGA-LUAD') or a legacy "
            "'metadata_cart' must be provided."
        )

    download_csv_path = samples_dir / "manifest.csv"
    download_df.to_csv(download_csv_path, index=False)

    if not include_clinical:
        return

    if project:
        clinical_df = query_tcga_clinical(project)
    elif metadata_clinical:
        clinical_df = pd.read_csv(
            Path(metadata_clinical).expanduser(), sep="\t"
        )
        # TCGA changed case_id to cases.case_id
        if "case_id" not in clinical_df.columns:
            if "cases.case_id" in clinical_df.columns:
                clinical_df = clinical_df.rename(
                    columns={"cases.case_id": "case_id"}
                )
            else:
                raise KeyError(
                    "Neither 'case_id' nor 'cases.case_id' found in "
                    "clinical TSV."
                )
    else:
        raise ValueError(
            "'metadata_clinical' is required in legacy mode when "
            "include_clinical=True."
        )

    # Deduplicate to one row per aliquot (Grn/Red pair -> one Sample_ID).
    id_cols = ["case_id", "Sample_ID"]
    if "sample_submitter_id" in download_df.columns:
        id_cols.append("sample_submitter_id")
    case_sample_df = download_df.drop_duplicates(
        subset=["Sample_ID"], keep="first"
    )[id_cols]

    annotation = (
        case_sample_df.merge(clinical_df, on="case_id", how="left")
        # Drop duplicates, replace '-- by NaN and drop empty entries
        .drop_duplicates(subset=["Sample_ID"], keep="first")
        .replace("'--", pd.NA)
        .dropna(axis=1, how="all")
    )
    lead_cols = [
        c
        for c in ("Sample_ID", "sample_submitter_id")
        if c in annotation.columns
    ]
    annotation = annotation[
        lead_cols + [c for c in annotation.columns if c not in lead_cols]
    ]

    annotation_name = meta or "annotation"
    annotation_csv_path = samples_dir / f"{annotation_name}.csv"
    annotation.to_csv(annotation_csv_path, index=False)


def download_tcga_idat(
    save_dir: Path,
    subdir: str,
    show_progress: bool = True,
) -> None:
    """Download missing TCGA IDAT files listed in the manifest.

    This function expects a `manifest.csv` file (generated by
    `make_tcga_metadata`) to be located in the dataset directory (e.g.,
    `<save_dir>/<subdir>/manifest.csv`). The manifest should list file IDs and
    filenames required for download.

    Args:
        save_dir: Directory to store idat files.

        subdir: Subdirectory name under `save_dir` for the dataset folder
            (must match the one used in `make_tcga_metadata`).

        show_progress: Whether to show download progress.
    """
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
    urls = [TCGA_DATA_URL.format(file_id=id_) for id_ in pending["id"]]
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


# -------------------------------------
# Assembly
# -------------------------------------


def make_dataset(
    dataset: dict[str, str | list[str]] | Iterable[str] | str,
) -> list[dict[str, str | list[str]]]:
    """Normalize dataset input into a list of standardized dictionaries.

    Accepts E-MTAB*, GSE*, or GSM* identifiers.
    Groups all GSMs (including those in dicts) into one GEO dataset
    with series='GEO'.

    Examples:
        >>> make_dataset("GSE1234")
        [{'source': 'geo', 'series': 'GSE1234', 'samples': 'all'}]

        >>> make_dataset(["E-MTAB-5678", "GSM1", "GSM2"])
        [
            {'source': 'ae', 'series': 'E-MTAB-5678', 'samples': 'all'},
            {'source': 'geo', 'series': 'GEO', 'samples': ['GSM1',
            'GSM2']}
        ]
    """
    if isinstance(dataset, dict | str):
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
        elif name.startswith("TCGA-"):
            datasets.append(
                {"source": "tcga", "project": name, "samples": "all"}
            )
        else:
            raise ValueError(
                f"Unrecognized dataset prefix '{name}'. Must start with "
                "'E-MTAB-', 'GSE', 'GSM', or 'TCGA-' (e.g. 'TCGA-LUAD')."
            )

    # Group all GSMs into a single dataset
    if geo_samples:
        datasets.insert(
            0,
            {"source": "geo", "series": "GEO", "samples": geo_samples},
        )

    return datasets


def _download_single_dataset(
    dataset: dict[str, Any],
    save_dir: str | Path,
    idat: bool = True,
    metadata: bool = True,
) -> None:
    """Helper: download IDAT/metadata for a single normalized dataset dict."""

    def to_path(p: str | Path) -> Path:
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
            if series_id == "GEO":
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
        project = dataset.get("project")
        metadata_cart = dataset.get("metadata_cart")
        metadata_clinical = dataset.get("metadata_clinical")
        if not project and not metadata_cart:
            raise ValueError(
                "TCGA dataset requires either 'project' (e.g. "
                "'TCGA-LUAD') or a legacy 'metadata_cart' "
                "(+ 'metadata_clinical')."
            )
        subdir = subdir or project or "TCGA"
        make_tcga_metadata(
            save_dir=save_dir,
            project=project,
            samples=samples,
            metadata_cart=to_path(metadata_cart) if metadata_cart else None,
            metadata_clinical=(
                to_path(metadata_clinical) if metadata_clinical else None
            ),
            subdir=subdir,
            meta=meta,
            include_clinical=metadata,
        )
        if idat:
            download_tcga_idat(save_dir=save_dir, subdir=subdir)
    else:
        raise ValueError(
            f"Invalid source: '{source}'. Expected 'ae', 'geo', or 'tcga'."
        )


def download_idats(
    dataset: dict[str, str | list[str]] | Iterable[str] | str,
    save_dir: str | Path,
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
         - project: TCGA project ID, e.g. "TCGA-LUAD" (recommended; fetches
           IDATs + clinical metadata live from the GDC API)
         - samples: "all" or list of case submitter IDs, e.g.
           "TCGA-05-4244" (optional, default "all"; enables partial
           per-case downloads of a project)
         - subdir: output folder under save_dir (optional, default
           <project>)
         - meta: annotation/metadata filename (optional, default "annotation")

       Legacy TCGA dicts (pre-downloaded GDC cart) may include instead:
         - metadata_cart: path to GDC metadata JSON (required)
         - metadata_clinical: path to clinical TSV (required)

    Notes:
        - All individual GEO sample IDs (`GSM*`) across strings or dicts are
          automatically grouped into a single GEO dataset with series
          `"GEO"`.
        - Optional `subdir` and `meta` parameters allow the user to control the
          folder and annotation filename for each dataset.

    Args:
        dataset: Dataset(s) to download. Can be:
            - A single string (series or sample)
            - A dict describing a dataset
            - A list of strings and/or dicts

        save_dir: Directory where downloaded files and metadata
            will be saved.

        idat: If True, download IDAT files.

        metadata: If True, download or generate metadata/annotation files.

    Examples:
        # Download a single GEO series
        >>> download_idats("GSE1234", "~/Downloads/geo_data")

        # Download a whole TCGA project (IDATs + clinical metadata via GDC)
        >>> download_idats("TCGA-LUAD", "~/Downloads/tcga_data")

        # Download only specific cases from a TCGA project (partial
        # download), with a custom folder and annotation name
        >>> download_idats({
        ...     "source": "tcga",
        ...     "project": "TCGA-LUAD",
        ...     "samples": ["TCGA-05-4384-01A", "TCGA-38-4631-01A"],
        ...     "subdir": "TCGA_NSCLC",
        ...     "meta": "tcga_annotation"
        ... }, "./tcga")

        # Legacy: TCGA dataset from a manually pre-downloaded GDC cart +
        # clinical TSV
        >>> download_idats({
        ...     "source": "tcga",
        ...     "metadata_cart": "cart.json",
        ...     "metadata_clinical": "clinical.tsv",
        ...     "subdir": "TCGA_NSCLC",
        ...     "meta": "tcga_annotation"
        ... }, "./tcga")

        # Download mixed datasets: AE, GEO series, individual GSM samples,
        # and a TCGA project
        >>> download_idats([
        ...     "E-MTAB-8542",
        ...     "GSE147391",
        ...     "GSM4180453",
        ...     "TCGA-LUAD",
        ...     "GSM4180454"
        ... ], "~/Downloads/mixed_data")

    """
    save_dir = Path(save_dir).expanduser()
    dataset_list: list[dict[str, str | list[str]]] = make_dataset(dataset)

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
    analysis_dir: str | Path,
    reference_dir: str | Path,
) -> None:
    """Prepare the directory structure and files for the tutorial.

    This function sets up the necessary directory structure, processes the
    tutorial data, and downloads required IDAT files for both analysis and
    reference.

    Args:
        analysis_dir: Path to the directory for storing analysis files.

        reference_dir: Path to the directory for storing reference files.
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
    logger.info("Missing %d tutorial analysis files", len(missing_analysis))
    _fetch_and_place(missing_analysis, analysis_dir)

    control = "Control (muscle tissue)"
    is_control = tutorial_df["Diagnosis"] == control
    missing_reference = _missing_files(
        reference_dir, tutorial_df[is_control]["Geo_File_ID"]
    )
    logger.info("Missing %d tutorial reference files", len(missing_reference))
    _fetch_and_place(missing_reference, reference_dir)
