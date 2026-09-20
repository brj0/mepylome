"""Tests for the downloader utility module inside mepylome."""

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from mepylome.utils.downloader import (
    TCGA_CASES_URL,
    TCGA_DATA_URL,
    TCGA_FILES_URL,
    _download_single_dataset,
    _first_attr_value,
    _geo_group,
    _get_val,
    _strip_ns,
    _text_of,
    _unique_add,
    download_arrayexpress_idat,
    download_arrayexpress_metadata,
    download_geo_idat,
    download_geo_idat_all_files,
    download_geo_idat_single_files,
    download_geo_metadata,
    download_idats,
    download_tcga_idat,
    list_target_methylation_projects,
    list_tcga_methylation_projects,
    make_dataset,
    make_tcga_metadata,
    parse_miniml_to_df,
    query_tcga_project_files,
)

# =============================================================================
# 1. Internal / Helper Function Tests
# =============================================================================


def test_geo_group() -> None:
    assert _geo_group("GSE12345") == "GSE12nnn"
    assert _geo_group("GSE123") == "GSEnnn"
    with pytest.raises(ValueError, match="geo_id seems too short"):
        _geo_group("GS")


def test_strip_ns() -> None:
    xml_data = "<root xmlns='http://test.com'><child>text</child></root>"
    root = ET.fromstring(xml_data)
    assert "}" in root.tag
    _strip_ns(root)
    assert root.tag == "root"
    assert root.find("child") is not None


def test_text_of() -> None:
    el = ET.Element("test")
    el.text = "  hello  "
    assert _text_of(el) == "hello"
    el.text = None
    assert _text_of(el) == ""


def test_first_attr_value() -> None:
    assert _first_attr_value({"a": " 1 ", "b": "2"}) == "1;2"
    assert _first_attr_value(None) == ""


def test_get_val() -> None:
    el = ET.Element("test")
    el.text = "node_text"
    assert _get_val(el) == "node_text"

    el_attr = ET.Element("test", attrib={"tag": "attr_text"})
    assert _get_val(el_attr) == "attr_text"


def test_unique_add() -> None:
    d: dict = {}
    _unique_add("key", "val1", d)
    assert d["key"] == "val1"
    _unique_add("key", "val2", d)
    assert d["key_1"] == "val2"
    _unique_add("key", "val3", d)
    assert d["key_2"] == "val3"


# =============================================================================
# 2. Dataset Normalization Tests (make_dataset)
# =============================================================================


def test_make_dataset_single_string() -> None:
    res = make_dataset("GSE1234")
    assert res == [{"source": "geo", "series": "GSE1234", "samples": "all"}]


def test_make_dataset_mixed_iterable() -> None:
    dataset_input = [
        "E-MTAB-1234",
        "GSM111",
        "GSM222",
        {"source": "tcga", "metadata_cart": "c.json"},
    ]
    res = make_dataset(dataset_input)  # type: ignore[arg-type]

    assert res[0] == {
        "source": "geo",
        "series": "GEO",
        "samples": ["GSM111", "GSM222"],
    }
    assert res[1] == {
        "source": "ae",
        "series": "E-MTAB-1234",
        "samples": "all",
    }
    assert res[2] == {"source": "tcga", "metadata_cart": "c.json"}


def test_make_dataset_target_string() -> None:
    res = make_dataset("TARGET-AML")
    assert res == [
        {"source": "target", "project": "TARGET-AML", "samples": "all"}
    ]


def test_make_dataset_tcga_and_target_mixed() -> None:
    res = make_dataset(["TCGA-LUAD", "TARGET-NBL", "GSE1234"])
    assert res == [
        {"source": "tcga", "project": "TCGA-LUAD", "samples": "all"},
        {"source": "target", "project": "TARGET-NBL", "samples": "all"},
        {"source": "geo", "series": "GSE1234", "samples": "all"},
    ]


def test_make_dataset_errors() -> None:
    with pytest.raises(TypeError):
        make_dataset(12345)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        make_dataset(["GSE123", 456])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="Unrecognized dataset prefix"):
        make_dataset("INVALID123")


# =============================================================================
# 3. Parsing & Metadata Generation Tests
# =============================================================================


def test_parse_miniml_to_df(tmp_path: Path) -> None:
    miniml_xml = """<?xml version="1.0" encoding="UTF-8"?>
    <MINiML xmlns="http://www.ncbi.nlm.nih.gov/geo/info/MINiML">
        <Sample iid="GSM1">
            <Status>Status_Val</Status>
            <Channel>
                <Characteristics tag="tissue">brain</Characteristics>
            </Channel>
            <Supplementary-Data>GSM1_Grn.idat.gz</Supplementary-Data>
        </Sample>
        <Sample iid="GSM2">
            <Status>Status_Val</Status>
            <Channel>
                <Characteristics tag="tissue">liver</Characteristics>
            </Channel>
            <Supplementary-Data>GSM2_Red.idat.gz</Supplementary-Data>
        </Sample>
    </MINiML>
    """
    xml_path = tmp_path / "GSE123.xml"
    xml_path.write_text(miniml_xml)
    csv_path = tmp_path / "annotation.csv"

    parse_miniml_to_df(xml_path, "GSE123", csv_path, samples="all")

    result = pd.read_csv(csv_path)
    assert len(result) == 2
    assert set(result["Sample_ID"]) == {"GSM1", "GSM2"}


def test_parse_miniml_no_samples(tmp_path: Path) -> None:
    xml_path = tmp_path / "empty.xml"
    xml_path.write_text("<MINiML></MINiML>")
    csv_path = tmp_path / "annotation.csv"
    with pytest.raises(ValueError, match="No <Sample> elements found"):
        parse_miniml_to_df(xml_path, "GSE123", csv_path)


# =============================================================================
# 4. GEO Downloading Operational Tests
# =============================================================================


@patch("mepylome.utils.downloader.download_file")
@patch("mepylome.utils.downloader.parse_miniml_to_df")
def test_download_geo_metadata(
    mock_parse: MagicMock, mock_download_file: MagicMock, tmp_path: Path
) -> None:
    def _fake_extract(member: str, path: Path, filter: str) -> None:
        (Path(path) / member).touch()

    with patch("tarfile.open") as mock_tar_open:
        mock_tar = MagicMock()
        mock_tar.extract.side_effect = _fake_extract
        mock_tar_open.return_value.__enter__.return_value = mock_tar

        download_geo_metadata("GSE12345", save_dir=tmp_path)

    mock_download_file.assert_called_once()
    mock_parse.assert_called_once_with(
        miniml_path=tmp_path / "GSE12345" / "GSE12345.xml",
        series_id="GSE12345",
        csv_path=tmp_path / "GSE12345" / "annotation.csv",
        samples=None,
    )


@patch("mepylome.utils.downloader.download_file")
@patch("tarfile.open")
def test_download_geo_idat_all_files(
    mock_tar_open: MagicMock, mock_download_file: MagicMock, tmp_path: Path
) -> None:
    series_id = "GSE12345"

    mock_tar = MagicMock()
    mock_tar_open.return_value.__enter__.return_value = mock_tar

    download_geo_idat_all_files(series_id, save_dir=tmp_path)
    mock_download_file.assert_called_once()
    part_dir = tmp_path / series_id / "idat.part"

    mock_tar.extractall.assert_called_once_with(
        path=part_dir,
        filter="data",
    )


@patch("mepylome.utils.downloader.download_files")
def test_download_geo_idat_single_files(
    mock_download_files: MagicMock, tmp_path: Path
) -> None:
    samples = ["GSM123_2019_R01C01", "GSM456_2019_R02C01"]
    download_geo_idat_single_files(
        "GSE123", save_dir=tmp_path, samples=samples
    )

    assert mock_download_files.called
    urls, paths = mock_download_files.call_args[0][:2]
    assert len(urls) == 4
    assert len(paths) == 4
    assert all(isinstance(p, Path) for p in paths)


@patch("mepylome.utils.downloader.download_geo_idat_all_files")
@patch("mepylome.utils.downloader.download_geo_idat_single_files")
def test_download_geo_idat_routing(
    mock_single: MagicMock, mock_all: MagicMock, tmp_path: Path
) -> None:
    download_geo_idat("GSE123", save_dir=tmp_path, samples="all")
    mock_all.assert_called_once()

    download_geo_idat("GSE123", save_dir=tmp_path, samples=["GSM123"])
    mock_single.assert_called_once()


# =============================================================================
# 5. ArrayExpress Downloading Operational Tests
# =============================================================================


@patch("mepylome.utils.downloader.download_file")
@patch("pandas.read_csv")
def test_download_arrayexpress_metadata(
    mock_read_csv: MagicMock, mock_download_file: MagicMock, tmp_path: Path
) -> None:
    series_id = "E-MTAB-1234"
    mock_df = pd.DataFrame(
        {"Array Data File": ["2015_R01C01_Grn.idat", "2015_R01C01_Red.idat"]}
    )
    mock_read_csv.return_value = mock_df

    download_arrayexpress_metadata(series_id, save_dir=tmp_path)
    mock_download_file.assert_called_once()
    csv_path = tmp_path / series_id / "annotation.csv"
    assert csv_path.exists()


@patch("requests.get")
@patch("mepylome.utils.downloader.download_files")
def test_download_arrayexpress_idat(
    mock_download_files: MagicMock, mock_get: MagicMock, tmp_path: Path
) -> None:
    series_id = "E-MTAB-1234"

    # Mock the JSON payload returned by BioStudies REST API
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "items": [
            {"path": "2015_R01C01_Grn.idat"},
            {"path": "2015_R01C01_Red.idat"},
        ],
        "pagination": {"offset": 0, "limit": 100, "total": 2},
    }
    mock_get.return_value = mock_response

    # Test downloading all samples
    download_arrayexpress_idat(series_id, save_dir=tmp_path, samples="all")
    assert mock_download_files.called
    urls = mock_download_files.call_args[1]["urls"]
    assert len(urls) == 2

    # Test missing sample raises ValueError
    with pytest.raises(ValueError, match="not found remotely"):
        download_arrayexpress_idat(
            series_id, save_dir=tmp_path, samples=["MissingSample"]
        )


# =============================================================================
# 6. TCGA Operational Tests
# =============================================================================


def test_make_tcga_metadata(tmp_path: Path) -> None:
    cart_json = tmp_path / "cart.json"
    clinical_tsv = tmp_path / "clinical.tsv"

    cart_data = [
        {
            "file_id": "id123",
            "file_name": "sample1_Grn.idat",
            "md5sum": "abc",
            "associated_entities": [{"case_id": "case_abc"}],
        }
    ]
    cart_json.write_text(json.dumps(cart_data))

    clinical_data = "case_id\tproject_id\ncase_abc\tTCGA-BRCA\n"
    clinical_tsv.write_text(clinical_data)

    make_tcga_metadata(
        save_dir=tmp_path,
        metadata_cart=cart_json,
        metadata_clinical=clinical_tsv,
        subdir="TCGA_TEST",
    )

    assert (tmp_path / "TCGA_TEST" / "manifest.txt").exists()
    annotation_file = tmp_path / "TCGA_TEST" / "annotation.csv"
    assert annotation_file.exists()

    df = pd.read_csv(annotation_file)
    assert df.loc[0, "Sample_ID"] == "sample1"
    assert df.loc[0, "project_id"] == "TCGA-BRCA"


@patch("mepylome.utils.downloader.download_files")
def test_download_tcga_idat(
    mock_download_files: MagicMock, tmp_path: Path
) -> None:
    subdir = "TCGA_TEST"
    samples_dir = tmp_path / subdir
    samples_dir.mkdir()

    manifest_df = pd.DataFrame(
        {"id": ["id1", "id2"], "filename": ["f1.idat", "f2.idat"]}
    )
    manifest_df.to_csv(samples_dir / "manifest.txt", index=False)

    cart_json = tmp_path / "dummy_cart.json"
    cart_json.write_text("[]")

    download_tcga_idat(save_dir=tmp_path, subdir=subdir)
    assert mock_download_files.called

    (samples_dir / "manifest.txt").unlink()
    with pytest.raises(FileNotFoundError):
        download_tcga_idat(save_dir=tmp_path, subdir=subdir)


# =============================================================================
# 7. TARGET Operational Tests (same GDC pipeline as TCGA)
# =============================================================================


def _filter_map(node: dict) -> dict:
    """Flatten a nested GDC filter into a {field: values} mapping."""
    if node["op"] in ("and", "or"):
        out: dict = {}
        for child in node["content"]:
            out.update(_filter_map(child))
        return out
    return {node["content"]["field"]: node["content"]["value"]}


def _gdc_file_hit(file_id: str, name: str, case: str, sample: str) -> dict:
    return {
        "file_id": file_id,
        "file_name": name,
        "md5sum": f"md5-{file_id}",
        "cases": [
            {"case_id": case, "samples": [{"submitter_id": sample}]},
        ],
    }


@patch("requests.post")
def test_list_methylation_projects_program(mock_post: MagicMock) -> None:
    buckets = [
        {"key": "TARGET-OS", "doc_count": 3},
        {"key": "TARGET-AML", "doc_count": 5},
        {"key": "TARGET-EMPTY", "doc_count": 0},
    ]
    mock_post.return_value.json.return_value = {
        "data": {
            "aggregations": {"cases.project.project_id": {"buckets": buckets}}
        }
    }

    assert list_target_methylation_projects() == ["TARGET-AML", "TARGET-OS"]
    fmap = _filter_map(mock_post.call_args.kwargs["json"]["filters"])
    assert fmap["cases.project.program.name"] == ["TARGET"]
    assert fmap["access"] == ["open"]

    # Default program is unchanged (TCGA).
    list_tcga_methylation_projects()
    fmap = _filter_map(mock_post.call_args.kwargs["json"]["filters"])
    assert fmap["cases.project.program.name"] == ["TCGA"]


@patch("mepylome.utils.downloader._gdc_post")
def test_query_project_files_target(mock_gdc_post: MagicMock) -> None:
    mock_gdc_post.return_value = [
        _gdc_file_hit(
            "u1", "201234567890_R01C01_Grn.idat", "c1", "TARGET-20-AAAAAA-09A"
        ),
        _gdc_file_hit(
            "u2", "201234567890_R01C01_Red.idat", "c1", "TARGET-20-AAAAAA-09A"
        ),
    ]

    df = query_tcga_project_files(
        "TARGET-AML", samples=["TARGET-20-AAAAAA", "TARGET-20-BBBBBB-14A"]
    )

    fmap = _filter_map(mock_gdc_post.call_args.kwargs["filters"])
    assert fmap["cases.project.project_id"] == ["TARGET-AML"]
    assert fmap["data_format"] == ["IDAT"]
    assert fmap["experimental_strategy"] == ["Methylation Array"]
    assert fmap["access"] == ["open"]
    # Case- and sample-level barcodes are both matched.
    wanted = ["TARGET-20-AAAAAA", "TARGET-20-BBBBBB-14A"]
    assert fmap["cases.submitter_id"] == wanted
    assert fmap["cases.samples.submitter_id"] == wanted

    assert list(df["id"]) == ["u1", "u2"]
    assert set(df["Sample_ID"]) == {"201234567890_R01C01"}
    assert set(df["sample_submitter_id"]) == {"TARGET-20-AAAAAA-09A"}


@patch("mepylome.utils.downloader.download_tcga_idat")
@patch("mepylome.utils.downloader.make_tcga_metadata")
def test_download_idats_target_routing(
    mock_meta: MagicMock, mock_idat: MagicMock, tmp_path: Path
) -> None:
    download_idats("TARGET-AML", save_dir=tmp_path)
    kwargs = mock_meta.call_args.kwargs
    assert kwargs["project"] == "TARGET-AML"
    assert kwargs["subdir"] == "TARGET-AML"
    assert kwargs["samples"] == "all"
    assert kwargs["include_clinical"] is True
    mock_idat.assert_called_once_with(save_dir=tmp_path, subdir="TARGET-AML")

    # idat=False skips the IDAT download, metadata=False skips clinical.
    mock_meta.reset_mock()
    mock_idat.reset_mock()
    download_idats("TARGET-AML", save_dir=tmp_path, idat=False)
    mock_idat.assert_not_called()
    download_idats("TARGET-AML", save_dir=tmp_path, metadata=False)
    assert mock_meta.call_args.kwargs["include_clinical"] is False
    mock_idat.assert_called_once()


@patch("mepylome.utils.downloader.download_tcga_idat")
@patch("mepylome.utils.downloader.make_tcga_metadata")
def test_download_single_dataset_target_options(
    mock_meta: MagicMock, mock_idat: MagicMock, tmp_path: Path
) -> None:
    # Partial download with custom folder and annotation name.
    _download_single_dataset(
        {
            "source": "target",
            "project": "TARGET-AML",
            "samples": ["TARGET-20-AAAAAA"],
            "subdir": "my_target",
            "meta": "my_annotation",
        },
        save_dir=tmp_path,
    )
    kwargs = mock_meta.call_args.kwargs
    assert kwargs["samples"] == ["TARGET-20-AAAAAA"]
    assert kwargs["subdir"] == "my_target"
    assert kwargs["meta"] == "my_annotation"
    mock_idat.assert_called_once_with(save_dir=tmp_path, subdir="my_target")

    # Legacy (cart) mode falls back to a program-specific folder name.
    _download_single_dataset(
        {"source": "target", "metadata_cart": "cart.json"}, save_dir=tmp_path
    )
    kwargs = mock_meta.call_args.kwargs
    assert kwargs["subdir"] == "TARGET"
    assert kwargs["metadata_cart"] == Path("cart.json")
    assert kwargs["project"] is None


def test_download_single_dataset_errors(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="TARGET dataset requires"):
        _download_single_dataset({"source": "target"}, save_dir=tmp_path)
    with pytest.raises(ValueError, match="TCGA dataset requires"):
        _download_single_dataset({"source": "tcga"}, save_dir=tmp_path)
    with pytest.raises(ValueError, match="'tcga', or 'target'"):
        _download_single_dataset({"source": "nope"}, save_dir=tmp_path)


@patch("mepylome.utils.downloader.download_files")
@patch("mepylome.utils.downloader._gdc_post")
def test_download_idats_target_end_to_end(
    mock_gdc_post: MagicMock, mock_download_files: MagicMock, tmp_path: Path
) -> None:
    """TARGET project -> manifest, annotation, IDAT downloads (mocked)."""
    sample_a, sample_b = "TARGET-20-AAAAAA-09A", "TARGET-20-BBBBBB-09A"
    file_hits = [
        _gdc_file_hit("g1", "111_R01C01_Grn.idat", "c1", sample_a),
        _gdc_file_hit("r1", "111_R01C01_Red.idat", "c1", sample_a),
        _gdc_file_hit("g2", "222_R02C01_Grn.idat", "c2", sample_b),
        _gdc_file_hit("r2", "222_R02C01_Red.idat", "c2", sample_b),
    ]
    case_hits = [
        {
            "case_id": "c1",
            "submitter_id": "TARGET-20-AAAAAA",
            "project": {"project_id": "TARGET-AML"},
            "demographic": {"gender": "female", "vital_status": "Alive"},
            "diagnoses": [
                {
                    "primary_diagnosis": "Acute myeloid leukemia, NOS",
                    "age_at_diagnosis": 3000,
                }
            ],
        },
        {
            "case_id": "c2",
            "submitter_id": "TARGET-20-BBBBBB",
            "project": {"project_id": "TARGET-AML"},
            "demographic": {"gender": "male", "vital_status": "Dead"},
            "diagnoses": [{"primary_diagnosis": "Acute myeloid leukemia"}],
        },
    ]

    def fake_gdc_post(url: str, **_: object) -> list[dict]:
        return {TCGA_FILES_URL: file_hits, TCGA_CASES_URL: case_hits}[url]

    mock_gdc_post.side_effect = fake_gdc_post

    download_idats("TARGET-AML", save_dir=tmp_path)

    dataset_dir = tmp_path / "TARGET-AML"
    manifest = pd.read_csv(dataset_dir / "manifest.txt")
    assert list(manifest["filename"]) == [
        "111_R01C01_Grn.idat",
        "111_R01C01_Red.idat",
        "222_R02C01_Grn.idat",
        "222_R02C01_Red.idat",
    ]

    # One annotation row per Grn/Red pair, with clinical data merged in.
    annotation = pd.read_csv(dataset_dir / "annotation.csv")
    assert list(annotation["Sample_ID"]) == ["111_R01C01", "222_R02C01"]
    assert list(annotation["sample_submitter_id"]) == [
        "TARGET-20-AAAAAA-09A",
        "TARGET-20-BBBBBB-09A",
    ]
    assert set(annotation["project.project_id"]) == {"TARGET-AML"}
    assert list(annotation["demographic.gender"]) == ["female", "male"]
    assert annotation.loc[0, "diagnoses.age_at_diagnosis"] == 3000

    # IDATs are fetched from the GDC data endpoint into <subdir>/idat.
    urls, paths = mock_download_files.call_args[0][:2]
    expected_ids = ["g1", "r1", "g2", "r2"]
    expected_urls = [TCGA_DATA_URL.format(file_id=i) for i in expected_ids]
    assert list(urls) == expected_urls
    assert {Path(p).parent for p in paths} == {dataset_dir / "idat"}


# =============================================================================
# 8. High-Level Hub Test (download_idats)
# =============================================================================


@patch("mepylome.utils.downloader._download_single_dataset")
def test_download_idats_entrypoint(
    mock_single_download: MagicMock, tmp_path: Path
) -> None:
    download_idats(
        ["GSE123", "E-MTAB-567"],
        save_dir=tmp_path,
        idat=True,
        metadata=False,
    )
    assert mock_single_download.call_count == 2

    first_call_ds = mock_single_download.call_args_list[0][1]["dataset"]
    assert first_call_ds["series"] == "GSE123"
    assert mock_single_download.call_args_list[0][1]["metadata"] is False
