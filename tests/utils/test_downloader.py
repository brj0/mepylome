"""Tests for the downloader utility module inside mepylome."""

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from mepylome.utils.downloader import (
    _first_attr_value,
    _geo_group,
    _get_tcga_series,
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
    make_dataset,
    make_tcga_metadata,
    parse_miniml_to_df,
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


def test_get_tcga_series(tmp_path: Path) -> None:
    test_file = tmp_path / "test.json"
    test_file.write_bytes(b"tcga_mock_data")
    series_name = _get_tcga_series(test_file)
    assert series_name.startswith("TCGA_")
    assert len(series_name) == 21  # 'TCGA_' + 16 hex chars


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
        "series": "GSE_MIXED",
        "samples": ["GSM111", "GSM222"],
    }
    assert res[1] == {
        "source": "ae",
        "series": "E-MTAB-1234",
        "samples": "all",
    }
    assert res[2] == {"source": "tcga", "metadata_cart": "c.json"}


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

    parse_miniml_to_df(xml_path, "GSE123", samples="all")
    csv_path = tmp_path / "annotation.csv"
    assert csv_path.exists()
    df = pd.read_csv(csv_path)
    assert len(df) == 2
    assert "Sample_ID" in df.columns
    assert df.loc[0, "Sample_ID"] == "GSM1"
    assert df.loc[0, "tissue"] == "brain"

    parse_miniml_to_df(
        xml_path, "GSE123", samples=["GSM2"], meta="custom_meta"
    )
    custom_csv = tmp_path / "custom_meta.csv"
    assert custom_csv.exists()
    df_filtered = pd.read_csv(custom_csv)
    assert len(df_filtered) == 1
    assert df_filtered.loc[0, "Sample_ID"] == "GSM2"


def test_parse_miniml_no_samples(tmp_path: Path) -> None:
    xml_path = tmp_path / "empty.xml"
    xml_path.write_text("<MINiML></MINiML>")
    with pytest.raises(ValueError, match="No <Sample> elements found"):
        parse_miniml_to_df(xml_path, "GSE123")


# =============================================================================
# 4. GEO Downloading Operational Tests
# =============================================================================


@patch("mepylome.utils.downloader.download_file")
@patch("mepylome.utils.downloader.parse_miniml_to_df")
def test_download_geo_metadata(
    mock_parse: MagicMock, mock_download_file: MagicMock, tmp_path: Path
) -> None:
    with patch("tarfile.open"):
        download_geo_metadata("GSE12345", save_dir=tmp_path)
        mock_download_file.assert_called_once()
        mock_parse.assert_called_once_with(
            tmp_path / "GSE12345" / "GSE12345.xml", "GSE12345", None, None
        )


@patch("mepylome.utils.downloader.download_file")
@patch("tarfile.open")
def test_download_geo_idat_all_files(
    mock_tar_open: MagicMock, mock_download_file: MagicMock, tmp_path: Path
) -> None:
    series_id = "GSE12345"
    idat_dir = tmp_path / series_id / "idat"

    mock_tar = MagicMock()
    mock_tar_open.return_value.__enter__.return_value = mock_tar

    download_geo_idat_all_files(series_id, save_dir=tmp_path)
    mock_download_file.assert_called_once()
    mock_tar.extractall.assert_called_once_with(path=idat_dir, filter="data")


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
    mock_response = MagicMock()
    mock_response.text = (
        '<a href="2015_R01C01_Grn.idat"></a>'
        '<a href="2015_R01C01_Red.idat"></a>'
    )
    mock_get.return_value = mock_response

    download_arrayexpress_idat(series_id, save_dir=tmp_path, samples="all")
    assert mock_download_files.called
    urls = mock_download_files.call_args[1]["urls"]
    assert len(urls) == 2

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

    assert (tmp_path / "TCGA_TEST" / "manifest.csv").exists()
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
    manifest_df.to_csv(samples_dir / "manifest.csv", index=False)

    cart_json = tmp_path / "dummy_cart.json"
    cart_json.write_text("[]")

    download_tcga_idat(
        save_dir=tmp_path, metadata_cart=cart_json, subdir=subdir
    )
    assert mock_download_files.called

    (samples_dir / "manifest.csv").unlink()
    with pytest.raises(FileNotFoundError):
        download_tcga_idat(
            save_dir=tmp_path, metadata_cart=cart_json, subdir=subdir
        )


# =============================================================================
# 7. High-Level Hub Test (download_idats)
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
