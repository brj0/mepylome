"""Tests for file utility functions."""

from collections.abc import Iterator
from io import BytesIO
from pathlib import Path
from typing import Any, Literal
from unittest.mock import MagicMock

import pytest
import requests
from _pytest.monkeypatch import MonkeyPatch

from mepylome.utils.files import (
    download_file,
    download_files,
    ensure_directory_exists,
    get_csv_file,
    get_resource_path,
    reset_file,
)


class MockResponse:
    """Mock requests response."""

    def __init__(self, content: bytes = b"hello world") -> None:
        self.content = content
        self.headers = {"content-length": str(len(content))}

    def raise_for_status(self) -> None:
        return None

    def iter_content(self, chunk_size: int = 8192) -> Iterator[bytes]:
        for i in range(0, len(self.content), chunk_size):
            yield self.content[i : i + chunk_size]

    def __enter__(self) -> "MockResponse":
        return self

    def __exit__(self, *args: object) -> Literal[False]:
        return False


def test_get_resource_path() -> None:
    path = get_resource_path("mepylome")
    assert isinstance(path, Path)
    assert path.exists()


def test_ensure_directory_exists(tmp_path: Path) -> None:
    path = tmp_path / "a" / "b" / "c"
    ensure_directory_exists(path)
    assert path.exists()
    assert path.is_dir()


def test_download_file_success(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    outfile = tmp_path / "file.txt"

    monkeypatch.setattr(
        requests,
        "get",
        lambda *args, **kwargs: MockResponse(b"abcdef"),
    )

    download_file(
        url="https://example.org/file.txt",
        save_path=outfile,
        show_progress=False,
    )

    assert outfile.read_bytes() == b"abcdef"


def test_download_file_skips_existing(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    outfile = tmp_path / "file.txt"
    outfile.write_text("existing")

    called = False

    def fake_get(*args: object, **kwargs: object) -> MockResponse:
        nonlocal called
        called = True
        return MockResponse()

    monkeypatch.setattr(requests, "get", fake_get)

    download_file(
        "https://example.org/file.txt",
        outfile,
        overwrite=False,
        show_progress=False,
    )

    assert not called


def test_download_file_overwrite_existing(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    outfile = tmp_path / "file.txt"
    outfile.write_text("old")

    monkeypatch.setattr(
        requests,
        "get",
        lambda *args, **kwargs: MockResponse(b"new"),
    )

    download_file(
        "https://example.org/file.txt",
        outfile,
        overwrite=True,
        show_progress=False,
    )

    assert outfile.read_bytes() == b"new"


def test_download_file_retries_then_succeeds(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    outfile = tmp_path / "file.txt"

    calls = {"n": 0}

    def fake_get(*args: object, **kwargs: object) -> MockResponse:
        calls["n"] += 1
        if calls["n"] < 3:
            raise requests.RequestException("fail")
        return MockResponse(b"ok")

    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr("mepylome.utils.files.time.sleep", lambda *_: None)

    download_file(
        "https://example.org/file.txt",
        outfile,
        show_progress=False,
        max_attempts=3,
        retry_delay=0,
    )

    assert outfile.read_bytes() == b"ok"


def test_download_file_raises_after_max_attempts(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    outfile = tmp_path / "file.txt"

    def fake_get(*args: object, **kwargs: object) -> MockResponse:
        raise requests.RequestException("fail")

    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr("mepylome.utils.files.time.sleep", lambda *_: None)

    with pytest.raises(RuntimeError):
        download_file(
            "https://example.org/file.txt",
            outfile,
            show_progress=False,
            max_attempts=2,
            retry_delay=0,
        )


def test_download_files_calls_download_file(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[Any] = []

    def fake_download(
        url: str,
        path: Path,
        overwrite: bool,
        show_progress: bool,
    ) -> None:
        calls.append((url, path, overwrite, show_progress))

    monkeypatch.setattr("mepylome.utils.files.download_file", fake_download)

    urls = ["u1", "u2"]
    paths = [tmp_path / "a", tmp_path / "b"]

    download_files(urls, paths, overwrite=True, show_progress=False)

    assert len(calls) == 2


def test_download_files_length_mismatch(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        download_files(["u1"], [tmp_path / "a", tmp_path / "b"])


def test_get_csv_file_from_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "test.csv"
    csv_path.write_bytes(b"a,b\n1,2\n")

    with get_csv_file(csv_path, "x.csv") as f:
        assert f.read() == b"a,b\n1,2\n"


def test_get_csv_file_from_bpm(tmp_path: Path) -> None:
    bpm_path = tmp_path / "test.bpm"
    bpm_path.write_bytes(b"manifest")

    with get_csv_file(bpm_path, "x.csv") as f:
        assert f.read() == b"manifest"


def test_get_csv_file_missing_in_zip(tmp_path: Path) -> None:
    import zipfile

    archive = tmp_path / "test.zip"

    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("a.csv", "x")

    with pytest.raises(FileNotFoundError):
        get_csv_file(archive, "missing.csv")


def test_get_csv_file_invalid_extension(tmp_path: Path) -> None:
    path = tmp_path / "x.txt"
    path.write_text("x")

    with pytest.raises(ValueError):
        get_csv_file(path, "x.csv")


def test_reset_file_seekable() -> None:
    buf = BytesIO(b"abcdef")
    buf.read(3)
    reset_file(buf)
    assert buf.tell() == 0


def test_reset_file_non_seekable() -> None:
    mock = MagicMock(spec=[])
    reset_file(mock)
