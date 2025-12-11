"""Pytest for manifest object."""

from mepylome.dtypes import Manifest, ProbeType
from mepylome.tests.helpers import TempManifest


def test_manifest() -> None:
    """Tests if dummy manifest is correctly created."""
    tmp_manifest = TempManifest()
    manifest = Manifest(raw_path=tmp_manifest.path)

    # Basic sanity checks
    assert str(manifest) is not None
    assert manifest.probe_info(ProbeType.ONE) is not None

    # Clean up
    manifest.proc_path.unlink()
    manifest.ctrl_path.unlink()
    manifest._pickle_path.unlink()
