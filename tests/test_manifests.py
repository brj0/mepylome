"""Unittest for manifest object."""

import unittest

from mepylome.dtypes import Manifest, ProbeType
from mepylome.tests.helpers import TempManifest


class TestManifests(unittest.TestCase):
    """Unittest for IDAT preprocessing."""

    def test_manifest(self):
        tmp_manifest = TempManifest()
        manifest = Manifest(raw_path=tmp_manifest.path)
        self.assertIsNotNone(str(manifest))
        self.assertIsNotNone(manifest.probe_info(ProbeType.ONE))

        # Clean up
        manifest.proc_path.unlink()
        manifest.ctrl_path.unlink()
        manifest._pickle_path.unlink()


if __name__ == "__main__":
    unittest.main()
