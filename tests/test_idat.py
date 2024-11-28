"""Unittest for IDAT parser."""

import unittest

import numpy as np

from mepylome.dtypes.idat import IdatParser
from mepylome.tests.helpers import TempIdatFile


class TestIdatParser(unittest.TestCase):
    """Unittest for IDAT parser."""

    def _check_idat_parser(self, test_data, gzipped):
        """Check the data consistency between writer and parser."""
        tmp_idat = TempIdatFile(test_data, gzipped)
        idat_file = IdatParser(tmp_idat.path)
        for key in tmp_idat.data:
            test_data = tmp_idat.data[key]
            parsed_data = getattr(idat_file, key)
            if isinstance(test_data, np.ndarray):
                self.assertTrue(
                    np.array_equal(test_data, parsed_data),
                    f"Arrays for key '{key}' do not match.",
                )
            else:
                self.assertEqual(
                    test_data,
                    parsed_data,
                    f"Values for key '{key}' do not match.",
                )

    def test_idat_parser(self):
        test_data_list = [
            {},
            {"illumina_ids": np.array([1, 10, 1000, 10000], dtype="<i4")},
            {"illumina_ids": np.arange(0, 10000, dtype="<i4")},
            {"illumina_ids": np.arange(100, 10000, dtype="<i4")},
            {
                "n_snps_read": 20,
                "illumina_ids": np.arange(1000, 1000 + 20, dtype="<i4"),
                "probe_means": np.arange(100, 100 + 20, dtype="<u2"),
                "std_dev": np.arange(2000, 2000 + 20, dtype="<u2"),
                "n_beads": np.arange(3, 3 + 20, dtype="<u1"),
                "mid_block": np.arange(400000, 400000 + 20, dtype="<i4"),
                "run_info": [
                    [
                        "time_0",
                        "type_0",
                        "pars_0",
                        "code_0",
                        "vers_0",
                    ],
                    [
                        "time_1",
                        "type_1",
                        "pars_1",
                        "code_1",
                        "vers_1",
                    ],
                ],
                "red_green": 1,
                "mostly_null": "asdf",
                "barcode": "ghij",
                "chip_type": "klmn",
                "mostly_a": "opqr",
                "unknown_1": "stuv",
                "unknown_2": "wxxz",
                "unknown_3": "1234",
                "unknown_4": "5678",
                "unknown_5": "90.,",
                "unknown_6": "<>-:",
                "unknown_7": "!=|#",
            },
        ]

        for test_data in test_data_list:
            for gzipped in [False, True]:
                self._check_idat_parser(test_data, gzipped)


if __name__ == "__main__":
    unittest.main()
