"""Contains the ArrayType used to represent different Illumina array types."""

from enum import Enum, unique
from pathlib import Path

from mepylome.dtypes.idat import IdatParser


def _find_valid_path(path):
    """Tries to find a valid IDAT file associated with the given path."""
    base_path = Path(path)
    if base_path.exists():
        return base_path
    grn_path = Path(str(base_path) + "_Grn.idat")
    if grn_path.exists():
        return grn_path
    grn_gz_path = Path(str(base_path) + "_Grn.idat.gz")
    if grn_gz_path.exists():
        return grn_gz_path
    msg = f"No valid file found for path: {path}"
    raise ValueError(msg)


@unique
class ArrayType(Enum):
    """Provides constants for the different Illumina array types.

    Enum representing different Illumina array types with method to infer
    type from probe count.
    """

    ILLUMINA_27K = "27k"
    ILLUMINA_450K = "450k"
    ILLUMINA_EPIC = "epic"
    ILLUMINA_EPIC_V2 = "epicv2"
    ILLUMINA_MSA48 = "msa48"
    ILLUMINA_MOUSE = "mouse"
    UNKNOWN = "unknown"

    def __str__(self):
        return self.value

    @classmethod
    def from_probe_count(cls, probe_count):
        """Infers array type based on the number of probes in an idat file."""
        if 622000 <= probe_count <= 623000:
            return cls.ILLUMINA_450K

        if 1050000 <= probe_count <= 1053000:
            return cls.ILLUMINA_EPIC

        if 1032000 <= probe_count <= 1033000:
            return cls.ILLUMINA_EPIC

        if 1100000 <= probe_count <= 1108000:
            return cls.ILLUMINA_EPIC_V2

        if 384400 <= probe_count <= 384600:
            return cls.ILLUMINA_MSA48

        if 55200 <= probe_count <= 55400:
            return cls.ILLUMINA_27K

        if 315000 <= probe_count <= 362000:
            return cls.ILLUMINA_MOUSE

        return cls.UNKNOWN

    @classmethod
    def from_idat(cls, path):
        """Infers array type from idat_file."""
        valid_path = _find_valid_path(path)
        probe_count = IdatParser(valid_path, array_type_only=True).n_snps_read
        return ArrayType.from_probe_count(probe_count)
