# Lib
from enum import Enum, unique

import logging

LOGGER = logging.getLogger(__name__)


@unique
class ArrayType(Enum):
    """This class stores meta data about array types, such as numbers of probes
    of each type, and how to guess the array from probes in idat files.
    """

    CUSTOM = "custom"
    ILLUMINA_27K = "27k"
    ILLUMINA_450K = "450k"
    ILLUMINA_EPIC = "epic"
    ILLUMINA_EPIC_V2 = "epicv2"
    ILLUMINA_MOUSE = "mouse"

    def __str__(self):
        return self.value

    @classmethod
    def from_probe_count(cls, probe_count):
        """Determines array type using number of probes counted in raw idat
        file. Returns array string.
        """
        if 622000 <= probe_count <= 623000:
            return cls.ILLUMINA_450K

        if 1050000 <= probe_count <= 1053000:
            return cls.ILLUMINA_EPIC

        if 1032000 <= probe_count <= 1033000:
            return cls.ILLUMINA_EPIC

        if 1100000 <= probe_count <= 1108000:
            return cls.ILLUMINA_EPIC_V2

        if 55200 <= probe_count <= 55400:
            return cls.ILLUMINA_27K

        if 315000 <= probe_count <= 362000:
            return cls.ILLUMINA_MOUSE

        raise ValueError(f"Unknown array type: {probe_count} probes detected")

    @property
    def num_probes(self):
        """Used to load normal cg+ch probes from start of manifest until this
        point. Then start control df.
        """
        probe_counts = {
            ArrayType.ILLUMINA_27K: 27578,
            ArrayType.ILLUMINA_450K: 485577,
            ArrayType.ILLUMINA_EPIC: 865918,
            ArrayType.ILLUMINA_MOUSE: 293199,  # MM285_v2 added 615 missing probes
        }
        return probe_counts.get(self)

    @property
    def num_controls(self):
        probe_counts = {
            # the manifest does not contain control probe data (illumina's site
            # included)
            ArrayType.ILLUMINA_27K: 0,
            ArrayType.ILLUMINA_450K: 850,
            ArrayType.ILLUMINA_EPIC: 635,
            ArrayType.ILLUMINA_MOUSE: 635,
        }
        return probe_counts.get(self)

    @property
    # not used anywhere in v1.5.0+
    def num_snps(self):
        probe_counts = {
            ArrayType.ILLUMINA_27K: 0,
            ArrayType.ILLUMINA_450K: 65,
            ArrayType.ILLUMINA_EPIC: 59,
            ArrayType.ILLUMINA_MOUSE: 1485,
        }
        return probe_counts.get(self)
