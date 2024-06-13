"""Contains the ArrayType used to represent different Illumina array types."""

from enum import Enum, unique


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
    ILLUMINA_MOUSE = "mouse"
    INVALID = "invalid"

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

        if 55200 <= probe_count <= 55400:
            return cls.ILLUMINA_27K

        if 315000 <= probe_count <= 362000:
            return cls.ILLUMINA_MOUSE

        msg = f"Unknown array type: {probe_count} probes detected"
        raise ValueError(msg)
