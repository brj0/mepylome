"""Module for handling probe types and probe channels.

Usage:
    pt = ProbeType.from_manifest_values("cg13869341", InfiniumDesignType.I)
"""

from enum import IntEnum, unique


@unique
class Channel(IntEnum):
    """Specifies the fluorescence color (red or green) for probes."""

    GRN = 0
    RED = 1

    def __str__(self):
        return self.value


@unique
class InfiniumDesignType(IntEnum):
    """The EPIC chip uses two array-designs."""

    I = 1
    II = 2


@unique
class ProbeType(IntEnum):
    """Represents probe type, distinguishing between regular and SNP probes.

    The probe type depends on the Infinium_Design_Type (I or II) and the probe
    name. Probes starting with 'cg' are regular probes, probes starting with
    'rs' are categorized as SNP probes.
    """

    ONE = 1
    TWO = 2
    SNP_ONE = 3
    SNP_TWO = 4
    CONTROL = 5

    def __str__(self):
        return self.value

    @staticmethod
    def from_manifest_values(name, infinium_type):
        """Method to determine ProbeType based on name and design type.

        Args:
            name (str): Probe name.
            infinium_type (str): Infinium design type ('I' or 'II').

        Returns:
            str: Probe type.
        """
        if name.startswith("rs"):
            if infinium_type == InfiniumDesignType.I:
                return ProbeType.SNP_ONE
            if infinium_type == InfiniumDesignType.II:
                return ProbeType.SNP_TWO
            return ProbeType.CONTROL
        if any(
            [
                name.startswith("ctl"),
                name.startswith("neg"),
                name.startswith("BSC"),
                name.startswith("NON"),
            ]
        ):
            return ProbeType.CONTROL

        if infinium_type == InfiniumDesignType.I:
            return ProbeType.ONE

        if infinium_type == InfiniumDesignType.II:
            return ProbeType.TWO

        # Mouse only - not tested
        if infinium_type in ("IR", "IG"):
            return ProbeType.ONE

        return ProbeType.CONTROL
