from enum import IntEnum, Enum, unique
import numpy as np


@unique
class Channel(IntEnum):
    """idat probes measure either a red or green fluorescence.  This specifies
    which to return within idat.py: red_idat or green_idat.
    """

    GRN = 0
    RED = 1

    def __str__(self):
        return self.value

    @property
    def is_green(self):
        return self == self.GRN

    @property
    def is_red(self):
        return self == self.RED



@unique
class InfiniumDesignType(IntEnum):
    I = 1
    II = 2


@unique
class ProbeType(IntEnum):
    """probes can either be type I or type II for CpG or Snp sequences.
    Control probes are used for background correction in different fluorescence
    ranges and staining efficiency.  Type I probes record EITHER a red or a
    green value.  Type II probes record both values together.  NOOB uses the
    red fluorescence on a green probe and vice versa to calculate background
    fluorescence.
    """

    ONE = 1
    TWO = 2
    SNP_ONE = 3
    SNP_TWO = 4
    CONTROL = 5
    # I was separating out mouse probes EARLY, here, but found they need to be
    # processed like all other probes, THEN removed in post-processing stage.
    # MOUSE_ONE = 'MouseI'
    # MOUSE_TWO = 'MouseII'

    def __str__(self):
        return self.value

    @staticmethod
    def from_manifest_values(name, infinium_type):
        """this function determines which of four kinds of probe goes with
        this name, using either the Infinium_Design_Type (I or II) or the name
        (starts with 'rs') and decides 'Control' is non of the above.
        """
        is_control = any(
            [
                name.startswith("rs"),
                name.startswith("ctl"),
                name.startswith("neg"),
                name.startswith("BSC"),
                name.startswith("NON"),
            ]
        )
        is_snp = name.startswith("rs")

        if is_control and is_snp:
            if infinium_type == InfiniumDesignType.I:
                return ProbeType.SNP_ONE
            elif infinium_type == InfiniumDesignType.II:
                return ProbeType.SNP_TWO
            else:
                return ProbeType.CONTROL
        elif is_control:
            return ProbeType.CONTROL

        elif infinium_type == InfiniumDesignType.I:
            return ProbeType.ONE

        elif infinium_type == InfiniumDesignType.II:
            return ProbeType.TWO

        # mouse only -- these are type I probes but Bret's files label them
        # this way
        elif infinium_type in ("IR", "IG"):
            return ProbeType.ONE

        return ProbeType.CONTROL


class Probe:
    """this doesn't appear to be instantiated anywhere in methylprep"""

    __slots__ = [
        "address",
        "illumina_id",
        "probe_type",
    ]

    def __init__(self, address, illumina_id, probe_type):
        self.address = address
        self.illumina_id = illumina_id
        self.probe_type = probe_type


class ProbeSubset:
    """used below in probes.py to define sub-sets of probes:
    foreground-(red|green|all), or (un)methylated probes
    """

    __slots__ = [
        "data_channel",
        "probe_address",
        "probe_channel",
        "probe_type",
    ]

    def __init__(self, data_channel, probe_address, probe_channel, probe_type):
        self.data_channel = data_channel
        self.probe_address = probe_address
        self.probe_channel = probe_channel
        self.probe_type = probe_type

    def __str__(self):
        return f"{self.probe_type}-{self.probe_channel}"

    @property
    def is_green(self):
        return self.data_channel.is_green

    @property
    def is_red(self):
        return self.data_channel.is_red

    @property
    def column_name(self):
        return self.probe_address.header_name

    # def get_probe_details(self, manifest):
    #    return manifest.get_probe_details(
    #        probe_type=self.probe_type,
    #        channel=self.probe_channel,
    #    )


