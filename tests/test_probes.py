"""Pytest for probe types and channels."""

import pytest

from mepylome.dtypes.probes import Channel, InfiniumDesignType, ProbeType

# ---------------------------------------------------------------------------
# Channel
# ---------------------------------------------------------------------------


def test_channel_values() -> None:
    assert Channel.GRN == 0
    assert Channel.RED == 1


def test_channel_str() -> None:
    assert str(Channel.GRN) == "0"
    assert str(Channel.RED) == "1"


# ---------------------------------------------------------------------------
# InfiniumDesignType
# ---------------------------------------------------------------------------


def test_infinium_design_type_values() -> None:
    assert InfiniumDesignType.TYPE_I == 1
    assert InfiniumDesignType.TYPE_II == 2


# ---------------------------------------------------------------------------
# ProbeType values and str
# ---------------------------------------------------------------------------


def test_probe_type_values() -> None:
    assert ProbeType.ONE == 1
    assert ProbeType.TWO == 2
    assert ProbeType.SNP_ONE == 3
    assert ProbeType.SNP_TWO == 4
    assert ProbeType.CONTROL == 5


def test_probe_type_str() -> None:
    assert str(ProbeType.ONE) == "1"
    assert str(ProbeType.TWO) == "2"
    assert str(ProbeType.SNP_ONE) == "3"
    assert str(ProbeType.SNP_TWO) == "4"
    assert str(ProbeType.CONTROL) == "5"


# ---------------------------------------------------------------------------
# ProbeType.from_manifest_values – regular cg probes
# ---------------------------------------------------------------------------


def test_cg_probe_type_i() -> None:
    pt = ProbeType.from_manifest_values(
        "cg13869341", InfiniumDesignType.TYPE_I
    )
    assert pt == ProbeType.ONE


def test_cg_probe_type_ii() -> None:
    pt = ProbeType.from_manifest_values(
        "cg00035864", InfiniumDesignType.TYPE_II
    )
    assert pt == ProbeType.TWO


# ---------------------------------------------------------------------------
# ProbeType.from_manifest_values – SNP probes
# ---------------------------------------------------------------------------


def test_rs_probe_type_i() -> None:
    pt = ProbeType.from_manifest_values(
        "rs10796216", InfiniumDesignType.TYPE_I
    )
    assert pt == ProbeType.SNP_ONE


def test_rs_probe_type_ii() -> None:
    pt = ProbeType.from_manifest_values("rs715359", InfiniumDesignType.TYPE_II)
    assert pt == ProbeType.SNP_TWO


def test_rs_probe_no_type_returns_control() -> None:
    """Probe rs with neither TYPE_I nor TYPE_II resolves to CONTROL."""
    # Passing an integer that is neither TYPE_I nor TYPE_II
    pt = ProbeType.from_manifest_values("rs000000", 99)
    assert pt == ProbeType.CONTROL


# ---------------------------------------------------------------------------
# ProbeType.from_manifest_values – control prefixes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "ctl_positive",
        "neg_control",
        "BSC_conversion",
        "NON_polymorphic",
    ],
)
def test_control_prefixes(name: str) -> None:
    pt = ProbeType.from_manifest_values(name, InfiniumDesignType.TYPE_I)
    assert pt == ProbeType.CONTROL


# ---------------------------------------------------------------------------
# ProbeType.from_manifest_values – unknown prefix falls through to CONTROL
# ---------------------------------------------------------------------------


def test_unknown_prefix_falls_to_control() -> None:
    """A probe name that doesn't match any known prefix defaults to CONTROL."""
    pt = ProbeType.from_manifest_values("xyz_unknown", 99)
    assert pt == ProbeType.CONTROL


# ---------------------------------------------------------------------------
# ProbeType.from_manifest_values – Mouse-specific IR/IG infinium types
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("infinium_type", ["IR", "IG"])
def test_mouse_probe_types(infinium_type: str) -> None:
    pt = ProbeType.from_manifest_values("cg_mouse_probe", infinium_type)
    assert pt == ProbeType.ONE
