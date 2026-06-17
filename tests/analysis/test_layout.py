"""Unit tests for the mepylome application layout builders."""

from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock, patch

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import pytest
from dash import html

from mepylome.analysis.layout import (
    UMAP_METRICS,
    get_all_genes,
    get_navbar,
    get_side_navigation,
)


@pytest.fixture
def mock_annotation() -> Generator[MagicMock, None, None]:
    """Fixture to mock Annotation.default_genes data frame."""
    with patch("mepylome.analysis.layout.Annotation") as mock_anno:
        mock_df = pd.DataFrame({"Name": ["BRCA1", "BRCA2", "TP53"]})
        mock_anno.default_genes.return_value = mock_df
        yield mock_anno


def test_get_all_genes(mock_annotation: MagicMock) -> None:
    """Test that get_all_genes parses the dataframe correctly."""
    genes = get_all_genes()
    assert genes == ["BRCA1", "BRCA2", "TP53"]
    mock_annotation.default_genes.assert_called_once()


def test_get_navbar() -> None:
    """Test that the navbar component tree structure is built cleanly."""
    navbar = get_navbar()

    assert isinstance(navbar, dbc.Navbar)
    assert navbar.color == "dark"
    assert navbar.dark is True

    container = navbar.children
    assert isinstance(container, dbc.Container)

    inner_elements = container.children
    assert any(isinstance(el, html.A) for el in inner_elements)
    assert any(isinstance(el, dbc.NavbarToggler) for el in inner_elements)


@pytest.mark.parametrize(
    "cpgs_array, expected_max_str, expected_val",
    [
        (np.array([1, 2, 3, 4, 5]), " (max. 5)", 5),
        (np.array([]), "", 1000),
    ],
)
def test_get_side_navigation_cpg_logic(
    cpgs_array: np.ndarray,
    expected_max_str: str,
    expected_val: int,
) -> None:
    """Test side navigation constraints handling for CpG array boundaries."""
    side_nav = get_side_navigation(
        sample_ids=["sample_1", "sample_2"],
        ids_to_highlight=["sample_1"],
        annotation_columns=["Age", "Gender"],
        analysis_dir=Path("/tmp/analysis"),
        annotation=Path("/tmp/annotation.csv"),
        reference_dir=Path("/tmp/reference"),
        test_dir=Path("/tmp/test"),
        output_dir=Path("/tmp/output"),
        cpgs=cpgs_array,
        n_cpgs=expected_val,
        prep="illumina",
        precalculate=True,
        cpg_selection="top",
        n_neighbors=15,
        metric="euclidean",
        min_dist=0.1,
        use_discrete_colors=True,
        custom_clfs=[{"name": "CustomRF"}],
    )

    assert isinstance(side_nav, dbc.Col)
    assert side_nav.width == {"size": 2}

    tabs_component = side_nav.children[0]
    setting_tab = tabs_component.children[0]

    content_strs = [str(child) for child in setting_tab.children]
    assert any(
        f"Number of CpG sites{expected_max_str}" in s for s in content_strs
    )


def test_get_side_navigation_color_and_clf_mapping() -> None:
    """Verify color scheme definitions and classifier options mapping."""
    side_nav = get_side_navigation(
        sample_ids=["s1"],
        ids_to_highlight=[],
        annotation_columns=["Batch"],
        analysis_dir=Path("."),
        annotation=Path("."),
        reference_dir=Path("."),
        test_dir=Path("."),
        output_dir=Path("."),
        cpgs=np.array([1, 2]),
        n_cpgs=2,
        prep="swan",
        precalculate=False,
        cpg_selection="random",
        n_neighbors=5,
        metric="cosine",
        min_dist=0.5,
        use_discrete_colors=False,
        custom_clfs=[{"name": "CustomGBC"}],
    )

    tabs_component = side_nav.children[0]

    highlight_tab = tabs_component.children[2]

    color_scheme_dropdown = next(
        child
        for child in highlight_tab.children
        if getattr(child, "id", None) == "umap-color-scheme"
    )
    assert color_scheme_dropdown.value == "continuous"

    classify_tab = tabs_component.children[4]
    clf_dropdown = next(
        child
        for child in classify_tab.children
        if getattr(child, "id", None) == "clf-clf-dropdown"
    )

    assert clf_dropdown.options["0"] == "CustomGBC"


def test_umap_metrics_constant() -> None:
    """Ensure the expected UMAP metrics lists remain defined and unchanged."""
    assert "euclidean" in UMAP_METRICS
    assert "manhattan" in UMAP_METRICS
    assert len(UMAP_METRICS) == 17
