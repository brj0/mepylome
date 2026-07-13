"""Tests for RFpurity tumor purity prediction."""

from typing import Any

import numpy as np
import pandas as pd
import pytest

import mepylome.dtypes.purity as purity_module
from mepylome.dtypes.beads import Manifest
from mepylome.dtypes.purity import predict_purity


class DummyModel:
    """Minimal sklearn-like model used for testing."""

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return deterministic predictions."""
        return X.mean(axis=1)


@pytest.fixture
def mock_models(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace RFpurify model loading with a dummy model."""
    bundle: dict[str, dict[str, Any]] = {
        "absolute": {
            "model": DummyModel(),
            "features": [
                "cg000001",
                "cg000002",
                "cg000003",
            ],
        },
        "estimate": {
            "model": DummyModel(),
            "features": [
                "cg000001",
                "cg000002",
                "cg000003",
            ],
        },
    }

    monkeypatch.setattr(
        purity_module,
        "_load_models",
        lambda: bundle,
    )


@pytest.fixture
def betas() -> pd.DataFrame:
    """Return a minimal beta-value test dataset."""
    return pd.DataFrame(
        {
            "cg000001": [0.1, 0.4],
            "cg000002": [0.2, 0.5],
            "cg000003": [0.3, 0.6],
        },
        index=[
            "sample1",
            "sample2",
        ],
    )


def test_predict_purity_returns_series(
    mock_models: None,
    betas: pd.DataFrame,
) -> None:
    """Prediction returns a named pandas Series."""
    scores = predict_purity(betas)

    assert isinstance(scores, pd.Series)
    assert scores.name == "purity_absolute"
    assert scores.index.tolist() == [
        "sample1",
        "sample2",
    ]
    assert np.all((scores >= 0) & (scores <= 1))


def test_predict_purity_estimate(
    mock_models: None,
    betas: pd.DataFrame,
) -> None:
    """The ESTIMATE model can be selected."""
    scores = predict_purity(
        betas,
        method="estimate",
    )

    assert isinstance(scores, pd.Series)
    assert scores.name == "purity_estimate"


def test_missing_probes_are_filled(
    mock_models: None,
) -> None:
    """Missing CpG probes are filled with the requested value."""
    betas = pd.DataFrame(
        {
            "cg000001": [0.2],
            "cg000003": [0.4],
        },
        index=[
            "sample",
        ],
    )

    scores = predict_purity(
        betas,
        fill=0.5,
    )

    expected = np.mean(
        [
            0.2,
            0.5,
            0.4,
        ]
    )

    assert np.isclose(
        scores.iloc[0],
        expected,
    )


def test_probe_order_is_corrected(
    mock_models: None,
) -> None:
    """Input CpG probe order does not affect prediction."""
    betas = pd.DataFrame(
        {
            "cg000003": [0.3],
            "cg000001": [0.1],
            "cg000002": [0.2],
        },
        index=[
            "sample",
        ],
    )

    scores = predict_purity(betas)

    assert np.isclose(
        scores.iloc[0],
        0.2,
    )


def test_invalid_method_raises(
    mock_models: None,
    betas: pd.DataFrame,
) -> None:
    """Invalid model names raise ValueError."""
    with pytest.raises(
        ValueError,
        match="method must be",
    ):
        predict_purity(
            betas,
            method="ABSOLUTE",  # type: ignore[arg-type]
        )


def test_predict_purity_epic_dummy_profiles() -> None:
    """Predict purity for synthetic EPIC beta profiles."""
    cpgs = Manifest("epic").data_frame.IlmnID

    betas = pd.DataFrame(
        np.vstack(
            [
                np.full(len(cpgs), 0.0),
                np.full(len(cpgs), 0.25),
                np.full(len(cpgs), 0.5),
                np.full(len(cpgs), 0.75),
                np.full(len(cpgs), 1.0),
            ]
        ),
        index=[
            "zero",
            "low",
            "middle",
            "high",
            "full",
        ],
        columns=cpgs,
    )

    scores = predict_purity(betas)

    expected = pd.Series(
        [
            0.711382,
            0.650700,
            0.526507,
            0.629469,
            0.734478,
        ],
        index=[
            "zero",
            "low",
            "middle",
            "high",
            "full",
        ],
        name="purity_absolute",
    )

    pd.testing.assert_series_equal(
        scores,
        expected,
        check_exact=False,
        atol=1e-6,
        rtol=1e-6,
    )
