"""Tests for normexp_signal."""

from importlib.metadata import PackageNotFoundError
from unittest.mock import patch

import numpy as np
import pytest
from scipy.stats import norm

from mepylome.utils.varia import (
    get_app_version,
    huber,
    normexp_get_xs,
    normexp_signal,
)


def test_get_app_version_returns_string() -> None:
    assert isinstance(get_app_version(), str)


def test_get_app_version_returns_unknown_on_missing_package() -> None:
    with patch(
        "mepylome.utils.varia.version", side_effect=PackageNotFoundError
    ):
        assert get_app_version() == "unknown"


def test_get_app_version_returns_mocked_version() -> None:
    with patch("mepylome.utils.varia.version", return_value="1.2.3"):
        assert get_app_version() == "1.2.3"


@pytest.mark.parametrize(
    ("y", "expected_mu", "expected_s"),
    [
        (
            np.array([1.0, 2.0, 3.0]),
            2.0,
            1.4826,
        ),
        (
            np.arange(100, dtype=float),
            49.5,
            37.065,
        ),
    ],
)
def test_huber_known_values(
    y: np.ndarray,
    expected_mu: float,
    expected_s: float,
) -> None:
    """Regression tests for known Huber estimates."""
    mu, s = huber(y, k=2.0, tol=1e-6)

    np.testing.assert_allclose(mu, expected_mu, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(s, expected_s, rtol=1e-12, atol=1e-12)


def test_huber_ignores_nan_values() -> None:
    """NaN values are removed before estimation."""
    y = np.array([1.0, 2.0, 3.0, np.nan])

    mu, s = huber(y)

    np.testing.assert_allclose(mu, 2.0)
    np.testing.assert_allclose(s, 1.4826)


def test_huber_is_robust_to_outlier() -> None:
    """Extreme values should not strongly affect Huber location."""
    y = np.array([1.0, 2.0, 3.0, 4.0, 1000.0])

    mu, _ = huber(y)

    assert mu < 10.0
    assert mu > 2.0


def test_huber_raises_when_mad_is_zero() -> None:
    """Constant input cannot produce a MAD scale."""
    y = np.array([5.0, 5.0, 5.0])

    with pytest.raises(
        ValueError,
        match="MAD is zero",
    ):
        huber(y)


def test_huber_preserves_scale_from_initial_mad() -> None:
    """Scale estimate is the MAD-based robust scale."""
    y = np.array([-1.0, 0.0, 1.0])

    _, s = huber(y)

    np.testing.assert_allclose(
        s,
        1.4826,
        rtol=1e-12,
        atol=1e-12,
    )


def normexp_signal_reference(par: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Reference implementation using scipy.stats."""
    mu = par[0]
    sigma = np.exp(par[1])
    sigma2 = sigma * sigma
    alpha = np.exp(par[2])

    if alpha <= 0:
        raise ValueError("alpha must be positive")
    if sigma <= 0:
        raise ValueError("sigma must be positive")

    mu_sf = x - mu - sigma2 / alpha
    log_dnorm = norm.logpdf(0, loc=mu_sf, scale=sigma)
    log_pnorm = norm.logsf(0, loc=mu_sf, scale=sigma)

    signal = mu_sf + sigma2 * np.exp(log_dnorm - log_pnorm)

    z = ~np.isnan(signal)
    if np.any(signal[z] < 0):
        signal[z] = np.maximum(signal[z], 1e-6)

    return signal


@pytest.mark.parametrize(
    "par",
    [
        np.array([100.0, np.log(200.0), np.log(300.0)]),
        np.array([0.0, np.log(1.0), np.log(10.0)]),
        np.array([500.0, np.log(50.0), np.log(1000.0)]),
    ],
)
def test_normexp_signal_matches_reference(par: np.ndarray) -> None:
    """Fast implementation matches scipy.stats implementation."""
    rng = np.random.default_rng(42)

    x = rng.uniform(0, 50000, size=10000)

    expected = normexp_signal_reference(par, x)
    observed = normexp_signal(par, x)

    np.testing.assert_allclose(
        observed,
        expected,
        rtol=1e-12,
        atol=1e-10,
    )


def test_normexp_signal_preserves_nan() -> None:
    """NaN values remain NaN."""
    par = np.array([100.0, np.log(200.0), np.log(300.0)])

    x = np.array([1000.0, np.nan, 5000.0])

    result = normexp_signal(par, x)

    assert np.isnan(result[1])
    assert np.isfinite(result[0])
    assert np.isfinite(result[2])


@pytest.mark.parametrize(
    "par",
    [
        np.array([0.0, np.log(1.0), -np.inf]),  # alpha = 0
        np.array([0.0, -np.inf, np.log(1.0)]),  # sigma = 0
    ],
)
def test_normexp_signal_invalid_parameters(par: np.ndarray) -> None:
    x = np.array([1000.0])

    with pytest.raises(ValueError):
        normexp_signal(par, x)


@pytest.mark.parametrize(
    ("par", "x", "expected"),
    [
        (
            np.array([1.0, np.log(2.0), np.log(3.0)]),
            np.array([4.0]),
            np.array([2.3735035872302235]),
        ),
        (
            np.array([600.0, 6.0, 9.0]),
            np.array([50.0, 10000.0, 3.0, 1.0, 150.0]),
            np.array(
                [
                    182.34695299,
                    9379.91446308,
                    175.20366233,
                    174.90984631,
                    199.20398827,
                ]
            ),
        ),
    ],
)
def test_normexp_signal_known_values(
    par: np.ndarray,
    x: np.ndarray,
    expected: np.ndarray,
) -> None:
    """Regression tests for known normexp_signal values."""
    result = normexp_signal(par, x)

    np.testing.assert_allclose(
        result,
        expected,
        rtol=1e-10,
        atol=1e-10,
    )


def test_normexp_get_xs_with_controls() -> None:
    """Regression test using controls to estimate parameters."""
    xf = np.array([[1.0, 2.0], [3.0, 4.0]])
    controls = np.array([[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]])

    result = normexp_get_xs(
        xf,
        controls,
        offset=50,
    )

    expected_xs = np.array(
        [
            [50.37097171, 50.43637800],
            [50.32146501, 50.37097171],
        ]
    )

    expected_param = np.array(
        [
            [6.0, 0.3937973, 2.30258509],
            [9.0, 0.3937973, 2.30258509],
        ]
    )

    np.testing.assert_allclose(
        result["xs"],
        expected_xs,
        rtol=1e-8,
        atol=1e-8,
    )

    np.testing.assert_allclose(
        result["param"],
        expected_param,
        rtol=1e-8,
        atol=1e-8,
    )


def test_normexp_get_xs_with_given_parameters() -> None:
    """Provided parameters are used without estimating from controls."""
    xf = np.array([[1.0, 2.0]])

    param = np.array(
        [
            [2.0, np.log(1.4826), np.log(10.0)],
        ]
    )

    result = normexp_get_xs(
        xf,
        param=param,
        offset=50,
    )

    assert result["xs"].shape == xf.shape

    np.testing.assert_allclose(
        result["param"],
        param,
    )


def test_normexp_get_xs_requires_controls_or_parameters() -> None:
    """Missing controls and parameters raises an error."""
    xf = np.array([[1.0, 2.0]])

    with pytest.raises(
        ValueError,
        match="controls.*param",
    ):
        normexp_get_xs(xf)


def test_normexp_get_xs_preserves_shape() -> None:
    """Output shape matches input signal shape."""
    rng = np.random.default_rng(42)

    xf = rng.uniform(0, 100, size=(5, 20))
    controls = rng.uniform(0, 100, size=(5, 50))

    result = normexp_get_xs(
        xf,
        controls,
    )

    assert result["xs"].shape == xf.shape
    assert result["param"].shape == (xf.shape[0], 3)


def test_normexp_get_xs_offset_is_applied() -> None:
    """Changing offset shifts xs by the same amount."""
    xf = np.array([[10.0, 20.0]])
    param = np.array([[5.0, np.log(2.0), np.log(3.0)]])

    result_0 = normexp_get_xs(
        xf,
        param=param,
        offset=0,
    )

    result_50 = normexp_get_xs(
        xf,
        param=param,
        offset=50,
    )

    np.testing.assert_allclose(
        result_50["xs"] - result_0["xs"],
        50.0,
    )
