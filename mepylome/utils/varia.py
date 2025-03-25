"""Contains auxiliary functions and a debug timer.

Usage:
    timer = Timer()
    timer.start()
    # process to be measured
    timer.stop("process_name")
"""

import logging
import socket
import tempfile
import time
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from uuid import uuid4

import numpy as np
import toml

from mepylome.utils.files import get_resource_path

__all__ = ["Timer", "normexp_get_xs", "MEPYLOME_TMP_DIR"]


def get_app_version():
    """Retrieve the app version from the package metadata."""
    try:
        return version("mepylome")
    except PackageNotFoundError:
        return "unknown"


logger = logging.getLogger(__name__)

version_str = get_app_version().replace(".", "_")
MEPYLOME_TMP_DIR = Path(tempfile.gettempdir()) / f"mepylome-{version_str}"
LOG_DIR = MEPYLOME_TMP_DIR / "log"

LOG_DIR.mkdir(parents=True, exist_ok=True)


def make_log_file(suffix):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = uuid4().hex[:8]
    log_file = LOG_DIR / f"{suffix}-{timestamp}-{unique_id}.log"
    log_file.touch(exist_ok=True)
    return log_file


class Timer:
    """Measures the time elapsed in milliseconds."""

    def __init__(self):
        self.time0 = time.time()

    def start(self):
        """Resets timer."""
        self.time0 = time.time()

    def stop(self, text=None):
        """Resets timer and return elapsed time."""
        delta_time = 1000 * (time.time() - self.time0)
        comment = "" if text is None else "(" + text + ")"
        print("Time passed:", delta_time, "ms", comment)
        self.time0 = time.time()
        return delta_time


def huber(y, k=1.5, tol=1.0e-6):
    """Compute Huber's M-estimator of location with MAD scale.

    Code adapted from
    https://github.com/cran/MASS/blob/master/R/huber.R

    Args:
        y (array-like): Vector of data values.
        k (float, optional): Winsorizes at `k` standard deviations. Default is
            1.5.
        tol (float, optional): Convergence tolerance. Default is 1.0e-6.

    Returns:
        tuple: A tuple containing:
            - mu (float): Location estimate (mean).
            - s (float): MAD scale estimate (standard deviation).

    Raises:
        ValueError: If the scale (MAD) is zero for the given sample.

    References:
        Huber, P. J. (1981). _Robust Statistics._ Wiley.
        Venables, W. N. and Ripley, B. D. (2002). _Modern Applied Statistics
            with S._ Fourth edition. Springer.

    Examples:
        >>> huber(chem)
    """
    y = y[~np.isnan(y)]
    mu = np.median(y)
    s = np.median(np.abs(y - mu)) * 1.4826
    if s == 0:
        msg = "Cannot estimate scale: MAD is zero for this sample"
        raise ValueError(msg)
    while True:
        yy = np.clip(y, mu - k * s, mu + k * s)
        mu1 = np.mean(yy)
        if np.abs(mu - mu1) < tol * s:
            break
        mu = mu1
    return mu, s


def normexp_signal(par, x):
    """Expected Signal Given Observed Foreground Under Normal+Exp Model.

    Adjust foreground intensities for observed background using Normal+Exp
    Model.

    Code adapted from:
    https://github.com/gangwug/limma/blob/master/R/background-normexp.R

    Args:
        par (array-like): Numeric vector containing the parameters of the
            Normal+Exp distribution.
                - par[0]: mu
                - par[1]: log sigma
                - par[2]: log alpha
        x (array-like): Numeric vector of (background corrected) intensities.

    Returns:
        array-like: Numeric vector containing adjusted intensities.

    Raises:
        ValueError: If alpha or sigma are non-positive.

    References:
        Ritchie, M. E., Silver, J., Oshlack, A., Silver, J., Holmes, M.,
        Diyagama, D., Holloway, A., and Smyth, G. K. (2007). A comparison of
        background correction methods for two-colour microarrays.
        _Bioinformatics_ 23, 2700-2707.
        <http://bioinformatics.oxfordjournals.org/content/23/20/2700>

        Silver, JD, Ritchie, ME, and Smyth, GK (2009). Microarray background
        correction: maximum likelihood estimation for the normal-exponential
        convolution. _Biostatistics_ 10, 352-363.
        <http://biostatistics.oxfordjournals.org/content/10/2/352>

    Examples:
        >>> normexp_signal([1, np.log(2), np.log(3)], 4)
        2.3735035872302235
    """
    from scipy.stats import norm

    mu = par[0]
    sigma = np.exp(par[1])
    sigma2 = sigma * sigma
    alpha = np.exp(par[2])
    if alpha <= 0:
        msg = "alpha must be positive"
        raise ValueError(msg)
    if sigma <= 0:
        msg = "sigma must be positive"
        raise ValueError(msg)
    mu_sf = x - mu - sigma2 / alpha
    log_dnorm = norm.logpdf(0, loc=mu_sf, scale=sigma)
    log_pnorm = norm.logsf(0, loc=mu_sf, scale=sigma)
    signal = mu_sf + sigma2 * np.exp(log_dnorm - log_pnorm)
    z = ~np.isnan(signal)
    if np.any(signal[z] < 0):
        logger.warning(
            "Limit of numerical accuracy reached with very low intensity or "
            "very high background:\nsetting adjusted intensities to small "
            "value"
        )
        signal[z] = np.maximum(signal[z], 1e-6)
    return signal


def normexp_get_xs(xf, controls=None, offset=50, param=None):
    """Used in NOOB.

    Adapted from
    https://github.com/hansenlab/minfi/blob/devel/R/preprocessNoob.R
    """
    n_probes = xf.shape[0]
    if param is None:
        if controls is None:
            msg = "'controls' or 'param' must be given"
            raise ValueError(msg)
        alpha = np.empty(n_probes)
        mu = np.empty(n_probes)
        sigma = np.empty(n_probes)
        for i in range(n_probes):
            mu[i], sigma[i] = huber(controls[i, :])
            alpha[i] = max(huber(xf[i, :])[0] - mu[i], 10)
        param = np.column_stack((mu, np.log(sigma), np.log(alpha)))
    result = np.empty(xf.shape)
    for i in range(n_probes):
        result[i, :] = normexp_signal(param[i], xf[i, :])
    return {
        "xs": result + offset,
        "param": param,
    }


def get_free_port(start_port):
    """Returns the first free port from start position."""
    port = start_port
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("localhost", port)) != 0:
                return port
            port += 1


def load_config():
    """Loads the configuration from the package's config.toml."""
    config_path = get_resource_path("mepylome", "data/config.toml")
    return toml.load(config_path)


CONFIG = load_config()
