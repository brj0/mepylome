import time
from scipy.stats import norm
import numpy as np

__all__ = ["Timer", "normexp_get_xs"]

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
    y = y[~np.isnan(y)]
    mu = np.median(y)
    s = np.median(np.abs(y - mu)) * 1.4826
    if s == 0:
        raise ValueError("Cannot estimate scale: MAD is zero for this sample")
    while True:
        yy = np.clip(y, mu - k * s, mu + k * s)
        mu1 = np.mean(yy)
        if np.abs(mu - mu1) < tol * s:
            break
        mu = mu1
    return mu, s


def normexp_signal(par, x):
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
        print(
            "Limit of numerical accuracy reached with very low intensity or "
            "very high background:\nsetting adjusted intensities to small "
            "value"
        )
        signal[z] = np.maximum(signal[z], 1e-6)
    return signal


def normexp_get_xs(xf, controls=None, offset=50, param=None):
    n_probes = xf.shape[0]
    if param is None:
        if controls is None:
            ValueError("'controls' or 'param' must be given")
        alpha = np.empty(n_probes)
        mu = np.empty(n_probes)
        sigma = np.empty(n_probes)
        for i in range(n_probes):
            mu[i], sigma[i] = huber(controls[i,:])
            alpha[i] = max(huber(xf[i, :])[0] - mu[i], 10)
        param = np.column_stack((mu, np.log(sigma), np.log(alpha)))
    result = np.empty(xf.shape)
    for i in range(n_probes):
        result[i, :] = normexp_signal(param[i], xf[i, :])
    return {
        "xs": result + offset,
        "param": param,
    }

