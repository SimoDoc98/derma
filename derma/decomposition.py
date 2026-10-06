"""Tonic/phasic decomposition backends, each a plain function returning (tonic, phasic).

References
----------
Greco A, Valenza G, Lanata A, Scilingo EP, Citi L (2016). cvxEDA: a convex optimization
approach to electrodermal activity processing. IEEE Trans Biomed Eng 63(4):797-804.
"""

import numpy as np


def moving_average(signal, window_samples):
    """
    Centred moving average that shrinks at the edges, as MATLAB ``movmean``.

    Parameters
    ----------
    signal : numpy.ndarray
        Input signal.
    window_samples : int
        Window length (samples). For an even length the window spans one more sample
        before the current one than after it, as in MATLAB.

    Returns
    -------
    numpy.ndarray
        Smoothed signal, same length as the input.
    """
    signal = np.asarray(signal, dtype=float)
    if window_samples <= 1:
        return signal.copy()

    n = len(signal)
    before = window_samples // 2
    after = window_samples - 1 - before
    idx = np.arange(n)
    lo = np.clip(idx - before, 0, n)
    hi = np.clip(idx + after + 1, 0, n)
    csum = np.concatenate(([0.0], np.cumsum(signal)))
    return (csum[hi] - csum[lo]) / (hi - lo)


def decompose(signal, fs, alpha, gamma, delta_knot, smoothing_sec=0.75):
    """
    Decompose EDA into tonic and phasic components with cvxEDA.

    The signal is smoothed with a short moving average against residual high-frequency
    noise, z-scored, decomposed, and the two components are brought back to uS.
    Requires the optional ``cvxeda`` package.

    Parameters
    ----------
    signal : numpy.ndarray
        Skin conductance (uS).
    fs : float
        Sampling frequency (Hz).
    alpha : float
        cvxEDA penalty on the sparse sudomotor driver (L1).
    gamma : float
        cvxEDA penalty on the tonic spline coefficients (L2).
    delta_knot : float
        Spacing of the tonic spline knots (s).
    smoothing_sec : float
        Length of the moving-average window (s); 0 disables the smoothing.

    Returns
    -------
    tonic : numpy.ndarray
        Tonic component (uS).
    phasic : numpy.ndarray
        Phasic component (uS).
    """
    try:
        import cvxeda
    except ImportError:
        raise ImportError("decompose() needs the optional cvxEDA backend: pip install 'derma[cvxeda]'") from None

    y = moving_average(signal, int(round(smoothing_sec * fs)))

    mu = float(np.mean(y))
    sigma = float(np.std(y)) if np.std(y) != 0 else 1.0
    y_norm = (y - mu) / sigma

    # Bateman time constants of the cvxEDA SCR model (Greco et al., 2016);
    # relative solver tolerance 1e-6, ample for EDA sampled at a few Hz
    phasic, _, tonic, _, _, _, _ = cvxeda.cvxEDA(y_norm, 1.0 / fs, tau0=2.0, tau1=0.7,
                                                 delta_knot=delta_knot, alpha=alpha, gamma=gamma,
                                                 options={'reltol': 1e-6, 'show_progress': False})

    return np.asarray(tonic) * sigma + mu, np.asarray(phasic) * sigma
