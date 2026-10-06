"""Decomposition backend: moving average and cvxEDA wrapper."""
import sys

import numpy as np
import pytest

from derma.decomposition import decompose, moving_average
from derma.generation import EDAGenerator
from derma.profiles import load_profile


def test_moving_average_matches_matlab_movmean():
    # Examples of the MATLAB movmean documentation (odd and even window)
    a = np.array([4, 8, 6, -1, -2, -3, -1, 3, 4, 5], dtype=float)
    np.testing.assert_allclose(moving_average(a, 3),
                               [6, 6, 13 / 3, 1, -2, -2, -1 / 3, 2, 4, 4.5])
    np.testing.assert_allclose(moving_average(a, 4),
                               [6, 6, 4.25, 2.75, 0, -1.75, -0.75, 0.75, 2.75, 4])
    np.testing.assert_array_equal(moving_average(a, 1), a)


def test_missing_cvxeda_gives_clear_error(monkeypatch):
    monkeypatch.setitem(sys.modules, 'cvxeda', None)
    with pytest.raises(ImportError, match=r"derma\[cvxeda\]"):
        decompose(np.ones(100), 4, alpha=8e-3, gamma=1e-1, delta_knot=10)


def test_decompose_synthetic_signal():
    pytest.importorskip('cvxeda')
    fs = 4
    raw = EDAGenerator(load_profile('CASE_HighArousal'), fs=fs).generate(duration_sec=300, seed=1)['raw']
    tonic, phasic = decompose(raw, fs, alpha=8e-3, gamma=1e-1, delta_knot=10)

    assert tonic.shape == raw.shape and phasic.shape == raw.shape
    # Phasic is a non-negative driver convolved with the SCR kernel
    assert np.min(phasic) > -1e-6
    # The two components explain the (smoothed) signal up to a small residual (uS)
    residual = moving_average(raw, 3) - tonic - phasic
    assert np.std(residual) < 0.1 * np.std(raw)
