"""Fit report on a profile estimated from synthetic signals (no dataset needed)."""
import csv
import sys

import numpy as np
import pytest

from derma.estimation import estimate_profile, extract_segments
from derma.generation import EDAGenerator
from derma.profiles import load_profile
from derma.report import PARAMETER_LABELS, TABLE_COLUMNS, fit_table, kernel_check, plot_fit, write_fit_report

FS = 4


@pytest.fixture(scope='module')
def estimated():
    generator = EDAGenerator(load_profile('CASE_HighArousal'), fs=FS)
    recipe = dict(conditions=None, min_segment_sec=60, max_segment_sec=300, onset_exclusion_sec=0)
    segments = [[], [], [], []]
    for seed in range(10):
        out = generator.generate(duration_sec=600, seed=seed)
        labels = np.zeros(len(out['raw']))
        for collection, part in zip(segments, extract_segments(out['raw'], out['tonic'], out['phasic'],
                                                                labels, recipe, FS)):
            collection.extend(part)
    profile, features = estimate_profile(*segments, fs=FS)
    return profile, features


def test_fit_table(estimated):
    profile, features = estimated
    rows = fit_table(features, profile)
    assert [r['parameter'] for r in rows] == list(PARAMETER_LABELS)
    for r in rows:
        assert set(r) == set(TABLE_COLUMNS)
        assert r['distribution'] == getattr(profile, r['parameter']).dist_name
        if r['n'] >= 2:
            assert r['q1'] <= r['median'] <= r['q3']
            assert 0 <= r['ks_d'] <= 1 and 0 <= r['ks_p'] <= 1


def test_kernel_check(estimated):
    _, features = estimated
    check = kernel_check(features, FS)
    assert check is not None
    assert check['tau2_s'] >= check['tau1_s'] > 0
    assert len(check['t']) == len(check['h_target']) == len(check['h_ode'])
    assert check['mse'] >= 0


def test_kernel_check_without_scrs():
    assert kernel_check({'scr_rise_time': np.array([]), 'scr_tau1': np.array([]),
                         'scr_tau_ratio': np.array([])}) is None


def test_missing_matplotlib_gives_clear_error(estimated, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, 'matplotlib', None)
    monkeypatch.setitem(sys.modules, 'matplotlib.figure', None)
    with pytest.raises(ImportError, match=r"derma\[report\]"):
        plot_fit(estimated[1], estimated[0], str(tmp_path / 'fit.png'))


def test_write_fit_report(estimated, tmp_path):
    pytest.importorskip('matplotlib')
    profile, features = estimated
    paths = write_fit_report(features, profile, str(tmp_path), 'synthetic', fs=FS)

    assert [p.rsplit('.', 1)[1] for p in paths] == ['csv', 'md', 'png']
    with open(paths[0], newline='') as f:
        assert len(list(csv.DictReader(f))) == len(PARAMETER_LABELS)
    with open(paths[1]) as f:
        assert 'KS D' in f.read()
    with open(paths[2], 'rb') as f:
        assert f.read(8) == b'\x89PNG\r\n\x1a\n'
