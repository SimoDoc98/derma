"""Estimation pipeline on synthetic signals (no dataset needed).

These tests check that the pipeline runs and returns valid profiles. They do not check that
the generative parameters are recovered: estimator and generator are known not to be
consistent yet.
"""
import numpy as np
import pytest

from derma.estimation import (NSSCR_FAMILIES, POSITIVE_FAMILIES, RECIPES, estimate_profile,
                              extract_segments, fit_best_distribution)
from derma.generation import EDAGenerator
from derma.profiles import DISTRIBUTION_FIELDS, DistributionParams, literature_profile, load_profile, save_profile

FS = 4


def test_recipes():
    assert RECIPES['MAUS_HighMWL']['onset_exclusion_sec'] == 40
    assert RECIPES['CASE_LowArousal']['onset_exclusion_sec'] == 0
    assert RECIPES['SAD']['conditions'] is None


def test_extract_segments_onset_and_chunking():
    # 100 s of 'rest', 400 s of 'task', 50 s of 'task' after a break
    labels = np.array(['rest'] * 400 + ['task'] * 1600 + ['rest'] * 40 + ['task'] * 200)
    signal = np.arange(len(labels), dtype=float)
    recipe = dict(conditions=['task'], min_segment_sec=60, max_segment_sec=300, onset_exclusion_sec=40)

    raws, tonics, phasics, artifacts = extract_segments(signal, signal, signal, labels, recipe, FS)

    # First block: 400 s - 40 s onset = 360 s -> chunks of 300 s and 60 s; second block too short
    assert [len(r) for r in raws] == [300 * FS, 60 * FS]
    assert raws[0][0] == 400 + 40 * FS
    assert all(not a.any() for a in artifacts)


def test_fit_best_distribution_families():
    rs = np.random.RandomState(0)
    nsscr = np.concatenate([np.zeros(5), rs.exponential(3.0, 60)])
    assert fit_best_distribution(nsscr, 'nsscr').dist_name in NSSCR_FAMILIES

    rise_time = rs.lognormal(1.0, 0.3, 60)
    best = fit_best_distribution(rise_time, 'scr_rise_time')
    assert best.dist_name in POSITIVE_FAMILIES and best.dist_name != 'halfnorm'
    assert best.bounds == (float(rise_time.min()), float(rise_time.max()))

    drift = rs.normal(-1e-3, 1e-3, 60)
    assert fit_best_distribution(drift, 'scl_drift').bounds[0] < 0


def test_estimate_profile_on_synthetic_recordings(tmp_path):
    generator = EDAGenerator(load_profile('MAUS_HighMWL'), fs=FS)
    recipe = dict(conditions=['task'], min_segment_sec=60, max_segment_sec=300, onset_exclusion_sec=40)
    labels = np.array(['rest'] * 60 * FS + ['task'] * 540 * FS)

    segments = [[], [], [], []]
    for seed in range(12):
        out = generator.generate(duration_sec=600, seed=seed)
        for collection, part in zip(segments, extract_segments(out['raw'], out['tonic'], out['phasic'],
                                                                labels, recipe, FS)):
            collection.extend(part)

    profile, features = estimate_profile(*segments, fs=FS)

    assert set(features) == set(DISTRIBUTION_FIELDS)
    for field in DISTRIBUTION_FIELDS:
        dist = getattr(profile, field)
        assert isinstance(dist, DistributionParams)
        assert np.all(np.isfinite(dist.bounds)) and dist.bounds[0] <= dist.bounds[1]
        assert dist.sample(5, random_state=np.random.RandomState(0)).shape == (5,)
    assert profile.scr_beta_var == 0.2

    save_profile(profile, str(tmp_path / 'estimated.json'))
    assert load_profile(str(tmp_path / 'estimated.json')) == profile


def test_empty_input_gives_literature_profile():
    profile, features = estimate_profile([], [], [])
    assert profile == literature_profile() and features == {}


@pytest.mark.parametrize('name', sorted(RECIPES))
def test_every_recipe_has_a_shipped_profile(name):
    load_profile(name)
