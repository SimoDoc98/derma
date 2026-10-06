"""Shipped profiles must hold exactly the values of the original thesis profiles."""
import numpy as np
import pytest

from derma.profiles import (DISTRIBUTION_FIELDS, DistributionParams, EDAProfile, available_profiles,
                            literature_profile, load_profile, save_profile)

# (distribution, params, bounds) of the original profile files, copied as numbers
EXPECTED = {
    'Literature': {
        'scl_drift': ('uniform', (-0.01, 0.02), (-0.01, 0.01)),
        'nsscr': ('uniform', (1.0, 24.0), (1.0, 25.0)),
        'scr_amp_sd': ('uniform', (0.1, 0.4), (0.1, 0.5)),
        'scr_rise_time': ('uniform', (0.5, 2.0), (0.5, 2.5)),
        'scr_tau1': ('uniform', (0.5, 1.5), (0.5, 2.0)),
        'scr_tau_ratio': ('uniform', (2.0, 8.0), (2.0, 10.0)),
        'scr_scl_ratio': ('uniform', (0.1, 0.9), (0.1, 1.0)),
        'scr_beta_var': 0.2, 'scl_period_range': (120.0, 600.0),
    },
    'CASE_LowArousal': {
        'scl_drift': ('t', (2.171220512715399, -0.0004923749897344377, 0.0009704419366937587), (-0.003968499028568364, 0.009414419239227492)),
        'nsscr': ('expon', (0.0, 1.2349054232690597), (0.0, 12.674616695059626)),
        'scr_amp_sd': ('gompertz', (0.12745708946118883, 0.0, 0.40199862420014654), (0.08478067815303802, 1.4583724737167358)),
        'scr_rise_time': ('norm', (3.696098539041672, 1.6758218424366), (0.25, 10.0)),
        'scr_tau1': ('norm', (2.090997900633317, 1.0040581465471052), (0.25, 5.0)),
        'scr_tau_ratio': ('norm', (1.6273512009171676, 0.2599956220832801), (1.0, 2.6791726791726798)),
        'scr_scl_ratio': ('lognorm', (0.6872526407241821, 0.0, 0.009590505622327328), (0.0034090252593159676, 0.051645733416080475)),
        'scr_beta_var': 0.2, 'scl_period_range': (120.0, 600.0),
    },
    'CASE_HighArousal': {
        'scl_drift': ('laplace', (-0.00029171753281051515, 0.0015333945246329551), (-0.00579470379951054, 0.009898445567885815)),
        'nsscr': ('expon', (0.0, 1.486375296450448), (0.0, 9.182608695652174)),
        'scr_amp_sd': ('norm', (0.7179968720035893, 0.34286031841618914), (0.10438691824674606, 1.8782343864440918)),
        'scr_rise_time': ('gamma', (7.749728448079791, 0.0, 0.4694852288319654), (1.25, 7.75)),
        'scr_tau1': ('gamma', (5.448972882147729, 0.0, 0.4023494349603379), (0.25, 5.125)),
        'scr_tau_ratio': ('lognorm', (0.2039428393833935, 0.0, 1.666404533284625), (1.0, 3.666269841269841)),
        'scr_scl_ratio': ('lognorm', (0.5434016406139878, 0.0, 0.011160656488078837), (0.004381699021905661, 0.05289045348763466)),
        'scr_beta_var': 0.2, 'scl_period_range': (120.0, 600.0),
    },
    'MAUS_LowMWL': {
        'scl_drift': ('t', (2.3021025572208895, -0.0008444273178383304, 0.0013651557458352588), (-0.010244082474572374, 0.0021469051190861027)),
        'nsscr': ('expon', (0.0, 2.616770802505385), (0.0, 9.30232558139535)),
        'scr_amp_sd': ('gamma', (3.958189926748989, 0.0, 0.23289831423432736), (0.09455768764019012, 2.850858688354492)),
        'scr_rise_time': ('lognorm', (0.33847432038496816, 0.0, 3.230081600497641), (1.9464285714285714, 7.0)),
        'scr_tau1': ('norm', (1.6961053139169784, 0.45274189956910493), (0.5833333333333334, 2.859375)),
        'scr_tau_ratio': ('norm', (1.753522264451728, 0.2194853003344199), (1.0, 2.252487244897959)),
        'scr_scl_ratio': ('gamma', (2.6201007936873584, 0.0, 0.01544681662057783), (0.005430514924228191, 0.1005912497639656)),
        'scr_beta_var': 0.2, 'scl_period_range': (120.0, 600.0),
    },
    'MAUS_HighMWL': {
        'scl_drift': ('t', (2.3729896277882503, -0.0010325430154977358, 0.0015397963101508915), (-0.012564804397793047, 0.002434903033756059)),
        'nsscr': ('expon', (0.0, 4.056720389810051), (0.0, 13.048543689320388)),
        'scr_amp_sd': ('norm', (0.85203413143754, 0.3215385239703404), (0.11344674229621887, 1.930709958076477)),
        'scr_rise_time': ('gamma', (8.157365401839497, 0.0, 0.3592181008420154), (0.5, 6.95)),
        'scr_tau1': ('norm', (1.606508631768384, 0.4306001373076161), (0.25, 3.0961538461538463)),
        'scr_tau_ratio': ('lognorm', (0.13386479866801856, 0.0, 1.7896258635640736), (1.3575757575757574, 2.6492063492063496)),
        'scr_scl_ratio': ('lognorm', (0.6738436131456972, 0.0, 0.032455608445823604), (0.0036730205174535513, 0.15850988030433655)),
        'scr_beta_var': 0.2, 'scl_period_range': (120.0, 600.0),
    },
    'SAD': {
        'scl_drift': ('t', (1.089469235887178, -0.000581182379829325, 0.0043358784452042515), (-0.19238595923005167, 0.33266950257408034)),
        'nsscr': ('halfnorm', (0.0, 5.66992680100806), (0.0, 13.2)),
        'scr_amp_sd': ('gamma', (10.878679435149035, 0.0, 0.10210993885993958), (0.09127543866634369, 3.880544424057007)),
        'scr_rise_time': ('lognorm', (0.2695504025242032, 0.0, 2.5893159282539315), (0.5, 9.0)),
        'scr_tau1': ('gamma', (14.692016272183556, 0.0, 0.11394016718070692), (0.25, 3.75)),
        'scr_tau_ratio': ('lognorm', (0.1483728067490914, 0.0, 1.7656240854108), (1.0, 4.440476190476191)),
        'scr_scl_ratio': ('johnsonsb', (0.645954256430231, 0.8305591151724656, 0.0, 0.22878259865218992), (0.003120789770036936, 0.21740518510341644)),
        'scr_beta_var': 0.2, 'scl_period_range': (120.0, 600.0),
    },
}


def test_available_profiles():
    assert available_profiles() == sorted(EXPECTED)


@pytest.mark.parametrize('name', sorted(EXPECTED))
def test_shipped_profile_matches_original(name):
    profile = load_profile(name)
    for field in DISTRIBUTION_FIELDS:
        dist_name, params, bounds = EXPECTED[name][field]
        dist = getattr(profile, field)
        assert dist.dist_name == dist_name
        assert tuple(dist.params) == params
        assert tuple(dist.bounds) == bounds
    assert profile.scr_beta_var == EXPECTED[name]['scr_beta_var']
    assert tuple(profile.scl_period_range) == EXPECTED[name]['scl_period_range']


def test_literature_profile_matches_shipped_file():
    built, shipped = literature_profile(), load_profile('Literature')
    for field in DISTRIBUTION_FIELDS:
        assert getattr(built, field).dist_name == getattr(shipped, field).dist_name
        np.testing.assert_allclose(getattr(built, field).params, getattr(shipped, field).params)
        np.testing.assert_allclose(getattr(built, field).bounds, getattr(shipped, field).bounds)


def test_save_load_roundtrip(tmp_path):
    profile = load_profile('MAUS_HighMWL')
    path = tmp_path / 'copy.json'
    save_profile(profile, str(path))
    assert load_profile(str(path)) == profile


def test_sample_reproducible_and_within_bounds():
    dist = DistributionParams('norm', (0.0, 1.0), (-0.5, 0.5))
    a = dist.sample(200, random_state=np.random.RandomState(7))
    b = dist.sample(200, random_state=np.random.RandomState(7))
    np.testing.assert_array_equal(a, b)
    assert a.shape == (200,) and np.all((a >= -0.5) & (a <= 0.5))


def test_default_beta_var():
    profile = EDAProfile(*[DistributionParams('uniform', (0, 1), (0, 1))] * len(DISTRIBUTION_FIELDS))
    assert profile.scr_beta_var == 0.2


def test_unknown_profile():
    with pytest.raises(FileNotFoundError):
        load_profile('no_such_profile')
