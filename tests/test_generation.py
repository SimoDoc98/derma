"""Generator checks that run without any dataset."""
import numpy as np
import pytest

from derma.generation import (EDAGenerator, crf_bach2010, fit_ode_kernel, ode3_impulse_response,
                              stable_seed)
from derma.profiles import available_profiles, load_profile

FS = 4


def test_stable_seed():
    # Values of the original implementation, used to seed the reference benchmark
    assert stable_seed('synth_benchmark_1') == 3238986315
    assert stable_seed('synth_benchmark_401') == 2730868254


def test_bach2010_kernel():
    # Canonical Bach (2010) shape at 20 Hz; reference values from the original implementation.
    # Loose tolerance: Nelder-Mead may stop at a slightly different point on other platforms.
    thetas, mse, t, h_target, h_ode = fit_ode_kernel(0.7013, 3.1487, 14.1257, 20)
    np.testing.assert_allclose(thetas, [5.3320695574674115, 7.402161339396784, 0.7780485481194469], rtol=1e-4)
    assert mse == pytest.approx(0.001540038433709768, rel=1e-3)
    assert len(t) == 1200
    # Positive coefficients: stable, non-oscillating system (real negative poles)
    assert np.all(thetas > 0)
    assert np.max(h_target) == pytest.approx(1.0) and np.max(h_ode) == pytest.approx(1.0)


def test_kernel_shapes_start_at_zero():
    t = np.linspace(0, 60, 240)
    crf = crf_bach2010(t, 1.0, 2.0, 4.0)
    h = ode3_impulse_response([1.0, 0.5, 0.25], t)
    assert crf[0] < 0.05 and h[0] == pytest.approx(0.0, abs=1e-12)


def test_same_seed_same_signal():
    generator = EDAGenerator(load_profile('Literature'), fs=FS)
    a = generator.generate(duration_sec=120, seed=42)
    b = generator.generate(duration_sec=120, seed=42)
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])
    c = generator.generate(duration_sec=120, seed=43)
    assert not np.array_equal(a['raw'], c['raw'])


@pytest.mark.parametrize('name', available_profiles())
def test_signal_properties(name):
    duration_sec = 300
    out = EDAGenerator(load_profile(name), fs=FS).generate(duration_sec=duration_sec, seed=stable_seed(name))
    n = duration_sec * FS

    for key in ('raw', 'tonic', 'phasic', 'sna_driver', 'artifacts'):
        assert out[key].shape == (n,)
    assert out['theta'].shape == (3,)

    # Conductance in uS: non-negative phasic, positive raw, raw = tonic + phasic
    assert np.all(out['phasic'] >= 0)
    assert np.min(out['raw']) > 0
    np.testing.assert_allclose(out['raw'], out['tonic'] + out['phasic'])
    assert not out['artifacts'].any()

    # Driver: no burst in the first 2 s, bursts at least 1 s apart, positive amplitudes
    onsets = np.nonzero(out['sna_driver'])[0]
    assert np.all(onsets >= 2 * FS)
    assert np.all(np.diff(onsets) >= FS)
    assert np.all(out['sna_driver'][onsets] > 0)
