"""
Equivalence with the original implementation (local only, never in CI).

Regenerates subjects of the original synthetic benchmark (30 min at 4 Hz) with the same
profile and seed, and compares every output with the reference files. Set
``DERMA_REFERENCE_DIR`` to the folder holding the reference ``synth_<profile>_S<id>.npz``
files; the tests are skipped otherwise.
"""
import os

import numpy as np
import pytest

from derma.generation import EDAGenerator, stable_seed
from derma.profiles import load_profile

REFERENCE_DIR = os.environ.get('DERMA_REFERENCE_DIR')

pytestmark = pytest.mark.skipif(not REFERENCE_DIR or not os.path.isdir(REFERENCE_DIR),
                                reason='DERMA_REFERENCE_DIR not set: reference benchmark not available')

# One subject per profile. The seed depends on the global subject id of the original
# benchmark: use these ids as they are, do not recompute them.
SUBJECTS = [('CASE_HighArousal', 1), ('CASE_LowArousal', 51), ('Literature', 101),
            ('MAUS_HighMWL', 201), ('MAUS_LowMWL', 251), ('SAD', 401)]

DURATION_SEC = 1800
FS = 4

# Measured differences: none with the SciPy version of the original runs (1.15),
# below 1e-15 (relative) with SciPy 1.18, from rounding in the convolutions.
# The tolerance leaves margin for other SciPy/NumPy versions while still catching
# any change in the random sequence or in the model.
RTOL = 1e-10
ATOL = 1e-12  # uS


@pytest.mark.parametrize('name, subject_id', SUBJECTS)
def test_benchmark_subject(name, subject_id):
    reference = np.load(os.path.join(REFERENCE_DIR, f'synth_{name}_S{subject_id:03d}.npz'))
    generator = EDAGenerator(load_profile(name), fs=FS)
    out = generator.generate(duration_sec=DURATION_SEC, seed=stable_seed(f'synth_benchmark_{subject_id}'))

    pairs = [('raw', 'eda_raw'), ('tonic', 'eda_tonic'), ('phasic', 'eda_phasic'),
             ('sna_driver', 'sna_driver'), ('theta', 'theta')]
    for key, ref_key in pairs:
        np.testing.assert_allclose(out[key], reference[ref_key], rtol=RTOL, atol=ATOL, err_msg=key)
