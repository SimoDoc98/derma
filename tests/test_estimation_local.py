"""
Re-estimation of the reference profiles from the original preprocessed recordings
(local only, never in CI).

Set ``DERMA_PROCESSED_DIR`` to the folder with the original ``<DATASET>_S<id>.npz`` files
(``eda_raw``, ``eda_tonic``, ``eda_phasic``, ``event_label``, ``artifact``); the tests are
skipped otherwise. The original estimation used the manual artifact masks stored in those
files and no onset exclusion, so ``onset_exclusion_sec`` is set to 0 here.
"""
import os

import numpy as np
import pytest

from derma.estimation import RECIPES, estimate_profile, extract_segments
from derma.profiles import DISTRIBUTION_FIELDS, load_profile

PROCESSED_DIR = os.environ.get('DERMA_PROCESSED_DIR')

pytestmark = pytest.mark.skipif(not PROCESSED_DIR or not os.path.isdir(PROCESSED_DIR),
                                reason='DERMA_PROCESSED_DIR not set: original recordings not available')

FS = 4
# Measured differences: none with the SciPy version of the original runs (1.15),
# about 1e-14 (relative) with SciPy 1.18. The tolerance leaves margin for other versions
# of the iterative MLE fits while still catching any change in features or families.
RTOL = 1e-9


@pytest.mark.parametrize('name', sorted(RECIPES))
def test_reestimated_profile_matches_frozen(name):
    recipe = dict(RECIPES[name], onset_exclusion_sec=0.0)
    files = sorted(f for f in os.listdir(PROCESSED_DIR) if f.startswith(recipe['dataset'] + '_') and f.endswith('.npz'))

    raws, tonics, phasics, artifacts = [], [], [], []
    for f in files:
        with np.load(os.path.join(PROCESSED_DIR, f), allow_pickle=True) as z:
            segments = extract_segments(z['eda_raw'], z['eda_tonic'], z['eda_phasic'], z['event_label'],
                                        recipe, FS, artifact=z['artifact'])
        for collection, part in zip((raws, tonics, phasics, artifacts), segments):
            collection.extend(part)

    profile, _ = estimate_profile(raws, tonics, phasics, artifacts, fs=FS)
    frozen = load_profile(name)
    for field in DISTRIBUTION_FIELDS:
        new, ref = getattr(profile, field), getattr(frozen, field)
        assert new.dist_name == ref.dist_name, field
        np.testing.assert_allclose(new.params, ref.params, rtol=RTOL, err_msg=field)
        np.testing.assert_allclose(new.bounds, ref.bounds, rtol=RTOL, err_msg=field)
