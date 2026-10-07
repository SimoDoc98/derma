"""
Prepared recordings against the signals of the original MATLAB preprocessing
(local only, never in CI).

Needs ``DERMA_DATA_DIR`` with the archives and the prepared recordings
(``python -m derma.datasets prepare``), and ``DERMA_REFERENCE_RAW_DIR`` with the original
exported recordings (``CASE_Sxx.mat``, ``MAUS_Sxx.mat``, ``SAD_Sxx.mat``: ``signal`` at
4 Hz, ``events_names``, ``events_intervals``). Skipped otherwise.

Differences by construction:
- the first and last 10 s differ, because DERMA extends the signal linearly before
  resampling while MATLAB ``resample`` pads with zeros; they are not compared;
- MAUS and SAD were smoothed with a 3-sample moving average in MATLAB: the same smoothing
  is applied here before comparing;
- CASE: the reference signals are in volts (raw DAQ output) and are converted to uS with
  the dataset conversion (24 V - 49.2); DERMA uses the interpolated recordings in uS.
"""
import io
import os
import zipfile

import numpy as np
import pytest

from derma.datasets import case, maus, sad
from derma.decomposition import moving_average

DATA_DIR = os.environ.get('DERMA_DATA_DIR')
REFERENCE_DIR = os.environ.get('DERMA_REFERENCE_RAW_DIR')

pytestmark = pytest.mark.skipif(not (DATA_DIR and REFERENCE_DIR and os.path.isdir(os.path.join(DATA_DIR, 'prepared'))
                                     and os.path.isdir(REFERENCE_DIR)),
                                reason='DERMA_DATA_DIR (prepared) or DERMA_REFERENCE_RAW_DIR not available')

EDGE = 40  # samples (10 s at 4 Hz) excluded at each end

# Maximum |difference| relative to the signal range, interior samples
TOL_MAUS = 1e-9   # same data and filter: floating-point rounding only
TOL_SAD = 1e-4    # the original conversion rounded the frame average to integer ADC units
TOL_CASE = 1e-3   # raw DAQ samples vs the dataset's interpolated recordings


def _reference(name):
    from scipy.io import loadmat
    return loadmat(os.path.join(REFERENCE_DIR, name + '.mat'), squeeze_me=True)


def _relative_error(signal, reference):
    n = min(len(signal), len(reference))
    diff = np.abs(signal[EDGE:n - EDGE] - reference[EDGE:n - EDGE])
    return diff.max() / np.ptp(reference[:n])


def test_maus():
    recordings = maus.load(DATA_DIR)
    subjects = sorted({r['subject'] for r in recordings})
    for i, subject in enumerate(subjects, start=1):
        reference = _reference(f'MAUS_S{i:02d}')
        names = [str(n) for n in np.atleast_1d(reference['events_names'])]
        for r in [r for r in recordings if r['subject'] == subject]:
            start, stop = reference['events_intervals'][names.index(str(r['labels'][0]))]
            segment = reference['signal'][start - 1:stop]
            assert len(segment) == len(r['eda']), r['name']
            assert _relative_error(moving_average(r['eda'], 3), segment) < TOL_MAUS, r['name']


def test_sad():
    for r in sad.load(DATA_DIR):
        # Original numbering: drive17a and drive17b were subjects 17 and 18
        number = 18 if r['name'].endswith('_b') else int(r['subject'])
        reference = _reference(f'SAD_S{number:02d}')['signal']
        assert len(reference) == len(r['eda']), r['name']
        assert _relative_error(moving_average(r['eda'], 3), reference) < TOL_SAD, r['name']


def _annotation_labels(archive, subject, n_samples):
    """Condition windows from the subject's own annotation file, as in the MATLAB export."""
    with archive.open(f'CASE_full/data/interpolated/annotations/sub_{subject}.csv') as f:
        video = np.loadtxt(io.TextIOWrapper(f), delimiter=',', skiprows=1)[1:, 3]
    labels = np.full(n_samples, 'undefined', dtype='<U16')
    labels[case.REST_SAMPLES[0] - 1:case.REST_SAMPLES[1]] = 'rest'
    for video_id, condition in case.VIDEOS.items():
        idx = np.where(video == video_id)[0] + 1
        labels[int(np.ceil(idx[0] / 5)) - 1:int(np.floor(idx[-1] / 5))] = condition
    return labels


def test_case():
    archive = zipfile.ZipFile(os.path.join(DATA_DIR, case.INFO['archive']))
    for r in case.load(DATA_DIR):
        subject = int(r['subject'])
        reference_us = 24 * _reference(f'CASE_S{subject:02d}')['signal'] - 49.2
        assert _relative_error(r['eda'], reference_us) < TOL_CASE, r['name']

        # Labels: condition windows from the subject's annotation file
        own = _annotation_labels(archive, subject, len(r['eda']))
        emotional = np.isin(own, list(case.VIDEOS.values())) | np.isin(r['labels'], list(case.VIDEOS.values()))
        assert np.mean(own[emotional] == r['labels'][emotional]) > 0.999, r['name']
