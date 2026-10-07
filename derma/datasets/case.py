"""CASE: continuously annotated signals of emotion, GSR at 1000 Hz (Sharma et al., 2019).

Each subject watched eight emotional videos (two amusing, two boring, two relaxing,
two scary) in an individual order, interleaved with blue screens. The interpolated
physiological files of the dataset give the GSR in uS and the video ID of every sample;
each subject becomes one prepared recording ``CASE_<subject>`` at 4 Hz, with the
condition of every sample in ``labels``.

Steps:
1. For each subject, read ``gsr`` (uS) and ``video`` from
   ``data/interpolated/physiological/sub_<n>.csv``.
2. Resample 1000 Hz -> 4 Hz.
3. Label the conditions: ``rest`` from 30 to 200 s (start of the session), then each
   emotional video from its first to its last sample.
4. Save one recording per subject.
"""

import io
import re

import numpy as np

from .common import FS_PREPARED, load_prepared, open_archive, print_status, resample, save_recording

INFO = dict(
    name='CASE',
    archive='CASE_full.zip',
    sha256='03818aa2e017e53ca7f7c9a39e9268ec6eeb7afd5ee245d5745f258e9e3c554e',
    page='https://springernature.figshare.com/articles/dataset/CASE_Dataset-full/8869157',
    doi='10.6084/m9.figshare.8869157',
    license='CC0 1.0',
    citation=('Sharma K, Castellini C, van den Broek EL, Albu-Schaeffer A, Schwenker F. A dataset of '
              'continuous affect annotations and physiological signals for emotion analysis. '
              'Scientific Data 6, 196 (2019). doi:10.1038/s41597-019-0209-0'),
)

FS_RAW = 1000.0  # Hz

PHYSIO_DIR = 'CASE_full/data/interpolated/physiological/'

# Video IDs of the emotional videos (metadata/videos_duration_num) and condition names
VIDEOS = {1: 'amusement-1', 2: 'amusement-2', 3: 'boredom-1', 4: 'boredom-2',
          5: 'relaxation-1', 6: 'relaxation-2', 7: 'fear-1', 8: 'fear-2'}

# Resting baseline: samples 121-800 at 4 Hz, i.e. 30-200 s from the start of the session
REST_SAMPLES = (121, 800)


def condition_labels(video, n_samples):
    """
    Condition of every 4 Hz sample, from the video ID of every 1000 Hz sample.

    Parameters
    ----------
    video : numpy.ndarray
        Video ID at 1000 Hz.
    n_samples : int
        Length of the 4 Hz recording.

    Returns
    -------
    numpy.ndarray of str
        ``rest``, a condition name, or ``undefined``.
    """
    decimation = int(FS_RAW / FS_PREPARED)
    labels = np.full(n_samples, 'undefined', dtype='<U16')
    labels[REST_SAMPLES[0] - 1:REST_SAMPLES[1]] = 'rest'
    for video_id, condition in VIDEOS.items():
        idx = np.where(video == video_id)[0] + 1  # 1-based, as the original windows
        if len(idx) == 0:
            continue
        # First and last 4 Hz samples fully inside the video
        start, stop = int(np.ceil(idx[0] / decimation)), int(np.floor(idx[-1] / decimation))
        labels[start - 1:stop] = condition
    return labels


def _subject_number(member):
    """Subject number of a ``sub_<n>.csv`` archive member."""
    return int(re.search(r'sub_(\d+)', member).group(1))


def status(data_dir=None):
    """Print whether the CASE archive is available, and where to download it."""
    return print_status(INFO, data_dir)


def prepare(data_dir=None, ignore_checksum=False):
    """
    Prepare the CASE recordings from the original archive.

    Parameters
    ----------
    data_dir : str, optional
        Data folder (default: ``DERMA_DATA_DIR``).
    ignore_checksum : bool
        Go on even if the archive SHA-256 does not match.

    Returns
    -------
    list of str
        Paths of the prepared recordings.
    """
    archive = open_archive(INFO, data_dir, ignore_checksum)
    members = [m for m in archive.namelist() if re.fullmatch(re.escape(PHYSIO_DIR) + r'sub_\d+\.csv', m)]
    # Numeric order of the subjects (sub_2 before sub_10)
    members.sort(key=_subject_number)

    paths = []
    for member in members:
        subject = f'{_subject_number(member):03d}'
        with archive.open(member) as f:
            text = io.TextIOWrapper(f, encoding='utf-8')
            header = text.readline().strip().split(',')
            columns = np.loadtxt(text, delimiter=',', usecols=(header.index('gsr'), header.index('video')))
        gsr, video = columns[:, 0], columns[:, 1]

        eda = resample(gsr, FS_RAW, FS_PREPARED)
        labels = condition_labels(video, len(eda))
        paths.append(save_recording(data_dir, 'CASE', f'CASE_{subject}', eda, subject, labels, member))
        print(f'  CASE {subject}: {len(eda) / FS_PREPARED / 60:.1f} min')

    return paths


def load(data_dir=None):
    """Prepared CASE recordings (see :func:`derma.datasets.common.load_prepared`)."""
    return load_prepared('CASE', data_dir)
