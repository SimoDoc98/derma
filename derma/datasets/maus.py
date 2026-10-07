"""MAUS: mental workload on the N-back task, wearable GSR at 256 Hz (Beh et al., 2021).

Each subject has a resting recording and six N-back trials (0-, 2-, 3-back, twice),
recorded separately. Every condition becomes its own prepared recording
``MAUS_<subject>_<condition>`` at 4 Hz; conditions are never concatenated.

Steps:
1. For each subject folder in ``Data/Raw_data``, read ``Resting_GSR`` from
   ``inf_resting.csv`` (skipped if empty) and the six trial columns of ``inf_gsr.csv``.
2. Resample 256 Hz -> 4 Hz.
3. Save one recording per condition.
"""

import csv
import io

import numpy as np

from .common import FS_PREPARED, load_prepared, open_archive, print_status, resample, save_recording

INFO = dict(
    name='MAUS',
    archive='MAUS.zip',
    sha256='c15fb4f7b54ceb63fea46166215891cedadd444b02b9c17f29252087aa5be08f',
    page='https://ieee-dataport.org/open-access/maus-dataset-mental-workload-assessment-n-back-task-using-wearable-sensor',
    doi='10.21227/q4td-yd35',
    license='CC BY 4.0',
    citation=('Beh WK, Wu YH, Wu AY. MAUS: A Dataset for Mental Workload Assessment on N-back Task '
              'Using Wearable Sensor. arXiv:2111.02561 (2021). IEEE DataPort, doi:10.21227/q4td-yd35.'),
)

FS_RAW = 256.0  # Hz

RAW_DIR = 'MAUS/Data/Raw_data/'

# Columns of inf_gsr.csv, in recording order, and their condition names
TRIALS = [('Trial 1:0back', '0back-1'), ('Trial 2:2back', '2back-1'), ('Trial 3:3back', '3back-1'),
          ('Trial 4:2back', '2back-2'), ('Trial 5:3back', '3back-2'), ('Trial 6:0back', '0back-2')]


def _read_columns(archive, member):
    """Columns of a CSV member of the archive, as float arrays keyed by header."""
    with archive.open(member) as f:
        rows = list(csv.reader(io.TextIOWrapper(f, encoding='utf-8')))
    header, body = rows[0], [r for r in rows[1:] if r]
    return {name: np.array([float(r[i]) for r in body]) for i, name in enumerate(header)}


def status(data_dir=None):
    """Print whether the MAUS archive is available, and where to download it."""
    return print_status(INFO, data_dir)


def prepare(data_dir=None, ignore_checksum=False):
    """
    Prepare the MAUS recordings from the original archive.

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
    subjects = sorted({m.split('/')[3] for m in archive.namelist()
                       if m.startswith(RAW_DIR) and m.count('/') >= 4 and m.split('/')[3]})

    paths = []
    for subject in subjects:
        recordings = []

        member = f'{RAW_DIR}{subject}/inf_resting.csv'
        rest = _read_columns(archive, member).get('Resting_GSR', np.array([]))
        if len(rest) > 0:
            recordings.append(('rest', rest, member))
        else:
            print(f'  MAUS {subject}: empty resting recording, skipped')

        member = f'{RAW_DIR}{subject}/inf_gsr.csv'
        trials = _read_columns(archive, member)
        recordings += [(condition, trials[column], member) for column, condition in TRIALS]

        for condition, gsr, member in recordings:
            eda = resample(gsr, FS_RAW, FS_PREPARED)
            labels = np.full(len(eda), condition)
            paths.append(save_recording(data_dir, 'MAUS', f'MAUS_{subject}_{condition}', eda, subject, labels, member))
        print(f'  MAUS {subject}: {len(recordings)} recordings')

    return paths


def load(data_dir=None):
    """Prepared MAUS recordings (see :func:`derma.datasets.common.load_prepared`)."""
    return load_prepared('MAUS', data_dir)
