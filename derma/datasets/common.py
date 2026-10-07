"""Shared helpers of the dataset loaders: data folder, archive check, resampling, storage.

Layout of the data folder (``data_dir`` argument or ``DERMA_DATA_DIR`` environment
variable)::

    data/
    ├── <original archive of each dataset, as downloaded>
    └── prepared/<DATASET>/<DATASET>_<subject>[_<condition>].npz

Each prepared file holds one recording: ``eda`` (uS), ``fs`` (Hz), ``subject``,
``labels`` (condition of every sample), ``dataset`` and ``source`` (archive member).
"""

import hashlib
import os
import warnings
import zipfile
from fractions import Fraction

import numpy as np
from scipy.signal import resample_poly

DATA_DIR_ENV = 'DERMA_DATA_DIR'

# Sampling frequency of every prepared recording (Hz)
FS_PREPARED = 4.0


def get_data_dir(data_dir=None):
    """
    Data folder: the argument if given, otherwise ``DERMA_DATA_DIR``.

    Parameters
    ----------
    data_dir : str, optional
        Path of the data folder.

    Returns
    -------
    str
        Path of the data folder.
    """
    data_dir = data_dir or os.environ.get(DATA_DIR_ENV)
    if not data_dir:
        raise ValueError(f'No data folder: pass data_dir or set the {DATA_DIR_ENV} environment variable.')
    return data_dir


def sha256(path):
    """SHA-256 hex digest of a file, read in chunks."""
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 22), b''):
            digest.update(chunk)
    return digest.hexdigest()


def print_status(info, data_dir=None):
    """
    Print whether the archive of a dataset is in the data folder, and where to get it.

    Parameters
    ----------
    info : dict
        ``name``, ``archive``, ``page``, ``doi``, ``license`` and ``citation`` of the dataset.
    data_dir : str, optional
        Data folder.

    Returns
    -------
    bool
        True if the archive is present.
    """
    data_dir = get_data_dir(data_dir)
    path = os.path.join(data_dir, info['archive'])
    prepared_dir = os.path.join(data_dir, 'prepared', info['name'])
    n_prepared = len([f for f in os.listdir(prepared_dir) if f.endswith('.npz')]) if os.path.isdir(prepared_dir) else 0

    present = os.path.isfile(path)
    print(f"{info['name']}: archive {'found' if present else 'NOT found'} ({path}); "
          f"{n_prepared} prepared recordings")
    if not present:
        print(f"  Download page: {info['page']}\n  DOI: {info['doi']}\n  License: {info['license']}\n"
              f"  Citation: {info['citation']}\n  Place {info['archive']} in {data_dir}, unmodified.")
    return present


def open_archive(info, data_dir=None, ignore_checksum=False):
    """
    Open the original archive of a dataset after checking its SHA-256.

    Parameters
    ----------
    info : dict
        ``archive`` file name and expected ``sha256`` of the dataset.
    data_dir : str, optional
        Data folder.
    ignore_checksum : bool
        Go on with a warning when the checksum does not match.

    Returns
    -------
    zipfile.ZipFile
        The open archive.
    """
    path = os.path.join(get_data_dir(data_dir), info['archive'])
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{path} not found. Download it from {info['page']}.")

    digest = sha256(path)
    if digest != info['sha256']:
        message = (f"{info['archive']}: SHA-256 {digest} differs from the expected {info['sha256']}. "
                   'The archive may be corrupted or a different version of the dataset.')
        if not ignore_checksum:
            raise ValueError(message + ' Use ignore_checksum=True (--ignore-checksum) to go on anyway.')
        warnings.warn(message + ' Going on as requested.')
    return zipfile.ZipFile(path)


def resample(signal, fs_in, fs_out=FS_PREPARED):
    """
    Polyphase resampling with a Kaiser-windowed anti-aliasing filter.

    The filter (Kaiser window, beta = 5, length proportional to the rate ratio) is the
    one of MATLAB ``resample``. The signal is extended linearly beyond its ends: padding
    with zeros would pull the first and last seconds towards zero, a spurious transient
    at the start of every recording.

    Parameters
    ----------
    signal : numpy.ndarray
        Input signal.
    fs_in, fs_out : float
        Input and output sampling frequencies (Hz).

    Returns
    -------
    numpy.ndarray
        Resampled signal.
    """
    ratio = Fraction(fs_out / fs_in).limit_denominator(1000)
    return resample_poly(np.asarray(signal, dtype=float), ratio.numerator, ratio.denominator,
                         window=('kaiser', 5.0), padtype='line')


def save_recording(data_dir, dataset, name, eda, subject, labels, source, fs=FS_PREPARED):
    """
    Save one prepared recording as ``prepared/<dataset>/<name>.npz``.

    Parameters
    ----------
    data_dir : str
        Data folder.
    dataset : str
        Dataset name (e.g. ``'MAUS'``).
    name : str
        Recording name, ``<DATASET>_<subject>[_<condition>]``.
    eda : numpy.ndarray
        Skin conductance (uS).
    subject : str
        Subject code (three digits).
    labels : numpy.ndarray of str
        Condition of every sample.
    source : str
        Archive member the recording comes from.
    fs : float
        Sampling frequency (Hz).

    Returns
    -------
    str
        Path of the saved file.
    """
    folder = os.path.join(get_data_dir(data_dir), 'prepared', dataset)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, name + '.npz')
    np.savez_compressed(path, eda=np.asarray(eda, dtype=float), fs=float(fs), subject=str(subject),
                        labels=np.asarray(labels, dtype=str), dataset=dataset, source=source)
    return path


def load_prepared(dataset, data_dir=None):
    """
    Read back the prepared recordings of a dataset.

    Parameters
    ----------
    dataset : str
        Dataset name (``'CASE'``, ``'MAUS'``, ``'SAD'``).
    data_dir : str, optional
        Data folder.

    Returns
    -------
    list of dict
        One dict per recording, sorted by name: ``name``, ``eda`` (uS), ``fs`` (Hz),
        ``subject``, ``labels``, ``dataset``, ``source``.
    """
    folder = os.path.join(get_data_dir(data_dir), 'prepared', dataset)
    if not os.path.isdir(folder):
        raise FileNotFoundError(f'No prepared {dataset} recordings in {folder}: run prepare first.')

    recordings = []
    for f in sorted(os.listdir(folder)):
        if f.endswith('.npz'):
            with np.load(os.path.join(folder, f)) as z:
                recordings.append(dict(name=f[:-4], eda=z['eda'], fs=float(z['fs']), subject=str(z['subject']),
                                       labels=z['labels'], dataset=str(z['dataset']), source=str(z['source'])))
    return recordings
