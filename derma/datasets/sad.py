"""SAD: stress in real-world driving, PhysioNet drivedb (Healey and Picard, 2005).

Each record is a drive with several signals in WFDB format; the hand GSR is used.
Records without a hand GSR channel are skipped. Drive 17 is split into two records
(``drive17a``, ``drive17b``): they become ``SAD_017_a`` and ``SAD_017_b``, same subject.

Steps:
1. Read the record list (``RECORDS``) and each header (``.hea``).
2. Find the hand GSR channel (name compared case-insensitively).
3. Read the signal file (``.dat``), average the samples of each frame and convert to
   physical units.
4. Resample from the frame frequency (15.5 Hz for most drives) to 4 Hz.
5. Save one recording per drive.
"""

import numpy as np

from .common import FS_PREPARED, load_prepared, open_archive, print_status, resample, save_recording

INFO = dict(
    name='SAD',
    archive='stress-recognition-in-automobile-drivers-1.0.0.zip',
    sha256='e381aa7c2c3fc69818158b17d396e1af43fadaabc62fab19add1699723cc5afb',
    page='https://physionet.org/content/drivedb/1.0.0/',
    doi='10.13026/C2SG6B',
    license='Open Data Commons Attribution License v1.0',
    citation=('Healey JA, Picard RW. Detecting stress during real-world driving tasks using physiological '
              'sensors. IEEE Transactions on Intelligent Transportation Systems 6(2), 156-166 (2005). '
              'Goldberger AL et al. PhysioBank, PhysioToolkit, and PhysioNet. Circulation 101(23), '
              'e215-e220 (2000).'),
)

RECORD_DIR = 'stress-recognition-in-automobile-drivers-1.0.0/'

GSR_CHANNEL = 'hand gsr'


def read_wfdb_channel(header_text, dat_bytes, channel):
    """
    One channel of a WFDB record stored in format 16 (16-bit samples, interleaved frames).

    Channels with several samples per frame are averaged within each frame, so the
    result is at the frame frequency.

    Parameters
    ----------
    header_text : str
        Content of the ``.hea`` file.
    dat_bytes : bytes
        Content of the ``.dat`` file (16-bit little-endian, all channels interleaved).
    channel : str
        Channel description, compared case-insensitively.

    Returns
    -------
    signal : numpy.ndarray or None
        Channel in physical units (None if the channel is not in the record).
    fs : float
        Frame frequency (Hz).
    """
    lines = [l for l in header_text.splitlines() if l.strip() and not l.startswith('#')]
    record_line = lines[0].split()
    n_signals, fs = int(record_line[1]), float(record_line[2].split('/')[0])

    samples_per_frame, gains, baselines, names = [], [], [], []
    for line in lines[1:1 + n_signals]:
        fields = line.split()
        fmt = fields[1]
        samples_per_frame.append(int(fmt.split('x')[1]) if 'x' in fmt else 1)
        # Gain field: "gain(baseline)/units"; baseline defaults to the ADC zero
        gain_field = fields[2].split('/')[0]
        gains.append(float(gain_field.split('(')[0]))
        baselines.append(float(gain_field.split('(')[1].rstrip(')')) if '(' in gain_field else float(fields[4]))
        names.append(' '.join(fields[8:]))

    matches = [i for i, name in enumerate(names) if name.lower() == channel.lower()]
    if not matches:
        return None, fs
    i = matches[0]

    frame_size = sum(samples_per_frame)
    samples = np.frombuffer(dat_bytes, dtype='<i2')
    frames = samples[:len(samples) // frame_size * frame_size].reshape(-1, frame_size)
    start = sum(samples_per_frame[:i])
    adc = frames[:, start:start + samples_per_frame[i]].astype(float).mean(axis=1)
    return (adc - baselines[i]) / gains[i], fs


def status(data_dir=None):
    """Print whether the SAD archive is available, and where to download it."""
    return print_status(INFO, data_dir)


def prepare(data_dir=None, ignore_checksum=False):
    """
    Prepare the SAD recordings from the original archive.

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
    records = archive.read(RECORD_DIR + 'RECORDS').decode().split()

    paths = []
    for record in records:
        header = archive.read(f'{RECORD_DIR}{record}.hea').decode()
        gsr, fs = read_wfdb_channel(header, archive.read(f'{RECORD_DIR}{record}.dat'), GSR_CHANNEL)
        if gsr is None:
            print(f'  SAD {record}: no hand GSR channel, skipped')
            continue

        # drive17a -> subject 017, part a
        subject = f'{int(record[5:7]):03d}'
        name = f'SAD_{subject}' + (f'_{record[7:]}' if len(record) > 7 else '')

        eda = resample(gsr, fs, FS_PREPARED)
        labels = np.full(len(eda), 'drive')
        paths.append(save_recording(data_dir, 'SAD', name, eda, subject, labels, f'{RECORD_DIR}{record}.dat'))
        print(f'  SAD {record}: {len(eda) / FS_PREPARED / 60:.1f} min')

    return paths


def load(data_dir=None):
    """Prepared SAD recordings (see :func:`derma.datasets.common.load_prepared`)."""
    return load_prepared('SAD', data_dir)
