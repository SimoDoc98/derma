"""Dataset loaders on small fake archives with the structure of the originals (invented values)."""
import os
import zipfile

import numpy as np
import pytest

from derma.datasets import case, maus, sad
from derma.datasets.__main__ import main
from derma.datasets.common import get_data_dir, load_prepared, resample, sha256


def _write_archive(path, members):
    with zipfile.ZipFile(path, 'w') as z:
        for name, content in members.items():
            z.writestr(name, content)
    return path


def _csv(header, columns):
    rows = [','.join(header)] + [','.join(f'{v:g}' for v in row) for row in zip(*columns)]
    return '\n'.join(rows) + '\n'


@pytest.fixture
def data_dir(tmp_path):
    return str(tmp_path)


def _accept_archive(module, data_dir, monkeypatch):
    """The fake archive has its own checksum: make the loader expect it."""
    monkeypatch.setitem(module.INFO, 'sha256', sha256(os.path.join(data_dir, module.INFO['archive'])))


# ---------------------------------------------------------------- MAUS

def _fake_maus(data_dir, seconds=70):
    n = int(256 * seconds)
    t = np.arange(n) / 256
    trial_header = [column for column, _ in maus.TRIALS]
    members = {
        'MAUS/Data/Raw_data/002/inf_resting.csv': _csv(['Resting_ECG', 'Resting_PPG', 'Resting_GSR'],
                                                       [np.zeros(n), np.zeros(n), 2.0 + 0.0 * t]),
        'MAUS/Data/Raw_data/002/inf_gsr.csv': _csv(trial_header, [3.0 + k + 0.0 * t for k in range(6)]),
        'MAUS/Data/Raw_data/003/inf_resting.csv': 'Resting_ECG,Resting_PPG,Resting_GSR\n',
        'MAUS/Data/Raw_data/003/inf_gsr.csv': _csv(trial_header, [1.0 + 0.0 * t] * 6),
    }
    _write_archive(os.path.join(data_dir, maus.INFO['archive']), members)


def test_maus_prepare(data_dir, monkeypatch, capsys):
    _fake_maus(data_dir)
    _accept_archive(maus, data_dir, monkeypatch)
    paths = maus.prepare(data_dir)

    names = sorted(os.path.basename(p)[:-4] for p in paths)
    assert len(names) == 13  # 7 + 6: empty rest of 003 skipped
    assert 'MAUS_002_rest' in names and 'MAUS_003_rest' not in names
    assert 'skipped' in capsys.readouterr().out

    recordings = {r['name']: r for r in maus.load(data_dir)}
    trial = recordings['MAUS_002_2back-1']
    assert trial['fs'] == 4.0 and len(trial['eda']) == 70 * 4
    assert set(trial['labels']) == {'2back-1'} and trial['subject'] == '002'
    # Constant 4 uS signal: preserved away from the filter edges
    np.testing.assert_allclose(trial['eda'][:], 4.0, rtol=1e-6)


# ---------------------------------------------------------------- SAD

def _wfdb(record, fs, signals, n_frames):
    """Header and 16-bit data of a fake record: signals = [(name, samples_per_frame, adc_values)]."""
    header = [f'{record} {len(signals)} {fs} {n_frames}']
    for name, spf, _ in signals:
        fmt = f'16x{spf}' if spf > 1 else '16'
        header.append(f'{record}.dat {fmt} 1000 16 0 0 0 0 {name}')
    frames = np.concatenate([np.tile(np.asarray(adc, dtype='<i2'), (n_frames, 1)) for _, _, adc in signals], axis=1)
    return '\n'.join(header) + '\n', frames.astype('<i2').tobytes()


def _fake_sad(data_dir):
    prefix = sad.RECORD_DIR
    members = {prefix + 'RECORDS': 'drive01\ndrive02\ndrive17a\n'}
    for record, channels in [('drive01', [('ECG', 1, [5]), ('hand GSR', 2, [1000, 3000])]),
                             ('drive02', [('ECG', 1, [5]), ('foot GSR', 2, [1000, 1000])]),
                             ('drive17a', [('hand GSr', 2, [4000, 4000]), ('ECG', 1, [5])])]:
        header, data = _wfdb(record, 15.5, channels, n_frames=int(15.5 * 120))
        members[prefix + record + '.hea'] = header
        members[prefix + record + '.dat'] = data
    _write_archive(os.path.join(data_dir, sad.INFO['archive']), members)


def test_sad_prepare(data_dir, monkeypatch, capsys):
    _fake_sad(data_dir)
    _accept_archive(sad, data_dir, monkeypatch)
    sad.prepare(data_dir)
    assert 'drive02: no hand GSR channel, skipped' in capsys.readouterr().out

    recordings = {r['name']: r for r in sad.load(data_dir)}
    assert sorted(recordings) == ['SAD_001', 'SAD_017_a']
    assert recordings['SAD_017_a']['subject'] == '017'
    # Frame average of 1000 and 3000 ADC units, gain 1000/uS -> 2 uS. The 8/31 polyphase
    # filter (15.5 -> 4 Hz) has a DC ripple of about 1e-5 between output phases.
    np.testing.assert_allclose(recordings['SAD_001']['eda'][:], 2.0, rtol=1e-4)
    np.testing.assert_allclose(recordings['SAD_017_a']['eda'][:], 4.0, rtol=1e-4)


def test_read_wfdb_channel_missing():
    header, data = _wfdb('rec', 15.5, [('ECG', 1, [1])], n_frames=10)
    signal, fs = sad.read_wfdb_channel(header, data, 'hand GSR')
    assert signal is None and fs == 15.5


# ---------------------------------------------------------------- CASE

def _fake_case(data_dir, seconds=60):
    n = seconds * 1000
    t = np.arange(n)
    video = np.full(n, 10.0)                       # start video
    video[(t >= 20000) & (t < 30000)] = 1          # amusing-1
    video[(t >= 30000) & (t < 40000)] = 11         # blue screen
    video[(t >= 40000) & (t < 50000)] = 3          # boring-1
    header = ['daqtime', 'ecg', 'bvp', 'gsr', 'rsp', 'skt', 'emg_zygo', 'emg_coru', 'emg_trap', 'video']
    members = {}
    for subject in (1, 2, 10):
        columns = [t] + [np.zeros(n)] * 2 + [np.full(n, 5.0 + subject)] + [np.zeros(n)] * 5 + [video]
        members[f'{case.PHYSIO_DIR}sub_{subject}.csv'] = _csv(header, columns)
    _write_archive(os.path.join(data_dir, case.INFO['archive']), members)


def test_case_prepare(data_dir, monkeypatch):
    _fake_case(data_dir)
    _accept_archive(case, data_dir, monkeypatch)
    case.prepare(data_dir)

    recordings = case.load(data_dir)
    assert [r['name'] for r in recordings] == ['CASE_001', 'CASE_002', 'CASE_010']
    rec = recordings[2]
    assert len(rec['eda']) == 60 * 4
    np.testing.assert_allclose(rec['eda'][:], 15.0, rtol=1e-6)

    labels = rec['labels']
    assert labels[0] == 'undefined'
    assert set(labels[80:120]) == {'amusement-1'}   # 20-30 s
    assert set(labels[160:200]) == {'boredom-1'}    # 40-50 s
    assert labels[130] == 'rest'                    # rest window from 30 s, outside the videos


# ---------------------------------------------------------------- common

def test_checksum_mismatch(data_dir):
    _fake_sad(data_dir)
    with pytest.raises(ValueError, match='ignore_checksum'):
        sad.prepare(data_dir)
    with pytest.warns(UserWarning, match='SHA-256'):
        sad.prepare(data_dir, ignore_checksum=True)
    assert len(load_prepared('SAD', data_dir)) == 2


def test_status(data_dir, capsys):
    assert not maus.status(data_dir)
    out = capsys.readouterr().out
    assert maus.INFO['page'] in out and maus.INFO['doi'] in out and maus.INFO['license'] in out
    _fake_maus(data_dir, seconds=5)
    assert maus.status(data_dir)


def test_data_dir_from_environment(monkeypatch, tmp_path):
    monkeypatch.setenv('DERMA_DATA_DIR', str(tmp_path))
    assert get_data_dir() == str(tmp_path)
    monkeypatch.delenv('DERMA_DATA_DIR')
    with pytest.raises(ValueError, match='DERMA_DATA_DIR'):
        get_data_dir()


def test_resample_rate():
    x = np.sin(2 * np.pi * 0.05 * np.arange(15500) / 15.5)
    assert len(resample(x, 15.5, 4.0)) == 4000
    assert len(resample(x[:1000], 1000.0, 4.0)) == 4


def test_command_line(data_dir, capsys):
    _fake_sad(data_dir)
    main(['status', 'sad', '--data-dir', data_dir])
    assert 'SAD: archive found' in capsys.readouterr().out
    with pytest.warns(UserWarning):
        main(['prepare', 'SAD', '--data-dir', data_dir, '--ignore-checksum'])
    assert 'SAD: 2 recordings prepared' in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main(['status', 'NOPE', '--data-dir', data_dir])
