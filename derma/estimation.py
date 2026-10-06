"""Parameter estimation from decomposed EDA and profile fitting (MLE, AIC).

Real recordings, already decomposed into tonic and phasic components, are cut into
segments of the conditions of interest. Each segment yields one value per model
parameter; the values of a population are summarized by the distribution family with
minimum AIC among a set of candidates.
"""

import warnings

import numpy as np
import scipy.stats as stats
from scipy.ndimage import label
from scipy.signal import find_peaks, peak_prominences

from .profiles import DistributionParams, EDAProfile, literature_profile

# Segmentation recipes of the reference profiles. Conditions are the event labels of the
# preprocessed recordings; segments shorter than min_segment_sec are discarded, longer
# ones are split into chunks of at most max_segment_sec.
# onset_exclusion_sec drops the start of each condition before segmentation: in MAUS the
# first ~40 s show elevated SCR activity at task onset, excluded to model the steady-state
# regime.
RECIPES = {
    'CASE_LowArousal': dict(dataset='CASE', conditions=['boredom-1', 'boredom-2', 'relaxation-1', 'relaxation-2'],
                            min_segment_sec=60.0, max_segment_sec=300.0, onset_exclusion_sec=0.0),
    'CASE_HighArousal': dict(dataset='CASE', conditions=['amusement-1', 'amusement-2', 'fear-1', 'fear-2'],
                             min_segment_sec=60.0, max_segment_sec=300.0, onset_exclusion_sec=0.0),
    'MAUS_LowMWL': dict(dataset='MAUS', conditions=['0back-1', '0back-2'],
                        min_segment_sec=60.0, max_segment_sec=300.0, onset_exclusion_sec=40.0),
    'MAUS_HighMWL': dict(dataset='MAUS', conditions=['2back-1', '2back-2', '3back-1', '3back-2'],
                         min_segment_sec=60.0, max_segment_sec=300.0, onset_exclusion_sec=40.0),
    'SAD': dict(dataset='SAD', conditions=None,
                min_segment_sec=60.0, max_segment_sec=300.0, onset_exclusion_sec=0.0),
}

# Candidate families for the parameters with positive support
POSITIVE_FAMILIES = ['lognorm', 'gamma', 'expon', 'norm', 'beta', 'gompertz', 'johnsonsb']
# NS.SCR keeps its zeros (segments without SCRs), so only families defined at zero
NSSCR_FAMILIES = ['expon', 'halfnorm', 'gamma', 'norm']
# SCL drift can be negative
REAL_FAMILIES = ['norm', 't', 'logistic', 'laplace']


def extract_segments(raw, tonic, phasic, labels, recipe, fs, artifact=None):
    """
    Cut a recording into the segments of the conditions listed in a recipe.

    Parameters
    ----------
    raw, tonic, phasic : numpy.ndarray
        Skin conductance and its components (uS).
    labels : numpy.ndarray
        Condition label of every sample.
    recipe : dict
        Entry of ``RECIPES`` (or a dict with the same keys).
    fs : float
        Sampling frequency (Hz).
    artifact : numpy.ndarray of bool, optional
        Artifact mask (True = artifact); all False if omitted.

    Returns
    -------
    raws, tonics, phasics, artifacts : list of numpy.ndarray
        One entry per segment.
    """
    if artifact is None:
        artifact = np.zeros(len(raw), dtype=bool)

    conditions = recipe['conditions']
    mask = np.ones(len(labels), dtype=bool) if conditions is None else np.isin(labels, conditions)

    min_len = int(recipe['min_segment_sec'] * fs)
    max_len = int(recipe['max_segment_sec'] * fs) if recipe.get('max_segment_sec') else None
    onset_len = int(recipe.get('onset_exclusion_sec', 0.0) * fs)

    blocks, n_blocks = label(mask)
    raws, tonics, phasics, artifacts = [], [], [], []
    for b in range(1, n_blocks + 1):
        idx = np.where(blocks == b)[0][onset_len:]
        if len(idx) < min_len:
            continue
        step = max_len if max_len is not None and len(idx) > max_len else len(idx)
        for start in range(0, len(idx), step):
            chunk = idx[start:start + step]
            if len(chunk) >= min_len:
                raws.append(raw[chunk])
                tonics.append(tonic[chunk])
                phasics.append(phasic[chunk])
                artifacts.append(artifact[chunk])

    return raws, tonics, phasics, artifacts


def extract_scr_features(phasic, fs, min_amplitude=0.05, min_distance_sec=0.5,
                         min_rise_time=0.1, max_rise_time=10.0):
    """
    Detect SCRs in a phasic component and measure amplitude and rise time.

    Parameters
    ----------
    phasic : numpy.ndarray
        Phasic component (uS).
    fs : float
        Sampling frequency (Hz).
    min_amplitude : float
        Minimum peak height (uS).
    min_distance_sec : float
        Minimum distance between consecutive peaks (s).
    min_rise_time, max_rise_time : float
        Physiological range of the rise time (s).

    Returns
    -------
    peaks, onsets : numpy.ndarray
        Sample indices of SCR peaks and onsets.
    amplitudes : numpy.ndarray
        Peak minus onset value (uS).
    rise_times : numpy.ndarray
        Onset-to-peak time (s).
    """
    peaks, _ = find_peaks(phasic, height=min_amplitude, distance=int(fs * min_distance_sec))
    if len(peaks) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    # Onset: left base of the peak prominence
    _, onsets, _ = peak_prominences(phasic, peaks)
    amplitudes = phasic[peaks] - phasic[onsets]
    rise_times = (peaks - onsets) / fs

    valid = (rise_times >= min_rise_time) & (rise_times <= max_rise_time)
    return peaks[valid], onsets[valid], amplitudes[valid], rise_times[valid]


def extract_features(raws, tonics, phasics, artifacts=None, fs=4):
    """
    One value of every model parameter per segment.

    Parameters
    ----------
    raws, tonics, phasics : list of numpy.ndarray
        Segments of skin conductance and its components (uS).
    artifacts : list of numpy.ndarray of bool, optional
        Artifact masks of the segments (True = artifact). SCRs whose peak or onset fall
        in an artifact are discarded; the drift is fitted on clean samples only.
    fs : float
        Sampling frequency (Hz).

    Returns
    -------
    dict of numpy.ndarray
        Finite values per parameter, keyed by the ``EDAProfile`` field names:
        ``scl_drift`` (uS/s), ``nsscr`` (events/min), ``scr_amp_sd`` (coefficient of
        variation of SCR amplitudes), ``scr_rise_time`` (s), ``scr_tau1`` (s),
        ``scr_tau_ratio`` and ``scr_scl_ratio`` (dimensionless).

    Notes
    -----
    tau_1 and tau_2 are measured as the time from the peak until the phasic falls below
    67% and 36.8% of the SCR amplitude (above the onset value), searched up to the next
    onset or 20 s; rho is the phasic over the tonic value at the peak. Segment values are
    means over the SCRs of the segment.
    """
    if artifacts is None:
        artifacts = [np.zeros_like(r, dtype=bool) for r in raws]

    values = {k: [] for k in ('scl_drift', 'nsscr', 'scr_amp_sd', 'scr_rise_time',
                              'scr_tau1', 'scr_tau_ratio', 'scr_scl_ratio')}

    # Means over segments without valid SCRs give NaN (dropped below): silence their warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        for raw, tonic, phasic, art in zip(raws, tonics, phasics, artifacts):
            tonic = np.asarray(tonic)
            phasic = np.asarray(phasic)
            art = np.asarray(art, dtype=bool)
            if len(art) != len(tonic):
                art = np.zeros(len(tonic), dtype=bool)
            valid_mask = ~art

            # At least 10 s of clean signal
            if np.sum(valid_mask) > fs * 10:
                time = np.arange(len(tonic)) / fs
                values['scl_drift'].append(np.polyfit(time[valid_mask], tonic[valid_mask], 1)[0])
            else:
                values['nsscr'].append(0)
                continue

            try:
                peaks, onsets, amps, rise_times = extract_scr_features(phasic, fs, min_amplitude=0.01)
                if len(peaks) == 0:
                    values['nsscr'].append(0)
                    continue

                # Discard SCRs with peak or onset inside an artifact
                keep = np.array([not (p >= len(art) or o >= len(art) or art[p] or art[o])
                                 for p, o in zip(peaks, onsets)], dtype=bool)
                peaks, onsets, amps, rise_times = peaks[keep], onsets[keep], amps[keep], rise_times[keep]

                clean_minutes = np.sum(valid_mask) / fs / 60.0
                values['nsscr'].append(len(peaks) / clean_minutes if clean_minutes > 0 else 0)
                values['scr_amp_sd'].append(np.nanstd(amps) / np.nanmean(amps) if np.nanmean(amps) > 0 else 0)

                tau1, tau_ratio, scl_ratio = [], [], []
                for p_idx, o_idx, amp in zip(peaks, onsets, amps):
                    # Decay window: up to the next onset, at most 20 s
                    next_onsets = onsets[onsets > p_idx]
                    seg_len = min(next_onsets[0] - p_idx, 20 * fs) if len(next_onsets) > 0 else 20 * fs
                    seg_len = int(np.clip(seg_len, 2, len(phasic) - p_idx))
                    seg = phasic[p_idx:p_idx + seg_len]

                    below_1 = np.where(seg < phasic[o_idx] + 0.67 * amp)[0]
                    below_2 = np.where(seg < phasic[o_idx] + 0.368 * amp)[0]
                    t1 = below_1[0] / fs if len(below_1) > 0 else np.nan
                    t2 = below_2[0] / fs if len(below_2) > 0 else np.nan
                    tau1.append(t1)
                    tau_ratio.append(t2 / t1 if t1 > 0 and t2 > 0 else np.nan)
                    scl_ratio.append(phasic[p_idx] / tonic[p_idx])

                values['scr_rise_time'].append(np.nanmean(rise_times))
                values['scr_tau1'].append(np.nanmean(tau1))
                values['scr_tau_ratio'].append(np.nanmean(tau_ratio))
                values['scr_scl_ratio'].append(np.nanmean(scl_ratio))

            except Exception:
                values['nsscr'].append(0)

    # No dtype cast: features keep the precision of the input signals
    features = {}
    for k, v in values.items():
        v = np.array(v)
        features[k] = v[np.isfinite(v)]
    return features


def fit_best_distribution(values, parameter):
    """
    Maximum-likelihood fit of the candidate families; the one with minimum AIC wins.

    Parameters
    ----------
    values : numpy.ndarray
        Segment values of one parameter.
    parameter : str
        ``EDAProfile`` field name: selects the candidate families and the data filter.

    Returns
    -------
    DistributionParams
        Best family, its parameters, and the observed range as bounds.
    """
    values = np.asarray(values)
    values = values[np.isfinite(values)]

    if parameter == 'scl_drift':
        families, fix_loc = REAL_FAMILIES, False
    elif parameter == 'nsscr':
        families, fix_loc = NSSCR_FAMILIES, True
    else:
        # Values at or near zero are measurement failures for positive parameters
        values = values[values > 0.001]
        families, fix_loc = POSITIVE_FAMILIES, True

    if len(values) < 2:
        return DistributionParams('norm', (0, 1), (-1, 1))

    best = DistributionParams('norm', (0, 1), (float(np.min(values)), float(np.max(values))))
    best_aic = np.inf
    for name in families:
        try:
            dist = getattr(stats, name)
            # Location fixed at 0 for positive parameters: more stable fits.
            # Poorly fitting candidates raise convergence warnings; AIC discards them anyway.
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                params = dist.fit(values, floc=0) if fix_loc and name != 'norm' else dist.fit(values)
            aic = 2 * len(params) + 2 * dist.nnlf(params, values)
            if aic < best_aic:
                best_aic = aic
                best = DistributionParams(name, params, best.bounds)
        except Exception:
            continue
    return best


def fit_profile(features):
    """
    Population profile from segment features.

    Parameters
    ----------
    features : dict of numpy.ndarray
        Output of :func:`extract_features`.

    Returns
    -------
    EDAProfile
        Best-fitting distribution of every parameter; sigma_beta fixed at 0.2.
    """
    return EDAProfile(**{name: fit_best_distribution(values, name) for name, values in features.items()},
                      scr_beta_var=0.2)


def estimate_profile(raws, tonics, phasics, artifacts=None, fs=4):
    """
    Estimate a population profile from decomposed segments.

    Parameters
    ----------
    raws, tonics, phasics : list of numpy.ndarray
        Segments of skin conductance and its components (uS), e.g. from
        :func:`extract_segments`.
    artifacts : list of numpy.ndarray of bool, optional
        Artifact masks of the segments.
    fs : float
        Sampling frequency (Hz).

    Returns
    -------
    profile : EDAProfile
        Fitted profile (the literature profile if no segment is given).
    features : dict of numpy.ndarray
        Segment values used for the fit, needed by the fit report.
    """
    if not raws:
        return literature_profile(), {}
    features = extract_features(raws, tonics, phasics, artifacts, fs)
    return fit_profile(features), features
