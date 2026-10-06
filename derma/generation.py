"""Synthetic EDA generation: SCR kernel calibration and tonic/phasic synthesis.

The phasic component is a sparse sudomotor driver convolved with the impulse response
of a third-order ODE (Bach et al., 2011), whose poles are calibrated so that the
response matches the canonical SCR shape of Bach et al. (2010). The tonic component
is a linear drift plus a slow oscillation, raised to a level set by the SCR/SCL ratio.

References
----------
Bach DR, Flandin G, Friston KJ, Dolan RJ (2010). Modelling event-related skin
conductance responses. Int J Psychophysiol 75(3):349-356.
Bach DR, Daunizeau J, Kuelzow N, Friston KJ, Dolan RJ (2011). Dynamic causal modeling
of spontaneous fluctuations in skin conductance. Psychophysiology 48(2):252-257.
"""

import hashlib

import numpy as np
from scipy import signal
from scipy.optimize import minimize


def stable_seed(key, n_bits=32):
    """
    Deterministic seed derived from a string key.

    Python's ``hash()`` is salted per process, so it cannot seed reproducible data.
    A BLAKE2b digest gives the same seed on every run, machine and Python version.

    Parameters
    ----------
    key : str
        Any string identifying the item to seed (e.g. ``'synth_benchmark_42'``).
    n_bits : int
        Width of the returned seed in bits.

    Returns
    -------
    int
        Seed in ``[0, 2**n_bits)``.
    """
    digest = hashlib.blake2b(key.encode('utf-8'), digest_size=8).digest()
    return int.from_bytes(digest, 'big') % (2 ** n_bits)


# =============================================================================
# SCR kernel: Bach (2010) canonical shape -> Bach (2011) third-order ODE
# =============================================================================

def crf_bach2010(t, rise_time_s, tau1_s, tau2_s):
    """
    Canonical SCR shape: Gaussian diffusion convolved with a bi-exponential decay.

    Parameters
    ----------
    t : numpy.ndarray
        Time axis (s), starting at 0.
    rise_time_s : float
        Rise time (s), used as the standard deviation of the Gaussian diffusion.
    tau1_s, tau2_s : float
        Fast and slow decay time constants (s).

    Returns
    -------
    numpy.ndarray
        SCR shape normalized to unit peak.
    """
    sigma = rise_time_s

    # Gaussian sweat diffusion, centred at 2.2 sigma so that it starts near zero at t = 0
    diffusion = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((t - sigma * 2.2) / sigma) ** 2)
    diffusion /= np.max(diffusion)

    # Bi-exponential recovery (sweat reabsorption)
    decay = np.exp(-t / tau1_s) + np.exp(-t / tau2_s)

    crf = signal.convolve(diffusion, decay, mode='full')[:len(t)]
    crf /= np.max(crf)
    return crf


def ode3_impulse_response(poles, t):
    """
    Impulse response of the third-order ODE x''' + theta1 x'' + theta2 x' + theta3 x = u.

    The system is parameterized by three positive rates p (poles at -p), so that it is
    stable and non-oscillatory by construction; the thetas follow from Vieta's formulas.

    Parameters
    ----------
    poles : array_like
        The three rates p1, p2, p3 (1/s).
    t : numpy.ndarray
        Time axis (s).

    Returns
    -------
    numpy.ndarray
        Impulse response normalized to unit peak (zeros if the system cannot be built).
    """
    p1, p2, p3 = poles
    theta1 = p1 + p2 + p3
    theta2 = p1 * p2 + p1 * p3 + p2 * p3
    theta3 = p1 * p2 * p3

    try:
        system = signal.lti([1], [1, theta1, theta2, theta3])
        _, y = signal.impulse(system, T=t)
        if np.max(y) > 0:
            y /= np.max(y)
        return y
    except Exception:
        return np.zeros_like(t)


def fit_ode_kernel(rise_time_s, tau1_s, tau2_s, fs):
    """
    Calibrate the ODE poles so that its impulse response matches the Bach (2010) shape.

    The error is computed after aligning the peaks of the two responses, so that the
    fit targets the SCR morphology and not its latency.

    Parameters
    ----------
    rise_time_s : float
        Target rise time (s).
    tau1_s, tau2_s : float
        Target fast and slow decay time constants (s).
    fs : float
        Sampling frequency (Hz).

    Returns
    -------
    thetas : numpy.ndarray
        ODE coefficients [theta1, theta2, theta3].
    mse : float
        Mean squared error between the aligned responses (unit-peak scale).
    t : numpy.ndarray
        Time axis of the fit (s), 60 s long.
    h_target : numpy.ndarray
        Target Bach (2010) shape.
    h_ode : numpy.ndarray
        ODE impulse response with the fitted poles, not aligned.
    """
    duration = 60.0
    t = np.linspace(0, duration, int(duration * fs))
    h_target = crf_bach2010(t, rise_time_s, tau1_s, tau2_s)
    idx_target = np.argmax(h_target)

    def cost(poles):
        h_ode = ode3_impulse_response(poles, t)
        shift = idx_target - np.argmax(h_ode)
        if shift > 0:
            h_aligned = np.pad(h_ode, (shift, 0), mode='constant')[:len(t)]
        else:
            h_aligned = np.pad(h_ode[-shift:], (0, -shift), mode='constant')
        return np.mean((h_target - h_aligned) ** 2)

    # Start from the inverse of the target time constants
    x0 = [1.0 / rise_time_s, 1.0 / tau1_s, 1.0 / tau2_s]
    res = minimize(cost, x0, bounds=[(1e-5, 100)] * 3, method='Nelder-Mead', tol=1e-4)

    p1, p2, p3 = res.x
    thetas = np.array([p1 + p2 + p3,
                       p1 * p2 + p1 * p3 + p2 * p3,
                       p1 * p2 * p3])
    return thetas, res.fun, t, h_target, ode3_impulse_response(res.x, t)


# =============================================================================
# Generator
# =============================================================================

class EDAGenerator:
    """
    Synthetic EDA from a population profile.

    Parameters
    ----------
    profile : derma.profiles.EDAProfile
        Distributions of the model parameters.
    fs : float
        Sampling frequency of the generated signals (Hz).
    """

    def __init__(self, profile, fs=4):
        self.profile = profile
        self.fs = fs

    def _generate_pulse_train(self, n_events, duration_sec, random_state, refractory_period=1.0):
        """
        Onset samples of the sudomotor bursts, at least ``refractory_period`` apart.

        Parameters
        ----------
        n_events : int
            Number of bursts to place.
        duration_sec : float
            Signal duration (s).
        random_state : numpy.random.RandomState
            Source of randomness.
        refractory_period : float
            Minimum interval between two bursts (s).

        Returns
        -------
        numpy.ndarray
            Sorted sample indices of the bursts (fewer than ``n_events`` if the
            signal is too crowded to place them all).
        """
        total_samples = int(duration_sec * self.fs)
        min_distance = int(refractory_period * self.fs)

        if n_events <= 0:
            return np.array([], dtype=int)

        impulses = []
        # Give up after 50 attempts per event when the signal is too crowded
        max_attempts = n_events * 50
        attempts = 0
        while len(impulses) < n_events and attempts < max_attempts:
            attempts += 1
            # No burst in the first 2 s
            candidate = random_state.randint(low=int(2 * self.fs), high=total_samples)
            if not any(abs(candidate - idx) < min_distance for idx in impulses):
                impulses.append(candidate)

        return np.array(sorted(impulses))

    def generate(self, duration_sec=60, seed=None):
        """
        Generate one synthetic subject.

        Parameters
        ----------
        duration_sec : float
            Signal duration (s).
        seed : int, optional
            Seed of the subject: the same seed and profile give the same signal.

        Returns
        -------
        dict
            ``raw``, ``tonic``, ``phasic``: signals (uS); ``sna_driver``: amplitudes of
            the main driver at the burst samples (uS); ``theta``: ODE coefficients of
            the SCR kernel; ``artifacts``: boolean mask (all False).
        """
        rs = np.random.RandomState(seed)

        total_samples = int(duration_sec * self.fs)
        time_axis = np.arange(total_samples) / self.fs

        # --- 1. Subject parameters (the order of the draws defines the sequence) ---
        p_scl_drift = self.profile.scl_drift.sample(random_state=rs)[0]
        p_scl_period = rs.uniform(self.profile.scl_period_range[0], self.profile.scl_period_range[1])
        # Oscillation amplitude A_osc (uS), not fitted: uniform in [0.01, 0.5] (Greco et al., 2016)
        p_scl_amp = rs.uniform(0.01, 0.5)

        p_nsscr = self.profile.nsscr.sample(random_state=rs)[0]
        p_amp_sd = self.profile.scr_amp_sd.sample(random_state=rs)[0]

        p_scr_rise = self.profile.scr_rise_time.sample(random_state=rs)[0]
        p_scr_tau1 = self.profile.scr_tau1.sample(random_state=rs)[0]
        p_scr_tau2 = p_scr_tau1 * self.profile.scr_tau_ratio.sample(random_state=rs)[0]

        p_scr_scl_ratio = self.profile.scr_scl_ratio.sample(random_state=rs)[0]
        p_beta_var = self.profile.scr_beta_var

        # --- 2. SCR kernels ---
        theta_opt, _, _, _, _ = fit_ode_kernel(p_scr_rise, p_scr_tau1, p_scr_tau2, self.fs)
        try:
            sys_ode = signal.lti([1], [1, theta_opt[0], theta_opt[1], theta_opt[2]])
        except Exception:
            # Canonical coefficients of Bach et al. (2011) if the fit fails
            sys_ode = signal.lti([1], [1, 2.1594, 3.9210, 0.9236])

        kernel_len = 60.0
        t_kernel = np.linspace(0, kernel_len, int(kernel_len * self.fs))
        _, h_base = signal.impulse(sys_ode, T=t_kernel)
        # Unit peak, so that driver amplitudes are SCR amplitudes in uS
        if np.max(h_base) > 0:
            h_base /= np.max(h_base)
        # Derivative kernel: first-order Taylor expansion of shape variability (Bach et al., 2013)
        h_deriv = np.gradient(h_base)

        # --- 3. Sudomotor driver ---
        n_events = int((p_nsscr / 60.0) * duration_sec)
        impulse_indices = self._generate_pulse_train(n_events, duration_sec, rs)

        u_main = np.zeros(total_samples)
        u_diff = np.zeros(total_samples)
        for idx in impulse_indices:
            amp = rs.lognormal(mean=0.0, sigma=p_amp_sd)
            beta = rs.normal(0, p_beta_var)
            u_main[idx] = amp
            u_diff[idx] = amp * beta

        # --- 4. Phasic component ---
        phasic = signal.convolve(u_main, h_base)[:total_samples] + signal.convolve(u_diff, h_deriv)[:total_samples]
        # SCRs cannot be negative
        phasic[phasic < 0] = 0

        # --- 5. Tonic component ---
        tonic_variation = p_scl_drift * time_axis + p_scl_amp * np.sin(2 * np.pi * (1.0 / p_scl_period) * time_axis)

        # Absolute level from the SCR/SCL ratio: SCL = mean active SCR amplitude / ratio
        if np.max(phasic) > 0:
            # "Active" phasic samples: above 0.05 uS
            mean_phasic_peak = np.mean(phasic[phasic > 0.05]) if np.any(phasic > 0.05) else np.max(phasic)
            target_scl_level = mean_phasic_peak / p_scr_scl_ratio
        else:
            target_scl_level = 5.0  # uS, when no SCR is generated

        # Physiological range of the skin conductance level (uS)
        target_scl_level = np.clip(target_scl_level, 1.0, 25.0)
        tonic = tonic_variation + target_scl_level

        # --- 6. Raw signal ---
        raw_signal = tonic + phasic

        # Conductance must stay positive
        min_val = np.min(raw_signal)
        if min_val < 0.1:
            shift = abs(min_val) + 0.5
            tonic += shift
            raw_signal += shift

        return {
            'raw': raw_signal,
            'tonic': tonic,
            'phasic': phasic,
            'sna_driver': u_main,
            'theta': theta_opt,
            'artifacts': np.zeros_like(raw_signal, dtype=bool),
        }
