"""Population profiles: fitted parameter distributions and their JSON files.

A profile holds, for each free parameter of the generative model, a ``scipy.stats``
distribution and the empirical bounds observed in the data. Synthetic subjects are
drawn from it by rejection sampling within those bounds.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
import scipy.stats as stats

PROFILE_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'profile_data')

# Distribution-valued fields of EDAProfile, in the order they are sampled by the generator
DISTRIBUTION_FIELDS = ('scl_drift', 'nsscr', 'scr_amp_sd', 'scr_rise_time',
                       'scr_tau1', 'scr_tau_ratio', 'scr_scl_ratio')


@dataclass
class DistributionParams:
    """
    A ``scipy.stats`` distribution restricted to empirical bounds.

    Parameters
    ----------
    dist_name : str
        Name of the distribution in ``scipy.stats`` (e.g. ``'lognorm'``).
    params : tuple
        Shape, location and scale parameters, in ``scipy.stats`` order.
    bounds : tuple of float
        Lower and upper bound of valid samples, in the units of the parameter.
    """
    dist_name: str
    params: tuple
    bounds: Tuple[float, float] = (-np.inf, np.inf)

    def sample(self, size=1, random_state=None):
        """
        Draw samples within ``bounds`` by rejection sampling.

        Parameters
        ----------
        size : int
            Number of samples to return.
        random_state : numpy.random.RandomState, optional
            Source of randomness. Each batch of candidates is drawn from it, so the
            sequence is reproducible for a given seed.

        Returns
        -------
        numpy.ndarray
            ``size`` samples, all inside ``bounds``.
        """
        dist = getattr(stats, self.dist_name)
        samples = np.array([])

        while len(samples) < size:
            # Oversample to absorb the rejected candidates
            n_needed = size - len(samples)
            candidates = dist.rvs(*self.params, size=max(10, n_needed * 2), random_state=random_state)
            valid = candidates[(candidates >= self.bounds[0]) & (candidates <= self.bounds[1])]
            samples = np.concatenate((samples, valid))

        return samples[:size]


@dataclass
class EDAProfile:
    """
    Distributions of the generative-model parameters of a population.

    Parameters
    ----------
    scl_drift : DistributionParams
        Linear drift slope of the skin conductance level m (uS/s).
    nsscr : DistributionParams
        Frequency of non-specific SCRs nu (events/min).
    scr_amp_sd : DistributionParams
        SCR amplitude variability sigma_A: log-normal sigma of the driver amplitudes.
    scr_rise_time : DistributionParams
        SCR rise time t_rise (s).
    scr_tau1 : DistributionParams
        Fast SCR decay time tau_1 (s).
    scr_tau_ratio : DistributionParams
        Ratio tau_2 / tau_1 between slow and fast decay (dimensionless).
    scr_scl_ratio : DistributionParams
        Ratio rho between mean SCR amplitude and tonic level (dimensionless).
    scr_beta_var : float
        Standard deviation sigma_beta of the derivative-kernel weights, i.e. the
        intra-subject variability of SCR shape (dimensionless). Fixed, not fitted.
    scl_period_range : tuple of float
        Range of the period T_osc of the tonic oscillation (s). Fixed, not fitted.
    provenance : dict
        Free-form metadata: data source, citation, estimation recipe, notes.
    """
    scl_drift: DistributionParams
    nsscr: DistributionParams
    scr_amp_sd: DistributionParams
    scr_rise_time: DistributionParams
    scr_tau1: DistributionParams
    scr_tau_ratio: DistributionParams
    scr_scl_ratio: DistributionParams
    scr_beta_var: float = 0.2
    scl_period_range: Tuple[float, float] = (120, 600)
    provenance: dict = field(default_factory=dict)


def literature_profile():
    """
    Profile built from parameter ranges reported in the literature.

    Every parameter is uniform within its range: drift (Bach et al., 2010), SCR rate
    and amplitude variability (Braithwaite et al., 2013), SCR morphology
    (Leiner et al., 2012), SCR/SCL ratio (Braithwaite et al., 2013).

    Returns
    -------
    EDAProfile
        Profile used when no real data is available.
    """
    def uniform(min_value, max_value):
        return DistributionParams('uniform', (min_value, max_value - min_value), (min_value, max_value))

    return EDAProfile(
        scl_drift=uniform(-0.01, 0.01),
        nsscr=uniform(1, 25),
        scr_amp_sd=uniform(0.1, 0.5),
        scr_rise_time=uniform(0.5, 2.5),
        scr_tau1=uniform(0.5, 2.0),
        scr_tau_ratio=uniform(2.0, 10.0),
        scr_scl_ratio=uniform(0.1, 1),
        scr_beta_var=0.2,
    )


def save_profile(profile, path):
    """
    Write a profile to a JSON file.

    Parameters
    ----------
    profile : EDAProfile
        Profile to save, including its ``provenance``.
    path : str
        Destination file.
    """
    content = {'parameters': {}}
    for name in DISTRIBUTION_FIELDS:
        dist = getattr(profile, name)
        content['parameters'][name] = {'distribution': dist.dist_name,
                                       'params': [float(p) for p in dist.params],
                                       'bounds': [float(b) for b in dist.bounds]}
    content['scr_beta_var'] = float(profile.scr_beta_var)
    content['scl_period_range'] = [float(p) for p in profile.scl_period_range]
    content['provenance'] = profile.provenance

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(content, f, indent=2)
        f.write('\n')


def load_profile(name_or_path):
    """
    Load a profile from a JSON file, or one of the profiles shipped with DERMA.

    Parameters
    ----------
    name_or_path : str
        Path to a JSON profile, or the name of a shipped profile
        (see :func:`available_profiles`).

    Returns
    -------
    EDAProfile
        The loaded profile.
    """
    path = name_or_path
    if not os.path.isfile(path):
        path = os.path.join(PROFILE_DATA_DIR, name_or_path + '.json')
        if not os.path.isfile(path):
            raise FileNotFoundError(f"No profile file or shipped profile named '{name_or_path}'. "
                                    f"Shipped profiles: {available_profiles()}")

    with open(path, encoding='utf-8') as f:
        content = json.load(f)

    distributions = {name: DistributionParams(p['distribution'], tuple(p['params']), tuple(p['bounds']))
                     for name, p in content['parameters'].items()}
    return EDAProfile(**distributions,
                      scr_beta_var=content['scr_beta_var'],
                      scl_period_range=tuple(content['scl_period_range']),
                      provenance=content.get('provenance', {}))


def available_profiles():
    """
    Names of the profiles shipped with DERMA.

    Returns
    -------
    list of str
        Names accepted by :func:`load_profile`.
    """
    return sorted(f[:-5] for f in os.listdir(PROFILE_DATA_DIR) if f.endswith('.json'))
