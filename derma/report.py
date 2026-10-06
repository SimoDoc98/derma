"""Goodness-of-fit report for fitted profiles.

For every parameter: fitted family, number of segments, median [IQR] and the
Kolmogorov-Smirnov statistic of the data against the fitted distribution. The SCR kernel
is checked on the median morphology: the ODE response fitted to the median
(t_rise, tau_1, tau_2) should reproduce the Bach (2010) shape. Figures need the optional
``matplotlib`` package (``pip install 'derma[report]'``).
"""

import csv
import os

import numpy as np
import scipy.stats as stats

from .generation import fit_ode_kernel

# Symbol and unit of every profile parameter
PARAMETER_LABELS = {
    'scl_drift': ('m', 'uS/s'),
    'nsscr': ('nu', 'events/min'),
    'scr_amp_sd': ('sigma_A', '-'),
    'scr_rise_time': ('t_rise', 's'),
    'scr_tau1': ('tau_1', 's'),
    'scr_tau_ratio': ('tau_2/tau_1', '-'),
    'scr_scl_ratio': ('rho', '-'),
}

TABLE_COLUMNS = ['parameter', 'symbol', 'unit', 'distribution', 'n', 'median', 'q1', 'q3', 'ks_d', 'ks_p']


def _fitted_values(features, parameter):
    """Values actually used by the fit: positive parameters drop values <= 0.001."""
    values = np.asarray(features.get(parameter, []), dtype=float)
    values = values[np.isfinite(values)]
    if parameter in ('scl_drift', 'nsscr'):
        return values
    return values[values > 0.001]


def fit_table(features, profile):
    """
    Goodness-of-fit table, one row per parameter.

    Parameters
    ----------
    features : dict of numpy.ndarray
        Segment values, as returned by :func:`derma.estimation.estimate_profile`.
    profile : derma.profiles.EDAProfile
        Fitted profile.

    Returns
    -------
    list of dict
        Keys of ``TABLE_COLUMNS``; median, quartiles in the unit of the parameter,
        ``ks_d`` and ``ks_p`` from a one-sample Kolmogorov-Smirnov test.
    """
    rows = []
    for parameter, (symbol, unit) in PARAMETER_LABELS.items():
        dist = getattr(profile, parameter)
        values = _fitted_values(features, parameter)
        row = dict(parameter=parameter, symbol=symbol, unit=unit, distribution=dist.dist_name,
                   n=len(values), median=np.nan, q1=np.nan, q3=np.nan, ks_d=np.nan, ks_p=np.nan)
        if len(values) >= 2:
            row['median'] = float(np.median(values))
            row['q1'], row['q3'] = (float(q) for q in np.percentile(values, [25, 75]))
            try:
                ks = stats.kstest(values, getattr(stats, dist.dist_name)(*dist.params).cdf)
                row['ks_d'], row['ks_p'] = float(ks.statistic), float(ks.pvalue)
            except Exception:
                pass
        rows.append(row)
    return rows


def kernel_check(features, fs=4):
    """
    Fit the ODE kernel to the median SCR morphology.

    Parameters
    ----------
    features : dict of numpy.ndarray
        Segment values.
    fs : float
        Sampling frequency (Hz).

    Returns
    -------
    dict or None
        Median ``rise_time_s``, ``tau1_s``, ``tau2_s`` (median tau_1 times median
        tau_2/tau_1), fitted ``thetas`` and ``mse``, time axis ``t`` (s), target
        ``h_target`` and peak-aligned ODE response ``h_ode``. None if the morphology
        cannot be computed.
    """
    medians = {}
    for parameter in ('scr_rise_time', 'scr_tau1', 'scr_tau_ratio'):
        values = _fitted_values(features, parameter)
        medians[parameter] = float(np.median(values)) if len(values) else np.nan

    rise_time = medians['scr_rise_time']
    tau1 = medians['scr_tau1']
    tau2 = tau1 * medians['scr_tau_ratio']
    if not all(np.isfinite(v) and v > 0 for v in (rise_time, tau1, tau2)):
        return None

    thetas, mse, t, h_target, h_ode = fit_ode_kernel(rise_time, tau1, tau2, fs)
    # Align the peaks, as in the fit cost
    shift = int(np.argmax(h_target)) - int(np.argmax(h_ode))
    if shift > 0:
        h_ode = np.pad(h_ode, (shift, 0), mode='constant')[:len(t)]
    else:
        h_ode = np.pad(h_ode[-shift:], (0, -shift), mode='constant')

    return dict(rise_time_s=rise_time, tau1_s=tau1, tau2_s=tau2, thetas=thetas, mse=float(mse),
                t=t, h_target=h_target, h_ode=h_ode)


def plot_fit(features, profile, path, fs=4, title=None):
    """
    Histograms with fitted densities and the kernel check, saved as one figure.

    Parameters
    ----------
    features : dict of numpy.ndarray
        Segment values.
    profile : derma.profiles.EDAProfile
        Fitted profile.
    path : str
        Output image file (e.g. ``.png``).
    fs : float
        Sampling frequency (Hz) for the kernel check.
    title : str, optional
        Figure title.
    """
    try:
        # Figure without pyplot: no change to the user's plotting backend
        from matplotlib.figure import Figure
    except ImportError:
        raise ImportError("plot_fit() needs matplotlib: pip install 'derma[report]'") from None

    table = {row['parameter']: row for row in fit_table(features, profile)}
    fig = Figure(figsize=(7, 9))
    axes = fig.subplots(4, 2)
    axes = axes.ravel()

    for ax, (parameter, (symbol, unit)) in zip(axes, PARAMETER_LABELS.items()):
        values = _fitted_values(features, parameter)
        ax.set_title(symbol, fontsize=10)
        if len(values) < 2:
            ax.text(0.5, 0.5, 'insufficient data', ha='center', va='center', transform=ax.transAxes)
            continue
        ax.hist(values, bins=min(30, max(8, len(values) // 3)), density=True, color='0.75', edgecolor='white')
        x = np.linspace(*ax.get_xlim(), 400)
        dist = getattr(profile, parameter)
        try:
            ax.plot(x, getattr(stats, dist.dist_name).pdf(x, *dist.params), color='C0', lw=1.8)
        except Exception:
            pass
        ax.text(0.97, 0.95, f"{dist.dist_name}\nKS D = {table[parameter]['ks_d']:.3f}",
                ha='right', va='top', fontsize=7, transform=ax.transAxes)
        ax.set_xlabel(unit, fontsize=8)
        ax.tick_params(labelsize=7)

    ax = axes[7]
    ax.set_title('SCR kernel (median morphology)', fontsize=10)
    check = kernel_check(features, fs)
    if check is None:
        ax.text(0.5, 0.5, 'morphology unavailable', ha='center', va='center', transform=ax.transAxes)
    else:
        ax.plot(check['t'], check['h_target'], 'k--', lw=1.5, label='Bach (2010)')
        ax.plot(check['t'], check['h_ode'], color='C0', lw=1.5, label='fitted ODE')
        ax.set_xlim(0, max(20, 12 * check['rise_time_s']))
        ax.text(0.97, 0.6, f"MSE = {check['mse']:.1e}", ha='right', fontsize=7, transform=ax.transAxes)
        ax.set_xlabel('s', fontsize=8)
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=7)

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def _fmt(x):
    """Compact number format for the Markdown table."""
    if not np.isfinite(x):
        return '-'
    if x != 0 and (abs(x) < 1e-2 or abs(x) >= 1e4):
        return f'{x:.2e}'
    return f'{x:.3f}'


def write_fit_report(features, profile, out_dir, name, fs=4):
    """
    Write the fit report of a profile: CSV table, Markdown table and figure.

    Parameters
    ----------
    features : dict of numpy.ndarray
        Segment values.
    profile : derma.profiles.EDAProfile
        Fitted profile.
    out_dir : str
        Output folder (created if missing).
    name : str
        Base name of the files: ``<name>_fit.csv``, ``<name>_fit.md``, ``<name>_fit.png``.
    fs : float
        Sampling frequency (Hz) for the kernel check.

    Returns
    -------
    list of str
        Paths of the written files.
    """
    os.makedirs(out_dir, exist_ok=True)
    rows = fit_table(features, profile)
    check = kernel_check(features, fs)

    csv_path = os.path.join(out_dir, f'{name}_fit.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=TABLE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    md_path = os.path.join(out_dir, f'{name}_fit.md')
    lines = [f'# Fit report: {name}', '',
             '| Parameter | Unit | Distribution | N | Median [IQR] | KS D | KS p |',
             '|---|---|---|---|---|---|---|']
    for r in rows:
        lines.append(f"| {r['symbol']} | {r['unit']} | {r['distribution']} | {r['n']} | "
                     f"{_fmt(r['median'])} [{_fmt(r['q1'])}, {_fmt(r['q3'])}] | {_fmt(r['ks_d'])} | {_fmt(r['ks_p'])} |")
    lines.append('')
    if check is None:
        lines.append('SCR kernel check: median morphology unavailable.')
    else:
        lines.append(f"SCR kernel check on the median morphology (t_rise = {check['rise_time_s']:.3f} s, "
                     f"tau_1 = {check['tau1_s']:.3f} s, tau_2 = {check['tau2_s']:.3f} s): "
                     f"theta = [{', '.join(f'{v:.4f}' for v in check['thetas'])}], MSE = {check['mse']:.2e}.")
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')

    paths = [csv_path, md_path]
    png_path = os.path.join(out_dir, f'{name}_fit.png')
    plot_fit(features, profile, png_path, fs=fs, title=name)
    paths.append(png_path)
    return paths
