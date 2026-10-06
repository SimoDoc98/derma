<p align="center">
  <img src="assets/derma-logo.png" alt="DERMA logo" width="360">
</p>

# DERMA

**D**ata-driven **E**lectrodermal **R**esponse **M**odeling **A**rchitecture: synthetic
electrodermal activity (EDA) calibrated on real recordings.

DERMA estimates interpretable tonic and phasic parameters from real EDA recordings,
summarizes them as population profiles (maximum-likelihood fits with AIC model
selection), and generates synthetic signals with known ground truth: tonic and phasic
components, sudomotor driver and the coefficients of the SCR kernel.

> **Status: early development.** The first releases port and generalize code developed
> during the author's PhD; the API is not stable yet.

## Contents

Available in v0.1:

- Synthetic EDA generation from literature ranges or from fitted profiles
  (`derma.generation`), with six reference profiles shipped with the package
  (`derma.profiles`)
- Parameter estimation and population profiles from decomposed recordings
  (`derma.estimation`), with an optional cvxEDA decomposition backend
  (`derma.decomposition`)
- Goodness-of-fit report for fitted profiles (`derma.report`)

Planned:

- Dataset loaders for public EDA datasets
- Subject-specific optimization of cvxEDA hyperparameters (S.O.M.A.)

## Installation

DERMA is not on PyPI yet. From a clone of this repository:

```bash
pip install -e .
pip install -e ".[cvxeda]"   # optional: cvxEDA decomposition backend
pip install -e ".[report]"   # optional: figures of the fit report (matplotlib)
```

## Datasets

DERMA never downloads or redistributes data. See [CONTRIBUTING.md](CONTRIBUTING.md) for
how datasets are handled.

## License

DERMA is released under the [MIT License](LICENSE).

The optional cvxEDA backend relies on the
[`cvxeda`](https://github.com/lciti/cvxEDA) package by L. Citi and A. Greco, which is
distributed under GPL-3.0 and is not included in DERMA. If you use it, please cite
Greco et al., "cvxEDA: a Convex Optimization Approach to Electrodermal Activity
Processing", *IEEE Trans. Biomed. Eng.*, 2016, doi:10.1109/TBME.2015.2474131.
