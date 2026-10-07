# Changelog

All notable changes to DERMA are documented in this file.

## [0.2.0] - Unreleased

### Notes
- The S.O.M.A. module (subject-specific optimization of decomposition hyperparameters),
  previously announced for this release, is postponed to a future release (v0.6). The
  delay allows it to be generalized into an algorithm-agnostic framework, applicable to
  multiple EDA decomposition methods rather than to cvxEDA only. The empty `derma.soma`
  placeholder module has been removed.

### Added
- `derma.datasets`: loaders for CASE, MAUS and SAD. Each loader checks the SHA-256 of the
  original archive, reads it without extracting, resamples the skin conductance to 4 Hz
  and saves one recording per subject (CASE, SAD) or per condition (MAUS) in
  `prepared/<DATASET>/<DATASET>_<subject>[_<condition>].npz`, read back by `load()`.
- Command line: `python -m derma.datasets status` and `python -m derma.datasets prepare`.
- Dataset page, DOI and license in the metadata of the reference profiles.

## [0.1.0] - 2026-10-06

First release: core of the generator and of the estimation pipeline, ported from the
original implementation of the author's PhD work.

### Added
- Project skeleton: package layout, MIT license, contribution guidelines on dataset
  handling, continuous integration.
- `derma.profiles`: profile data structures, literature profile, JSON loading and saving.
- Six frozen reference profiles (Literature, CASE low/high arousal, MAUS low/high mental
  workload, SAD) with provenance metadata.
- `derma.generation`: SCR kernel calibration (Bach 2010 shape fitted by a third-order ODE,
  Bach 2011) and the synthetic EDA generator, ported from the original implementation
  with identical outputs for the same seed.
- `derma.estimation`: per-segment parameter extraction, maximum-likelihood fit with AIC
  selection, segmentation recipes of the reference profiles with the new
  `onset_exclusion_sec` option (40 s for MAUS). Re-estimating the reference profiles from
  the original preprocessed recordings reproduces them exactly.
- `derma.decomposition`: `decompose()` wrapper around the optional `cvxeda` package,
  with 0.75 s moving-average smoothing.
- `derma.report`: goodness-of-fit table (CSV and Markdown), histograms with fitted
  densities and SCR kernel check (PNG, optional matplotlib).
- `CITATION.cff`.
