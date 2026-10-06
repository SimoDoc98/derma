# Changelog

All notable changes to DERMA are documented in this file.

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
