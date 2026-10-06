# Changelog

All notable changes to DERMA are documented in this file.

## [Unreleased]

### Added
- Project skeleton: package layout, MIT license, contribution guidelines on dataset
  handling, continuous integration.
- `derma.profiles`: profile data structures, literature profile, JSON loading and saving.
- Six frozen reference profiles (Literature, CASE low/high arousal, MAUS low/high mental
  workload, SAD) with provenance metadata.
- `derma.generation`: SCR kernel calibration (Bach 2010 shape fitted by a third-order ODE,
  Bach 2011) and the synthetic EDA generator, ported from the original implementation
  with identical outputs for the same seed.

### Changed (with respect to the original implementation)
- Random numbers come from an explicit `numpy.random.RandomState` per subject instead of
  the global NumPy state; the sequence of draws, and therefore the output, is unchanged.
- The default `scr_beta_var` of `EDAProfile` is 0.2 instead of 0.1, matching the value
  used by every reference profile. Shipped profiles are not affected.
