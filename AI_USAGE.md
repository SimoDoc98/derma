# Use of AI tools

This file summarises, by category, how AI assistance has been used in the development
of DERMA. It is updated whenever a new kind of use appears and before each release.

## Tools

- Claude Code, model Claude Opus 5.5 (`claude-opus-5-5`), used from October 2026.

## Scope of assistance

- **Analysis of the original code**: inventory of the original PhD code to be ported
  (generator, estimator, S.O.M.A., dataset preprocessing), numerical checks on the
  original outputs, and a check that the `cvxeda` 1.1.0 package reproduces the original
  cvxEDA results.
- **Project scaffolding**: repository structure, packaging configuration, initial tests
  and CI workflow.
- **Porting and refactoring of the author's original code**: port of the generator, the
  SCR kernel calibration and the profile data structures from the original PhD code,
  conversion of the reference profiles to JSON, and equivalence tests showing that the
  ported code reproduces the original outputs.
- **Documentation**: first drafts of `README.md`, `CONTRIBUTING.md` and `CHANGELOG.md`.

## Human review

All AI-assisted outputs (code, tests, documentation) are reviewed, edited where needed
and validated by the author, who makes all design decisions (e.g., license, package
layout, profile file format, dataset handling rules).