# Changelog

All notable project changes are documented here.

## [Unreleased]

## [0.1.1] - 2026-08-09

### Fixed

- Restored the best validation model and matching optimizer state before
  advancing to the next curriculum stage.
- Saved the restored best state in `latest.pt`, keeping resumed and
  uninterrupted training aligned.

### Changed

- Limited push-triggered CI to `main`; pull requests now receive one validation
  run instead of duplicate push and pull-request runs.
- Updated CI actions to Node.js 24-compatible releases.

### Documentation

- Added a code-level architecture and extension guide.
- Added repository status, release, citation, and development links to the
  README.

### Metadata

- Replaced placeholder contributor metadata with the project author's name.

## [0.1.0] - 2026-08-09

### Added

- Bitset-backed triangle-free graph construction with independent verification.
- Exact Turán formulas and a direct oracle constructor.
- Uniform-random and least-degree search baselines.
- A custom permutation-equivariant PyTorch edge policy.
- Elite-trajectory curriculum training with resumable checkpoints.
- Five-seed evaluation through unseen graph size `n = 24`.
- Episode-level results, aggregate tables, figures, checkpoints, trajectories,
  and a complete MVP research report.
- Automated tests, linting, formatting checks, and GitHub Actions CI.

[Unreleased]: https://github.com/ernestterjyan/neural-extremal-graph-search/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/ernestterjyan/neural-extremal-graph-search/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/ernestterjyan/neural-extremal-graph-search/releases/tag/v0.1.0
