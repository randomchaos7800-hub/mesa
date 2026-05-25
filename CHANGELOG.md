# Changelog

## Unreleased

- Tightened `schema v2` metadata requirements with explicit domain, source profile, and review provenance fields.
- Expanded the curated v2 gold dataset to `60` items and added a public dev split (`dataset/mesa_v2_dev.json`).
- Added split-aware manifests, a curated review log, and statistical reporting helpers.
- Added benchmark spec, evaluation protocol, dataset governance, benchmark card, baseline reporting, release-process, and leaderboard docs.
- Added machine-readable citation metadata in `CITATION.cff`.

## 0.3.2

- Added schema v2 with annotated dataset format, validators, and migrators.
- Added `run_benchmark_v2()` with observable storage/retrieval/answer metrics.
- Added typed v2 scorers for all current task/answer formats.
- Added observable trace hooks to reference adapters.
- Added curated `dataset/mesa_v2.json` and `dataset/version_v2.json`.
- Added v2 dataset annotation guidance.
- Fixed package version/export mismatch.
- Improved legacy ROUGE-1 fallback so missing `rouge-score` no longer collapses scores to `0.0`.
