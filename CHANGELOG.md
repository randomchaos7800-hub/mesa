# Changelog

## Unreleased

- Added the **real world tests** split: `dataset/mesa_realworld.json` (+ manifest) — set 1 is
  delivery-stop routing (12 items, all 9 task types) probing operational state that mutates
  mid-shift: stop reordering with retracted revisions, near-identical address interference,
  jointly-binding time windows, double-superseded delivery windows, and equipment-fault causal
  chains. Reference floors: Echo 0.250 (0.0 on update/interference/temporal/adversarial), Null 0.083.
- Fixed `run_benchmark_v2()` crashing on a plain-string `dataset_path` (the README quickstart form);
  paths are now coerced to `Path`.
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
