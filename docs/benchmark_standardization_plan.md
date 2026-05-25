# MESA Benchmark Standardization Plan

This document turns the benchmark roadmap into a repo-native execution plan.
It is scoped around the current codebase and organized as a sequence of tracked
releases from `v0.4` through `v1.0`.

## Objective

Make MESA credible as a standard benchmark by improving:

- dataset scale
- domain diversity
- scoring rigor
- benchmark governance
- anti-overfitting protocol
- baseline comparability
- external reproducibility

The guiding shift is:

- `schema v1` remains a legacy compatibility path
- `schema v2` becomes the only official benchmark path

## Current Baseline

As of `0.3.2`:

- official benchmark-capable path exists in `schema v2`
- curated v2 gold dataset: `dataset/mesa_v2.json` with `19` items
- annotated v2 pool: `dataset/mesa_v2_annotated.json` with `120` items
- stage-aware scoring exists for storage, retrieval, and answer quality
- legacy composite scorer still exists for `schema v1`

## Release Plan

## v0.4: Official Benchmark Identity

Goal: make the repo unambiguous about what the benchmark is.

### Code and doc changes

- Update [README.md](/home/dino/mesa-benchmark/README.md:1)
  - lead with `schema v2`
  - move `schema v1` into a clear legacy section
  - add links to benchmark spec, roadmap, and issue plan
- Add [docs/benchmark_spec.md](/home/dino/mesa-benchmark/docs/benchmark_spec.md:1)
  - official task definitions
  - official metrics
  - official run requirements
  - disclosure requirements for published results
- Update [METHODOLOGY.md](/home/dino/mesa-benchmark/METHODOLOGY.md:1)
  - split into `legacy v1` and `official v2`
  - de-emphasize composite scoring
- Update [AUDIT.md](/home/dino/mesa-benchmark/AUDIT.md:1)
  - point open items at this roadmap
- Add [docs/result_reporting.md](/home/dino/mesa-benchmark/docs/result_reporting.md:1)
  - required metadata
  - baseline report format
  - confidence interval expectations

### Acceptance criteria

- `schema v2` is the default benchmark story in repo docs
- `schema v1` is explicitly labeled legacy everywhere
- a new contributor can identify the official benchmark path in under 2 minutes

## v0.5: Gold Dataset Expansion

Goal: move from a tiny curated set to a minimally serious benchmark slice.

### Dataset targets

- expand `dataset/mesa_v2.json` from `19` to `75-100` items
- reach at least `8-10` curated items per task type
- raise weakest categories first:
  - `adversarial`
  - `temporal`
  - `update`
  - `update/interference`
  - `causal`

### Code and data changes

- Expand [dataset/mesa_v2.json](/home/dino/mesa-benchmark/dataset/mesa_v2.json:1)
- Update [dataset/version_v2.json](/home/dino/mesa-benchmark/dataset/version_v2.json:1)
- Extend [docs/annotation_guidelines.md](/home/dino/mesa-benchmark/docs/annotation_guidelines.md:1)
  - review checklist
  - rejection criteria
  - distractor quality rules
- Add [docs/dataset_review_checklist.md](/home/dino/mesa-benchmark/docs/dataset_review_checklist.md:1)
- Update [tests/test_dataset.py](/home/dino/mesa-benchmark/tests/test_dataset.py:1)
  - enforce stronger type coverage thresholds
  - validate new metadata fields

### Acceptance criteria

- curated v2 gold set reaches at least `75` items
- every task type has enough volume for stable per-type reporting
- all gold items pass validation plus human review checklist

## v0.6: Multi-Domain and Multi-Author Coverage

Goal: stop looking like a benchmark over one user profile.

### Dataset targets

- add explicit domain coverage for:
  - personal
  - infrastructure
  - workplace/project
  - education/research
  - finance-lite
  - health-lite
- add at least `3-5` distinct source profiles or writing styles
- no single domain should exceed roughly `40%` of the curated gold set

### Code and data changes

- Extend [dataset/schema_v2.json](/home/dino/mesa-benchmark/dataset/schema_v2.json:1)
  - make `domain` required
  - add `source_profile`
  - add `annotator_id`
  - add `reviewer_id`
- Update [mesa/dataset/validators.py](/home/dino/mesa-benchmark/mesa/dataset/validators.py:1)
  - validate domain/source-profile metadata
- Update [dataset/mesa_v2_annotated.json](/home/dino/mesa-benchmark/dataset/mesa_v2_annotated.json:1)
  - enrich source metadata
- Update [tests/test_schema_v2.py](/home/dino/mesa-benchmark/tests/test_schema_v2.py:1)
  - enforce new required metadata

### Acceptance criteria

- benchmark reports can break down results by domain
- curated gold set visibly spans multiple domains and source profiles

## v0.7: Governance and Annotation Rigor

Goal: make the dataset defensible outside the author.

### Process targets

- double-annotation for new gold items
- adjudication log for disagreements
- explicit retirement and replacement policy
- inter-annotator agreement reporting

### Code and doc changes

- Expand [docs/annotation_guidelines.md](/home/dino/mesa-benchmark/docs/annotation_guidelines.md:1)
  - detailed task-specific rules
- Add [docs/dataset_governance.md](/home/dino/mesa-benchmark/docs/dataset_governance.md:1)
  - who can approve items
  - how disagreements are resolved
  - how versions are cut
- Add [dataset/review_log_v2.json](/home/dino/mesa-benchmark/dataset/review_log_v2.json:1)
  - item review metadata
  - adjudication notes
- Update [mesa/dataset/manifest.py](/home/dino/mesa-benchmark/mesa/dataset/manifest.py:1)
  - support richer release metadata
- Add [docs/release_process.md](/home/dino/mesa-benchmark/docs/release_process.md:1)

### Acceptance criteria

- every new gold item has reviewer provenance
- benchmark releases have auditable curation history

## v0.8: Dev/Test Split and Hidden Evaluation

Goal: make overfitting materially harder.

### Dataset targets

- create public `dev` split
- create hidden `test` split
- balance each split across:
  - task type
  - domain
  - session format
  - distractor density

### Code and data changes

- Add [dataset/mesa_v2_dev.json](/home/dino/mesa-benchmark/dataset/mesa_v2_dev.json:1)
- Add hidden test manifest support
  - [dataset/version_v2.json](/home/dino/mesa-benchmark/dataset/version_v2.json:1)
  - [mesa/dataset/manifest.py](/home/dino/mesa-benchmark/mesa/dataset/manifest.py:1)
- Update [mesa/runner.py](/home/dino/mesa-benchmark/mesa/runner.py:1)
  - enforce official-run metadata
  - distinguish public dev vs official test mode
- Add [docs/evaluation_protocol.md](/home/dino/mesa-benchmark/docs/evaluation_protocol.md:1)
  - official submission rules
  - hidden-test policy
- Add integration tests in [tests/test_runner.py](/home/dino/mesa-benchmark/tests/test_runner.py:1)
  - manifest handling
  - split routing
  - output metadata

### Acceptance criteria

- official results are generated from a hidden test path or equivalent submission flow
- public development no longer happens on the full gold set

## v0.9: Baselines and Reporting

Goal: make MESA easy to compare against.

### Baseline targets

Add at least one result for each class:

- long-context-only baseline
- naive RAG baseline
- vector-store memory baseline
- structured memory baseline
- commercial assistant baseline
- open memory-agent framework baseline
- intentionally weak baseline

### Code and doc changes

- Add [docs/baseline_reporting_template.md](/home/dino/mesa-benchmark/docs/baseline_reporting_template.md:1)
- Add [docs/baselines.md](/home/dino/mesa-benchmark/docs/baselines.md:1)
- Add [docs/benchmark_card_template.md](/home/dino/mesa-benchmark/docs/benchmark_card_template.md:1)
- Update [mesa/runner.py](/home/dino/mesa-benchmark/mesa/runner.py:1)
  - include cost/latency fields where available
- Add statistical helpers under:
  - [mesa/scoring/stats.py](/home/dino/mesa-benchmark/mesa/scoring/stats.py:1)
- Add tests in:
  - [tests/test_scoring_v2.py](/home/dino/mesa-benchmark/tests/test_scoring_v2.py:1)
  - [tests/test_runner.py](/home/dino/mesa-benchmark/tests/test_runner.py:1)

### Acceptance criteria

- benchmark reports include uncertainty and stage-level summaries
- outside users can reproduce a baseline from documented settings

## v1.0: Public Standard Candidate

Goal: make MESA adoptable beyond this repo.

### Release targets

- stable `schema v2` benchmark spec
- fixed public dev split
- fixed hidden test split
- stable scorer version
- public benchmark card
- citation-ready release

### Code and doc changes

- Add [CITATION.cff](/home/dino/mesa-benchmark/CITATION.cff:1)
- Add [docs/benchmark_card.md](/home/dino/mesa-benchmark/docs/benchmark_card.md:1)
- Update [CHANGELOG.md](/home/dino/mesa-benchmark/CHANGELOG.md:1)
  - benchmark release notes
- Add leaderboard or submission docs:
  - [docs/leaderboard.md](/home/dino/mesa-benchmark/docs/leaderboard.md:1)
- Add packaging and release metadata in:
  - [pyproject.toml](/home/dino/mesa-benchmark/pyproject.toml:1)
  - [mesa/__init__.py](/home/dino/mesa-benchmark/mesa/__init__.py:1)

### Acceptance criteria

- MESA has a stable benchmark release another group can cite and run
- at least a few non-author baselines exist

## Cross-Cutting Technical Work

These items span multiple releases.

### Scoring

- keep legacy `llm_judge()` advisory only in [mesa/scorer.py](/home/dino/mesa-benchmark/mesa/scorer.py:1)
- continue improving typed scoring in:
  - [mesa/scoring/answer_types.py](/home/dino/mesa-benchmark/mesa/scoring/answer_types.py:1)
  - [mesa/scoring/grounding.py](/home/dino/mesa-benchmark/mesa/scoring/grounding.py:1)
- add confidence intervals and significance helpers in:
  - [mesa/scoring/stats.py](/home/dino/mesa-benchmark/mesa/scoring/stats.py:1)

### Adapter contract

- keep the observable contract centered in [mesa/adapter.py](/home/dino/mesa-benchmark/mesa/adapter.py:1)
- consider async support after v0.8
- consider stricter official-run enforcement in [mesa/runner.py](/home/dino/mesa-benchmark/mesa/runner.py:1)

### Validation

- keep schema and validator logic aligned across:
  - [dataset/schema_v2.json](/home/dino/mesa-benchmark/dataset/schema_v2.json:1)
  - [mesa/dataset/validators.py](/home/dino/mesa-benchmark/mesa/dataset/validators.py:1)
  - [tests/test_schema_v2.py](/home/dino/mesa-benchmark/tests/test_schema_v2.py:1)

## Dataset Milestones

Use these milestones instead of growing the dataset without quality controls.

### Milestone A

- curated gold set: `40` items
- all 9 task types represented
- at least `3-4` items in weakest categories

### Milestone B

- curated gold set: `75` items
- every task type at `8+`
- domain metadata complete
- at least `35%` multi-session in `temporal`
- at least `50%` multi-session in `causal`

### Milestone C

- curated gold set: `100` items
- every task type at `10+`
- all gold items have annotator and reviewer metadata
- dev/test split ready

### Milestone D

- curated gold set: `150-200` items
- domain-balanced
- hidden test set operational
- baseline matrix populated

## Definition of Done for Standardization

MESA is not a standard benchmark when:

- only the author can explain the dataset
- the gold set is too small to stabilize metrics
- the full test set is public and used for tuning
- baselines are sparse or underdocumented

MESA is closer to a standard benchmark when:

- v2 is the only official benchmark path
- the dataset is large enough and diverse enough
- governance is documented
- official test evaluation is harder to overfit
- outside groups can reproduce results without private context

---

## Mike v2 Baseline Run — 2026-05-25

**Run:** `results/run_v2_2026-05-25_05-27.json`
**Dataset:** mesa_v2.json, 14 items, all task types
**Adapter:** MikeAdapter (full relay + session store pipeline)

| Metric | Score |
|---|---|
| Answer correct | 0.50 (7/14) |
| Answer grounded | 0.14 (2/14) |
| Storage fact recall | 0.77 |
| Unsupported claim items | 0.86 |

**By type:**
| Type | n | Correct | Grounded |
|---|---|---|---|
| recall/single | 1 | 0.0 | 0.0 |
| recall/preference | 1 | 1.0 | 0.0 |
| recall/constraint | 1 | 0.0 | 0.0 |
| adversarial | 1 | 0.0 | 0.0 |
| temporal | 2 | 0.5 | 0.5 |
| update | 2 | 1.0 | 0.0 |
| update/interference | 2 | 1.0 | 0.5 |
| synthesis/multi | 2 | 0.5 | 0.0 |
| causal | 2 | 0.0 | 0.0 |

**Scorer issues identified (see gap analysis below)**
