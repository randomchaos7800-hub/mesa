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

- Update [README.md](../README.md)
  - lead with `schema v2`
  - move `schema v1` into a clear legacy section
  - add links to benchmark spec, roadmap, and issue plan
- Add [docs/benchmark_spec.md](../docs/benchmark_spec.md)
  - official task definitions
  - official metrics
  - official run requirements
  - disclosure requirements for published results
- Update [METHODOLOGY.md](../METHODOLOGY.md)
  - split into `legacy v1` and `official v2`
  - de-emphasize composite scoring
- Update [AUDIT.md](../AUDIT.md)
  - point open items at this roadmap
- Add [docs/result_reporting.md](../docs/result_reporting.md)
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

- Expand [dataset/mesa_v2.json](../dataset/mesa_v2.json)
- Update [dataset/version_v2.json](../dataset/version_v2.json)
- Extend [docs/annotation_guidelines.md](../docs/annotation_guidelines.md)
  - review checklist
  - rejection criteria
  - distractor quality rules
- Add [docs/dataset_review_checklist.md](../docs/dataset_review_checklist.md)
- Update [tests/test_dataset.py](../tests/test_dataset.py)
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

- Extend [dataset/schema_v2.json](../dataset/schema_v2.json)
  - make `domain` required
  - add `source_profile`
  - add `annotator_id`
  - add `reviewer_id`
- Update [mesa/dataset/validators.py](../mesa/dataset/validators.py)
  - validate domain/source-profile metadata
- Update [dataset/mesa_v2_annotated.json](../dataset/mesa_v2_annotated.json)
  - enrich source metadata
- Update [tests/test_schema_v2.py](../tests/test_schema_v2.py)
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

- Expand [docs/annotation_guidelines.md](../docs/annotation_guidelines.md)
  - detailed task-specific rules
- Add [docs/dataset_governance.md](../docs/dataset_governance.md)
  - who can approve items
  - how disagreements are resolved
  - how versions are cut
- Add [dataset/review_log_v2.json](../dataset/review_log_v2.json)
  - item review metadata
  - adjudication notes
- Update [mesa/dataset/manifest.py](../mesa/dataset/manifest.py)
  - support richer release metadata
- Add [docs/release_process.md](../docs/release_process.md)

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

- Add [dataset/mesa_v2_dev.json](../dataset/mesa_v2_dev.json)
- Add hidden test manifest support
  - [dataset/version_v2.json](../dataset/version_v2.json)
  - [mesa/dataset/manifest.py](../mesa/dataset/manifest.py)
- Update [mesa/runner.py](../mesa/runner.py)
  - enforce official-run metadata
  - distinguish public dev vs official test mode
- Add [docs/evaluation_protocol.md](../docs/evaluation_protocol.md)
  - official submission rules
  - hidden-test policy
- Add integration tests in [tests/test_runner.py](../tests/test_runner.py)
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

- Add [docs/baseline_reporting_template.md](../docs/baseline_reporting_template.md)
- Add [docs/baselines.md](../docs/baselines.md)
- Add [docs/benchmark_card_template.md](../docs/benchmark_card_template.md)
- Update [mesa/runner.py](../mesa/runner.py)
  - include cost/latency fields where available
- Add statistical helpers under:
  - [mesa/scoring/stats.py](../mesa/scoring/stats.py)
- Add tests in:
  - [tests/test_scoring_v2.py](../tests/test_scoring_v2.py)
  - [tests/test_runner.py](../tests/test_runner.py)

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

- Add [CITATION.cff](../CITATION.cff)
- Add [docs/benchmark_card.md](../docs/benchmark_card.md)
- Update [CHANGELOG.md](../CHANGELOG.md)
  - benchmark release notes
- Add leaderboard or submission docs:
  - [docs/leaderboard.md](../docs/leaderboard.md)
- Add packaging and release metadata in:
  - [pyproject.toml](../pyproject.toml)
  - [mesa/__init__.py](../mesa/__init__.py)

### Acceptance criteria

- MESA has a stable benchmark release another group can cite and run
- at least a few non-author baselines exist

## Cross-Cutting Technical Work

These items span multiple releases.

### Scoring

- keep legacy `llm_judge()` advisory only in [mesa/scorer.py](../mesa/scorer.py)
- continue improving typed scoring in:
  - [mesa/scoring/answer_types.py](../mesa/scoring/answer_types.py)
  - [mesa/scoring/grounding.py](../mesa/scoring/grounding.py)
- add confidence intervals and significance helpers in:
  - [mesa/scoring/stats.py](../mesa/scoring/stats.py)

### Adapter contract

- keep the observable contract centered in [mesa/adapter.py](../mesa/adapter.py)
- consider async support after v0.8
- consider stricter official-run enforcement in [mesa/runner.py](../mesa/runner.py)

### Validation

- keep schema and validator logic aligned across:
  - [dataset/schema_v2.json](../dataset/schema_v2.json)
  - [mesa/dataset/validators.py](../mesa/dataset/validators.py)
  - [tests/test_schema_v2.py](../tests/test_schema_v2.py)

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

---

## Mike v2 Full Run — 2026-05-25 (Official Baseline)

**Run:** `results/run_v2_2026-05-25_14-08.json`
**Dataset:** mesa_v2.json, 76 items, all task types
**Adapter:** MikeAdapter (full relay + session store pipeline)

| Metric | 14-item pilot | 76-item baseline |
|---|---|---|
| Answer correct | 0.50 | **0.26** |
| Answer grounded | 0.14 | **0.16** |
| Storage fact recall | 0.77 | **0.31** |
| Storage forbidden hits | — | **0.34** |

**By type:**
| Type | n | Correct | Grounded |
|---|---|---|---|
| update/interference | 10 | 0.90 | 0.70 |
| update | 11 | 0.45 | 0.09 |
| synthesis/multi | 7 | 0.29 | 0.00 |
| recall/preference | 5 | 0.20 | 0.00 |
| causal | 12 | 0.17 | 0.08 |
| temporal | 13 | 0.08 | 0.15 |
| adversarial | 8 | 0.00 | 0.00 |
| recall/single | 5 | 0.00 | 0.00 |
| recall/constraint | 5 | 0.00 | 0.00 |

**Key findings:**
- update/interference is Mike's dominant skill — tracks superseded vs current state well
- Storage recall collapsed (0.77 → 0.31) on complex multi-session items — real architectural gap
- Adversarial 0/8 — Mike's refusal phrasing never matches `_REFUSAL_PATTERNS` (known scorer bug)
- recall/single, recall/constraint at 0.0 — scorer AND/OR bug (mesa #1) inflating false negatives
- 14-item pilot was optimistic; skewed toward Mike's strengths
- **0.26 is the honest floor. Target for next Mike eval: ≥0.40 after scorer fixes land.**
