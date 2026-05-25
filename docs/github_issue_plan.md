# MESA Roadmap Issue Plan

This document is a copy-paste-ready breakdown for GitHub issues and milestones.
Create one milestone per release target: `v0.4`, `v0.5`, `v0.6`, `v0.7`,
`v0.8`, `v0.9`, and `v1.0`.

## Milestone: v0.4

### Issue: Make schema v2 the official benchmark path

Checklist:

- [ ] Update `README.md` to lead with `schema v2`
- [ ] Move `schema v1` into a clearly labeled legacy section
- [ ] Link roadmap and benchmark spec from the README
- [ ] Ensure examples default to `dataset/mesa_v2.json`

### Issue: Publish benchmark spec

Checklist:

- [ ] Add `docs/benchmark_spec.md`
- [ ] Define official tasks and answer formats
- [ ] Define official metrics
- [ ] Define what qualifies as an official run
- [ ] Define what must be reported with benchmark results

### Issue: Split methodology into official vs legacy

Checklist:

- [ ] Update `METHODOLOGY.md`
- [ ] Move schema-v1 composite details into a legacy section
- [ ] Document schema-v2 stage-level metrics
- [ ] Clarify that legacy LLM judging is advisory only

## Milestone: v0.5

### Issue: Expand curated v2 gold dataset to 40 items

Checklist:

- [ ] Promote reviewed items from `dataset/mesa_v2_annotated.json`
- [ ] Keep all 9 task types represented
- [ ] Prioritize weakest categories first
- [ ] Update `dataset/version_v2.json`

### Issue: Expand curated v2 gold dataset to 75+ items

Checklist:

- [ ] Reach at least 75 curated items
- [ ] Reach at least 8 items per task type
- [ ] Replace weak items instead of only appending
- [ ] Re-run dataset validation and tests

### Issue: Add dataset review checklist

Checklist:

- [ ] Add `docs/dataset_review_checklist.md`
- [ ] Define acceptance and rejection criteria
- [ ] Define distractor quality rules
- [ ] Define evidence sufficiency review rules

## Milestone: v0.6

### Issue: Require domain metadata in schema v2

Checklist:

- [ ] Update `dataset/schema_v2.json`
- [ ] Add required `domain` field
- [ ] Add `source_profile`
- [ ] Add `annotator_id`
- [ ] Add `reviewer_id`
- [ ] Update validators and tests

### Issue: Diversify curated gold set across domains

Checklist:

- [ ] Add items from personal domain
- [ ] Add items from infrastructure domain
- [ ] Add items from workplace/project domain
- [ ] Add items from education/research domain
- [ ] Add items from finance-lite domain
- [ ] Add items from health-lite domain
- [ ] Verify no single domain dominates the dataset

### Issue: Increase multi-session coverage

Checklist:

- [ ] Raise multi-session share in `temporal`
- [ ] Raise multi-session share in `causal`
- [ ] Raise multi-session share in `update`
- [ ] Add tests or reports tracking session-format coverage

## Milestone: v0.7

### Issue: Formalize annotation governance

Checklist:

- [ ] Expand `docs/annotation_guidelines.md`
- [ ] Add `docs/dataset_governance.md`
- [ ] Define review roles and approval rules
- [ ] Define item retirement and replacement policy

### Issue: Add review log and adjudication history

Checklist:

- [ ] Add `dataset/review_log_v2.json`
- [ ] Track annotator and reviewer provenance
- [ ] Track adjudication notes for disputed items
- [ ] Add tests for review-log shape if applicable

### Issue: Add release process documentation

Checklist:

- [ ] Add `docs/release_process.md`
- [ ] Define dataset versioning process
- [ ] Define scorer versioning process
- [ ] Define release checklist

## Milestone: v0.8

### Issue: Create public dev split

Checklist:

- [ ] Add `dataset/mesa_v2_dev.json`
- [ ] Balance split by task type
- [ ] Balance split by domain
- [ ] Balance split by session format
- [ ] Update manifest metadata

### Issue: Add hidden test evaluation protocol

Checklist:

- [ ] Define hidden test policy in `docs/evaluation_protocol.md`
- [ ] Update `mesa/dataset/manifest.py` for split-aware metadata
- [ ] Update `mesa/runner.py` for official-run metadata enforcement
- [ ] Add runner tests for split handling

### Issue: Mark full-gold self-eval as non-official

Checklist:

- [ ] Update README and methodology docs
- [ ] Distinguish dev vs official test reporting
- [ ] Ensure published examples do not imply full-gold self-eval is official

## Milestone: v0.9

### Issue: Add baseline reporting template

Checklist:

- [ ] Add `docs/baseline_reporting_template.md`
- [ ] Add `docs/benchmark_card_template.md`
- [ ] Define required hardware/model/prompt disclosure
- [ ] Define latency and cost reporting rules

### Issue: Add statistical reporting helpers

Checklist:

- [ ] Add `mesa/scoring/stats.py`
- [ ] Implement confidence interval helpers
- [ ] Implement per-type summary helpers
- [ ] Add tests

### Issue: Populate first baseline matrix

Checklist:

- [ ] Run long-context-only baseline
- [ ] Run naive RAG baseline
- [ ] Run vector-store baseline
- [ ] Run structured memory baseline
- [ ] Run one commercial assistant baseline
- [ ] Run one weak baseline
- [ ] Document all runs in `docs/baselines.md`

## Milestone: v1.0

### Issue: Cut stable benchmark release

Checklist:

- [ ] Freeze schema version
- [ ] Freeze scorer version
- [ ] Freeze dev split
- [ ] Freeze hidden test split
- [ ] Publish benchmark card

### Issue: Add citation metadata

Checklist:

- [ ] Add `CITATION.cff`
- [ ] Add citation section to the README
- [ ] Add release citation to docs

### Issue: Publish leaderboard or submission process

Checklist:

- [ ] Add `docs/leaderboard.md`
- [ ] Define submission format
- [ ] Define result verification rules
- [ ] Define update cadence

## Labels

Recommended labels:

- `roadmap`
- `benchmark`
- `dataset`
- `scoring`
- `docs`
- `governance`
- `baselines`
- `infra`
- `good first issue`
- `release-blocker`

## Tracking Rules

For every roadmap issue:

- link the owning milestone
- link the relevant spec or roadmap section
- name the exact files in scope
- define acceptance criteria in the issue body
- avoid combining dataset growth and scorer changes in one issue unless they are inseparable
