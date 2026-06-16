# MESA Dataset Governance

This document defines how MESA benchmark data is curated, reviewed, versioned,
and retired.

## Roles

- `annotator`: drafts or revises benchmark items
- `reviewer`: checks correctness, answerability, and task-type fit
- `maintainer`: approves dataset releases and version updates

In the current repo state, a single person may fill multiple roles, but the
fields are kept separate so the process can scale beyond one maintainer.

## Required fields for curated v2 items

Every curated item should carry:

- `metadata.annotator_id`
- `metadata.reviewer_id`
- `metadata.review_status`
- `metadata.domain`
- `metadata.source_profile`

## Review statuses

- `annotated`: drafted but not yet curated into the public gold set
- `reviewed`: checked and approved for curated use
- `adjudicated`: reviewed after disagreement or ambiguity
- `provisional`: temporarily included while awaiting stronger review

## Promotion policy

Items should be promoted from the annotated pool into the curated set only if:

- they validate against `schema_v2`
- they pass the dataset review checklist
- they improve benchmark coverage or quality
- they are at least as strong as the weakest curated item of that type

## Retirement policy

Items should be retired or replaced when:

- the question is ambiguous
- the answer depends on outside knowledge
- the evidence is insufficient
- the task type is mismatched
- the item is trivially gameable

Retired items should be documented in release notes or review logs rather than
silently disappearing.

## Review log

Curated review provenance lives in:

- [dataset/review_log_v2.json](../dataset/review_log_v2.json)

## Release rule

Dataset releases should update:

- the dataset file
- the version manifest
- the review log if curated items changed
- tests that enforce current coverage expectations
