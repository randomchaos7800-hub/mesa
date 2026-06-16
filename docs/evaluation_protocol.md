# MESA Evaluation Protocol

This document defines how MESA benchmark results should be generated and
interpreted.

## Public dev vs hidden test

MESA is moving toward two tracks:

- `dev_public`: for adapter iteration and local debugging
- `test_hidden`: for official benchmark comparison

The public repo includes a dev split:

- [dataset/mesa_v2_dev.json](../dataset/mesa_v2_dev.json)

The hidden-test manifest is represented publicly only as a template:

- [dataset/version_v2_test_hidden.template.json](../dataset/version_v2_test_hidden.template.json)

The actual hidden test dataset should not be committed to the public repo.

## Official-run expectations

An official run should:

- use `schema v2`
- include manifest-backed dataset metadata
- disclose model and backend configuration
- disclose whether traces were fully available

## Non-official runs

The following are non-official by default:

- full public gold self-evaluation
- legacy v1 composite runs
- runs missing manifest or trace disclosure

## Reporting split labels

Every reported run should clearly label one of:

- `dev_public`
- `full_gold_public`
- `test_hidden`
- `legacy`

## Result integrity

Do not report public-dev results as if they were hidden-test leaderboard scores.
