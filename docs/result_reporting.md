# MESA Result Reporting

This document defines the minimum reporting standard for benchmark runs.

## Required Metadata

Every reported run should include:

- `mesa_version`
- `schema_version`
- `dataset_name`
- `dataset_version`
- `dataset_path`
- `adapter`
- `task_count`
- `run_mode`

If an LLM or external model is used, also report:

- model name
- model revision or release date
- provider or serving stack
- prompt or system-prompt policy summary
- retrieval backend configuration

## Official v2 Reporting

Official benchmark reporting should be based on `schema v2`.

Report these stage-level summaries:

- storage recall/precision
- retrieval recall/precision
- answer correctness rate
- answer grounded rate
- abstention correctness rate
- unsupported-claim item rate

Also report:

- per-task-type breakdown
- dataset version
- whether traces were complete or partial

## Legacy v1 Reporting

Legacy `schema v1` reporting is allowed for historical comparison only.

If you report it, label it clearly as:

- `legacy`
- `non-official`
- `composite-scored`

Do not present legacy composite scores as the primary benchmark result.

## Baseline Reporting Format

Each baseline should disclose:

- system name
- model name
- memory backend
- retrieval method
- dataset and version
- schema version
- hardware
- latency notes
- cost notes if relevant
- known limitations

## Comparison Rules

Do not compare benchmark runs as if they were equivalent when they differ on:

- schema version
- dataset version
- public-dev vs official-test split
- adapter observability
- model family or retrieval stack without disclosure

## Recommended Statistics

When possible, report:

- confidence intervals
- per-type sample counts
- session-format breakdown
- domain breakdown once domain metadata is standardized

## Minimum Table Shape

For official `schema v2` results, a minimum summary table should include:

- system
- dataset version
- items
- storage recall
- retrieval recall
- answer correctness
- answer groundedness
- abstention correctness

## Narrative Reporting Rules

Good benchmark reports say:

- what system was evaluated
- what evidence path was available
- what benchmark release was used
- what metric families improved or regressed

Bad benchmark reports say:

- only one overall score
- nothing about dataset version or traces
- nothing about model/backend configuration
