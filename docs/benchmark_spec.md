# MESA Benchmark Spec

This document defines the official MESA benchmark contract.

## Official Status

MESA has two runnable paths:

- `schema v2`: official benchmark path
- `schema v1`: legacy compatibility path

Only `schema v2` results should be treated as official benchmark results.

## Benchmark Scope

MESA evaluates memory systems across three observable stages:

- `storage`: what facts were written from the source sessions
- `retrieval`: what evidence was surfaced for the question
- `answer`: whether the final answer was correct and grounded

The benchmark is designed for memory-agent systems, not just base language
models. It evaluates the pipeline around the model, not only the model itself.

## Official Task Types

The official task taxonomy is:

- `recall/single`
- `recall/preference`
- `recall/constraint`
- `synthesis/multi`
- `temporal`
- `update`
- `update/interference`
- `adversarial`
- `causal`

## Official Dataset Format

Official runs must use `schema v2` items validated against:

- [dataset/schema_v2.json](../dataset/schema_v2.json)

Official gold data currently lives in:

- [dataset/mesa_v2.json](../dataset/mesa_v2.json)
- [dataset/version_v2.json](../dataset/version_v2.json)

Each official item must include:

- a typed `task_type`
- a typed `answer_format`
- `sessions`
- `gold_memory`
- `gold_answer`

## Official Adapter Contract

All adapters must implement:

- `reset()`
- `inject()` or `inject_session()`
- `ask()`

Official observable runs should also implement:

- `get_writes()`
- `ask_with_trace()`

Optional debugging support:

- `get_retrieved_context()`

The contract is defined in:

- [mesa/adapter.py](../mesa/adapter.py)

## Official Metrics

Official reporting should use stage-level v2 metrics rather than a single
headline composite score.

### Storage metrics

- `required_fact_recall`
- `required_fact_precision`
- `forbidden_fact_hits`
- `non_required_fact_hits`
- `unannotated_write_count`

### Retrieval metrics

- `required_fact_recall`
- `required_fact_precision`
- `forbidden_fact_hits`
- `non_required_fact_hits`
- `unannotated_retrieval_count`

### Answer metrics

- `correct`
- `grounded`
- `unsupported_claims`
- `abstention_correct`

### Run-level summaries

Official run summaries should include:

- aggregate storage metrics
- aggregate retrieval metrics
- aggregate answer metrics
- per-task-type summaries

## Official Run Requirements

An official run should:

- use `run_benchmark_v2()`
- use a validated `schema v2` dataset
- expose retrieval trace support through `ask_with_trace()`
- report dataset version metadata
- disclose the adapter and memory architecture
- disclose model/backend details if any LLMs are used
- disclose whether traces were available or partially missing

An official run should not:

- present `schema v1` composite scores as benchmark-grade headline metrics
- omit retrieval trace support while claiming official comparability
- compare results across different dataset versions without disclosure
- mix public-dev and official-test results in one table without labeling

## Legacy Path

The legacy path remains available for backwards compatibility:

- [mesa/scorer.py](../mesa/scorer.py)
- `run_benchmark()`
- `dataset/mesa_v1.json`

Legacy outputs may still be useful for historical comparison, but they are not
the official benchmark definition.

## Required Disclosure for Published Results

Any published result should include:

- MESA version
- dataset path
- dataset version
- schema version
- adapter class
- model name and revision, if applicable
- retrieval backend details
- whether the run is official `schema v2` or legacy `schema v1`
- any missing trace capability or non-standard scoring behavior
