# MESA Benchmark Card

## Benchmark identity

- benchmark name: MESA
- benchmark release: `v0.9-prep`
- schema version: `2`
- curated dataset version: `0.3.0`
- public dev version: `0.3.0-dev`

## Intended use

MESA is designed to evaluate memory-agent systems using observable stage-level
metrics across storage, retrieval, and final answering.

It is intended to support:

- comparison of memory-system behavior
- diagnosis of memory pipeline failures
- measurement of groundedness and abstention quality

It is not intended to support:

- claims about general intelligence
- claims about model safety in the absence of memory traces
- fair cross-version comparison without split and manifest disclosure

## Metric families

- storage recall/precision and forbidden-hit metrics
- retrieval recall/precision and forbidden-hit metrics
- answer correctness, groundedness, unsupported claims, and abstention quality

## Dataset composition

Current curated public gold set:

- `76` items
- all `9` task types represented
- public dev split: `27` items
- hidden local test split scaffolded outside version control: `18` items

## Known limitations

- the benchmark still leans heavily on a single author/source family
- multi-session coverage is improved but still concentrated in a subset of task types
- hidden-test operation exists locally but is not yet externally hosted

## Reporting requirements

Use the reporting rules in:

- [docs/result_reporting.md](../docs/result_reporting.md)
- [docs/evaluation_protocol.md](../docs/evaluation_protocol.md)
