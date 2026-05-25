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

- `60` items
- all `9` task types represented
- public dev split: `27` items

## Known limitations

- remaining curated items are still dominated by single-session examples
- domain diversity is still weaker than the long-term target
- hidden-test operation is scaffolded, not yet externally hosted

## Reporting requirements

Use the reporting rules in:

- [docs/result_reporting.md](/home/dino/mesa-benchmark/docs/result_reporting.md:1)
- [docs/evaluation_protocol.md](/home/dino/mesa-benchmark/docs/evaluation_protocol.md:1)
