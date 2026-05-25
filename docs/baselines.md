# MESA Baselines

This file is the home for benchmark baselines once they are rerun under the
official v2 reporting contract.

## Current v2 reference baselines

These are lightweight reference baselines on the public dev split:

- dataset: `dataset/mesa_v2_dev.json`
- dataset version: `0.4.0-dev`
- split: `dev_public`

| System | Correct | Grounded | Abstention | Storage recall | Retrieval recall | Unsupported claim items |
|---|---:|---:|---:|---:|---:|---:|
| `NullAdapter` | 0.1111 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `EchoAdapter` | 0.2593 | 0.9259 | 0.3333 | 0.4167 | 0.4167 | 0.0741 |
| `DictAdapter` | 0.1481 | 0.7778 | 1.0000 | 0.1667 | 0.1667 | 0.2222 |

Interpretation:

- `NullAdapter` is the refusal floor: it abstains cleanly but fails nearly every answerable item.
- `EchoAdapter` is a smoke-test baseline: it benefits from raw-context exposure but still fails on selective retrieval and abstention.
- `DictAdapter` is a weak structured-memory baseline: it can extract simple facts but collapses on multi-hop and interference-heavy items.

## Target baseline classes

- long-context-only baseline
- naive RAG baseline
- vector-store memory baseline
- structured memory baseline
- commercial assistant baseline
- intentionally weak baseline

## Expected metadata

See:

- [docs/baseline_reporting_template.md](/home/dino/mesa-benchmark/docs/baseline_reporting_template.md:1)
- [docs/result_reporting.md](/home/dino/mesa-benchmark/docs/result_reporting.md:1)
