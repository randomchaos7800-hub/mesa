# MESA Baselines

This file is the home for benchmark baselines under the official v2 reporting contract.

## Full gold set — v2 baselines (2026-05-25)

Run against the full curated gold set. No LLM judge (scoring: EM + ROUGE-1).

- dataset: `dataset/mesa_v2.json`
- dataset version: `0.5.0`
- split: `full_gold_public`
- n_items: 94
- public results: `public_results/`

| System | Correct | Grounded | Abstention | Storage recall | Notes |
|---|---:|---:|---:|---:|---|
| **Hermes (AIAgent)** | **0.351** | **0.319** | **0.375** | **0.337** | Hermes framework + Genesis local model; long-context mode |
| **Mike (relay)** | **0.263** | **0.158** | **0.000** | **0.309** | Production memory agent; 76-item run (v0.4.0 dataset) |
| `EchoAdapter` | 0.224 | 0.908 | 0.125 | 0.265 | Long-context floor: raw text in, raw text out |
| `DictAdapter` | 0.158 | 0.605 | 0.875 | 0.118 | Naive "X is Y" pattern matching |
| `NullAdapter` | 0.105 | 1.000 | 1.000 | 0.000 | Refusal floor: always "I don't have that" |

**Reading the table:**

- Hermes leads at 0.351 correct — 9 points above Mike and 13 above EchoAdapter. Both run conversation history injection; Hermes wins because it can reason across the context and apply LLM understanding rather than raw text matching.
- Hermes' abstention rate (0.375) is the highest of the real systems — it correctly refuses on adversarial items at a rate Mike doesn't (Mike: 0.000).
- Mike scores 0.263 and is the only system with active write/recall memory — scores above are on the 94-item v0.5.0 dataset except Mike which ran on the earlier 76-item v0.4.0 set. Mike needs a rerun on v0.5.0 to be directly comparable.
- EchoAdapter's `grounded` score (0.908) looks great but is trivially earned — it retrieves everything, so everything is "grounded." Hermes grounded at 0.319 means it's being selective and still correct.
- NullAdapter scores 0.105 (not 0.0) because adversarial items reward correct refusal.

**By-type breakdown — Hermes vs Mike vs EchoAdapter (94-item set):**

| Type | n | Hermes | Mike* | Echo | Hermes−Echo |
|---|---:|---:|---:|---:|---:|
| update/interference | 10 | 0.50 | 0.90 | 0.40 | +0.10 |
| recall/single | 10 | 0.50 | — | 0.00 | +0.50 |
| recall/constraint | 10 | 0.50 | — | 0.00 | +0.50 |
| update | 11 | 0.36 | 0.45 | 0.36 | 0.00 |
| recall/preference | 10 | 0.40 | 0.20 | 0.00 | +0.40 |
| adversarial | 8 | 0.38 | 0.00 | 0.12 | +0.26 |
| synthesis/multi | 10 | 0.20 | 0.29 | 0.14 | +0.06 |
| causal | 12 | 0.25 | 0.17 | 0.25 | 0.00 |
| temporal | 13 | 0.15 | 0.08 | 0.31 | −0.16 |

*Mike scores from 76-item v0.4.0 run; direct comparison only valid for types present in both datasets.

Hermes' biggest wins over EchoAdapter: recall/* items (0.50 vs 0.00) where raw-text retrieval fails but LLM reasoning can extract the correct fact. Temporal is the weak spot for both — temporal items reward exact date recall which requires faithful memory, not reasoning.

## Dev split reference baselines

These are the earlier reference numbers on the 27-item dev split:

- dataset: `dataset/mesa_v2_dev.json`
- dataset version: `0.4.0-dev`
- split: `dev_public`

| System | Correct | Grounded | Abstention | Storage recall | Retrieval recall |
|---|---:|---:|---:|---:|---:|
| `NullAdapter` | 0.1111 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| `EchoAdapter` | 0.2593 | 0.9259 | 0.3333 | 0.4167 | 0.4167 |
| `DictAdapter` | 0.1481 | 0.7778 | 1.0000 | 0.1667 | 0.1667 |

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
