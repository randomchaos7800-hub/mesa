# MESA Baselines

This file is the home for benchmark baselines under the official v2 reporting contract.

## Full gold set — v2 baselines (2026-05-25)

### v0.7.0 — 361 items (current)

Run against the full 361-item gold set. No LLM judge (scoring: EM + ROUGE-1).

- dataset: `dataset/mesa_v2.json`
- dataset version: `0.7.0`
- split: `full_gold_public`
- n_items: 361 (109 reviewed + 252 provisional)
- public results: `public_results/`

| System | Correct | Grounded | Abstention | Storage recall | Notes |
|---|---:|---:|---:|---:|---|
| **Mike (relay)** | **0.249** | **0.136** | **0.000** | **0.119** | Production memory agent; full relay pipeline; run 2026-05-25 |
| **Hermes (AIAgent)** | **0.235** | **0.303** | **0.105** | **0.098** | Hermes framework + Genesis local model; long-context mode; run 2026-05-25 |

**Reading the table:**

- Both systems dropped from the 94-item v0.5.0 numbers — the 252 new synthetic items are harder (no real accumulated memory backing them, more diverse domains).
- Mike edges Hermes on overall correct (0.249 vs 0.235) but Hermes leads on grounding (0.303 vs 0.136) — Mike answers more questions but with less retrievable support.
- Hermes is the only system with non-zero adversarial abstention (0.105); Mike still scores 0.000 on adversarial — never refuses.
- Mike domain gap: old curated items (personal, infra, ops) score 30–83%; new synthetic domains score 0–19%. Mike's relay benefits from real accumulated memory on familiar topics.

**By-type breakdown — Mike vs Hermes (361-item v0.7.0):**

| Type | n | Mike | Hermes | Mike−Hermes |
|---|---:|---:|---:|---:|
| recall/constraint | 39 | 0.546 | 0.636 | −0.09 |
| recall/preference | 39 | 0.364 | 0.545 | −0.18 |
| recall/single | 40 | 0.450 | 0.400 | +0.05 |
| synthesis/multi | 40 | 0.417 | 0.333 | +0.08 |
| temporal | 43 | 0.333 | 0.267 | +0.07 |
| update/interference | 39 | 0.256 | 0.103 | +0.15 |
| causal | 42 | 0.214 | 0.071 | +0.14 |
| update | 41 | 0.098 | 0.146 | −0.05 |
| adversarial | 38 | 0.000 | 0.105 | −0.11 |

Hermes wins on recall/* (LLM reasoning extracts facts better). Mike wins on synthesis/multi, update/interference, temporal, causal — areas where accumulated real memory and active tool use help.

---

### v0.5.0 — 94 items (archived)

Run against the earlier 94-item curated gold set. No LLM judge (scoring: EM + ROUGE-1).

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
