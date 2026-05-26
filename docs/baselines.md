# MESA Baselines

This file is the home for benchmark baselines under the official v2 reporting contract.

## Full gold set — v2 baselines (2026-05-25)

Run against the full curated gold set. No LLM judge (scoring: EM + ROUGE-1).

- dataset: `dataset/mesa_v2.json`
- dataset version: `0.4.0`
- split: `full_gold_public`
- n_items: 76
- public results: `public_results/`

| System | Correct | Grounded | Abstention | Storage recall | Notes |
|---|---:|---:|---:|---:|---|
| **Mike (relay)** | **0.263** | **0.158** | **0.000** | **0.309** | Production memory agent; official baseline |
| `EchoAdapter` | 0.224 | 0.908 | 0.125 | 0.265 | Long-context floor: raw text in, raw text out |
| `DictAdapter` | 0.158 | 0.605 | 0.875 | 0.118 | Naive "X is Y" pattern matching |
| `NullAdapter` | 0.105 | 1.000 | 1.000 | 0.000 | Refusal floor: always "I don't have that" |

**Reading the table:**

- Mike scores 0.26 correct and is the only system with real memory — it can answer questions EchoAdapter gets wrong because the answer isn't literally in the injected text.
- EchoAdapter beats Mike on `grounded` (0.91 vs 0.16) because MESA's grounding metric checks whether the answer is supported by retrieved memory. Echo trivially retrieves everything; Mike retrieves selectively and its retrieval trace is sparse.
- NullAdapter scores 0.105 (not 0.0) because adversarial items (8 of 76) reward correct refusal — NullAdapter refuses everything, so it aces adversarial and fails everything else.
- The gap between Mike (0.26) and EchoAdapter (0.22) on `correct` is real but small — the benchmark is hard, and most of the gap lives in `update/interference` (Mike 0.90 vs Echo 0.40).

**By-type breakdown — Mike vs EchoAdapter:**

| Type | n | Mike correct | Echo correct | Delta |
|---|---:|---:|---:|---:|
| update/interference | 10 | 0.90 | 0.40 | +0.50 |
| update | 11 | 0.45 | 0.36 | +0.09 |
| synthesis/multi | 7 | 0.29 | 0.14 | +0.14 |
| causal | 12 | 0.17 | 0.25 | −0.08 |
| temporal | 13 | 0.08 | 0.31 | −0.23 |
| recall/preference | 5 | 0.20 | 0.00 | +0.20 |
| recall/single | 5 | 0.00 | 0.00 | 0.00 |
| recall/constraint | 5 | 0.00 | 0.00 | 0.00 |
| adversarial | 8 | 0.00 | 0.12 | −0.12 |

Mike's lead is concentrated in update/interference. EchoAdapter beats Mike on temporal because temporal items often ask "what did you discuss on date X?" and the raw conversation text answers that directly. Mike's adversarial score of 0.00 is the critical gap — it doesn't abstain correctly when it should.

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
