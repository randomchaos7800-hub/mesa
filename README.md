# MESA — Memory Eval Benchmark

A reproducible benchmark framework for **personal AI memory systems**.

Most AI memory benchmarks test context-window retention or external RAG pipelines. MESA tests the full loop: **inject → extract → store → retrieve → answer**. The question types are grounded in real failure modes observed in production AI companion systems.

Planning docs:

- [Benchmark Standardization Plan](docs/benchmark_standardization_plan.md)
- [GitHub Issue Plan](docs/github_issue_plan.md)
- [Benchmark Spec](docs/benchmark_spec.md)
- [Result Reporting](docs/result_reporting.md)
- [Evaluation Protocol](docs/evaluation_protocol.md)
- [Dataset Governance](docs/dataset_governance.md)
- [Benchmark Card](docs/benchmark_card.md)

---

## Official benchmark path

MESA has two runnable paths, but only one official benchmark path:

- `schema v2`: official observable benchmark
- `schema v1`: legacy compatibility runner

If you are adopting MESA now, use:

- [dataset/mesa_v2.json](dataset/mesa_v2.json)
- [dataset/mesa_v2_dev.json](dataset/mesa_v2_dev.json) for public development
- `run_benchmark_v2()`
- the stage-level metrics in `storage`, `retrieval`, and `answer`

Use the legacy v1 path only for backwards comparison against older results.

For serious evaluation workflow, use:

- `dataset/mesa_v2_dev.json` for public iteration
- a hidden test split outside version control for private comparison

## What it tests

Nine question types, each targeting a distinct failure mode:

| Type | Description | What can fail |
|------|-------------|---------------|
| `recall/single` | Single fact stated once | Extraction or retrieval |
| `recall/preference` | Preference/opinion expressed | Subjective fact extraction |
| `recall/constraint` | Hard rule stated once, noise follows | Rule persistence under interference |
| `synthesis/multi` | Requires combining 2+ facts | Multi-hop retrieval |
| `temporal` | Involves when something happened | Date metadata survival |
| `update` | Fact stated, then superseded | Conflict resolution |
| `update/interference` | Original + similar-but-different facts added after | Original not overwritten by similar data |
| `adversarial` | Fact never stated | Hallucination resistance |
| `causal` | A + B → C (causal chain) | Multi-hop reasoning |

---

## Install

```bash
pip install rouge-score openai  # runtime deps
pip install pytest jsonschema   # for tests
```

Published package name:

```bash
pip install mesa-memory-eval
```

Or clone and install in editable mode:

```bash
git clone https://github.com/randomchaos7800-hub/mesa
cd mesa
pip install -e ".[dev]"
```

Optional adapter deps:
```bash
pip install -e ".[chroma]"   # ChromaAdapter
pip install -e ".[mem0]"     # Mem0Adapter
```

---

## Quickstart

MESA currently exposes two benchmark paths:

- `schema v2`: official observable runner (`run_benchmark_v2`)
- `schema v1`: legacy composite-scoring runner (`run_benchmark`)

**1. Implement the adapter interface:**

```python
from mesa import MemoryAdapter

class MyAdapter(MemoryAdapter):
    def reset(self):
        self.memory.clear()

    def inject(self, sessions: list[dict]):
        # sessions is a list of {"role": "user"|"assistant", "content": str}
        for turn in sessions:
            self.memory.ingest(turn)

    def ask(self, question: str) -> str:
        return self.memory.query(question)
```

**2. Run the official v2 benchmark:**

```python
from mesa.runner import run_benchmark_v2

results = run_benchmark_v2(
    adapter=MyAdapter(),
    dataset_path="dataset/mesa_v2.json",
    official_run=True,
)

first = results["results"][0]
print(first["storage"]["metrics"])
print(first["retrieval"]["metrics"])
print(first["answer"]["metrics"])
print(results["summary"])
```

Official v2 CLI:

```bash
mesa-benchmark \
  --adapter examples.simple_adapter.EchoAdapter \
  --dataset dataset/mesa_v2.json \
  --schema-version 2 \
  --official-run
```

**3. Legacy v1 quickstart, for backwards comparison only:**

```python
from mesa.runner import run_benchmark

results = run_benchmark(
    adapter=MyAdapter(),
    dataset_path="dataset/mesa_v1.json",
    no_llm_judge=True,
)

print(f"Avg composite: {results['avg_composite']:.4f}")
print(f"Pass rate:     {results['pass_rate_50pct']:.1%}")
```

Legacy CLI:

```bash
mesa-benchmark --adapter my_package.MyAdapter --schema-version 1 --dataset dataset/mesa_v1.json --no-llm-judge
```

**4. Try the example or reference adapters:**

```bash
# EchoAdapter: returns the raw injected context (smoke test)
python -m mesa.runner \
    --adapter examples.simple_adapter.EchoAdapter \
    --dataset dataset/fixtures/sample.json \
    --no-llm-judge

# NullAdapter: always refuses (adversarial baseline)
python -m mesa.runner \
    --adapter examples.simple_adapter.NullAdapter \
    --dataset dataset/fixtures/sample.json \
    --no-llm-judge

# KeywordAdapter: LLM extraction + TF-IDF retrieval (no vector DB required)
python -m mesa.runner \
    --adapter adapters.keyword_adapter.KeywordAdapter \
    --dataset dataset/fixtures/sample.json \
    --no-llm-judge
```

See `adapters/` for `KeywordAdapter`, `Mem0Adapter`, and `ChromaAdapter` — each wraps a different memory architecture and can be used as a starting point for your own implementation.

---

## Scoring

### Official v2 scoring

The v2 runner reports per-item metrics in three stages:

- `storage`: required fact recall/precision, forbidden hits, extra or unannotated writes
- `retrieval`: required fact recall/precision, forbidden hits, extra or unannotated retrieved facts
- `answer`: correctness, grounding, unsupported claims, and abstention correctness

Supported typed answer scorers in v2:

- `single_fact`
- `abstention`
- `temporal`
- `update_current`
- `update/interference`
- `multi_fact`
- `causal`

V2 is the official benchmark path because it is much harder to game than the legacy composite scorer.

Official benchmark comparisons should report these v2 metrics, not a single composite score.

### Legacy v1 scoring

Three dimensions, combined into a single composite score per item:

**Exact match (0.0–1.0)**
Normalized substring and word-overlap. Handles markdown, punctuation, date format variants (`"March 1, 2026"` ≡ `"2026-03-01"`), and multi-answer expected strings.

**ROUGE-1 F1 (0.0–1.0)**
Unigram recall/precision F1 with stemming. Captures partial credit for answers that share vocabulary without exact containment.

**LLM judge (0.0–1.0)**
Advisory 0–3 rubric from an OpenAI-compatible model, normalized to 0.0–1.0. Useful for legacy causal and preference items where the correct answer can be phrased many ways, but not trusted as the primary metric path.

**Composite weights:**

| Mode | Exact | LLM judge | ROUGE-1 |
|------|-------|-----------|---------|
| With judge | 0.40 | 0.30 | 0.30 |
| No judge | 0.55 | — | 0.45 |

**Pass threshold**: composite ≥ 0.50

**Adversarial items** are scored via refusal detection (`is_refusal(predicted)`) rather than exact match. A correct system says it doesn't know; a hallucinating system fabricates an answer and scores 0.

The legacy judge runs with a separate system prompt, strict JSON output, and deterministic settings, but it remains a compatibility feature for schema-v1. It is not the official benchmark metric path.

---

## Dataset

Official dataset:

- [dataset/mesa_v2.json](dataset/mesa_v2.json) — curated v2 gold dataset for observable diagnostic runs
- [dataset/mesa_v2_dev.json](dataset/mesa_v2_dev.json) — public dev split for iteration and smoke-comparison
- [dataset/version_v2.json](dataset/version_v2.json) — dataset manifest and version metadata
- [dataset/version_v2_dev.json](dataset/version_v2_dev.json) — public dev manifest
- [dataset/fixtures/sample_v2.json](dataset/fixtures/sample_v2.json) — small v2 smoke-test fixture set

Legacy dataset:

- [dataset/mesa_v1.json](dataset/mesa_v1.json) — legacy v1 dataset used for composite-scored historical runs

Legacy v1 item shape:

```json
{
  "id": "mesa-recall-single-0001",
  "type": "recall/single",
  "question": "What speed in tokens per second was the new inference rig running at?",
  "expected_answer": "About 70 tokens per second",
  "sessions": [
    {"role": "user", "content": "The new rig is hitting about 70 tokens per second..."},
    {"role": "assistant", "content": "Got it — 70 tok/s, solid for that hardware."}
  ],
  "metadata": {"source": "sessions", "date_added": "2026-04-21"}
}
```

**Multi-session format** (for temporal items with facts spread across dates):

```json
{
  "id": "mesa-temporal-0002",
  "type": "temporal",
  "sessions": [
    {
      "date": "2026-02-01",
      "turns": [{"role": "user", "content": "I drink coffee every morning."}]
    },
    {
      "date": "2026-02-14",
      "turns": [{"role": "user", "content": "Switched to tea. Coffee is done."}]
    }
  ]
}
```

The runner calls `adapter.inject_session(turns, session_date)` once per session in order. The default `MemoryAdapter.inject_session()` delegates to `inject()`, ignoring the date. Override it if your system stores per-session timestamps.

`dataset/fixtures/sample.json` — legacy smoke-test dataset for the v1 path.

## Security note

Running a third-party adapter means executing arbitrary Python code from that adapter path. Treat adapter classes as fully trusted local code.

---

## Baselines

Current baseline tables are mostly historical and still centered on the legacy v1 dataset. Official benchmark reporting should move toward v2 stage-level result tables with explicit dataset versioning.

| System | Dataset | Items | Composite | Pass rate | Notes |
|--------|---------|-------|-----------|-----------|-------|
| Mike (full relay, Gemma-4 26B Q4¹, RTX 5060 Ti) | mesa_v1 (112 items) | 112 | 0.4592 | 43.8% | No LLM judge. MikeAdapter (adapters/mike_adapter.py). 2026-04-21 |
| supergemma (Gemma-4 26B Q4¹, RTX 5060 Ti) | mesa_v1 (100 items) | 100 | 0.4377 | 41% | No LLM judge. Local inference rig, 2026-04-21 |
| supergemma (Gemma-4 26B Q4¹, RTX 5060 Ti) | fixtures | 9 | 0.7275 | 100% | No LLM judge. 2026-04-21 |

¹ **Abliterated model.** The Gemma-4 26B Q4 variant used in all runs above has had its refusal direction removed (abliteration). The base model will attempt to answer any question rather than refusing. **All adversarial scores therefore reflect the agent's instruction-layer guardrails only** — there is no model-level safety training to lean on. A system that scores well on `adversarial` items with this model is doing so through its own constitution and prompting, not inherited model behavior. For Mike specifically: 4 of 5 adversarial items scored as correct refusals, entirely from Mike's system prompt and relay logic. The 5th item failed because the model attempted a tool call rather than refusing, producing a malformed output (`<|channel><tool_call|>`) — a formatting artifact of the abliterated model under tool-use pressure, not a hallucinated answer.

**No human baseline is included.** A valid human baseline requires a cold reader — someone who has never seen the source conversations and answers only from the injected session text. The dataset is drawn from real conversations between the author and Mike, a personal AI companion. This material is deeply inside baseball: the author is the world's foremost expert on Mike's behavior, failure modes, and internal context. Any answer the author gives is contaminated by that knowledge, making the resulting score meaningless as a reference point. A human baseline will be added if and when a qualified cold reader is available to run the full sample blind.

**By type (supergemma, mesa_v1, 100 items, no judge):**

| Type | supergemma | Mike (full relay) |
|------|------------|-------------------|
| update/interference | 0.6153 | 0.6919 |
| update | 0.4319 | 0.5682 |
| temporal | 0.5967 | 0.5016 |
| recall/single | 0.3411 | 0.4836 |
| recall/constraint | 0.4855 | 0.4758 |
| adversarial | 0.4000 | 0.4000 |
| synthesis/multi | 0.4138 | 0.4072 |
| recall/preference | 0.4251 | 0.3897 |
| causal | 0.3278 | 0.3252 |

Mike's full relay pipeline was benchmarked on 112 items (dataset grew since supergemma run). Mike outperforms on `update` and `recall/single` — likely because his accumulated memory helps with facts from real prior conversations. Worse on `temporal` and `recall/preference` where tool calls sometimes distract from the injected session context.

---

## Running tests

```bash
pytest tests/ -v
```

---

## Known issues

**Adversarial scores on abliterated models test agent constitution, not model safety.** If your inference model has had refusal training removed, adversarial items measure only what your system prompt and relay logic enforce. Document this when reporting baselines.

**Legacy LLM judge is advisory, not authoritative.** It improves schema-v1 signal on causal and preference items, but it is still a prompt-based grader and should not be treated as a leaderboard-grade metric. Prefer schema-v2 runs for serious comparisons.

**Multi-session temporal items are underrepresented.** The schema and runner fully support multi-session injection (facts spread across dated sessions), but more items are needed to make the `by_session_format` breakdown statistically meaningful.

---

## Contributing

The gold dataset and scorer are the most valuable parts to improve. Contributions welcome:

- **New gold items**: Add or refine official items via `dataset/mesa_v2.json` and `dataset/schema_v2.json`. Multi-session temporal, causal, adversarial, and update/interference items are the highest-value gaps.
- **Legacy compatibility**: If you need `schema v1` for historical comparisons, keep changes clearly scoped and labeled as legacy.
- **Scorer improvements**: Typed v2 scoring and groundedness checks are the main scoring surface. PRs for better deterministic equivalence detection are welcome.
- **Adapter examples**: Real implementations (LangChain memory, MemGPT, custom vector stores) would be valuable references.
- **Baselines**: If you run MESA against a system, include dataset version, schema version, and benchmark metadata in your PR.

---

## Citation

If you use MESA in research, a mention in your methodology is appreciated:

```
Vitale Dynamics MESA v1 (2026). Memory Eval Benchmark for Personal AI Systems.
https://github.com/randomchaos7800-hub/mesa
```

Machine-readable citation metadata:

- [CITATION.cff](CITATION.cff)

---

## License

MIT
