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

Clone and install in editable mode:

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
mesa-benchmark run \
  --adapter examples.simple_adapter.EchoAdapter \
  --dataset dataset/mesa_v2.json \
  --schema-version 2 \
  --official-run
```

**Daily fast probes**

For quick iteration loops against a local inference endpoint:

```bash
mesa-benchmark run \
  --adapter adapters.keyword_adapter.KeywordAdapter \
  --dataset dataset/mesa_probes.json \
  --schema-version 2
```

This prints a one-page failure taxonomy + explicit `adapter_scope` tagging.
It is a fast diagnostic path, not an official baseline. `mesa_probes.json` has
no dataset manifest.

Compare two probe runs (pure Python, file-based):

```python
from mesa.runner import compare_probe_runs
delta = compare_probe_runs("results/run_v2_...json", "results/run_v2_...json")
print(delta["failure_deltas"])
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
mesa-benchmark run --adapter my_package.MyAdapter --schema-version 1 --dataset dataset/mesa_v1.json --no-llm-judge
```

**4. Try the example or reference adapters:**

```bash
# EchoAdapter: returns the raw injected context (smoke test)
mesa-benchmark run \
    --adapter examples.simple_adapter.EchoAdapter \
    --dataset dataset/fixtures/sample.json \
    --no-llm-judge

# NullAdapter: always refuses (adversarial baseline)
mesa-benchmark run \
    --adapter examples.simple_adapter.NullAdapter \
    --dataset dataset/fixtures/sample.json \
    --no-llm-judge

# KeywordAdapter: LLM extraction + TF-IDF retrieval (no vector DB required)
mesa-benchmark run \
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

All baselines below are on the v2 schema. The v1 composite-scored runs are archived in `docs/baselines.md`.

### v0.7.0 — full 361-item gold set (2026-05-25)

Scoring: EM + ROUGE-1, no LLM judge. 95% CI via 500-iteration bootstrap. Dataset: 109 reviewed + 252 provisional items across 8 domains and all 9 task types.

| System | Correct | 95% CI | Grounded | Abstention | Storage recall |
|---|---:|:---:|---:|---:|---:|
| **Mike** (production relay) | **0.249** | [0.195, 0.312] | 0.136 | 0.000 | 0.119 |
| **Hermes** (AIAgent, long-context) | **0.235** | [0.185, 0.294] | 0.303 | 0.105 | 0.098 |
| `EchoAdapter` (floor) | 0.217 | [0.167, 0.276] | 0.846 | 0.105 | 0.117 |
| `NullAdapter` (refusal floor) | 0.172 | [0.122, 0.231] | 1.000 | 1.000 | 0.000 |
| `DictAdapter` (pattern floor) | 0.154 | [0.113, 0.204] | 0.624 | 0.605 | 0.039 |

**Why v2 scores are lower than earlier numbers:** The 94-item v0.5.0 set was curated from real production conversations where Mike had genuine memory. The 252 new items are synthetic, cover 7 new domains, and require the system to work purely from injected context — no memory backing. That is a harder, and more honest, test. The full-set score is the real story.

**Why real systems are only 3–5 points above EchoAdapter:** EchoAdapter injects the full conversation as its "answer" and returns everything verbatim. On EM+ROUGE without a semantic judge, raw retrieval competes well against reasoning. The gap widens on `recall/*` types (where extraction matters) and `adversarial` (where Echo doesn't abstain but Mike/Hermes sometimes do). Semantic scoring (open issues #5, #6) will increase this gap.

**NullAdapter at 0.172:** The dataset has 38 adversarial items (10.5%) that reward correct refusal. NullAdapter scores 1.000 on those and 0.000 on everything else. Any real system must clear 0.172 to prove it adds value beyond "always refuse."

---

#### Per-type breakdown — all systems (361 items)

| Type | n | Mike | Hermes | Echo | Null | Dict |
|---|---:|---:|---:|---:|---:|---:|
| recall/constraint | 39 | **0.546** | **0.636** | 0.545 | 0.000 | 0.091 |
| recall/preference | 39 | 0.364 | **0.545** | 0.182 | 0.000 | 0.000 |
| recall/single | 40 | **0.450** | 0.400 | 0.400 | 0.000 | 0.100 |
| synthesis/multi | 40 | **0.417** | 0.333 | **0.417** | 0.000 | 0.083 |
| temporal | 43 | **0.333** | 0.267 | 0.279 | 0.000 | 0.000 |
| update/interference | 39 | **0.256** | 0.103 | 0.103 | 0.000 | 0.128 |
| causal | 42 | **0.214** | 0.071 | **0.214** | 0.000 | 0.000 |
| update | 41 | 0.098 | **0.146** | 0.098 | 0.000 | 0.000 |
| adversarial | 38 | 0.000 | 0.105 | 0.105 | **1.000** | 0.605 |

**Type diagnostics:**
- `recall/*` — Hermes leads here; LLM reasoning extracts facts that raw-text retrieval misses. Mike's real accumulated memory helps on familiar topics.
- `adversarial` — Mike scores 0.000 (never refuses). Hermes and Echo both score 0.105 — Hermes by design, Echo by incidentally returning "I don't know" when context is absent.
- `update` / `update/interference` — Mike's active relay tools handle updates better than Hermes' context-injection mode.
- `causal` / `synthesis/multi` — Mike's multi-tool reasoning matches Echo on causal; Hermes falls behind both.
- `temporal` — both real systems beat Echo slightly, but all struggle. Exact date recall requires faithful memory, not reasoning.

#### Per-domain breakdown — Mike vs Hermes

| Domain | n | Mike | Hermes | Notes |
|---|---:|---:|---:|---|
| developer-workflow | 6 | **0.667** | — | Old curated items; real memory backing |
| health (curated) | 6 | **0.833** | — | Old curated items |
| workplace (curated) | 7 | **0.571** | — | Old curated items |
| personal (curated) | 23 | 0.478 | — | Old curated items |
| infrastructure (curated) | 12 | 0.500 | — | Old curated items |
| operations (curated) | 26 | 0.308 | — | Old curated items |
| health_fitness_medical | 36 | 0.000 | — | New synthetic items |
| workplace_professional_projects | 36 | 0.063 | — | New synthetic items |
| infrastructure_devops_tech_operations | 36 | 0.125 | — | New synthetic items |
| education_research_learning | 36 | 0.125 | — | New synthetic items |
| legal_contracts_compliance | 36 | 0.125 | — | New synthetic items |
| personal_lifestyle | 36 | 0.125 | — | New synthetic items |
| finance_money_investments | 36 | 0.188 | — | New synthetic items |

**Domain gap:** Mike scores 30–83% on old curated items (drawn from real conversations he was part of), and 0–19% on the 252 new synthetic items (no real memory backing). This is the clearest signal in the v0.7.0 data: Mike's relay benefits substantially from accumulated real memory, not just context injection. Hermes' per-domain breakdown follows a similar pattern.

---

### v0.5.0 — 94-item curated set (archived)

See `docs/baselines.md` for the archived v0.5.0 table (Hermes 0.351, Mike 0.263, Echo/Null/Dict reference adapters).

---

**No human baseline is included.** A valid human baseline requires a cold reader with no prior knowledge of the source material. The dataset is drawn from real conversations; the author's knowledge contaminates any score they could produce. A human baseline will be added if a qualified cold reader is available.

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
