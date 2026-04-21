# MESA — Memory Eval Benchmark

A reproducible benchmark framework for **personal AI memory systems**.

Most AI memory benchmarks test context-window retention or external RAG pipelines. MESA tests the full loop: **inject → extract → store → retrieve → answer**. The question types are grounded in real failure modes observed in production AI companion systems.

---

## What it tests

Seven question types, each targeting a distinct failure mode:

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

Or clone and install in editable mode:

```bash
git clone https://github.com/randomchaos7800-hub/mmeb
cd mmeb
pip install -e ".[dev]"
```

---

## Quickstart

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

**2. Run the benchmark:**

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

Or from the CLI:
```bash
python -m mesa.runner --adapter my_package.MyAdapter --no-llm-judge
```

**3. Try the example adapters first:**

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
```

---

## Scoring

Three dimensions, combined into a single composite score per item:

**Exact match (0.0–1.0)**
Normalized substring and word-overlap. Handles markdown, punctuation, date format variants (`"March 1, 2026"` ≡ `"2026-03-01"`), and multi-answer expected strings.

**ROUGE-1 F1 (0.0–1.0)**
Unigram recall/precision F1 with stemming. Captures partial credit for answers that share vocabulary without exact containment.

**LLM judge (0.0 or 1.0)**
Binary YES/NO verdict from an OpenAI-compatible model. Useful for nuanced causal and preference questions where the correct answer can be phrased many ways.

**Composite weights:**

| Mode | Exact | LLM judge | ROUGE-1 |
|------|-------|-----------|---------|
| With judge | 0.40 | 0.30 | 0.30 |
| No judge | 0.55 | — | 0.45 |

**Pass threshold**: composite ≥ 0.50

**Adversarial items** are scored via refusal detection (`is_refusal(predicted)`) rather than exact match. A correct system says it doesn't know; a hallucinating system fabricates an answer and scores 0.

---

## Dataset

`dataset/mesa_v1.json` — 20 hand-curated items from real AI companion conversations. Covers all 7 question types. Each item:

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

`dataset/fixtures/sample.json` — 5 hand-crafted smoke-test items, one per primary type. If your system can't pass these, something is fundamentally broken.

---

## Baselines

| System | Dataset | Items | Composite | Pass rate | Notes |
|--------|---------|-------|-----------|-----------|-------|
| supergemma (Gemma-4 26B Q4, RTX 5060 Ti) | mesa_v1 | 20 | 0.3444 | 30% | No LLM judge. Local inference rig, 2026-04-21 |
| supergemma (Gemma-4 26B Q4, RTX 5060 Ti) | fixtures | 5 | 0.7275 | 100% | No LLM judge. 2026-04-21 |

**By type (mesa_v1, no judge):**

| Type | Score |
|------|-------|
| update | 0.5938 |
| recall/single | 0.4222 |
| causal | 0.3162 |
| recall/preference | 0.2829 |
| adversarial | 0.0000 * |

*adversarial 0.00: the two adversarial items were answerable via filesystem tools, not a hallucination failure. See [Known Issues](#known-issues).*

---

## Running tests

```bash
pytest tests/ -v
```

---

## Known issues

**Adversarial items in v1 are tool-resolvable.** Two of the adversarial items can be answered by a system with filesystem access. Future versions will use questions about things that cannot be looked up anywhere.

**LLM judge needed for causal/preference.** Without the judge, causal and preference items score low even when the answer is semantically correct — the no-judge scorer penalizes paraphrasing. Run with `--llm-judge` and a local model for better signal on these types.

**Single-session injection only.** v1 items inject one conversation window. Multi-session temporal reasoning (facts spread across days) is not yet represented.

---

## Contributing

The gold dataset and scorer are the most valuable parts to improve. Contributions welcome:

- **New gold items**: Add items via `dataset/mesa_v1.json` with the schema in `dataset/schema.json`. All 7 question types need more coverage, especially `synthesis/multi` and `temporal`.
- **Scorer improvements**: The exact match scorer is conservative. PRs for better semantic equivalence detection (without requiring an LLM) are welcome.
- **Adapter examples**: Real implementations (LangChain memory, MemGPT, custom vector stores) would be valuable references.
- **Baselines**: If you run MESA against a system, open a PR to add it to the baselines table.

---

## Citation

If you use MESA in research, a mention in your methodology is appreciated:

```
Vitale Dynamics MESA v1 (2026). Memory Eval Benchmark for Personal AI Systems.
https://github.com/randomchaos7800-hub/mmeb
```

---

## License

MIT
