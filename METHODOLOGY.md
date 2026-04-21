# MESA Methodology

## Overview

MESA (Memory Eval Suite for Agents) is a white-box evaluation framework for personal AI memory pipelines. Unlike black-box evals that test the language model in isolation, MESA tests the full loop: session injection → fact extraction → knowledge graph storage → retrieval-augmented response.

## Motivation

Published benchmarks like LongMemEval test context-window retention or external RAG. They don't test the custom extraction→storage→retrieval loop that production memory agents use. MESA fills that gap with questions built from real AI companion conversations, so scores reflect actual usage patterns rather than synthetic benchmarks.

## Pipeline Under Test

```
sessions ← inject_session() [once per session, in chronological order]
     ↓
extract_to_memory()           # Two-stage extraction (LLM → Mem0 compare)
     ↓
knowledge graph (memory/)     # entities + facts + timeline
     ↓
relay.respond(question)       # Relay with memory context injection
     ↓
score(predicted, expected)
```

Each benchmark item runs in a fully isolated temporary directory — real conversation history is never written to.

## Session Formats

### Single-session items
The majority of items inject one contiguous conversation window. The adapter's `inject(turns)` method is called once.

### Multi-session items
Some items (especially `temporal`, `update`, and `update/interference`) inject facts spread across multiple dated sessions. The runner calls `inject_session(turns, session_date)` once per session in chronological order.

The default `MemoryAdapter.inject_session()` delegates to `inject()`, ignoring the date. Adapters that tag memories with conversation dates should override `inject_session()` to pass the date through to their storage layer.

Multi-session items test behaviors that single-session items cannot:
- **Cross-session retention**: a fact from session 1 must still be retrievable in session 3
- **Cross-session update**: a fact changed in session 2 must supersede the session 1 value
- **Cross-session interference**: similar facts from later sessions must not corrupt earlier ones
- **Temporal anchoring**: date metadata from `session_date` must survive into the retrieval layer

## Question Types

### recall/single
Single fact stated once in the injected session. Tests baseline extraction fidelity: did the pipeline extract the fact and can the relay retrieve it?

### recall/preference
A preference or behavioral opinion expressed by the user (e.g. "I hate long explanations"). Tests whether extraction captures subjective preferences, not just objective facts.

### recall/constraint
A hard rule or constraint stated by the user (e.g. "never exceed $20/month, no exceptions"). Subsequent turns reference related but different values, acting as noise. Tests whether the relay retains the constraint as a *rule* rather than just a fact — specifically whether it survives interference from similar data stated later.

### synthesis/multi
The correct answer requires combining two or more facts from the injected sessions. Tests whether the relay can reason across multiple extracted facts rather than surfacing a single stored value.

### temporal
Involves a date or time relationship ("when did", "how many days ago"). Tests whether temporal metadata survives extraction and whether the relay can report timing correctly. Multi-session temporal items inject facts across dated sessions; the answer is often the date from a specific session.

### update
A fact is stated, then explicitly superseded within the same session (single-session) or a later session (multi-session). Tests conflict resolution: the superseded fact should be marked inactive and the new fact should be the authoritative answer.

### update/interference
A fact is stated, then multiple similar-but-not-identical facts are added in later turns or sessions. The question asks for the *original* fact. Tests for memory interference: does the similar new data overwrite or blur the original? Harder than `update` because there is no explicit supersession.

### adversarial
The question asks about something never stated in any injected session. `sessions: []` — zero context is injected. The relay must say it doesn't know rather than hallucinate. Scoring treats any refusal/uncertainty response as a pass, any fabricated answer as a fail.

### causal
Answering correctly requires both fact A and fact B, neither of which individually implies C. Tests whether the relay can perform multi-hop reasoning over extracted facts.

## Composite Scoring

Three dimensions are combined into a single composite score per item.

**Exact match (0.0–1.0)**  
Normalized substring and word-overlap scoring. Handles markdown, punctuation, date format variants, and multi-answer expected strings. Scores: 1.0 for full containment, 0.8 for ≥80% word overlap, 0.5 for ≥50% overlap, 0.0 otherwise.

**ROUGE-1 F1 (0.0–1.0)**  
Unigram recall/precision F1 using the `rouge-score` library with stemming. Captures partial credit for answers that share significant vocabulary with the expected answer without exact substring match.

**LLM judge (0.0–1.0)**  
Graded verdict (0–3) from a local inference server (OpenAI-compatible API), normalized to 0.0–1.0:
- 3 → 1.00 — fully correct: all key information matches
- 2 → 0.67 — mostly correct: minor omission or imprecision
- 1 → 0.33 — partially correct: some relevant info present but incomplete
- 0 → 0.00 — wrong: hallucinated, missing key facts, or refused without cause

The graded scale captures partial credit that binary YES/NO misses, particularly on causal and preference questions.

**Composite weights:**

| Mode | Exact | LLM judge | ROUGE-1 |
|------|-------|-----------|---------|
| Full | 0.40 | 0.30 | 0.30 |
| No judge | 0.55 | — | 0.45 |

When the LLM judge is unavailable or skipped, its weight redistributes proportionally between exact and ROUGE-1.

**Pass threshold**: composite ≥ 0.50

## Cross-Session Reporting

The runner includes per-item session metadata in result JSON:

- `session_format`: `"single"` or `"multi"`
- `session_count`: number of sessions injected (1 for single-session items)

The summary includes a `by_session_format` breakdown comparing composite scores and pass rates between single-session and multi-session items. This makes it possible to track whether a system's cross-session retention regresses as multi-session items are added.

## Data Provenance

### Handcrafted fixtures (`dataset/fixtures/sample.json`)
10 items written by hand, covering all 9 question types plus one multi-session temporal example. Used as the smoke-test dataset — if these don't pass, something is fundamentally broken.

### Mined candidates (`dataset/candidates/`)
Generated by `build_dataset.py` using the local LLM to produce question/answer pairs from two sources:

- **sessions.db** — Real AI companion conversation history. Opened read-only. The LLM is given a window of conversation turns and asked to generate diverse Q/A pairs.
- **LIGHTHOUSE** — Behavioral self-documentation (corrections, identity, patterns). Provides a rich source of preference and behavioral facts.

All candidates are stored in dated batch directories and never auto-added to the gold dataset.

### Gold dataset (`dataset/mesa_v1.json`)
Human-reviewed via `review.py`. Every item is manually approved before entering the gold set. This gatekeeping ensures:
- Expected answers are actually correct
- Questions are answerable from the injected sessions
- Adversarial items are genuinely unanswerable
- Multi-session items have valid date sequences

## Memory Isolation Design

Each benchmark item runs in a fresh `tempfile.TemporaryDirectory`. Before the relay is imported, three module-level paths are patched:

```python
storage.MEMORY_ROOT → tmpdir
session_log.SESSIONS_DIR → tmpdir/sessions/
sessions_mod.DEFAULT_DB_PATH → tmpdir/sessions.db
```

This ensures the real conversation database is never touched during a benchmark run. Each item's memory state starts empty — the only facts available to the relay are those extracted from the item's injected sessions.

## Limitations

- **Extraction quality depends on the local model.** If the model fails to extract a fact (parse error, truncation, hallucinated output), the relay will have no memory to retrieve from. This inflates false negatives. The `stored_facts()` diagnostic in the runner helps distinguish extraction failure from retrieval failure.
- **Temporal items require date propagation.** For single-session items, date markers are embedded in the first user turn per LongMemEval convention. For multi-session items, the `session_date` parameter in `inject_session()` carries the date — but only if the adapter overrides the default implementation to store it.
- **Multi-session temporal items are underrepresented.** The schema and runner fully support multi-session injection, but more items are needed to make the `by_session_format` breakdown statistically meaningful.
- **LLM judge requires a running inference server.** Without `--llm-judge`, causal and preference items score lower than their true accuracy because the no-judge scorer penalizes paraphrasing.

## Versioning

The schema file (`dataset/schema.json`) is versioned. Breaking changes to the item format require a new schema version and migration script. The run output JSON includes the dataset path so results are traceable to the exact dataset version.
