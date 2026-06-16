"""MESA benchmark runner.

Loops over a dataset, calls your MemoryAdapter for each item, scores responses,
and writes results to a JSON file.

Usage:
    from mesa.runner import run_benchmark
    from my_adapter import MyAdapter

    results = run_benchmark(
        dataset_path="dataset/mesa_v1.json",
        adapter=MyAdapter(),
        no_llm_judge=True,
    )

Or from the CLI:
    python -m mesa.runner --adapter examples.simple_adapter.EchoAdapter
    python -m mesa.runner --adapter my_package.MyAdapter --no-llm-judge
"""

import argparse
import importlib
import json
import logging
import statistics
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Optional

from mesa.adapter import MemoryAdapter
from mesa.dataset.manifest import load_dataset_manifest
from mesa.dataset.migrators import upgrade_v1_item
from mesa.scoring.answer_types import (
    score_abstention_answer,
    score_causal_answer,
    score_constraint_answer,
    score_multi_fact_answer,
    score_preference_answer,
    score_single_fact_answer,
    score_temporal_answer,
    score_update_interference_answer,
    score_update_current_answer,
)
from mesa.scoring.deterministic import match_fact_ids
from mesa.scoring.stats import bootstrap_mean_ci
from mesa.scorer import exact_match, rouge1_f1, composite, is_refusal, llm_judge

logger = logging.getLogger(__name__)

DEFAULT_DATASET = Path(__file__).parent.parent / "dataset" / "mesa_v1.json"
DEFAULT_DATASET_V2 = Path(__file__).parent.parent / "dataset" / "mesa_v2.json"
RESULTS_DIR = Path("results")


def _is_multi_session(sessions: list) -> bool:
    """Return True if sessions is multi-session format (list of {date, turns})."""
    return bool(sessions) and isinstance(sessions[0], dict) and "turns" in sessions[0]


def _inject(adapter: "MemoryAdapter", sessions: list) -> None:
    """Dispatch single-session or multi-session injection to the adapter."""
    if _is_multi_session(sessions):
        for session in sessions:
            adapter.inject_session(session["turns"], session_date=session.get("date"))
    else:
        adapter.inject(sessions)


def _normalize_v2_sessions(sessions: list[dict]) -> list[dict]:
    """Convert v2 session objects into the runner injection shape."""
    return [
        {
            "date": session.get("date"),
            "turns": session["turns"],
        }
        for session in sessions
    ]


def _collect_writes(adapter: MemoryAdapter) -> list[dict] | None:
    """Collect structured writes, falling back to legacy stored_facts()."""
    writes = adapter.get_writes()
    if writes is not None:
        return [
            {
                "memory_id": write.memory_id,
                "text": write.text,
                "metadata": write.metadata,
            }
            for write in writes
        ]

    facts = adapter.stored_facts()
    if facts is None:
        return None
    return [
        {
            "memory_id": None,
            "text": fact,
            "metadata": {"source": "stored_facts"},
        }
        for fact in facts
    ]


def _collect_answer_trace(adapter: MemoryAdapter, question: str) -> tuple[dict, bool]:
    """Collect answer output, preferring structured trace when available."""
    trace = adapter.ask_with_trace(question)
    if trace is not None:
        return {
            "answer": trace.answer,
            "retrieved": None if trace.retrieved is None else [
                {
                    "memory_id": item.memory_id,
                    "text": item.text,
                    "score": item.score,
                    "metadata": item.metadata,
                }
                for item in trace.retrieved
            ],
            "metadata": trace.metadata,
        }, True

    answer = adapter.ask(question)
    return {"answer": answer, "retrieved": None, "metadata": {}}, False


def _collect_session_texts(item: dict) -> list[str]:
    """Flatten session turns into plain text for grounding fallback."""
    texts = []
    for session in item.get("sessions", []):
        for turn in session.get("turns", []):
            texts.append(turn.get("content", ""))
    return texts


def _score_item_v2(item: dict, writes: list[dict] | None, answer_trace: dict) -> tuple[dict, dict, dict, list[str]]:
    """Score a v2 item across storage, retrieval, and answer stages."""
    gold_memory = item["gold_memory"]
    atomic_facts = gold_memory.get("atomic_facts", [])
    required_fact_ids = set(gold_memory.get("required_fact_ids", []))
    forbidden_fact_ids = set(gold_memory.get("forbidden_fact_ids", []))

    write_texts = [write["text"] for write in (writes or [])]
    retrieved_items = answer_trace["retrieved"] or []
    retrieved_texts = [entry["text"] for entry in retrieved_items]

    stored_fact_ids = match_fact_ids(write_texts, atomic_facts) if write_texts else set()
    retrieved_fact_ids = match_fact_ids(retrieved_texts, atomic_facts) if retrieved_texts else set()
    stored_required_hits = len(stored_fact_ids & required_fact_ids)
    retrieved_required_hits = len(retrieved_fact_ids & required_fact_ids)
    stored_non_required_hits = len(stored_fact_ids - required_fact_ids)
    retrieved_non_required_hits = len(retrieved_fact_ids - required_fact_ids)
    unannotated_write_count = max(0, len(write_texts) - len(stored_fact_ids))
    unannotated_retrieval_count = max(0, len(retrieved_texts) - len(retrieved_fact_ids))

    storage_metrics = {
        "required_fact_recall": round(stored_required_hits / len(required_fact_ids), 4)
        if required_fact_ids else None,
        "required_fact_precision": round(stored_required_hits / len(write_texts), 4)
        if write_texts else None,
        "forbidden_fact_hits": len(stored_fact_ids & forbidden_fact_ids),
        "non_required_fact_hits": stored_non_required_hits,
        "unannotated_write_count": unannotated_write_count,
        "matched_fact_ids": sorted(stored_fact_ids),
    }
    retrieval_metrics = {
        "required_fact_recall": round(retrieved_required_hits / len(required_fact_ids), 4)
        if required_fact_ids else None,
        "required_fact_precision": round(retrieved_required_hits / len(retrieved_texts), 4)
        if retrieved_texts else None,
        "forbidden_fact_hits": len(retrieved_fact_ids & forbidden_fact_ids),
        "non_required_fact_hits": retrieved_non_required_hits,
        "unannotated_retrieval_count": unannotated_retrieval_count,
        "matched_fact_ids": sorted(retrieved_fact_ids),
    }

    evidence_texts = retrieved_texts or write_texts or _collect_session_texts(item)
    answer_format = item["answer_format"]
    if answer_format == "abstention":
        answer_metrics = score_abstention_answer(answer_trace["answer"], item["gold_answer"], evidence_texts)
    elif answer_format == "preference":
        answer_metrics = score_preference_answer(answer_trace["answer"], item["gold_answer"], evidence_texts)
    elif answer_format == "constraint":
        answer_metrics = score_constraint_answer(answer_trace["answer"], item["gold_answer"], evidence_texts)
    elif answer_format == "single_fact":
        answer_metrics = score_single_fact_answer(answer_trace["answer"], item["gold_answer"], evidence_texts)
    elif answer_format == "temporal":
        answer_metrics = score_temporal_answer(answer_trace["answer"], item["gold_answer"], evidence_texts)
    elif answer_format == "update_current":
        answer_metrics = score_update_current_answer(answer_trace["answer"], item["gold_answer"], evidence_texts)
    elif answer_format == "multi_fact":
        if item["task_type"] == "causal":
            answer_metrics = score_causal_answer(answer_trace["answer"], item["gold_answer"], evidence_texts)
        else:
            answer_metrics = score_multi_fact_answer(answer_trace["answer"], item["gold_answer"], evidence_texts)
    elif item["task_type"] == "update/interference":
        answer_metrics = score_update_interference_answer(answer_trace["answer"], item["gold_answer"], evidence_texts)
    else:
        answer_metrics = {
            "correct": None,
            "grounded": None,
            "unsupported_claims": [],
            "missing_required": [],
            "forbidden_mentions": [],
            "abstention_correct": None,
        }

    failures = []
    if storage_metrics["required_fact_recall"] not in (None, 1.0):
        failures.append("missing_required_storage_fact")
    if retrieval_metrics["required_fact_recall"] not in (None, 1.0) and retrieved_items:
        failures.append("missing_required_retrieval_fact")
    if storage_metrics["forbidden_fact_hits"]:
        failures.append("stored_forbidden_fact")
    if retrieval_metrics["forbidden_fact_hits"]:
        failures.append("retrieved_forbidden_fact")
    if storage_metrics["non_required_fact_hits"] or storage_metrics["unannotated_write_count"]:
        failures.append("stored_extra_fact")
    if retrieval_metrics["non_required_fact_hits"] or retrieval_metrics["unannotated_retrieval_count"]:
        failures.append("retrieved_extra_fact")
    if answer_metrics["correct"] is False:
        failures.append("incorrect_answer")
    if answer_metrics["grounded"] is False:
        failures.append("unsupported_answer_claim")
    if answer_metrics["abstention_correct"] is False:
        failures.append("unclean_abstention")

    return storage_metrics, retrieval_metrics, answer_metrics, failures


def _average_metric(values: list[float | bool | None]) -> float | None:
    """Average numeric and boolean metrics, skipping missing values."""
    normalized = []
    for value in values:
        if value is None:
            continue
        if isinstance(value, bool):
            normalized.append(1.0 if value else 0.0)
        else:
            normalized.append(float(value))
    if not normalized:
        return None
    return round(sum(normalized) / len(normalized), 4)


def _summarize_v2_results(results: list[dict]) -> dict:
    """Build run-level v2 metric summaries."""
    storage = {
        "required_fact_recall": _average_metric([r["storage"]["metrics"]["required_fact_recall"] for r in results]),
        "required_fact_precision": _average_metric([r["storage"]["metrics"]["required_fact_precision"] for r in results]),
        "forbidden_fact_hits": round(sum(r["storage"]["metrics"]["forbidden_fact_hits"] for r in results) / len(results), 4),
        "non_required_fact_hits": round(sum(r["storage"]["metrics"]["non_required_fact_hits"] for r in results) / len(results), 4),
        "unannotated_write_count": round(sum(r["storage"]["metrics"]["unannotated_write_count"] for r in results) / len(results), 4),
    }
    retrieval = {
        "required_fact_recall": _average_metric([r["retrieval"]["metrics"]["required_fact_recall"] for r in results]),
        "required_fact_precision": _average_metric([r["retrieval"]["metrics"]["required_fact_precision"] for r in results]),
        "forbidden_fact_hits": round(sum(r["retrieval"]["metrics"]["forbidden_fact_hits"] for r in results) / len(results), 4),
        "non_required_fact_hits": round(sum(r["retrieval"]["metrics"]["non_required_fact_hits"] for r in results) / len(results), 4),
        "unannotated_retrieval_count": round(sum(r["retrieval"]["metrics"]["unannotated_retrieval_count"] for r in results) / len(results), 4),
    }
    answer = {
        "correct_rate": _average_metric([r["answer"]["metrics"]["correct"] for r in results]),
        "grounded_rate": _average_metric([r["answer"]["metrics"]["grounded"] for r in results]),
        "abstention_correct_rate": _average_metric([r["answer"]["metrics"]["abstention_correct"] for r in results]),
        "unsupported_claim_items": round(sum(1 for r in results if r["answer"]["metrics"]["unsupported_claims"]) / len(results), 4),
    }
    by_type = {}
    for result in results:
        by_type.setdefault(result["task_type"], []).append(result)
    by_type_summary = {}
    for task_type, items in by_type.items():
        by_type_summary[task_type] = {
            "n": len(items),
            "correct_rate": _average_metric([r["answer"]["metrics"]["correct"] for r in items]),
            "grounded_rate": _average_metric([r["answer"]["metrics"]["grounded"] for r in items]),
        }
    by_domain = {}
    by_session_format = {}
    for result in results:
        domain = result.get("metadata", {}).get("domain", "unknown")
        session_format = result.get("session_format", "single")
        by_domain.setdefault(domain, []).append(result)
        by_session_format.setdefault(session_format, []).append(result)
    by_domain_summary = {}
    for domain, items in by_domain.items():
        by_domain_summary[domain] = {
            "n": len(items),
            "correct_rate": _average_metric([r["answer"]["metrics"]["correct"] for r in items]),
            "grounded_rate": _average_metric([r["answer"]["metrics"]["grounded"] for r in items]),
        }
    by_session_summary = {}
    for session_format, items in by_session_format.items():
        by_session_summary[session_format] = {
            "n": len(items),
            "correct_rate": _average_metric([r["answer"]["metrics"]["correct"] for r in items]),
            "grounded_rate": _average_metric([r["answer"]["metrics"]["grounded"] for r in items]),
        }
    confidence_intervals = {
        "answer.correct_rate": bootstrap_mean_ci([r["answer"]["metrics"]["correct"] for r in results]),
        "answer.grounded_rate": bootstrap_mean_ci([r["answer"]["metrics"]["grounded"] for r in results]),
        "answer.abstention_correct_rate": bootstrap_mean_ci([r["answer"]["metrics"]["abstention_correct"] for r in results]),
    }
    return {
        "storage": storage,
        "retrieval": retrieval,
        "answer": answer,
        "by_type": by_type_summary,
        "by_domain": by_domain_summary,
        "by_session_format": by_session_summary,
        "confidence_intervals": confidence_intervals,
    }


def _run_item_once(
    adapter: "MemoryAdapter",
    item: dict,
    sessions: list,
    question: str,
    item_id: str,
    effective_trace_required: bool,
    official_run: bool,
) -> dict:
    """One inject+score cycle. Raises ValueError on trace violations."""
    adapter.reset()
    _inject(adapter, sessions)
    writes = _collect_writes(adapter)

    t0 = time.time()
    try:
        answer_trace, _has_trace = _collect_answer_trace(adapter, question)
    except Exception as e:
        logger.warning(f"  Adapter error for {item_id}: {e}")
        answer_trace = {"answer": "", "retrieved": None, "metadata": {"error": str(e)}}
    elapsed = round(time.time() - t0, 2)

    write_trace_available = writes is not None
    retrieval_trace_available = answer_trace["retrieved"] is not None
    observable = write_trace_available or retrieval_trace_available

    if effective_trace_required and not observable:
        raise ValueError(f"Adapter does not expose trace hooks required for v2 run: {item_id}")
    if official_run and not retrieval_trace_available:
        raise ValueError(f"Official v2 runs require retrieval trace support: {item_id}")

    storage_metrics, retrieval_metrics, answer_metrics, failures = _score_item_v2(item, writes, answer_trace)
    retrieval_metrics["trace_available"] = retrieval_trace_available
    storage_metrics["trace_available"] = write_trace_available
    if not retrieval_trace_available:
        failures = list(failures) + ["retrieval_trace_missing"]

    return {
        "writes": writes,
        "answer_trace": answer_trace,
        "elapsed": elapsed,
        "storage_metrics": storage_metrics,
        "retrieval_metrics": retrieval_metrics,
        "answer_metrics": answer_metrics,
        "failures": failures,
        "write_trace_available": write_trace_available,
        "retrieval_trace_available": retrieval_trace_available,
        "observable": observable,
    }


def run_benchmark_v2(
    adapter: MemoryAdapter,
    dataset_path: Optional[Path] = None,
    trace_required: bool = False,
    limit: Optional[int] = None,
    quiet: bool = False,
    official_run: bool = False,
    n_runs: int = 1,
    dry_run: bool = False,
) -> dict:
    """Run the v2 benchmark scaffolding with structured traces.

    This is the official MESA v2 benchmark runner.
    """
    if dataset_path is None:
        dataset_path = DEFAULT_DATASET_V2

    manifest = load_dataset_manifest(dataset_path)
    if official_run and manifest is None:
        raise ValueError("Official v2 runs require a dataset manifest")
    effective_trace_required = trace_required or official_run
    with open(dataset_path) as f:
        dataset = json.load(f)

    if limit:
        dataset = dataset[:limit]

    if not dataset:
        raise ValueError(f"Dataset is empty or all items filtered out (path={dataset_path})")

    # Scope / contamination contract enforcement (Improvement #2)
    adapter_scope = adapter.get_scope() if hasattr(adapter, "get_scope") else getattr(adapter, "scope", "full_production")
    if official_run and adapter_scope != "pure_injection":
        raise ValueError(
            f"Official v2 runs require scope='pure_injection'; adapter reports '{adapter_scope}'. "
            "Use a properly isolated adapter for official baselines."
        )

    if dry_run:
        by_type: dict[str, int] = {}
        for raw in dataset:
            t = raw.get("task_type", raw.get("type", "unknown"))
            by_type[t] = by_type.get(t, 0) + 1
        total = len(dataset) * n_runs
        print(f"\nmesa dry-run: {len(dataset)} items × {n_runs} run(s) = {total} total")
        print(f"  Dataset : {dataset_path}")
        print(f"  Adapter : {adapter.__class__.__name__}")
        print(f"  Schema  : v2")
        print(f"\n  Type breakdown:")
        for t, n in sorted(by_type.items()):
            print(f"    {t:<30} {n}")
        return {"run_id": "dry-run", "n_items": len(dataset), "dry_run": True}

    results = []
    for idx, item in enumerate(dataset):
        if item.get("version") != "2":
            item = upgrade_v1_item(item)

        item_id = item.get("id", f"item-{idx:04d}")
        question = item["question"]
        sessions = _normalize_v2_sessions(item.get("sessions", []))

        if not quiet:
            logger.info(f"[v2 {idx+1}/{len(dataset)}] {item_id} ({item.get('task_type', 'unknown')})")

        run_corrects: list[float] = []
        first: dict = {}
        for _ in range(max(1, n_runs)):
            run = _run_item_once(
                adapter, item, sessions, question, item_id,
                effective_trace_required, official_run,
            )
            if not first:
                first = run
            c = run["answer_metrics"].get("correct")
            if c is not None:
                run_corrects.append(float(c))

        answer_metrics = dict(first["answer_metrics"])
        if n_runs > 1 and run_corrects:
            correct_mean = round(sum(run_corrects) / len(run_corrects), 4)
            correct_std = round(statistics.stdev(run_corrects), 4) if len(run_corrects) > 1 else 0.0
            answer_metrics["correct"] = correct_mean >= 0.5
            answer_metrics["correct_mean"] = correct_mean
            answer_metrics["correct_std"] = correct_std

        results.append(
            {
                "id": item_id,
                "version": item["version"],
                "task_type": item["task_type"],
                "answer_format": item["answer_format"],
                "metadata": item.get("metadata", {}),
                "session_format": "multi" if len(sessions) > 1 else "single",
                "session_count": len(sessions) if sessions else 0,
                "adapter_scope": adapter_scope,
                "observable": first["observable"],
                "write_trace_available": first["write_trace_available"],
                "retrieval_trace_available": first["retrieval_trace_available"],
                "n_runs": n_runs,
                "storage": {
                    "writes": first["writes"],
                    "metrics": first["storage_metrics"],
                },
                "retrieval": {
                    "retrieved": first["answer_trace"]["retrieved"],
                    "metrics": first["retrieval_metrics"],
                },
                "answer": {
                    "text": first["answer_trace"]["answer"],
                    "metadata": first["answer_trace"]["metadata"],
                    "metrics": answer_metrics,
                },
                "failures": first["failures"],
                "elapsed_s": first["elapsed"],
            }
        )

    return {
        "run_id": datetime.now().strftime("%Y-%m-%d_%H-%M"),
        "dataset": str(dataset_path),
        "dataset_name": manifest.get("dataset_name") if manifest else dataset_path.stem,
        "dataset_split": manifest.get("split", "unspecified") if manifest else "unspecified",
        "schema_version": "2",
        "dataset_version": manifest.get("dataset_version") if manifest else None,
        "benchmark_release": manifest.get("benchmark_release") if manifest else None,
        "n_items": len(results),
        "trace_required": effective_trace_required,
        "official_run": official_run,
        "adapter_scope": adapter_scope,
        "observable_rate": _average_metric([r["observable"] for r in results]),
        "summary": _summarize_v2_results(results),
        "results": results,
    }


def run_benchmark(
    adapter: MemoryAdapter,
    dataset_path: Optional[Path] = None,
    no_llm_judge: bool = True,
    judge_client=None,
    judge_model: str = "local",
    type_filter: Optional[str] = None,
    limit: Optional[int] = None,
    quiet: bool = False,
) -> dict:
    """Run the full benchmark and return a summary dict.

    Args:
        adapter: Your MemoryAdapter implementation.
        dataset_path: Path to the benchmark JSON (default: dataset/mesa_v1.json).
        no_llm_judge: Skip LLM judge scoring (reweights composite to 0.55/0.45).
        judge_client: OpenAI-compatible client for LLM judge calls (required if not no_llm_judge).
        judge_model: Model name for LLM judge (default: "local").
        type_filter: Only run items of this question type.
        limit: Max items to evaluate.
        quiet: Suppress per-item log output.

    Returns:
        Summary dict with keys: run_id, n_items, avg_composite, pass_rate_50pct,
        by_type, results (list of per-item result dicts).
    """
    if dataset_path is None:
        dataset_path = DEFAULT_DATASET

    with open(dataset_path) as f:
        dataset = json.load(f)

    if type_filter:
        dataset = [d for d in dataset if d.get("type") == type_filter]

    if limit:
        dataset = dataset[:limit]

    if not dataset:
        raise ValueError(f"Dataset is empty or all items filtered out (path={dataset_path})")

    results = []

    for idx, item in enumerate(dataset):
        item_id = item.get("id", f"item-{idx:04d}")
        q_type = item.get("type", "unknown")
        question = item["question"]
        expected = item["expected_answer"]
        sessions = item.get("sessions", [])

        if not quiet:
            logger.info(f"[{idx+1}/{len(dataset)}] {item_id} ({q_type})")

        # --- Run the adapter ---
        is_multi = _is_multi_session(sessions)
        session_count = len(sessions) if is_multi else 1
        adapter.reset()
        _inject(adapter, sessions)
        facts = adapter.stored_facts()

        t0 = time.time()
        try:
            predicted = adapter.ask(question)
        except Exception as e:
            logger.warning(f"  Adapter error for {item_id}: {e}")
            predicted = ""
        elapsed = round(time.time() - t0, 2)

        # --- Score ---
        is_adversarial = q_type == "adversarial"

        if is_adversarial:
            em = is_refusal(predicted)
            r1 = em
            judge_result = {"score": None, "verdict": "SKIPPED", "reason": "adversarial-refusal-check"}
            if not no_llm_judge and judge_client:
                judge_result = llm_judge(predicted, expected, question, judge_client, judge_model)
            comp = composite(em, r1, judge_result["score"], use_llm_judge=not no_llm_judge)
        else:
            em = exact_match(predicted, expected)
            r1 = rouge1_f1(predicted, expected)
            judge_result = {"score": None, "verdict": "SKIPPED", "reason": ""}
            if not no_llm_judge and judge_client:
                judge_result = llm_judge(predicted, expected, question, judge_client, judge_model)
            comp = composite(em, r1, judge_result["score"], use_llm_judge=not no_llm_judge)

        result = {
            "id": item_id,
            "type": q_type,
            "session_format": "multi" if is_multi else "single",
            "session_count": session_count,
            "question": question,
            "expected": expected,
            "predicted": predicted,
            "stored_facts": facts,
            "scores": {
                "exact": em,
                "rouge1": r1,
                "llm_judge": judge_result["score"],
                "composite": comp,
            },
            "judge": judge_result,
            "elapsed_s": elapsed,
        }
        results.append(result)

        if not quiet:
            verdict_str = f"judge={judge_result['verdict']}" if not no_llm_judge else "no-judge"
            logger.info(f"  exact={em:.2f} rouge={r1:.2f} composite={comp:.2f} [{verdict_str}] {elapsed}s")

    # Summary
    composites = [r["scores"]["composite"] for r in results]
    avg_composite = round(sum(composites) / len(composites), 4) if composites else 0.0
    pass_rate = round(sum(1 for c in composites if c >= 0.5) / len(composites), 4) if composites else 0.0

    by_type: dict[str, list[float]] = {}
    for r in results:
        by_type.setdefault(r["type"], []).append(r["scores"]["composite"])
    type_scores = {t: round(sum(v) / len(v), 4) for t, v in by_type.items()}

    by_fmt: dict[str, list[float]] = {}
    for r in results:
        by_fmt.setdefault(r["session_format"], []).append(r["scores"]["composite"])
    fmt_summary = {
        fmt: {
            "n": len(v),
            "avg_composite": round(sum(v) / len(v), 4),
            "pass_rate": round(sum(1 for c in v if c >= 0.5) / len(v), 4),
        }
        for fmt, v in by_fmt.items()
    }

    return {
        "run_id": datetime.now().strftime("%Y-%m-%d_%H-%M"),
        "dataset": str(dataset_path),
        "n_items": len(results),
        "llm_judge": not no_llm_judge,
        "avg_composite": avg_composite,
        "pass_rate_50pct": pass_rate,
        "by_type": type_scores,
        "by_session_format": fmt_summary,
        "results": results,
    }


# ------------------------------------------------------------------
# Improvement #3 — Probe battery helpers (KISS, file-based, loud)
# These live here so they are defined before main() uses them.
# ------------------------------------------------------------------

def print_probe_taxonomy(summary: dict) -> None:
    """Print a compact one-page failure taxonomy when running the probes set.
    Designed for fast morning debugging loops against the tower.
    """
    print("\n" + "=" * 64)
    print("  MESA PROBE TAXONOMY (fast deterministic view)")
    print("=" * 64)
    print(f"  Items: {summary.get('n_items', 0)}   Scope: {summary.get('adapter_scope', 'n/a')}")
    print()

    failures_count = {}
    for r in summary.get("results", []):
        for f in r.get("failures", []):
            failures_count[f] = failures_count.get(f, 0) + 1

    if failures_count:
        print("  Top failure modes (count):")
        for f, c in sorted(failures_count.items(), key=lambda x: -x[1])[:6]:
            print(f"    {f:<35} {c}")
    else:
        print("  No explicit failures recorded.")

    print()
    print("  By type (answer.correct):")
    for t, data in summary.get("summary", {}).get("by_type", {}).items():
        cr = data.get("correct_rate") if isinstance(data, dict) else None
        if cr is not None:
            print(f"    {t:<25} correct_rate={cr:.2f}")

    print("\n  Quick diagnosis tips:")
    print("  - scope=full_production   → prior memory / tools may influence outcomes")
    print("  - retrieval_trace_missing → implement ask_with_trace on the adapter")
    print("  - missing_required_*      → extraction or storage layer dropped facts")
    print("  - unsupported_answer_claim → grounding failure (hallucination)")
    print("=" * 64 + "\n")


def compare_probe_runs(path_a: str, path_b: str, label_a: str = "A", label_b: str = "B") -> dict:
    """Diff two probe run JSONs by failure type. Pure file-based, no LLM.
    Returns a simple delta report you can print or save.
    """
    with open(path_a) as f:
        a = json.load(f)
    with open(path_b) as f:
        b = json.load(f)

    fa = {}
    for r in a.get("results", []):
        for f in r.get("failures", []):
            fa[f] = fa.get(f, 0) + 1

    fb = {}
    for r in b.get("results", []):
        for f in r.get("failures", []):
            fb[f] = fb.get(f, 0) + 1

    all_failures = sorted(set(fa) | set(fb))
    deltas = {}
    for f in all_failures:
        deltas[f] = fa.get(f, 0) - fb.get(f, 0)

    report = {
        "a": path_a,
        "b": path_b,
        "n_a": a.get("n_items"),
        "n_b": b.get("n_items"),
        "adapter_scope_a": a.get("adapter_scope"),
        "adapter_scope_b": b.get("adapter_scope"),
        "failure_deltas": deltas,
        "only_in_a": [f for f in fa if f not in fb],
        "only_in_b": [f for f in fb if f not in fa],
    }
    return report


def doctor(
    adapter_path: Optional[str] = None,
    dataset_path: Optional[Path] = None,
    judge_url: Optional[str] = None,
) -> bool:
    """Pre-flight checks: adapter importable, dataset valid, judge reachable."""
    checks: list[tuple[str, Optional[bool], str]] = []

    if adapter_path:
        try:
            a = _load_adapter(adapter_path)
            a.reset()
            checks.append(("adapter importable", True, a.__class__.__name__))
        except Exception as e:
            checks.append(("adapter importable", False, str(e)))
    else:
        checks.append(("adapter", None, "not specified"))

    dp = dataset_path or DEFAULT_DATASET_V2
    try:
        with open(dp) as f:
            data = json.load(f)
        checks.append(("dataset valid JSON", True, f"{len(data)} items — {dp.name}"))
    except Exception as e:
        checks.append(("dataset valid JSON", False, str(e)))

    if judge_url:
        try:
            urllib.request.urlopen(f"{judge_url.rstrip('/')}/models", timeout=5)
            checks.append(("judge endpoint reachable", True, judge_url))
        except Exception as e:
            checks.append(("judge endpoint reachable", False, f"{judge_url}: {e}"))
    else:
        checks.append(("judge endpoint", None, "not configured — LLM judge disabled"))

    print("\n" + "=" * 58)
    print("  MESA DOCTOR")
    print("=" * 58)
    all_pass = True
    for name, status, detail in checks:
        if status is True:
            icon = "✓"
        elif status is False:
            icon = "✗"
            all_pass = False
        else:
            icon = "·"
        print(f"  {icon} {name:<32} {detail}")
    print("=" * 58)
    print(f"  {'PASS — ready to run' if all_pass else 'FAIL — fix issues above'}")
    print("=" * 58 + "\n")
    return all_pass


def score_results(results_path: str, dataset_path: Optional[Path] = None) -> dict:
    """Re-score an existing results JSON without re-running the adapter.

    Useful when the scorer changes and you want to apply new metrics to old runs.
    Writes are preserved from the original run; only the metric computation is repeated.
    """
    with open(results_path) as f:
        existing = json.load(f)

    if dataset_path is None and existing.get("dataset"):
        candidate = Path(existing["dataset"])
        dataset_path = candidate if candidate.exists() else DEFAULT_DATASET_V2
    if dataset_path is None:
        dataset_path = DEFAULT_DATASET_V2

    with open(dataset_path) as f:
        raw_dataset = json.load(f)

    item_lookup: dict[str, dict] = {}
    for raw in raw_dataset:
        item = raw if raw.get("version") == "2" else upgrade_v1_item(raw)
        if "id" in item:
            item_lookup[item["id"]] = item

    rescored = []
    for r in existing.get("results", []):
        item_id = r["id"]
        item = item_lookup.get(item_id)
        if item is None:
            logger.warning(f"score: item {item_id!r} not in dataset — keeping original scores")
            rescored.append(r)
            continue

        raw_writes = r.get("storage", {}).get("writes") or []
        writes: Optional[list[dict]] = [
            {"memory_id": w.get("memory_id"), "text": w["text"], "metadata": w.get("metadata", {})}
            for w in raw_writes
        ] if raw_writes else None
        answer_trace = {
            "answer": r.get("answer", {}).get("text", ""),
            "retrieved": r.get("retrieval", {}).get("retrieved"),
            "metadata": r.get("answer", {}).get("metadata", {}),
        }
        storage_metrics, retrieval_metrics, answer_metrics, failures = _score_item_v2(item, writes, answer_trace)
        retrieval_metrics["trace_available"] = r.get("retrieval_trace_available", False)
        storage_metrics["trace_available"] = r.get("write_trace_available", False)
        rescored.append({
            **r,
            "storage": {**r.get("storage", {}), "metrics": storage_metrics},
            "retrieval": {**r.get("retrieval", {}), "metrics": retrieval_metrics},
            "answer": {**r.get("answer", {}), "metrics": answer_metrics},
            "failures": failures,
        })

    manifest = load_dataset_manifest(dataset_path)
    new_summary = _summarize_v2_results(rescored) if rescored else {}
    return {
        **existing,
        "run_id": existing.get("run_id", "") + "_rescored",
        "dataset": str(dataset_path),
        "dataset_name": manifest.get("dataset_name") if manifest else dataset_path.stem,
        "dataset_version": manifest.get("dataset_version") if manifest else None,
        "summary": new_summary,
        "results": rescored,
    }


def _load_adapter(dotted_path: str) -> MemoryAdapter:
    """Import and instantiate an adapter from a dotted module path.

    E.g. "examples.simple_adapter.EchoAdapter" → EchoAdapter()
    """
    module_path, class_name = dotted_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls()


def _add_run_args(p: argparse.ArgumentParser) -> None:
    """Shared arguments for the 'run' subcommand."""
    p.add_argument("--adapter", required=True, help="Dotted path to MemoryAdapter, e.g. examples.simple_adapter.EchoAdapter")
    p.add_argument("--dataset", default=str(DEFAULT_DATASET_V2), help="Path to dataset JSON")
    p.add_argument("--schema-version", choices=["1", "2"], default="2")
    p.add_argument("--trace-required", action="store_true")
    p.add_argument("--official-run", action="store_true")
    p.add_argument("--no-llm-judge", action="store_true", default=True)
    p.add_argument("--llm-judge", dest="no_llm_judge", action="store_false")
    p.add_argument("--judge-url", default=None)
    p.add_argument("--judge-model", default="local")
    p.add_argument("--filter", dest="type_filter", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--output", default="results")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--n-runs", type=int, default=1, metavar="N",
                   help="Run each item N times and report variance (default: 1)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print run plan without executing")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    parser = argparse.ArgumentParser(description="MESA benchmark")
    sub = parser.add_subparsers(dest="command")

    # ── run ───────────────────────────────────────────────────────────────────
    run_p = sub.add_parser("run", help="Run the benchmark")
    _add_run_args(run_p)

    # ── doctor ────────────────────────────────────────────────────────────────
    doc_p = sub.add_parser("doctor", help="Pre-flight checks before a run")
    doc_p.add_argument("--adapter", default=None, help="Dotted path to adapter (optional)")
    doc_p.add_argument("--dataset", default=str(DEFAULT_DATASET_V2))
    doc_p.add_argument("--judge-url", default=None)

    # ── score ─────────────────────────────────────────────────────────────────
    score_p = sub.add_parser("score", help="Re-score an existing results JSON")
    score_p.add_argument("results", help="Path to results JSON from a prior run")
    score_p.add_argument("--dataset", default=None, help="Dataset path (defaults to path stored in results)")
    score_p.add_argument("--output", default="results")

    args = parser.parse_args()

    # ── doctor ────────────────────────────────────────────────────────────────
    if args.command == "doctor":
        ok = doctor(
            adapter_path=args.adapter,
            dataset_path=Path(args.dataset) if args.dataset else None,
            judge_url=args.judge_url,
        )
        sys.exit(0 if ok else 1)

    # ── score ─────────────────────────────────────────────────────────────────
    if args.command == "score":
        dataset_path = Path(args.dataset) if args.dataset else None
        rescored = score_results(args.results, dataset_path=dataset_path)
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"run_v2_{rescored['run_id']}.json"
        with open(out_path, "w") as f:
            json.dump(rescored, f, indent=2)
        n = rescored.get("n_items", len(rescored.get("results", [])))
        cr = rescored.get("summary", {}).get("answer", {}).get("correct_rate", "n/a")
        print(f"\nmesa score: {n} items rescored")
        print(f"  correct_rate: {cr}")
        print(f"Results → {out_path}")
        return

    # ── run (default) ─────────────────────────────────────────────────────────
    if args.command not in ("run", None):
        parser.print_help()
        sys.exit(1)

    # Resolve adapter — required for run
    if not hasattr(args, "adapter") or args.adapter is None:
        parser.error("--adapter is required for 'run'")

    adapter = _load_adapter(args.adapter)

    if args.schema_version == "2":
        dataset_path = Path(args.dataset)
        summary = run_benchmark_v2(
            adapter=adapter,
            dataset_path=dataset_path,
            trace_required=args.trace_required,
            limit=args.limit,
            quiet=args.quiet,
            official_run=args.official_run,
            n_runs=args.n_runs,
            dry_run=args.dry_run,
        )

        if summary.get("dry_run"):
            return

        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"run_v2_{summary['run_id']}.json"
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)

        n_runs_label = f" × {args.n_runs} runs" if args.n_runs > 1 else ""
        print(f"\n{'='*60}")
        print(f"  Schema : v2{n_runs_label}")
        print(f"  Items  : {summary['n_items']}")
        print(f"  Split  : {summary['dataset_split']}")
        print(f"  Trace  : {'required' if summary['trace_required'] else 'preferred'}")
        print(f"  Scope  : {summary.get('adapter_scope', 'unknown')}")
        print(f"  Correct: {summary['summary']['answer']['correct_rate']}")
        print(f"  Grounded: {summary['summary']['answer']['grounded_rate']}")
        print(f"{'='*60}")
        print(f"Results → {out_path}")

        dataset_stem = Path(args.dataset).stem.lower()
        if "probe" in dataset_stem or "sample" in dataset_stem:
            print_probe_taxonomy(summary)
        return

    # v1 legacy path
    judge_client = None
    if not args.no_llm_judge:
        from openai import OpenAI
        if not args.judge_url:
            parser.error("--judge-url required when --llm-judge is set")
        judge_client = OpenAI(base_url=args.judge_url, api_key="none")

    summary = run_benchmark(
        adapter=adapter,
        dataset_path=Path(args.dataset) if Path(args.dataset) != DEFAULT_DATASET_V2 else DEFAULT_DATASET,
        no_llm_judge=args.no_llm_judge,
        judge_client=judge_client,
        judge_model=args.judge_model,
        type_filter=args.type_filter,
        limit=args.limit,
        quiet=args.quiet,
    )

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"run_{summary['run_id']}.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Items : {summary['n_items']}")
    print(f"  Avg   : {summary['avg_composite']:.4f}")
    print(f"  Pass  : {summary['pass_rate_50pct']:.1%}")
    print(f"  By type:")
    for t, s in sorted(summary["by_type"].items()):
        print(f"    {t:<25} {s:.4f}")
    print(f"  By session format:")
    for fmt, s in sorted(summary["by_session_format"].items()):
        print(f"    {fmt:<10} n={s['n']:3d}  avg={s['avg_composite']:.4f}  pass={s['pass_rate']:.1%}")
    print(f"{'='*60}")
    print(f"Results → {out_path}")


if __name__ == "__main__":
    main()
