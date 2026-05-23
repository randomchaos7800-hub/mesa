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
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from mesa.adapter import MemoryAdapter
from mesa.dataset.migrators import upgrade_v1_item
from mesa.scoring.answer_types import (
    score_abstention_answer,
    score_causal_answer,
    score_multi_fact_answer,
    score_single_fact_answer,
    score_temporal_answer,
    score_update_interference_answer,
    score_update_current_answer,
)
from mesa.scoring.deterministic import match_fact_ids
from mesa.scorer import exact_match, rouge1_f1, composite, is_refusal, llm_judge

logger = logging.getLogger(__name__)

DEFAULT_DATASET = Path(__file__).parent.parent / "dataset" / "mesa_v1.json"
DEFAULT_DATASET_V2 = Path(__file__).parent.parent / "dataset" / "fixtures" / "sample_v2.json"
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
            "retrieved": [
                {
                    "memory_id": item.memory_id,
                    "text": item.text,
                    "score": item.score,
                    "metadata": item.metadata,
                }
                for item in (trace.retrieved or [])
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


def run_benchmark_v2(
    adapter: MemoryAdapter,
    dataset_path: Optional[Path] = None,
    trace_required: bool = False,
    limit: Optional[int] = None,
    quiet: bool = False,
) -> dict:
    """Run the v2 benchmark scaffolding with structured traces.

    This runner intentionally stops at trace capture. Official v2 scoring will
    plug into the returned envelope in a later change.
    """
    if dataset_path is None:
        dataset_path = DEFAULT_DATASET_V2

    with open(dataset_path) as f:
        dataset = json.load(f)

    if limit:
        dataset = dataset[:limit]

    if not dataset:
        raise ValueError(f"Dataset is empty or all items filtered out (path={dataset_path})")

    results = []
    for idx, item in enumerate(dataset):
        if item.get("version") != "2":
            item = upgrade_v1_item(item)

        item_id = item.get("id", f"item-{idx:04d}")
        question = item["question"]
        sessions = _normalize_v2_sessions(item.get("sessions", []))

        if not quiet:
            logger.info(f"[v2 {idx+1}/{len(dataset)}] {item_id} ({item.get('task_type', 'unknown')})")

        adapter.reset()
        _inject(adapter, sessions)
        writes = _collect_writes(adapter)

        t0 = time.time()
        try:
            answer_trace, has_trace = _collect_answer_trace(adapter, question)
        except Exception as e:
            logger.warning(f"  Adapter error for {item_id}: {e}")
            answer_trace = {"answer": "", "retrieved": None, "metadata": {"error": str(e)}}
            has_trace = False
        elapsed = round(time.time() - t0, 2)

        observable = writes is not None or has_trace
        if trace_required and not observable:
            raise ValueError(f"Adapter does not expose trace hooks required for v2 run: {item_id}")

        storage_metrics, retrieval_metrics, answer_metrics, failures = _score_item_v2(item, writes, answer_trace)

        results.append(
            {
                "id": item_id,
                "version": item["version"],
                "task_type": item["task_type"],
                "answer_format": item["answer_format"],
                "observable": observable,
                "storage": {
                    "writes": writes,
                    "metrics": storage_metrics,
                },
                "retrieval": {
                    "retrieved": answer_trace["retrieved"],
                    "metrics": retrieval_metrics,
                },
                "answer": {
                    "text": answer_trace["answer"],
                    "metadata": answer_trace["metadata"],
                    "metrics": answer_metrics,
                },
                "failures": failures,
                "elapsed_s": elapsed,
            }
        )

    return {
        "run_id": datetime.now().strftime("%Y-%m-%d_%H-%M"),
        "dataset": str(dataset_path),
        "schema_version": "2",
        "n_items": len(results),
        "trace_required": trace_required,
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


def _load_adapter(dotted_path: str) -> MemoryAdapter:
    """Import and instantiate an adapter from a dotted module path.

    E.g. "examples.simple_adapter.EchoAdapter" → EchoAdapter()
    """
    module_path, class_name = dotted_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    parser = argparse.ArgumentParser(description="MESA benchmark runner")
    parser.add_argument("--adapter", required=True, help="Dotted path to MemoryAdapter class, e.g. examples.simple_adapter.EchoAdapter")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET), help="Path to dataset JSON")
    parser.add_argument("--schema-version", choices=["1", "2"], default="1", help="Dataset/runner schema version")
    parser.add_argument("--trace-required", action="store_true", help="Require observable trace hooks for schema v2 runs")
    parser.add_argument("--no-llm-judge", action="store_true", default=True, help="Skip LLM judge (default: on)")
    parser.add_argument("--llm-judge", dest="no_llm_judge", action="store_false", help="Enable LLM judge")
    parser.add_argument("--judge-url", default=None, help="OpenAI-compatible base URL for LLM judge")
    parser.add_argument("--judge-model", default="local", help="Model name for LLM judge")
    parser.add_argument("--filter", dest="type_filter", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default="results", help="Results directory")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    adapter = _load_adapter(args.adapter)

    if args.schema_version == "2":
        dataset_path = Path(args.dataset)
        if dataset_path == DEFAULT_DATASET:
            dataset_path = DEFAULT_DATASET_V2
        summary = run_benchmark_v2(
            adapter=adapter,
            dataset_path=dataset_path,
            trace_required=args.trace_required,
            limit=args.limit,
            quiet=args.quiet,
        )
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"run_v2_{summary['run_id']}.json"
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n{'='*60}")
        print(f"  Schema: v2")
        print(f"  Items : {summary['n_items']}")
        print(f"  Trace : {'required' if summary['trace_required'] else 'preferred'}")
        print(f"{'='*60}")
        print(f"Results → {out_path}")
        return

    judge_client = None
    if not args.no_llm_judge:
        from openai import OpenAI
        if not args.judge_url:
            parser.error("--judge-url required when --llm-judge is set")
        judge_client = OpenAI(base_url=args.judge_url, api_key="none")

    summary = run_benchmark(
        adapter=adapter,
        dataset_path=Path(args.dataset),
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
