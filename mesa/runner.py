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
from mesa.scorer import exact_match, rouge1_f1, composite, is_refusal, llm_judge

logger = logging.getLogger(__name__)

DEFAULT_DATASET = Path(__file__).parent.parent / "dataset" / "mesa_v1.json"
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

    return {
        "run_id": datetime.now().strftime("%Y-%m-%d_%H-%M"),
        "dataset": str(dataset_path),
        "n_items": len(results),
        "llm_judge": not no_llm_judge,
        "avg_composite": avg_composite,
        "pass_rate_50pct": pass_rate,
        "by_type": type_scores,
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
    for t, s in sorted(summary["by_type"].items()):
        print(f"    {t:<25} {s:.4f}")
    print(f"{'='*60}")
    print(f"Results → {out_path}")


if __name__ == "__main__":
    main()
