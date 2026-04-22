"""Run the MESA benchmark against Mike's full relay pipeline."""

import json
import logging
import sys
from pathlib import Path

# Put mesa-benchmark on path so mesa package and adapters resolve
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

from adapters.mike_adapter import MikeAdapter
from mesa.runner import run_benchmark

DATASET = Path(__file__).parent / "dataset" / "mesa_v1.json"
RESULTS_DIR = Path(__file__).parent / "results"


def main():
    logger.info("Initializing MikeAdapter...")
    adapter = MikeAdapter()

    logger.info(f"Running MESA v1 (100 items, no LLM judge)...")
    summary = run_benchmark(
        adapter=adapter,
        dataset_path=DATASET,
        no_llm_judge=True,
        quiet=False,
    )

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"mike_{summary['run_id']}.json"
    out.write_text(json.dumps(summary, indent=2))

    print(f"\n{'='*60}")
    print(f"  Mike — MESA v1 Results")
    print(f"  Items : {summary['n_items']}")
    print(f"  Avg   : {summary['avg_composite']:.4f}")
    print(f"  Pass  : {summary['pass_rate_50pct']:.1%}")
    print(f"\n  By type:")
    for t, s in sorted(summary["by_type"].items(), key=lambda x: -x[1]):
        bar = "█" * int(s * 20)
        print(f"    {t:<25} {s:.4f}  {bar}")
    print(f"\n  By session format:")
    for fmt, s in sorted(summary["by_session_format"].items()):
        print(f"    {fmt:<10} n={s['n']:3d}  avg={s['avg_composite']:.4f}  pass={s['pass_rate']:.1%}")
    print(f"{'='*60}")
    print(f"Results → {out}")


if __name__ == "__main__":
    main()
