# Contributing

## Priorities

High-value contributions are:

- curated additions to `dataset/mesa_v2.json`
- better primary-source baselines
- adapter implementations with observable retrieval traces
- scorer improvements that increase falsifiability, not just score smoothness

## Dataset changes

For v2 dataset work:

1. Follow [docs/annotation_guidelines.md](/home/dino/mesa-benchmark/docs/annotation_guidelines.md:1).
2. Prefer promoting items from `dataset/mesa_v2_annotated.json` into `dataset/mesa_v2.json` rather than inventing synthetic examples.
3. Keep `dataset/version_v2.json` in sync:
   - `item_count`
   - `task_types`
4. Add or update tests if the curated dataset coverage changes.

## Code changes

- Preserve the legacy `run_benchmark()` path unless explicitly removing it in a major version.
- New adapter work should implement:
  - `get_writes()`
  - `ask_with_trace()`
- Prefer deterministic scoring over opaque judge behavior when possible.

## Validation

Run:

```bash
/home/dino/mesa/.venv/bin/pytest tests -q
```

For v2 dataset or runner changes, also smoke test:

```bash
python3 -m mesa.runner --adapter examples.simple_adapter.NullAdapter --schema-version 2 --limit 1
```
