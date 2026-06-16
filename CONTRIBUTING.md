# Contributing

## Priorities

High-value contributions are:

- curated additions to `dataset/mesa_v2.json`
- better primary-source baselines
- adapter implementations with observable retrieval traces
- scorer improvements that increase falsifiability, not just score smoothness

## Dataset changes

For v2 dataset work:

1. Follow [docs/annotation_guidelines.md](docs/annotation_guidelines.md).
2. Use [docs/dataset_review_checklist.md](docs/dataset_review_checklist.md) before promoting items.
3. Respect [docs/dataset_governance.md](docs/dataset_governance.md).
4. Keep required metadata fields populated:
   - `domain`
   - `source_profile`
   - `annotator_id`
   - `reviewer_id`
   - `review_status`
5. Keep `dataset/review_log_v2.json` in sync with curated item changes.
6. Prefer promoting items from `dataset/mesa_v2_annotated.json` into `dataset/mesa_v2.json` rather than inventing synthetic examples.
7. Keep dataset manifests in sync:
   - `item_count`
   - `task_types`
   - `split`
   - `dataset_version`
8. Add or update tests if the curated dataset coverage changes.

## Code changes

- Preserve the legacy `run_benchmark()` path unless explicitly removing it in a major version.
- New adapter work should implement:
  - `get_writes()`
  - `ask_with_trace()`
- Prefer deterministic scoring over opaque judge behavior when possible.

## Validation

Run:

```bash
pytest tests/ -q
```

For v2 dataset or runner changes, also smoke test:

```bash
mesa-benchmark run --adapter examples.simple_adapter.NullAdapter --schema-version 2 --limit 1
```
