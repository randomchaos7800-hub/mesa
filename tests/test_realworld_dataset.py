"""Tests for the real world tests split (delivery-stop routing set)."""

import json
from pathlib import Path

import pytest

from mesa.dataset.manifest import infer_manifest_path, load_dataset_manifest
from mesa.runner import run_benchmark_v2

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET = REPO_ROOT / "dataset" / "mesa_realworld.json"
MANIFEST = REPO_ROOT / "dataset" / "version_realworld.json"

ALL_TASK_TYPES = {
    "recall/single",
    "recall/preference",
    "recall/constraint",
    "synthesis/multi",
    "temporal",
    "update",
    "update/interference",
    "adversarial",
    "causal",
}


def _load_items():
    with open(DATASET) as f:
        return json.load(f)


def test_realworld_dataset_exists_and_parses():
    items = _load_items()
    assert len(items) == 12


def test_realworld_ids_unique_and_patterned():
    items = _load_items()
    ids = [item["id"] for item in items]
    assert len(ids) == len(set(ids))
    for item_id in ids:
        assert item_id.startswith("mesa-rw-delivery-")


def test_realworld_covers_all_nine_task_types():
    items = _load_items()
    assert {item["task_type"] for item in items} == ALL_TASK_TYPES


def test_realworld_update_items_forbid_superseded_facts():
    """Update items must mark superseded facts forbidden so stale answers score down."""
    items = _load_items()
    for item in items:
        if item["task_type"] not in ("update", "update/interference"):
            continue
        gm = item["gold_memory"]
        statuses = {f["fact_id"]: f["status"] for f in gm["atomic_facts"]}
        non_active = [fid for fid, s in statuses.items() if s in ("superseded", "distractor")]
        assert non_active, f"{item['id']} has no superseded/distractor facts"
        assert gm["forbidden_fact_ids"], f"{item['id']} forbids nothing"
        for fid in gm["forbidden_fact_ids"]:
            assert statuses[fid] != "active", f"{item['id']} forbids an active fact"


def test_realworld_adversarial_expects_abstention():
    items = _load_items()
    adversarial = [i for i in items if i["task_type"] == "adversarial"]
    assert adversarial
    for item in adversarial:
        assert item["gold_answer"]["abstention_expected"] is True


def test_realworld_manifest_inferred_and_loads():
    assert infer_manifest_path(DATASET) == MANIFEST
    manifest = load_dataset_manifest(DATASET)
    assert manifest is not None
    assert manifest["item_count"] == len(_load_items())
    assert manifest["schema_version"] == "2"


def test_realworld_official_run_with_string_path():
    """run_benchmark_v2 must accept a plain string dataset path (README quickstart)."""
    from examples.simple_adapter import EchoAdapter

    results = run_benchmark_v2(
        adapter=EchoAdapter(),
        dataset_path=str(DATASET),
        official_run=True,
        quiet=True,
    )
    assert len(results["results"]) == 12
    answer = results["summary"]["answer"]
    assert 0.0 <= answer["correct_rate"] <= 1.0
