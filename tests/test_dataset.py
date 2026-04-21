"""Tests for dataset schema and fixture integrity."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

REPO_ROOT = Path(__file__).parent.parent
SCHEMA_PATH = REPO_ROOT / "dataset" / "schema.json"
FIXTURES_PATH = REPO_ROOT / "dataset" / "fixtures" / "sample.json"
GOLD_PATH = REPO_ROOT / "dataset" / "mmeb_v1.json"

VALID_TYPES = {
    "recall/single", "recall/preference", "synthesis/multi",
    "temporal", "update", "adversarial", "causal",
}


def _validate_item(item: dict) -> list[str]:
    errors = []
    if not item.get("id", "").startswith("mmeb-"):
        errors.append(f"id must start with 'mmeb-': {item.get('id')}")
    if item.get("type") not in VALID_TYPES:
        errors.append(f"invalid type: {item.get('type')}")
    if not item.get("question", "").strip():
        errors.append("empty question")
    if not item.get("expected_answer", "").strip():
        errors.append("empty expected_answer")
    if not isinstance(item.get("sessions", []), list):
        errors.append("sessions must be a list")
    if item.get("type") == "adversarial" and item.get("sessions"):
        errors.append("adversarial items must have empty sessions")
    return errors


class TestSchema:
    def test_exists(self):
        assert SCHEMA_PATH.exists()

    def test_valid_json(self):
        schema = json.loads(SCHEMA_PATH.read_text())
        assert schema["type"] == "array"
        required = set(schema["items"]["required"])
        assert {"id", "type", "question", "expected_answer", "sessions"} <= required


class TestFixtures:
    def test_exists(self):
        assert FIXTURES_PATH.exists()

    def test_has_5_items(self):
        assert len(json.loads(FIXTURES_PATH.read_text())) == 5

    def test_covers_key_types(self):
        items = json.loads(FIXTURES_PATH.read_text())
        types = {i["type"] for i in items}
        assert "recall/single" in types
        assert "recall/preference" in types
        assert "adversarial" in types

    def test_all_valid(self):
        for item in json.loads(FIXTURES_PATH.read_text()):
            assert _validate_item(item) == [], f"{item.get('id')}: {_validate_item(item)}"

    def test_unique_ids(self):
        items = json.loads(FIXTURES_PATH.read_text())
        ids = [i["id"] for i in items]
        assert len(ids) == len(set(ids))

    def test_adversarial_empty_sessions(self):
        for item in json.loads(FIXTURES_PATH.read_text()):
            if item["type"] == "adversarial":
                assert item["sessions"] == []

    def test_non_adversarial_has_sessions(self):
        for item in json.loads(FIXTURES_PATH.read_text()):
            if item["type"] != "adversarial":
                assert len(item["sessions"]) > 0, f"{item['id']} has no sessions"


class TestGoldDataset:
    def test_exists(self):
        assert GOLD_PATH.exists()

    def test_valid_json(self):
        assert isinstance(json.loads(GOLD_PATH.read_text()), list)

    def test_all_valid(self):
        items = json.loads(GOLD_PATH.read_text())
        if not items:
            pytest.skip("mmeb_v1.json is empty")
        for item in items:
            assert _validate_item(item) == [], f"{item.get('id')}: {_validate_item(item)}"

    def test_unique_ids(self):
        items = json.loads(GOLD_PATH.read_text())
        if not items:
            pytest.skip("mmeb_v1.json is empty")
        ids = [i["id"] for i in items]
        assert len(ids) == len(set(ids))
