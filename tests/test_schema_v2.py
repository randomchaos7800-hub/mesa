"""Focused tests for v2 dataset validation helpers."""

import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from mesa.dataset.validators import validate_gold_answer, validate_gold_memory, validate_v2_item_structure

FIXTURES_V2_PATH = Path(__file__).parent.parent / "dataset" / "fixtures" / "sample_v2.json"
GOLD_V2_PATH = Path(__file__).parent.parent / "dataset" / "mesa_v2.json"


def _load_items():
    return json.loads(FIXTURES_V2_PATH.read_text())


def _load_gold_items():
    return json.loads(GOLD_V2_PATH.read_text())


class TestSchemaV2Validators:
    def test_valid_fixture_item_has_no_errors(self):
        item = _load_items()[0]
        errors = []
        errors.extend(validate_v2_item_structure(item))
        errors.extend(validate_gold_memory(item))
        errors.extend(validate_gold_answer(item))
        assert errors == []

    def test_unknown_required_fact_id_fails(self):
        item = copy.deepcopy(_load_items()[0])
        item["gold_memory"]["required_fact_ids"] = ["f999"]
        errors = validate_gold_memory(item)
        assert errors == ["required_fact_ids references unknown fact_id: f999"]

    def test_answerable_item_requires_canonical_answer(self):
        item = copy.deepcopy(_load_items()[0])
        item["gold_answer"]["canonical_answers"] = []
        errors = validate_gold_answer(item)
        assert "answerable items must provide at least one canonical answer" in errors

    def test_gold_dataset_item_has_no_errors(self):
        item = _load_gold_items()[0]
        errors = []
        errors.extend(validate_v2_item_structure(item))
        errors.extend(validate_gold_memory(item))
        errors.extend(validate_gold_answer(item))
        assert errors == []
