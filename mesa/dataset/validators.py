"""Lightweight validators for MESA dataset schema v2."""

VALID_REVIEW_STATUSES = {"annotated", "reviewed", "adjudicated", "provisional"}


def validate_v2_item_structure(item: dict) -> list[str]:
    errors = []
    required = {
        "id",
        "version",
        "task_type",
        "answer_format",
        "question",
        "sessions",
        "gold_memory",
        "gold_answer",
        "metadata",
    }
    missing = sorted(required - set(item))
    if missing:
        errors.append(f"missing required fields: {', '.join(missing)}")
    if item.get("version") != "2":
        errors.append(f"version must be '2': {item.get('version')!r}")
    if not isinstance(item.get("sessions"), list):
        errors.append("sessions must be a list")
    metadata = item.get("metadata")
    if not isinstance(metadata, dict):
        errors.append("metadata must be an object")
        return errors
    for key in (
        "difficulty",
        "domain",
        "source_profile",
        "distractor_density",
        "annotator_id",
        "reviewer_id",
        "review_status",
    ):
        if key not in metadata:
            errors.append(f"metadata missing required field: {key}")
    if metadata.get("difficulty") not in {"easy", "medium", "hard"}:
        errors.append(f"invalid metadata.difficulty: {metadata.get('difficulty')!r}")
    if metadata.get("distractor_density") not in {"low", "medium", "high"}:
        errors.append(f"invalid metadata.distractor_density: {metadata.get('distractor_density')!r}")
    if not isinstance(metadata.get("domain"), str) or not metadata.get("domain", "").strip():
        errors.append("metadata.domain must be a non-empty string")
    if not isinstance(metadata.get("source_profile"), str) or not metadata.get("source_profile", "").strip():
        errors.append("metadata.source_profile must be a non-empty string")
    if not isinstance(metadata.get("annotator_id"), str) or not metadata.get("annotator_id", "").strip():
        errors.append("metadata.annotator_id must be a non-empty string")
    if not isinstance(metadata.get("reviewer_id"), str) or not metadata.get("reviewer_id", "").strip():
        errors.append("metadata.reviewer_id must be a non-empty string")
    if metadata.get("review_status") not in VALID_REVIEW_STATUSES:
        errors.append(f"invalid metadata.review_status: {metadata.get('review_status')!r}")
    temporal_gap_days = metadata.get("temporal_gap_days")
    if temporal_gap_days is not None and (not isinstance(temporal_gap_days, int) or temporal_gap_days < 0):
        errors.append("metadata.temporal_gap_days must be null or a non-negative integer")
    return errors


def validate_gold_memory(item: dict) -> list[str]:
    errors = []
    gold_memory = item.get("gold_memory", {})
    atomic_facts = gold_memory.get("atomic_facts", [])
    fact_ids = [fact.get("fact_id") for fact in atomic_facts]
    if len(fact_ids) != len(set(fact_ids)):
        errors.append("gold_memory.atomic_facts contains duplicate fact_id values")
    fact_id_set = set(fact_ids)
    for key in ("required_fact_ids", "forbidden_fact_ids"):
        for fact_id in gold_memory.get(key, []):
            if fact_id not in fact_id_set:
                errors.append(f"{key} references unknown fact_id: {fact_id}")
    return errors


def validate_gold_answer(item: dict) -> list[str]:
    errors = []
    gold_answer = item.get("gold_answer", {})
    canonical_answers = gold_answer.get("canonical_answers", [])
    abstention_expected = gold_answer.get("abstention_expected")
    if not isinstance(canonical_answers, list):
        errors.append("gold_answer.canonical_answers must be a list")
    if not isinstance(abstention_expected, bool):
        errors.append("gold_answer.abstention_expected must be a boolean")
    if abstention_expected is False and not canonical_answers:
        errors.append("answerable items must provide at least one canonical answer")
    return errors
