"""Migration helpers between dataset schema versions."""


def infer_answer_format(task_type: str) -> str:
    """Infer a v2 answer_format from a legacy v1 task type."""
    mapping = {
        "recall/single": "single_fact",
        "recall/preference": "preference",
        "recall/constraint": "constraint",
        "synthesis/multi": "multi_fact",
        "temporal": "temporal",
        "update": "update_current",
        "update/interference": "single_fact",
        "adversarial": "abstention",
        "causal": "multi_fact",
    }
    return mapping.get(task_type, "single_fact")


def upgrade_v1_item(item: dict) -> dict:
    """Wrap a v1 item in the v2 envelope with placeholder annotations."""
    sessions = item.get("sessions", [])
    if sessions and isinstance(sessions[0], dict) and "turns" in sessions[0]:
        migrated_sessions = [
            {
                "session_id": f"s{idx + 1}",
                "date": session.get("date"),
                "turns": session["turns"],
            }
            for idx, session in enumerate(sessions)
        ]
    else:
        migrated_sessions = []
        if sessions:
            migrated_sessions.append(
                {
                    "session_id": "s1",
                    "date": None,
                    "turns": sessions,
                }
            )

    expected = item.get("expected_answer", "")
    return {
        "id": item["id"],
        "version": "2",
        "task_type": item["type"],
        "answer_format": infer_answer_format(item["type"]),
        "question": item["question"],
        "sessions": migrated_sessions,
        "gold_memory": {
            "atomic_facts": [],
            "required_fact_ids": [],
            "forbidden_fact_ids": [],
        },
        "gold_answer": {
            "canonical_answers": [expected] if expected else [],
            "must_include": [],
            "must_not_include": [],
            "abstention_expected": item.get("type") == "adversarial",
        },
        "metadata": item.get("metadata", {}),
    }
