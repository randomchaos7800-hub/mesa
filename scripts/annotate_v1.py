#!/usr/bin/env python3
"""
Annotate v1 dataset items with atomic_facts and produce a v2 dataset.

Rules by task type:
  recall/*       — 1 active fact from expected_answer; required=[f1]
  temporal       — 1 active fact; required=[f1]; must_include=[date token]
  synthesis/multi — facts split per sentence from expected_answer; required=all
  causal         — same as synthesis/multi
  update         — f1=old value (superseded), f2=new value (active from answer);
                   required=[f2], forbidden=[f1]
  update/interf. — f1=first mentioned (active/expected); f2..=distractors;
                   required=[f1], forbidden=[f2..]
  adversarial    — no facts; abstention_expected=True
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V1_PATH = ROOT / "dataset" / "mesa_v1.json"
OUT_PATH = ROOT / "dataset" / "mesa_v2_annotated.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_flat_turns(sessions):
    """Return a flat list of {role, content} dicts regardless of v1 structure."""
    if not sessions:
        return []
    first = sessions[0]
    if isinstance(first, dict) and "turns" in first:
        turns = []
        for s in sessions:
            turns.extend(s.get("turns", []))
        return turns
    return [t for t in sessions if isinstance(t, dict) and "role" in t]


def user_turns(sessions):
    return [t["content"] for t in get_flat_turns(sessions) if t.get("role") == "user"]


def split_sentences(text):
    """Split text into non-empty sentences."""
    parts = re.split(r"(?<=[.;])\s+", text.strip())
    return [p.strip(" .,;") for p in parts if len(p.strip()) > 4]


def make_fact(fid, text, status, source_sessions=None, aliases=None, slots=None):
    fact = {
        "fact_id": fid,
        "text": text,
        "status": status,
        "source_sessions": source_sessions or ["s1"],
    }
    if aliases:
        fact["aliases"] = aliases
    if slots:
        fact["slots"] = slots
    return fact


def find_must_include(expected_answer):
    """Extract short key phrases to use as must_include strings."""
    # Strip leading qualifiers like "I don't know", "The X is..."
    text = expected_answer.strip()
    # For short answers just return the whole thing
    if len(text) < 60:
        return [text]
    # For longer answers return the first sentence
    sentences = split_sentences(text)
    return [sentences[0]] if sentences else [text[:60]]


# ---------------------------------------------------------------------------
# Per-type annotators
# ---------------------------------------------------------------------------

def annotate_recall(item):
    answer = item.get("expected_answer", "").strip()
    facts = [make_fact("f1", answer, "active")]
    return facts, ["f1"], [], find_must_include(answer), []


def annotate_temporal(item):
    answer = item.get("expected_answer", "").strip()
    # Extract date token if present
    date_match = re.search(r"\d{4}[-/]\d{2}[-/]\d{2}|\d{4}-\d{2}-\d{2}", answer)
    must_include = [date_match.group(0)] if date_match else find_must_include(answer)
    facts = [make_fact("f1", answer, "active")]
    return facts, ["f1"], [], must_include, []


def annotate_multi(item):
    answer = item.get("expected_answer", "").strip()
    sentences = split_sentences(answer)
    if len(sentences) < 2:
        # Comma-split for compound facts like "Mistral 7B, using about 4GB of VRAM"
        parts = [p.strip() for p in re.split(r",\s*(?:and\s+)?", answer) if len(p.strip()) > 4]
        sentences = parts if len(parts) >= 2 else [answer]
    facts = [make_fact(f"f{i+1}", s, "active") for i, s in enumerate(sentences)]
    required = [f"f{i+1}" for i in range(len(facts))]
    must_include = [s for s in sentences if len(s) <= 60][:3]
    return facts, required, [], must_include, []


def annotate_update(item):
    answer = item.get("expected_answer", "").strip()
    turns = get_flat_turns(item.get("sessions", []))

    # Find the first user turn that asserts an old state
    old_value = ""
    for turn in turns:
        if turn.get("role") == "user":
            old_value = turn["content"].strip()
            break

    # If the expected answer is very short it may just be the new value
    # Build superseded fact from first user assertion, active from expected
    if old_value and old_value.lower() != answer.lower():
        facts = [
            make_fact("f1", old_value[:200], "superseded"),
            make_fact("f2", answer, "active"),
        ]
        required = ["f2"]
        forbidden = ["f1"]
    else:
        facts = [make_fact("f1", answer, "active")]
        required = ["f1"]
        forbidden = []

    must_include = find_must_include(answer)
    return facts, required, forbidden, must_include, []


def annotate_update_interference(item):
    """First mentioned value is the answer; subsequent ones are distractors."""
    answer = item.get("expected_answer", "").strip()
    turns = get_flat_turns(item.get("sessions", []))
    user_msgs = [t["content"] for t in turns if t.get("role") == "user"]

    facts = []
    required = []
    forbidden = []

    # f1 = the expected answer (first mentioned value)
    facts.append(make_fact("f1", answer, "active"))
    required.append("f1")

    # Subsequent user turns may introduce interfering values
    fid = 2
    for msg in user_msgs[1:]:
        # Only add if it's clearly a different named value (not the expected answer)
        if answer.lower() not in msg.lower() and len(msg.strip()) < 300:
            distr_text = msg.strip()[:200]
            facts.append(make_fact(f"f{fid}", distr_text, "distractor"))
            forbidden.append(f"f{fid}")
            fid += 1
            if fid > 5:  # cap
                break

    must_include = find_must_include(answer)
    return facts, required, forbidden, must_include, []


def annotate_adversarial(item):
    answer = item.get("expected_answer", "").strip()
    # No atomic facts to recall — abstention expected
    must_not_include = []
    # If expected says "I don't know" keep forbidden list empty;
    # the abstention scorer handles it
    return [], [], [], [], must_not_include


# ---------------------------------------------------------------------------
# Main converter
# ---------------------------------------------------------------------------

ANNOTATORS = {
    "recall/single": annotate_recall,
    "recall/preference": annotate_recall,
    "recall/constraint": annotate_recall,
    "temporal": annotate_temporal,
    "synthesis/multi": annotate_multi,
    "causal": annotate_multi,
    "update": annotate_update,
    "update/interference": annotate_update_interference,
    "adversarial": annotate_adversarial,
}


def infer_answer_format(task_type):
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


def convert_item(item):
    task_type = item.get("type") or item.get("task_type", "recall/single")
    annotator = ANNOTATORS.get(task_type, annotate_recall)
    facts, required_ids, forbidden_ids, must_include, must_not_include = annotator(item)

    sessions = item.get("sessions", [])
    turns = get_flat_turns(sessions)
    if not turns:
        turns = [{"role": "user", "content": "[No prior context provided]"}]
    migrated_sessions = [{"session_id": "s1", "date": None, "turns": turns}]

    answer = item.get("expected_answer", "").strip()
    abstention = task_type == "adversarial"

    return {
        "id": item["id"],
        "version": "2",
        "task_type": task_type,
        "answer_format": infer_answer_format(task_type),
        "question": item["question"],
        "sessions": migrated_sessions,
        "gold_memory": {
            "atomic_facts": facts,
            "required_fact_ids": required_ids,
            "forbidden_fact_ids": forbidden_ids,
        },
        "gold_answer": {
            "canonical_answers": [] if abstention else [answer],
            "must_include": must_include,
            "must_not_include": must_not_include,
            "abstention_expected": abstention,
        },
        "metadata": {
            k: v for k, v in item.get("metadata", {}).items()
            if k in {"difficulty", "domain", "distractor_density", "temporal_gap_days", "notes"}
        },
    }


def main():
    with open(V1_PATH) as f:
        v1 = json.load(f)

    converted = [convert_item(item) for item in v1]

    with open(OUT_PATH, "w") as f:
        json.dump(converted, f, indent=2)

    # Stats
    by_type = {}
    for item in converted:
        t = item["task_type"]
        by_type[t] = by_type.get(t, 0) + 1

    print(f"Annotated {len(converted)} items → {OUT_PATH.relative_to(ROOT)}")
    for t, n in sorted(by_type.items()):
        print(f"  {t}: {n}")

    # Spot-check
    errors = 0
    for item in converted:
        gm = item["gold_memory"]
        fact_ids = {f["fact_id"] for f in gm["atomic_facts"]}
        for fid in gm["required_fact_ids"] + gm["forbidden_fact_ids"]:
            if fid not in fact_ids:
                print(f"ERROR {item['id']}: {fid} not in atomic_facts", file=sys.stderr)
                errors += 1
    if errors == 0:
        print("Integrity check: OK")


if __name__ == "__main__":
    main()
