"""Semantic equivalence scorer for MESA — no LLM required.

Combines three signals:
  1. Word overlap (normalized Jaccard on stemmed tokens)
  2. Edit distance (SequenceMatcher ratio)
  3. Key entity match (exact match on extracted entities/names/numbers)

Outputs a score in [0.0, 1.0].

No external dependencies beyond the Python standard library.
"""

import re
from difflib import SequenceMatcher


def _normalize(text: str) -> str:
    """Lowercase, strip markdown, collapse whitespace, remove punctuation."""
    text = text.lower().strip()
    text = re.sub(r'[*_`#>\[\]()~]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s-]', '', text)
    return text


def _tokenize(text: str) -> list[str]:
    """Split normalized text into tokens, removing common stop words."""
    STOP_WORDS = {
        'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'shall', 'can', 'need', 'must', 'to', 'of',
        'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through',
        'during', 'before', 'after', 'above', 'below', 'between', 'and', 'but',
        'or', 'nor', 'not', 'so', 'yet', 'both', 'either', 'neither', 'each',
        'every', 'all', 'any', 'few', 'more', 'most', 'other', 'some', 'such',
        'no', 'only', 'own', 'same', 'than', 'too', 'very', 'just', 'about',
        'also', 'then', 'this', 'that', 'these', 'those', 'it', 'its', 'i',
        'me', 'my', 'we', 'our', 'you', 'your', 'he', 'him', 'his', 'she',
        'her', 'they', 'them', 'their', 'what', 'which', 'who', 'whom', 'when',
        'where', 'why', 'how'
    }
    tokens = _normalize(text).split()
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 1]


def _simple_stem(word: str) -> str:
    """Very simple stemming — remove common suffixes."""
    if len(word) <= 3:
        return word
    for suffix in ['ing', 'ly', 'ed', 'es', 's', 'tion', 'ment', 'ness', 'able', 'ible']:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[:-len(suffix)]
    return word


def _word_overlap_score(expected: str, actual: str) -> float:
    """Jaccard similarity on stemmed tokens."""
    exp_tokens = set(_simple_stem(t) for t in _tokenize(expected))
    act_tokens = set(_simple_stem(t) for t in _tokenize(actual))
    if not exp_tokens or not act_tokens:
        return 0.0
    intersection = exp_tokens & act_tokens
    union = exp_tokens | act_tokens
    return len(intersection) / len(union)


def _edit_distance_score(expected: str, actual: str) -> float:
    """Normalized edit distance using SequenceMatcher (ratio)."""
    return SequenceMatcher(None, _normalize(expected), _normalize(actual)).ratio()


def _extract_entities(text: str) -> set[str]:
    """Extract potential entities: numbers, dates, proper nouns, specific terms."""
    entities = set()
    normalized = _normalize(text)
    # Numbers
    for m in re.finditer(r'\b\d+\.?\d*\b', normalized):
        entities.add(m.group())
    # Month names with dates
    for m in re.finditer(r'\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d+\b', normalized):
        entities.add(m.group())
    # Slash dates
    for m in re.finditer(r'\b\d{1,2}/\d{1,2}/\d{2,4}\b', normalized):
        entities.add(m.group())
    # ISO dates
    for m in re.finditer(r'\b\d{4}-\d{2}-\d{2}\b', normalized):
        entities.add(m.group())
    # Dollar amounts
    for m in re.finditer(r'\$[\d,]+\.?\d*', normalized):
        entities.add(m.group())
    # IP addresses
    for m in re.finditer(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', normalized):
        entities.add(m.group())
    # Version numbers
    for m in re.finditer(r'\bv?\d+\.\d+\.\d+\b', normalized):
        entities.add(m.group())
    return entities


def _entity_match_score(expected: str, actual: str) -> float:
    """Score based on entity overlap. Entities are critical for factual accuracy."""
    exp_entities = _extract_entities(expected)
    act_entities = _extract_entities(actual)
    if not exp_entities:
        return 1.0  # No entities to check, neutral
    if not act_entities:
        return 0.0  # Expected entities but found none
    matched = len(exp_entities & act_entities)
    return matched / len(exp_entities)


def semantic_equivalence(expected: str, actual: str) -> float:
    """Compute semantic equivalence score between expected and actual answers.

    Combines three signals:
      - word_overlap: 30% (Jaccard on stemmed tokens)
      - edit_distance: 50% (SequenceMatcher ratio)
      - entity_match:  20% (exact entity overlap)

    Edit distance gets the highest weight because it captures word order
    and paraphrase better than token-level Jaccard. Entity match is a
    bonus signal for factual accuracy on numbers/dates/names.

    Returns:
        A score in [0.0, 1.0].
    """
    if not expected.strip():
        return 0.0
    if not actual.strip():
        return 0.0

    # Exact match gets full score
    if _normalize(expected) == _normalize(actual):
        return 1.0

    w = _word_overlap_score(expected, actual)
    e = _edit_distance_score(expected, actual)
    en = _entity_match_score(expected, actual)

    return round(0.30 * w + 0.50 * e + 0.20 * en, 4)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    tests = [
        # (expected, actual, expected_min_score)
        ("The server is homeserver", "homeserver is the server", 0.7),
        ("The server is homeserver", "The server is homeserver", 1.0),
        ("Ibuprofen was prescribed for dehydration headaches", "They were prescribed ibuprofen for the headache", 0.55),
        ("Kernel update failed causing outage", "The kernel update failed and took the server down", 0.55),
        ("$30/month", "thirty dollars a month", 0.3),
        ("homeserver", "I don't know", 0.0),
        ("", "anything", 0.0),
        ("anything", "", 0.0),
    ]

    print("Semantic Equivalence Scorer — Self-Test")
    print("=" * 60)
    all_pass = True
    for expected, actual, min_score in tests:
        score = semantic_equivalence(expected, actual)
        passed = score >= min_score - 0.05
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  [{status}] score={score:.4f} (min={min_score})")
        print(f"        expected: {expected!r}")
        print(f"        actual:   {actual!r}")
        print()

    if all_pass:
        print("All tests passed.")
    else:
        print("Some tests failed.")
