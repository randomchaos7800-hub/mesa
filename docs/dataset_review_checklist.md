# MESA Dataset Review Checklist

Use this checklist before promoting any item from the annotated pool into the
curated v2 gold dataset.

## Core validity

- [ ] The item validates against `dataset/schema_v2.json`
- [ ] `gold_memory` contains the right atomic facts
- [ ] `required_fact_ids` are sufficient to answer the question
- [ ] `forbidden_fact_ids` capture stale facts or distractors where relevant
- [ ] `gold_answer` matches the intended task type and answer format

## Answerability

- [ ] The question is answerable from the injected sessions alone
- [ ] The answer does not depend on outside world knowledge
- [ ] The answer is not trivially guessable without the intended evidence
- [ ] The expected answer is not ambiguous or under-specified

## Task-type fit

- [ ] `recall/single`: one main fact is being tested
- [ ] `recall/preference`: the item is actually about a preference or posture
- [ ] `recall/constraint`: the item encodes a rule, not just a fact
- [ ] `synthesis/multi`: the answer requires combining multiple facts
- [ ] `temporal`: the time/date aspect is essential to correctness
- [ ] `update`: there is a clear superseded fact and a current fact
- [ ] `update/interference`: similar distractors exist without true supersession
- [ ] `adversarial`: the correct behavior is abstention, not guessing
- [ ] `causal`: the answer requires a real causal explanation, not restating one fact

## Distractor quality

- [ ] Distractors are plausible enough to test retrieval precision
- [ ] Distractors do not accidentally make the answer ambiguous
- [ ] Update/interference items include realistic confusers
- [ ] Adversarial items do not leak the answer through wording

## Session quality

- [ ] The session text naturally supports the intended facts
- [ ] Multi-session items use dates only when they matter
- [ ] Multi-session ordering is coherent
- [ ] There is no accidental contradiction unless the item is testing it

## Promotion gate

- [ ] The item improves coverage for an underrepresented task type, domain, or session format
- [ ] The item is at least as strong as the weakest currently curated item of that type
- [ ] The item has been reviewed by a second person or explicitly marked as single-review

## Preferred promotion order

When choosing between many valid items, prioritize:

1. underrepresented task types
2. multi-session temporal and causal items
3. stronger adversarial coverage
4. new domains or source profiles
5. better distractor construction
