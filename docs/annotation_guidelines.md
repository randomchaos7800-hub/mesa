# MESA v2 Annotation Guidelines

## Goal

Annotate benchmark items so the evaluator can distinguish:

- storage failure
- retrieval failure
- answer failure
- hallucination / unsupported answering

Each item must be auditable from the session text alone.

## Required fields

Every v2 item must include:

- `task_type`
- `answer_format`
- `sessions`
- `gold_memory`
- `gold_answer`

## Atomic facts

`gold_memory.atomic_facts` should be minimal and testable.

Good:

- `The user's primary home server is homebase.`
- `The VPN IP of homebase is 10.0.0.10.`

Bad:

- `The user has a home server called homebase with an IP and some hardware details.`

Each atomic fact should capture one thing that could plausibly be stored or missed independently.

## Fact statuses

Use:

- `active`: should be available as valid memory
- `superseded`: was true earlier but should not be returned as current
- `distractor`: present in the sessions but should not answer the question
- `unsupported`: reserved for future adversarial/poison annotations

## Required vs forbidden facts

- `required_fact_ids`: the minimum evidence needed to answer correctly
- `forbidden_fact_ids`: facts that must not drive the answer for this item

For update items:

- current fact goes in `required_fact_ids`
- stale fact goes in `forbidden_fact_ids`

For interference items:

- target fact goes in `required_fact_ids`
- confuser facts go in `forbidden_fact_ids`

## Gold answer

`gold_answer` should include:

- `canonical_answers`: acceptable canonical outputs
- `must_include`: critical answer components
- `must_not_include`: stale/confuser/forbidden components
- `abstention_expected`: only `true` when the system should refuse

Use `must_include` for the smallest decisive answer components.

Examples:

- preference: `["terse"]`
- constraint: `["$20", "no exceptions"]`
- causal: `["85%", "80%"]`

## Task-specific notes

### `recall/preference`

Annotate the preference in normalized form, not just surface wording.

Example:

- fact: `The user prefers terse responses.`
- aliases: `["keep it short", "short and direct"]`

### `recall/constraint`

Include both the rule and any exception boundary.

Example:

- fact: `The cloud storage budget cap is $20/month with no exceptions.`

### `temporal`

Normalize the target date in `canonical_answers`, but keep alternate date phrasings in aliases or canonical variants.

### `update`

Represent both the stale fact and the current fact explicitly.

### `causal`

Split the cause chain into separate facts whenever possible.

Do not hide both halves inside one gold fact unless the underlying session text is inseparable.

## Review bar

Do not add an item to `mesa_v2.json` unless:

- the answer is fully supported by session text
- required and forbidden facts are clearly separated
- a cold reader could reproduce the annotation
- the item has a clear failure mode worth measuring
