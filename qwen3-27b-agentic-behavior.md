# Qwen3-27B Agentic Task Execution — Field Observations

**Hardware:** RTX 5060 Ti 16GB GDDR7 (single GPU), vLLM Genesis stack, max_model_len=65536  
**Proxy:** cha0tiktower:8010 (litellm proxy, model alias "local")  
**Agent harness:** Hermes gateway, OpenAI tool-call loop, max_turns=90  
**Task:** Multi-file codebase audit with deliverables (pytest run, write AUDIT.md, fix bugs)  
**Observation date:** 2026-05-08

---

## Summary

Qwen3-27B is capable of correct diagnosis on long agentic tasks but **will not commit to writes**. It reads every file 3–5 times, correctly identifies problems each time, and then starts reading again rather than executing fixes. Human intervention was required to apply fixes the model had fully understood for 90+ minutes.

---

## Observed Behaviors

### 1. Infinite pre-write loop

The model diagnosed all three pytest failures correctly within the first 20 minutes:
- `rouge-score` not installed → add `skipif` guard to tests
- `mesa-adversarial-0006` has sessions but schema forbids it → schema fix
- (Third failure fully described with exact line numbers)

It then read every involved file again. And again. Across 3+ sessions and one context compaction, it never wrote a single line to disk. Total time before human takeover: ~2.5 hours.

**Pattern:** "Now let me read the remaining files I need, then make all the fixes" → reads everything → "Now I have a complete picture, let me make the fixes" → reads everything again.

This is not a context issue — the model retained correct diagnosis across compaction. It is a decisiveness failure: the model treats reading as lower-risk than writing, so it keeps reading.

### 2. Context compaction threshold bug (infrastructure)

Root cause of slow compaction: the litellm proxy at `:8010` returns:
```json
{"id": "local", "object": "model", "owned_by": "local"}
```
No `max_model_len`. The Hermes model metadata resolver falls through all detection methods and lands on the **128K default fallback**.

Result: compaction threshold = 0.35 × 128K = **44,800 tokens** instead of the correct 0.35 × 65K = **22,937 tokens**.

The model was accumulating ~35–40K tokens of file reads without triggering compaction. Fix: pin `context_length: 65536` in the agent config under the `model:` key.

### 3. Post-compaction restart loop

After compaction fires, the model receives:
1. A [CONTEXT COMPACTION] summary of prior turns
2. The PINNED TASK SPEC (verbatim task description)

The local Qwen3-27B summarizer produced low-quality compaction summaries — "Goal: audit the codebase, Next steps: read source files" — instead of preserving actual progress ("pytest ran, 3 failures found, rouge fix pending"). When the model saw the task spec again, it restarted from the beginning.

The iterative compaction path (`_previous_summary` set) showed some improvement on the second compaction but the model still restarted reading rather than continuing with fixes.

### 4. Delegation as workaround

In one iteration, the model spontaneously delegated file reading to a subagent (`delegate` tool, 89.7s). This was the correct behavior — offloading reading kept the main context lean. The model then moved directly to fix mode. This suggests the model *can* break the loop with the right framing, but doesn't do it reliably.

---

## Fixes Applied

| Fix | Location | Effect |
|---|---|---|
| `context_length: 65536` in config | `/home/hermes/.hermes/config.yaml` | Compaction fires at 22,937 tokens instead of 44,800 |
| `protect_last_n: 8` (was 20) | same | Fewer reading-loop turns survive in tail; more gets summarized |
| Summary routing to local | same | Compaction hits cha0tiktower:8010, not OpenRouter (no credits) |

---

## Practical Conclusions

**Qwen3-27B is not suitable as the primary model for long-horizon agentic tasks involving file writes.** It excels at:
- Codebase analysis and diagnosis (consistently correct)
- Structured reasoning about failures
- Short-horizon tool use (single read → single write)

It fails at:
- Committing to writes after extended analysis
- Breaking out of pre-write confirmation loops
- Long-horizon task completion without human nudges

**Threshold:** Tasks under ~10 tool calls work fine. Past ~20 tool calls without a write, the model enters a permanent reading loop.

**Workaround options:**
1. Use a more decisive model (Claude Sonnet/Opus) for the main agent; keep Qwen3-27B for compaction summarization only
2. Constrain prompts to force immediate action: "make ONLY these three changes, do not read any other files"
3. Monitor for read-without-write patterns and intervene after N consecutive reads

---

## Compaction Summary Quality

The local Qwen3-27B summarizer (same model via same proxy) produced poor compaction summaries under the structured prompt. Summary budget was ~3,276 tokens (5% of 65K context). The structured template (Goal / Progress / Done / In Progress / Next Steps) was not reliably followed — summaries collapsed to goal restatement + "read source files" as next step.

This is a compaction quality failure separate from the agentic decisiveness failure. Using a higher-quality model (even a faster cloud model) for summarization only would likely improve continuity across compactions significantly.
