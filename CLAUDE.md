# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

## 5. Project Conventions: pydantic & uv

**Data models — use pydantic, not dataclasses.**

- All shared data objects are `pydantic.BaseModel`s (see `alltap/types.py`), not `@dataclass`. This gives validation, type coercion, value-based equality, and JSON (de)serialization.
- Make value objects immutable with `model_config = ConfigDict(frozen=True)` where they shouldn't change after construction.
- For models that must hold non-pydantic types (e.g. a numpy image buffer), set `model_config = ConfigDict(arbitrary_types_allowed=True)` rather than falling back to a dataclass.
- In hot paths (per-frame, per-landmark) where the data comes from a trusted source like MediaPipe, construct with `Model.model_construct(...)` to skip validation and protect the CPU/latency budget. Validation still applies to config-driven and deserialized data.

**Environment & dependencies — use uv, not pip/requirements.txt.**

- Dependencies live in `pyproject.toml`, locked in `uv.lock` (both committed). There is no `requirements.txt` or `setup.py`.
- Add dependencies with `uv add <pkg>` (and `uv add --dev <pkg>` for dev-only tools); never hand-edit `pyproject.toml` deps when `uv add` will do it.
- Install/sync with `uv sync`.
- Run everything through uv: `uv run python -m alltap.main`, `uv run pytest`.
