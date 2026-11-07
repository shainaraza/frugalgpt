---
name: implement-step
description: Routine for building the next FrugalGPT roadmap step — locate the step and its paper section, implement a small documented module, write and run tests, review against the paper, update ROADMAP.md, and explain the design. Use whenever starting or continuing a step in ROADMAP.md.
---

# Implement one roadmap step

Follow these phases in order. Do **one** step only.

## 1. Orient
- Read `ROADMAP.md`. Find the first step that isn't ✅ (or the step the user named). Mark it 🔄.
- Read `AGENTS.md` for decisions and conventions.
- Use the `paper-reference` skill to read the relevant paper section and the matching official-code behaviour.

## 2. Plan (brief, in chat)
- What the module does, its public API (function signatures), and which paper section / equation each maps to.
- Any choice not fixed by the paper or code: state the options and your recommendation.
- If the step costs money (OpenAI calls) or downloads > 100 MB, give the estimate and **wait for approval**.

## 3. Implement
- Small module in `src/frugalgpt/` (or `scripts/`). Docstrings cite the paper (`"""Sec. 4, Eq. 2: …"""`).
- Readable over clever. Type hints on public functions. Seeds fixed (default 2024).
- Don't copy from `reference/`. Write our own code.

## 4. Test
- Add `tests/test_<module>.py`. Prefer small synthetic cases whose answer can be checked by hand, plus
  edge cases (empty input, ties, budget infeasible, last LLM always accepted…).
- Run `uv run pytest -q` and `uv run ruff check`. Fix until green. Report real output; never claim a pass you didn't see.

## 5. Review
- For steps that implement paper logic (metrics, pricing, cascade, MPI, optimizer, scorer), launch the
  `paper-fidelity-reviewer` subagent on the new files. Fix confirmed issues; mention anything you chose not to change.

## 6. Close
- In `ROADMAP.md`: mark the step ✅ and add a dated one-line entry to the Log (include key numbers, if any).
- If a new decision was made, add it to the decision table in `AGENTS.md`; if a new paper/code difference
  was found, add it to `docs/paper_vs_code.md`.
- In chat, give a short summary:
  - what was built (files)
  - how it maps to the paper
  - design choices + trade-offs and alternatives considered
  - test results
  - the proposed next step. **Stop and wait for approval.**
