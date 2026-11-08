---
name: frugalgpt
description: How to work on this FrugalGPT re-implementation — build one ROADMAP.md step with three user checkpoints (plan, results, next step), and where to look up the paper, the official code, and the authors' numbers. Use whenever starting or continuing a roadmap step, or when implementing, testing or checking any FrugalGPT component (data, metrics, pricing, cascade, MPI, optimizer, scorer, baselines).
---

# FrugalGPT: how to build a step

Do **one** roadmap step at a time. Stop at each 🛑 checkpoint, show the user what you have, end with a
clear question, and wait for their answer. Never skip a checkpoint.

## Where to look things up

| Need | File |
|---|---|
| Current step, sources, links | `ROADMAP.md` |
| Rules, decisions, conventions | `AGENTS.md` |
| Paper method, equations, results to reproduce | `docs/paper_summary.md` |
| What the official code does differently | `docs/paper_vs_code.md` |
| Exact paper wording | `paper.pdf` (extract text with `pypdf` if needed) |
| Authors' code (read only, never copy) | `reference/FrugalGPT/src/` |
| Authors' published numbers | `reference/FrugalGPT/summary/*.csv` |

Missing `reference/`? `git clone --depth 1 https://github.com/stanford-futuredata/FrugalGPT reference/FrugalGPT`

Also read any **✍️ My notes** section in these files: the user's own additions.

## Paper rules

- Paper and code disagree → **default to the code**; put the paper version behind a config flag.
- Thresholds are **score thresholds** (`accept if score > τ`). Convert any `1 − τ` in the reference code.
- Budget = **average cost per query**. Paper tables report **totals over the test set**.
- Unspecified in both paper and code → say so and propose a choice. Never invent a "paper value".
- Docstrings cite the paper: `"""Sec. 4, Eq. 2: …"""`.

## Steps

1. **Orient.** Find the first step in `ROADMAP.md` that isn't ✅ (or the one the user named); mark it 🔄.
   Read the matching paper section and official code.
2. **Plan.** In chat: what the module does, its functions, which files change, paper section for each,
   open choices with your recommendation, any cost (OpenAI calls, downloads > 100 MB).
   Read-only exploration is fine; no code yet.
   🛑 **Checkpoint 1 — plan.** Wait for approval or changes.
3. **Implement.** Small module in `src/frugalgpt/` (or `scripts/`). Readable, type hints, seed 2024.
4. **Test.** `tests/test_<module>.py` with small hand-checkable cases and edge cases (empty input,
   ties, infeasible budget, last LLM always accepted…). Run `uv run pytest -q` and `uv run ruff check`
   until green. Report the real output.
5. **Review.** For paper logic, run the `paper-fidelity-reviewer` agent on the new files.
   🛑 **Checkpoint 2 — results.** Show test results, key numbers, and each reviewer finding with your
   proposal (fix / don't fix, why). Wait for the user's decision, then apply it.
6. **Close.** In `ROADMAP.md` mark ✅ and add a dated log line with key numbers. New decision →
   `AGENTS.md`; new paper/code difference → `docs/paper_vs_code.md`. Summarize in chat: files built,
   paper mapping, design choices and trade-offs, test results, proposed next step.
   🛑 **Checkpoint 3 — next step.** Remind the user they can commit and push. Wait for approval.
