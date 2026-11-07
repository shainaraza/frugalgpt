---
name: paper-reference
description: FrugalGPT paper details — equations, pipeline components, hyperparameters, datasets, expected results, and where the official code differs from the paper. Use when implementing, testing, or checking any FrugalGPT component (scorer, cascade, MPI, optimizer, cost model, metrics, baselines) or comparing our numbers to the paper.
---

# FrugalGPT paper reference

The knowledge lives in tool-neutral docs. Read what the task needs:

| Need | Read |
|---|---|
| Method, equations (cost model, Eq. 1, Eq. 2), setup, results to reproduce | `docs/paper_summary.md` |
| What the official code actually does (scorer output, accept rule, optimizer, hyperparameters, data format) | `docs/paper_vs_code.md` |
| Exact wording of a passage | `paper.pdf` (PDF page rendering may be unavailable; extract text with `pypdf` if needed) |
| Ground-truth behaviour of the authors' implementation | `reference/FrugalGPT/src/FrugalGPT/` (read only; never copy code) |
| The authors' published numbers to compare against | `reference/FrugalGPT/summary/*.csv` |

If `reference/FrugalGPT/` is missing, restore it with
`git clone --depth 1 https://github.com/stanford-futuredata/FrugalGPT reference/FrugalGPT`.

## Rules

- When the paper and the code disagree, **default to the code** and offer the paper's version behind a
  config flag. The decision table is in `AGENTS.md`.
- Thresholds are always **score thresholds** (`accept if score > τ`). If you find a `1 − τ` form in the
  reference code, convert it.
- Budget = **average cost per query**. Paper tables report **totals over the test set**.
- When citing, use the paper's section / equation numbers (e.g. "Sec. 4, Eq. 2") in docstrings.
- If something is unspecified in both the paper and the code, say so and propose a choice. Don't invent a
  "paper value".
