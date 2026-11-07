# Roadmap

Status: ✅ done · 🔄 in progress · ⬜ not started

## Sources

- Official code (GitHub): https://github.com/stanford-futuredata/FrugalGPT — local clone in `reference/FrugalGPT/`
- Cached LLM responses (GitHub release, used in step 2):
  - https://github.com/lchen001/DataHolder/releases/download/v0.0.1/HEADLINES.zip
  - https://github.com/lchen001/DataHolder/releases/download/v0.0.1/HEADLINES.sqlite
  - https://github.com/lchen001/DataHolder/releases/download/v0.0.1/qa_cache.sqlite
- Official results to check against: [`summary/`](https://github.com/stanford-futuredata/FrugalGPT/tree/main/summary)
  (e.g. `summary_HEADLINES_e8_full_2024.csv`)
- Paper: https://openreview.net/forum?id=cSimKw5p6R

## Steps

| # | Step | Status | Paper ref | Done when |
|---|---|---|---|---|
| 0 | Project docs: `AGENTS.md`, `docs/`, skills, reviewer agent, settings | ✅ | — | Files in place |
| 1 | Skeleton: `pyproject.toml` (uv, py3.12), package layout, `README.md`, ruff/pytest config, `.gitignore`, `git init` | ✅ | — | `uv run pytest` runs (0 tests OK) |
| 2 | Data: download HEADLINES cached responses ([DataHolder release](https://github.com/lchen001/DataHolder/releases/tag/v0.0.1)) → `correct` / `cost` tables | ⬜ | Sec. 5 | Per-LLM accuracy and cost match the official [`summary/`](https://github.com/stanford-futuredata/FrugalGPT/tree/main/summary) CSVs |
| 3 | Metrics: normalization + exact match | ⬜ | Sec. 3 (reward r) | Unit tests incl. edge cases |
| 4 | Pricing: cost model `c_i(p)` and the 2023 / 2024 price tables | ⬜ | Sec. 3, Tables 1 & 3 | Recomputed costs match the cached `cost` column |
| 5 | Cascade simulation on tables (order + thresholds → per-query answer and cost) | ⬜ | Sec. 4 | Hand-checked synthetic tests |
| 6 | Optimizer: chain search + threshold search under a budget; MPI matrix and pruning | ⬜ | Eq. 1, Eq. 2 | Finds the known optimum on synthetic data |
| 7 | Baselines: best single LLM; log-prob threshold (GPT-J → GPT-4) | ⬜ | Fig. 4 | Curves produced |
| 8 | Scorer: DistilBERT P(correct), training loop on MPS, per-LLM scorers (adds torch + transformers deps) | ⬜ | Sec. 4 | Scorer AUC reported per LLM |
| 9 | Reproduce HEADLINES: Table 2 row, Fig. 2 case study, Fig. 3 curve, Fig. 4 | ⬜ | Sec. 5 | Within a few points of the paper; differences explained |
| 10 | Other datasets: OVERRULING, AGNEWS, COQA, SCIQ | ⬜ | Table 2, Fig. 3 | Table 2 reproduced |
| 11 | Fresh experiment with current OpenAI models (live collection, our key) | ⬜ | Fig. 5 analogue | Cost estimated and approved, then trade-off curve |
| 12 | Ablations: scorer backbone, label-distribution shift | ⬜ | Fig. 7 | Plots |

## Log

- **2026-11-07** — Read paper + official code; documented differences in `docs/paper_vs_code.md`. Step 0 done.
- **2026-11-07** — Step 1: uv project (Python 3.12, hatchling), `src/frugalgpt/` package, README, ruff + pytest config, `git init`. Deps kept minimal (numpy, pandas, pyyaml); heavy ML deps added in step 8. `uv run pytest`: 1 passed; ruff: clean.
