# Architecture

How the code is organised and how data moves through it. For the paper method see
[`paper_summary.md`](paper_summary.md); for where the official code differs see
[`paper_vs_code.md`](paper_vs_code.md).

## Data flow

LLM responses are collected once. Everything after that runs offline on three
`n_queries × n_llms` tables per split.

```
cached responses (CSV, release 0.0.1)
        │  data.py  (+ metrics.py for exact match, pricing.py for cost checks)
        ▼
  correct, cost tables          data/tables/<DS>_{train,test}.npz
        │  scorer.py  (one DistilBERT per LLM)
        ▼
  score table                   data/scores/<DS>_{train,test}.npz
        │
        │  optimizer.py  — on TRAIN: best chain + thresholds under a budget (Eq. 1)
        │     └─ mpi.py  — optional chain pruning (Eq. 2)
        ▼
  Cascade(order, thresholds)
        │  cascade.py  — run the strategy on TEST tables
        ▼
  accuracy, mean cost per query
        │  experiment.py  — repeat over budgets; baselines.py for comparison
        ▼
  trade-off curve, Table 2, case study   outputs/<DS>/   (plots.py)
```

The strategy is always learned on train and reported on test.

## Modules (`src/frugalgpt/`)

| Module | Role | Main entry points | Paper |
|---|---|---|---|
| `data` | Cached responses → `correct` / `cost` tables | `download_dataset`, `build_tables`, `load_tables` | Sec. 5 |
| `metrics` | Answer normalisation and exact match (the 0/1 reward) | `normalize_answer`, `exact_match_array` | Sec. 3 |
| `pricing` | Cost of one call: input / output tokens × price + fixed fee | `compute_cost`, `recompute_costs` | Sec. 3, Tables 1 & 3 |
| `scorer` | Trains one scorer per LLM; score = P(correct) | `train_scorer`, `score_texts`, `logits_to_score` | Sec. 4 |
| `cascade` | A strategy and its simulation on tables | `Cascade`, `stop_index`, `simulate` | Sec. 4 |
| `mpi` | Maximum performance improvement matrix, top-`m` chain pruning | `mpi_matrix`, `pruned_chains` | Eq. 2 |
| `optimizer` | Chain search × threshold search under an average-cost budget | `optimize`, `search_thresholds`, `quantiles_to_thresholds` | Eq. 1 |
| `baselines` | Best single LLM; score-threshold curve | `best_single_llm`, `threshold_curve` | Fig. 4 |
| `experiment` | Train → test protocol over budgets; Table 2 and case-study read-outs | `tradeoff`, `case_study`, `cost_to_match` | Sec. 5 |
| `plots` | Figures | `plot_tradeoff`, `plot_case_study`, `plot_baseline` | Figs. 2–4 |

### Key objects

- **`Cascade(order, thresholds)`** — `order` is a tuple of LLM column indices; `thresholds` has
  `len(order) − 1` entries. Accept LLM `order[t]` if `score > thresholds[t]`; the last LLM is always
  accepted. Cost is cumulative: a query pays for every LLM it was sent to.
- **`Split(correct, cost, score)`** — one split's three tables, the input to `experiment`.
- **`OptimizeResult` / `ChainResult`** — the chosen `Cascade` with its train accuracy and mean cost.

### Optimizer in one paragraph

`optimize` loops over candidate chains (all ordered tuples of length `L`, or the MPI-pruned set).
For each chain, `search_thresholds` searches **quantiles** of the scores rather than raw thresholds
(a 40-point grid per step, then Nelder-Mead), converts the best quantiles to score thresholds, and
returns the most accurate point within budget. The chain with the highest train accuracy wins.
The optimizer re-implements the cascade's accept rule internally for speed, so the two must stay in
sync (`test_grid_objective_matches_pointwise` checks this).

## Parameters

| Parameter | Where it is defined | How to set it |
|---|---|---|
| Chain length `L` (default 3) | `optimize(length=…)`, passed through `tradeoff` / `case_study` | `run_tradeoff.py --length` |
| Chain search (`exhaustive` / `mpi_pruned`) | `optimize(chain_search=…)` | `run_tradeoff.py --chain-search` |
| Thresholds | Learned by `optimize`; or build `Cascade(...)` by hand | — |
| Threshold grid | `optimizer.GRID_POINTS`, `Q_LOW`, `Q_HIGH` | constants |
| Budget | Average cost per query (USD); a list per run | `run_tradeoff.py` budget grid |
| Scorer hyper-parameters | `ScorerConfig` | `configs/scorer.yaml` |

## Scripts (`scripts/`)

Run in this order for a dataset `<DS>`:

1. `prepare_data.py --dataset <DS>` — download the cached responses, build `data/tables/`.
2. `train_scorers.py --dataset <DS>` — one scorer per LLM → `data/scores/` (resumable).
3. `run_tradeoff.py --dataset <DS>` — optimise on train, evaluate on test → `outputs/<DS>/`.
4. `make_table2.py` — combine datasets into one Table 2.

`check_pricing.py` and `check_dataset.py` are sanity checks against the cached data.

## Tests (`tests/`)

One file per module (`test_<module>.py`), built on small synthetic tables whose answers can be
worked out by hand. Run with `uv run pytest`.
