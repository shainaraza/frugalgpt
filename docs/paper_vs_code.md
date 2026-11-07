# Paper vs. Official Code

Notes from reading the authors' implementation
(https://github.com/stanford-futuredata/FrugalGPT; local clone in `reference/FrugalGPT/`).
Files read: `src/FrugalGPT/{llmcascade,llmchain,optimizer,scoring}.py`, `src/service/utils.py`.
Where the paper is vague or disagrees with the code, the code is what produced the published numbers.

---

## 1. Summary of differences

| Topic | Paper | Official code |
|---|---|---|
| **Scorer output** | Max of softmax | `prob[1]` = **P(correct)** (`scoring.py: get_score`) |
| **Accept rule** | `g_i(q,a) > τ_i` | Uses distance `d = 1 − score` and accepts if `d < τ_i`, i.e. `score > 1 − τ_i` (`optimizer.py: compute_dist`, `llmchain.py: predict_one`). This is why Fig. 2(a) shows 0.04/0.63 while the text says 0.96/0.37. |
| **Scorer input text** | Query + answer | Few-shot prompt prefix stripped; keeps only the last `Q: … A:` block, then `" " + answer` (`llmcascade.py: scorer_text`) |
| **Scorer training** | Not given | HF `Trainer`, `distilbert-base-uncased`, 8 epochs, batch 8 (eval 64), warmup 500, weight decay 0.01, max_len 512, seed 2024. **55% of the scorer's data held out as validation**; best epoch by val accuracy kept (`load_best_model_at_end`). |
| **Chain search** | MPI pruning, top-m (m=3) | **Exhaustive**: `itertools.permutations(models, L)` for `L = L_max` only; no MPI pruning (`llmchain.py: train`) |
| **Threshold search** | Grid cells + quadratic fit + QP | `scipy.optimize.brute` with `Ns=40` points per axis, then `scipy.optimize.fmin` (Nelder-Mead) refine (`optimizer.py: optimize`) |
| **Threshold parameterization** | Thresholds in `[0,1]` | Searches over **quantiles** `q_i ∈ (1e-5, 1−1e-5)`. Threshold `i` = `(1−q_i)`-quantile of distances among queries that reached step `i`. Quantile vectors that are not non-decreasing are rejected. |
| **Last LLM** | — | Always accepted (threshold fixed at 1) |
| **Infeasible chain** | — | If the first LLM's average cost alone exceeds the budget, the chain is skipped |
| **Reward** | Not given | Exact match after SQuAD-style normalization: lowercase, strip punctuation, remove articles (a/an/the), collapse whitespace (`utils.py: normalize_answer`) |
| **Generation params** | — | `max_tokens=50, temperature=0.1, stop=['\n']` |
| **Score ties** | — | Optional tiny random noise (`score_noise_injection`, eps=1e-8) |
| **Train split inside `train()`** | — | A further 1% of training data split off (`test_size=0.01`) and unused for the cascade |

---

## 2. Threshold search, precisely (`optimizer.py`)

For a fixed chain of `d` LLMs, it builds three `n × d` matrices on the training queries:

- `L_mat[n, j]` = 1 if LLM `j` in the chain is correct on query `n`
- `C_mat[n, j]` = **cumulative** cost of calling LLMs `0..j` on query `n`
- `d_mat[n, j]` = `1 − score` of LLM `j`'s answer on query `n`

Given thresholds `δ`, query `n` stops at the **first** `j` with `d_mat[n,j] < δ_j`
(with `δ_last = 1`, so the last LLM always stops). Then

```
accuracy = Σ_n L_mat[n, stop(n)]
cost     = Σ_n C_mat[n, stop(n)]
objective: maximize accuracy  s.t.  cost ≤ budget · n   (infeasible → penalty 10000)
```

So the **budget is an average cost per query**.

---

## 3. Data released by the authors

- FrugalGPT repo, GitHub release `0.0.1`: **cached generations** for 12 LLM APIs (2023):
  `{AGNEWS, COQA, HEADLINES, OVERRULING, SCIQ}_{train,test}.csv`, 17 MB (OVERRULING) to 362 MB (SCIQ) per file.
  Columns: `<index>, query, ref_answer, model name, answer, cost, max_tokens, temperature, stop`.
  `query` holds the full few-shot prompt; `answer` usually has a leading space (e.g. `" down"`).
- `data/HEADLINES/{train,test}.csv`, `data/COQA/…` in the repo: `query, ref_answer, _id` (query = last `Q: … A:` only).
- `config/prompt/<DATASET>/prefix_e*.txt`: few-shot prompts (HEADLINES: 8 examples).
- `config/serviceinfo.json`: per-model price constants (`cost_input`, `cost_output`, `cost_fixed` per token / request).
- `config/serviceidmap.json`: model-name → id map.
- `summary/*.csv`: the authors' per-model / per-budget results (useful to check our numbers).
- Separate repo `lchen001/DataHolder` releases: SQLite caches and trained scorer checkpoints (≈1 GB zips; not needed).

---

## 4. Our implementation decisions

- **Default = match the code.** Paper-only variants go behind config flags:
  - `scorer.output: p_correct | max_softmax`
  - `optimizer.chain_search: exhaustive | mpi_pruned` (with `m`)
  - `optimizer.threshold_search: quantile_grid | grid_qp`
- Thresholds are stored and reported as **score thresholds** (`accept if score > τ`), never `1 − τ`.
- Our own plain PyTorch training loop instead of HF `Trainer`, with the same hyperparameters.
