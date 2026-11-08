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
| **Scorer training** | Not given | HF `Trainer`, `distilbert-base-uncased`, 8 epochs, batch 8 (eval 64), warmup 500, weight decay 0.01, max_len 512, seed 2024. lr 5e-5 (default), linear warmup/decay, grad clip 1.0, no decay on bias/LayerNorm (defaults). **55% of the scorer's data held out as validation** (library default `score_test_size=0.55`, used by `FrugalGPT_gen_tradeoff_HEADLINES.ipynb`; the 2024-models notebook `…HEADLINES2024.ipynb` passes 0.45); the epoch with the **lowest validation loss** is kept (`load_best_model_at_end` without `metric_for_best_model` → `eval_loss`). |
| **Chain search** | MPI pruning, top-m (m=3) | **Exhaustive**: `itertools.permutations(models, L)` for `L = L_max` only; no MPI pruning (`llmchain.py: train`) |
| **Threshold search** | Grid cells + quadratic fit + QP | `scipy.optimize.brute` with `Ns=40` points per axis, then `scipy.optimize.fmin` (Nelder-Mead) refine (`optimizer.py: optimize`) |
| **Threshold parameterization** | Thresholds in `[0,1]` | Searches over **quantiles** `q_i ∈ (1e-5, 1−1e-5)`. Threshold `i` = `(1−q_i)`-quantile of distances among queries that reached step `i`. Quantile vectors that are not non-decreasing are rejected. |
| **Last LLM** | — | Inference (`llmchain.py: predict_one`, `llmcascade.py: get_completion`): always accepted. Training objective (`optimizer.py: f`): `δ_last = 1` on `d = 1 − score`, so a last-step score of exactly 0 is **never accepted** and that query counts as 0 accuracy and 0 cost. We always accept the last LLM (`cascade.py`), i.e. follow inference. |
| **Infeasible chain** | — | If the first LLM's average cost alone exceeds the budget, the chain is skipped |
| **Reward** | Not given | Exact match after SQuAD-style normalization: lowercase, strip punctuation, remove articles (a/an/the), collapse whitespace (`utils.py: normalize_answer`) |
| **Generation params** | — | `max_tokens=50, temperature=0.1, stop=['\n']` |
| **Tiny scores** | — | Works on `d = 1 − score`, so scores below ~1e-16 all become `d = 1.0` and are treated as tied. We work on scores directly and keep them distinct (no effect on realistic scores). |
| **No feasible chain** | — | `train` starts from `acc = −1` and a fully infeasible chain reports `−10000/n`, so with `n > 10000` training queries it can return a chain that never meets the budget. We return `None`. |
| **Budget boundary** | — | `cost > budget · n` on float sums. We use the same rule with a `1e-12` relative tolerance so summation order cannot flip a point exactly on the budget. |
| **Score ties** | — | Optional tiny random noise (`score_noise_injection`, eps=1e-8) |
| **Train split inside `train()`** | — | A further 1% of training data split off (`test_size=0.01`) and unused for the cascade |

---

## 2. Threshold search, precisely (`optimizer.py`)

For a fixed chain of `d` LLMs, it builds three `n × d` matrices on the training queries:

- `L_mat[n, j]` = 1 if LLM `j` in the chain is correct on query `n`
- `C_mat[n, j]` = **cumulative** cost of calling LLMs `0..j` on query `n`
- `d_mat[n, j]` = `1 − score` of LLM `j`'s answer on query `n`

Given thresholds `δ`, query `n` stops at the **first** `j` with `d_mat[n,j] < δ_j`
(with `δ_last = 1`, so the last LLM stops unless its score is exactly 0; see §1 "Last LLM"). Then

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

**Release CSVs vs `summary/headlines_e8_perf_seed_9_2024updatedmodel.csv` (checked in step 2).**
- Same queries: the release train/test splits contain exactly the 5000/5000 queries of `data/HEADLINES/{train,test}.csv`.
- Same reward: our normalization + exact match is identical to `service/utils.py::normalize_answer`.
- Train accuracy differs by up to 0.58 pts (GPT-FSQ), and train cost by up to 0.04%. Since the queries and
  the scoring rule are identical, the summary must have been computed from a slightly different response
  cache (the older `api_performance/headlines/9_train` run, which isn't in the repo), not from the release CSVs.
- `Test_acc` for the 2023 models in that file is a placeholder (`0.1` for every model), and `Test_cost` is
  wrong for J1 / GPT-3 / GPT-4, so only the train columns can be compared.
- The file also lists models that aren't in the release (J1-I, FA-Q, FA-C, FT-GPT-C, Dolly).

---

## 3b. Cost recomputation (`pricing.py`, step 4)

The release CSVs store each call's `cost` but not its token counts. To check the Sec. 3 cost model we
re-tokenize the cached prompt and answer (`tiktoken`) and compare. Prices match Table 1 / Table 3 and
`config/serviceinfo.json` exactly. Counting rules (fitted to HEADLINES, `COUNTING_2023`):

| Models | Rule | Exact share (train / test) |
|---|---|---|
| GPT-C, GPT-3 | `r50k_base` / `p50k_base` tokens | 100% / 100% |
| GPT-4 | `cl100k_base`, +7 prompt tokens (OpenAI chat format, one user message) | 100% / 100% |
| ChatGPT | as GPT-4, and +1 output token (fitted; cause not documented) | 100% / 100% |
| GPT-J | GPT-2 tokens, +1 output for the `\n` stop token | 100% / 100% |
| GPT-FSQ | as GPT-J, and −1 prompt token per `"\n\n"` (fitted) | 100% / 100% |
| J1-L/G/Jumbo | input free; GPT-2 tokens +1 stop (AI21's tokenizer differs) | ≥ 99.9% |
| GPT-Neo | GPT-2 tokens +1 stop (own BPE, approximation) | total −0.4% |
| CoHere, CoHere-M | **characters**: `len(prompt)` + a constant 2 output | 100% / 100% |

- Billed output is capped at `max_tokens = 50`: an answer cut off at the limit has no stop token.
- **CoHere bug, reproduced on purpose:** the official `_get_io_tokens` bills `len(context)/1000` and
  `len(completion)/1000` at $0.0025 each, but `completion` is the result dict `{raw, completion}`, so the
  output always counts as 2 characters, whatever the answer. Price is $2.5 per 1M characters (the paper
  reports ≈ $10 per 1M tokens).
- Later steps use the **cached** `cost` table, not recomputed costs: that is what produced the paper's
  numbers, and GPT-Neo / J1 are not exact.

---

## 3c. Baselines (`baselines.py`, step 7)

- **Best single LLM** (Table 2, Fig. 3 dots): same as the official `evaluate.py: compute_score`, i.e. mean exact
  match and mean cached `cost` per model. HEADLINES test: GPT-4 is best, 85.56%, total $33.31 (paper: $33.1).
- **Simple threshold baseline (Fig. 4)**: paper only. The official repo has **no implementation**; the only trace
  is `"logprobs": 1` in API requests and an empty `result_dict['logprobs']` in `llmcascade.py`. The release CSVs
  have **no log-prob column**, so GPT-J's mean token log-prob cannot be recovered offline (and GPT-J's 2023 API
  is no longer available to us). `threshold_curve` therefore takes any per-query score; the real Fig. 4 curve is a
  known gap. In Step 9 we draw it with the Step 8 scorer's GPT-J score as a stand-in, so it compares an L = 2 chain
  with fixed models against the optimized cascade, not log-prob vs. learned scorer.
- **Empty answers**: a `-inf` score (no tokens) is always escalated, for every threshold (`-inf > τ` is false).
  Not specified in the paper.
- **Budget**: `best_single_llm` uses the optimizer's `over_budget` (1e-12 relative tolerance) so both sides of a
  comparison agree on feasibility.
- **Units**: functions return mean cost per query (the Eq. 1 budget unit); `ThresholdCurve.total_cost` and
  `CascadeResult.total_cost` give the test-set totals that Table 2 / Fig. 3 / Fig. 4 plot.

---

## 3d. Scorer (`scorer.py`, step 8)

- Plain PyTorch loop reproducing the HF `Trainer` defaults the official code relies on (AdamW lr 5e-5, β = (0.9,
  0.999), ε = 1e-8; linear warmup 500 then linear decay to 0; grad-norm clip 1.0; weight decay 0.01 except bias and
  LayerNorm; dropout on during training; best epoch by validation loss).
- **Padding**: official pads every text to the longest in the whole set; we pad per batch. Same logits up to float
  noise (attention mask).
- **Validation split**: official `train_test_split(test_size=0.55)` has no `random_state`, but it is deterministic:
  `scoring.py` calls `set_seed(2024)` at import and every `Trainer.__init__` reseeds. The permutation therefore
  depends on call order (the 1% split in `llmcascade.py`, the previous LLM's training) and on DataFrame row order,
  so it cannot be reproduced bit-for-bit. Ours: seed 2024, same sizes (`ceil(0.55 n)` held out), same split for
  every LLM.
- **Head init**: official loads a fresh pretrained model per LLM; its new head depends on the torch RNG state left
  by the previous training run (deterministic, order-dependent). We seed once (2024) and deep-copy the same
  initial model for every LLM.
- **Truncation**: texts are cut at 512 tokens on the right, as in the official code. If the query's own block
  is longer, the answer itself is cut off (COQA 3.8%, SCIQ 1.0% of inputs; see §3f).
- **Softmax precision**: we compute `softmax` in float64; the official `get_score` uses float32 per text, which
  saturates to exactly 1.0 once the logit gap exceeds ≈ 17 and so creates ties we don't have. Negligible.
- **Resume safety**: each per-LLM score file stores a fingerprint of the queries, that LLM's answers and the
  config; stale files are retrained, never combined.
- **Train-split scores are optimistic**: the cascade optimizer uses scores on the train split, 45% of which the
  scorer was trained on. Official code does the same; we follow it (possible ablation later).

---

## 3e. Experiments (`experiment.py`, `scripts/run_tradeoff.py`, step 9)

- **Protocol** = official `FrugalGPT_gen_tradeoff_HEADLINES.ipynb`: per budget, learn chain + thresholds on train,
  simulate on test; budgets are mean cost per query, results reported as test-set totals.
- **Official curves use the 2024 pool.** Both HEADLINES notebooks (also the one without `2024` in its name) call
  gpt-4o-mini / gpt-4o / gpt-4-turbo / Llama-3-70B / Gemma-2-9B, and the FrugalGPT rows of
  `summary/headlines_e8_perf_seed_9_2024updatedmodel.csv` are that run (its first point is gpt-4o-mini alone).
  There is no released 2023 FrugalGPT curve, so we compare with the paper's 2023 numbers (Fig. 2, Table 2).
- **Budget grid**: the official `linspace(3.5e-5, 3.6e-3, 20)` was chosen for the 2024 pool. We keep its points
  above the cheapest 2023 LLM (GPT-J, 5.5e-5 per query), add 10 log-spaced points from 1.001 × GPT-J's cost to
  4e-4, and the Fig. 2 budget. A budget equal to the cheapest LLM's cost is never feasible: the quantile search
  starts at 1e-5, so the lowest-scored queries are always passed on.
- **Fig. 2 budget** = best single LLM's mean **train** cost / 5 (paper: "1/5 of GPT-4's cost"); train and test
  differ by < 0.01%.
- **Read-outs on test**: Table 2 picks the cheapest grid point whose test accuracy ≥ GPT-4's, and Fig. 4's
  baseline sweeps its threshold over test-score quantiles. This is reading points off the curves, as the paper's
  figures do. The Table 2 point is **selected on noisy test accuracy**, so it is optimistic by selection: a point
  that matches by a few queries by chance wins. Since step 10, `table2.json` also reports the match margin (in
  queries), its paired-bootstrap support, and a **robust** read-out: the cheapest point that is ≥ the reference in
  ≥ 90% of 2000 paired bootstrap resamples of the test set (`experiment.robust_cost_to_match`).
- **Fig. 4 stand-in score**: GPT-J's DistilBERT scorer instead of its mean token log-prob (§3c). A learned scorer
  is a stronger signal than raw log-probs, so our baseline saves more than the paper's (~60%).

---

## 3f. Other datasets (step 10)

- **Few-shot prefix stripped before scoring.** The official notebooks prepend `config/prompt/<DS>/prefix_eN.txt`
  to each query and pass the same string to the cascade, which removes it before `scorer_text`. So the scorer
  sees only the query's own block. This matters for prompts without `Q:` (OVERRULING, COQA, SCIQ use
  `Context: … Question: … Answer:`). Without it, COQA / SCIQ scorer inputs are 900–1100 tokens, and after
  truncation to 512 **all inputs are identical** (only the few-shot block survives).
- **Prefix taken from the data.** The repo's prefix files belong to the 2024 runs and differ slightly from the
  2023 cached prompts: AGNEWS "falls into" vs. "follows into", COQA doubled quotes, SCIQ an extra blank line.
  So no 2023 prompt starts with them. We derive the shared block from the train prompts instead (longest
  common start, cut at its last blank line; `scorer.prompt_prefix`). It equals the official file exactly on
  HEADLINES and OVERRULING, and 100% of test prompts start with it on every dataset.
- **Truncation after stripping:** COQA scorer inputs have a median of 365 tokens and 3.8% exceed 512 (their
  answer is cut off, as in the official code). SCIQ: median 88, 1.0% over 512. Other datasets: 0%.
- **Repeated prompts (COQA, SCIQ).** Some distinct dataset items have identical prompt text: SCIQ 1 per
  split, with the same reference answer; COQA 15 (train) / 9 (test) prompts (10 extra test rows), mostly with
  different reference answers (one COQA test group repeats its reference). Responses were cached by prompt, so
  every copy has the same answer and cost. `LLMCascade` evaluates each row, so `build_tables` keeps every copy as
  its own query (the j-th copy is matched across LLMs; row order is identical for every LLM). Sizes: COQA
  3991 / 3992, SCIQ 5839 / 5840. The official **2024** notebooks differ: `convert_and_merge_dataframes`
  rebuilds each query from its last `\n\n` block and merges rows with identical (last block, reference) via
  `groupby(...).first()` (`examples/FrugalGPT_gen_tradeoff_SCIQ.ipynb` cell 8, `…COQA2024.ipynb` cell 9; the
  COQA 2024 notebook also reshuffles train/test with `random_state=42`). That would drop 1 SCIQ row per split
  and 1 COQA test row; we keep the 2023 release rows as they are. This also explains why the 2024 prefix files
  differ from the 2023 prompts.
- **COQA train/test overlap (from the release).** 21 COQA test prompts (0.5%) also appear verbatim in train, so
  the scorer saw their exact text with a label. SCIQ, AGNEWS, OVERRULING and HEADLINES have none. We keep the
  official split and report a COQA sensitivity check without those rows.
- **Table 2 reference = best LLM on train.** On OVERRULING test, GPT-3 (95.9%) edges out GPT-4 (95.6%); on train
  GPT-4 is best, and the paper's Table 2 uses GPT-4 ($9.7 ≈ our $9.68). Choosing on train also avoids picking
  anything by looking at test.
- **Budget grid** for the other datasets: 30 log-spaced points (`--n-log`) from 1.01 × the cheapest LLM's train
  cost to the reference LLM's train cost, plus 1/5 of the reference cost. (1.001 × was never feasible, see §3e.)
- **Prefix check.** `train_scorers.py` stops if any train or test prompt does not start with the derived prefix
  (`scorer.prefix_coverage`), since `scorer_text` would otherwise silently keep the few-shot block.
- **Score/row alignment.** Combined `data/scores/<DS>_<split>.npz` files store a hash of the split's queries
  (`scorer.queries_hash`); `run_tradeoff.py` refuses scores whose rows do not match the tables.
- **AGNEWS saves more than the paper (open difference).** Review found no leakage, and our scorer text equals the
  official one on 100% of AGNEWS test rows. Contributors we can measure: (1) Table 2 selection on test noise
  (the plain match wins by 2 queries; see the robust read-out); (2) a train → test score shift: scorers are
  more confident on train (GPT-3 AUC 0.860 vs 0.728), so the train-learned thresholds escalate fewer test
  queries and test cost is 12.6% below train cost at the matching budget (official protocol, cf. §3d). Weak
  LLMs get very high scorer AUCs on AGNEWS (GPT-NeoX 0.963, GPT-J 0.938): the DistilBERT scorer learns the
  4-way topic and checks the answer. What the 2023 paper run did differently is unknown.

---

## 4. Our implementation decisions

- **Default = match the code.** Paper-only variants go behind config flags:
  - `scorer.output: p_correct | max_softmax`
  - `optimizer.chain_search: exhaustive | mpi_pruned` (with `m`)
  - `optimizer.threshold_search: quantile_grid | grid_qp`
- Thresholds are stored and reported as **score thresholds** (`accept if score > τ`), never `1 − τ`.
- Our own plain PyTorch training loop instead of HF `Trainer`, with the same hyperparameters.
