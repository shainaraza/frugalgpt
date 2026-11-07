# FrugalGPT — Paper Summary

**Paper:** *FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance*
Lingjiao Chen, Matei Zaharia, James Zou — Transactions on Machine Learning Research (TMLR), 12/2024
OpenReview: https://openreview.net/forum?id=cSimKw5p6R · Local copy: [`../paper.pdf`](../paper.pdf)
Official code/data: https://github.com/stanford-futuredata/FrugalGPT

---

## 1. Core idea in one paragraph

LLM APIs differ in price by up to two orders of magnitude, and no single LLM is best on every query.
FrugalGPT builds an **LLM cascade**: send the query to a cheap LLM first, use a small learned
**scorer** to judge the answer, and only escalate to a more expensive LLM if the score is below a
threshold. The order of LLMs and the per-LLM thresholds are chosen by an optimizer to **maximize
accuracy subject to an average-cost budget**. On five datasets this matches the best single LLM
(GPT-4 / GPT-3) at **52–98% lower cost**, or improves accuracy by up to ~4% at equal cost.

Two empirical observations motivate it (Sec. 1):
1. **A small model can judge answer quality.** DistilBERT predicts whether an LLM's answer is correct well enough to drive routing.
2. **No LLM is universally better.** On HEADLINES, Llama 3 (70B) is right on 5% of queries where GPT-4 Turbo is wrong, and the two agree on 87%.

---

## 2. Problem statement (Sec. 3)

- Queries `q ~ Q`, correct answer `a`.
- An LLM marketplace of `K` APIs `f_1 … f_K`. Each maps a prompt to an answer.
- **Cost model** for API `i` on prompt `p`:

  ```
  c_i(p) = c̃_{i,2} · ‖f_i(p)‖  +  c̃_{i,1} · ‖p‖  +  c̃_{i,0}
           (output tokens)        (input tokens)    (fixed per request)
  ```

- **Goal (budget-aware LLM usage):** choose a strategy `s` to

  ```
  maximize   E[ r(q, â(s, q)) ]        (expected reward, e.g. exact-match accuracy)
  subject to E[ c(s, q) ] ≤ b          (average cost per query ≤ budget b)
  ```

---

## 3. The FrugalGPT pipeline (Sec. 4)

Three components, applied in a loop:

| Component | Role | Instantiation in the paper |
|---|---|---|
| **LLM router** | Pick the next LLM to call, given the last one called | A fixed (query-agnostic) permutation `σ`, e.g. GPT-J → J1-L → GPT-4. At most `L` LLMs per query. |
| **Generation scorer** `g_i(q, a) ∈ [0,1]` | Estimate quality of LLM `i`'s answer | DistilBERT + linear head (768 → 2), softmax. **One scorer trained per LLM** on (query + that LLM's answer, is-correct label). |
| **Stop judger** | Accept the answer or go back to the router | Threshold rule: accept if `g_i(q, a) > τ_i`. |

**Inference for one query:**

```
for t = 1 .. L:
    k   = σ^(t)(∅)              # t-th LLM in the chain
    a_k = f_k(q)                # pay c_k(q)
    if g_k(q, a_k) > τ_k or t == L:
        return a_k
```

Cost of a query = sum of costs of **every** LLM called up to and including the one accepted.

---

## 4. Optimization (Eq. 1)

Jointly choose the chain `σ` and thresholds `τ` to maximize expected reward under the budget.
The paper shows this is **NP-hard** even with perfect scorers (proof in the appendix, which is not in our PDF).

Two approximations:

1. **Search-space pruning via MPI (Eq. 2).**
   *Maximum Performance Improvement*:
   `MPI(k1, k2) = Pr[ r(q, f_{k1}(q)) > r(q, f_{k2}(q)) ]`
   = how often `k1` is right when `k2` is wrong.
   After LLM `k`, only consider the **top-m** LLMs by `MPI(·, k)` as the next LLM.
   Complexity drops from `O(K^L)` to `O(m^L)`. The paper uses `m = 3`.

2. **Threshold search.** For a fixed chain: split `[0,1]^L` into equal grid cells, approximate the
   objective in each cell by a quadratic fitted from the cell's corner values, solve each cell with a
   QP solver, and keep the best.
   *(The official code does something different; see [`paper_vs_code.md`](paper_vs_code.md).)*

---

## 5. Experimental setup (Sec. 5)

- **Chain length** `L = 3`.
- **Split:** each dataset randomly split **50% train** (learn the scorer and cascade) / **50% test**.
- **Datasets:**

  | Dataset | Task | Best single LLM (2023) | Cost savings at equal accuracy |
  |---|---|---|---|
  | HEADLINES | 4-way classification (gold price direction: up/down/neutral/none) | GPT-4 | **98.3%** ($33.1 → $0.6) |
  | OVERRULING | 2-way classification (legal: is it overruling?) | GPT-4 | 73.3% ($9.7 → $2.6) |
  | COQA | Free-form reading-comprehension QA | GPT-3 | 59.2% ($72.5 → $29.6) |
  | AGNEWS | 4-way news topic classification | GPT-4 | 75.4% ($64.6 → $15.9) |
  | SCIQ | Science QA | GPT-3 | 52.3% ($132.4 → $63.1) |

- **2023 LLM pool (Table 1):** 14 APIs from OpenAI, AI21, Cohere, Textsynth, Databricks and ForeFrontAI,
  e.g. GPT-J ($0.2 / $5 per 1M in/out tokens) up to GPT-4 ($30 / $60).
- **2024 LLM pool (Table 3):** GPT-4 Turbo, GPT-4o, GPT-4o-mini, Claude 3.5 Sonnet, Gemini 1.5 Pro/Flash/Flash-8B,
  Jamba 1.5 Large, Llama 3 (70B), Gemma 2 (9B). Evaluated on HEADLINES and SCIQ (Fig. 5), with >90% savings.

### Key results / figures to reproduce

| Item | What it shows |
|---|---|
| **Fig. 2** | HEADLINES case study, budget $6.5 (1/5 of GPT-4). Learned chain GPT-J → J1-L → GPT-4. −80% cost, +1.5% accuracy vs GPT-4. Only 16.6% of queries reach GPT-4. MPI heatmap. |
| **Table 2** | Cost needed to match the best single LLM, per dataset. |
| **Fig. 3** | Accuracy-vs-cost trade-off curves (HEADLINES, OVERRULING, COQA). |
| **Fig. 4** | Baseline comparison: "simple threshold" = GPT-J, accept if mean token log-prob > threshold, else GPT-4. Saves ~60% vs FrugalGPT's >90%. |
| **Fig. 5** | Same trade-offs with the 2024 models. |
| **Fig. 7a** (appendix) | Robustness to label-distribution shift on HEADLINES. |
| **Fig. 7b** (appendix) | Scorer backbone ablation: ALBERT (11M) < DistilBERT (67M) ≈ BERT (110M), at budget = 10% of GPT-4. |

Note: dollar amounts in Table 2 / figures are **totals over the test set**, not per query.

---

## 6. Gaps and inconsistencies in the paper

1. The **appendix is not included** in our PDF (dataset details, NP-hardness proof, Table 5, Fig. 7).
2. **Not specified:** prompts / few-shot examples, temperature, grading rule for free-form QA, scorer
   training hyperparameters, grid resolution of the threshold search.
3. **Scorer output:** the text says the *max* softmax value is the score. That is classifier confidence,
   not P(correct), and would rate a confident "wrong" prediction highly.
4. **Fig. 2(a) vs text:** the figure shows thresholds 0.04 / 0.63, the text says 0.96 / 0.37 (exact
   complements). The official code compares `score > 1 − τ`, which explains it.
5. Most **2023 APIs are discontinued** (GPT-Curie, J1, Textsynth models, Dolly, ForeFront), so those
   results can only be reproduced from the authors' cached responses.
