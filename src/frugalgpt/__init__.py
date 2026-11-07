"""FrugalGPT: budget-aware LLM cascades.

Re-implementation of Chen, Zaharia & Zou, "FrugalGPT: How to Use Large Language Models While
Reducing Cost and Improving Performance", TMLR 2024.

A cascade calls LLMs one after another, cheapest first. After each call a small learned scorer
estimates P(answer is correct); if it exceeds that LLM's threshold the answer is returned,
otherwise the next LLM is called (Sec. 4). The order and thresholds are chosen to maximize
accuracy under an average-cost budget (Eq. 1).

Everything except `frugalgpt.llm` works offline on cached responses arranged as
(n_queries x n_llms) tables of correctness, cost and score.
"""

__version__ = "0.1.0"

DEFAULT_SEED = 2024
