# FrugalGPT

A clean re-implementation of **FrugalGPT** — *How to Use Large Language Models While Reducing Cost
and Improving Performance* (Chen, Zaharia, Zou; TMLR 2024,
[OpenReview](https://openreview.net/forum?id=cSimKw5p6R)).

FrugalGPT answers each query with an **LLM cascade**: call a cheap LLM first, score its answer with a
small learned model (DistilBERT → P(correct)), and escalate to a more expensive LLM only if the score
is below a per-model threshold. The order of LLMs and the thresholds are optimized to maximize
accuracy under an average-cost budget.

## Quick start

```bash
uv sync            # create .venv and install dependencies
uv run pytest      # run the tests
uv run ruff check  # lint
```

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/). Runs on macOS (Apple-silicon GPU via MPS) or CPU.

## Design

Offline-first. LLM responses are collected once and cached; every other stage (scorer training,
cascade simulation, optimization, baselines, plots) runs on `(n_queries × n_llms)` tables of
**correctness**, **cost** and **score**. Only `frugalgpt.llm` calls external APIs.

| Module | Paper |
|---|---|
| `data` — datasets and cached responses → tables | Sec. 5 |
| `metrics` — answer normalization, exact match | reward `r`, Sec. 3 |
| `pricing` — cost model `c_i(p)` | Sec. 3, Tables 1 & 3 |
| `scorer` — generation scorer `g_i(q, a)` | Sec. 4 |
| `cascade` — run a chain with thresholds | Sec. 4 |
| `mpi` — maximum performance improvement, pruning | Eq. 2 |
| `optimizer` — chain + threshold search under a budget | Eq. 1 |
| `baselines` — best single LLM, log-prob threshold | Fig. 4 |

Status: see [`ROADMAP.md`](ROADMAP.md). Notes on the paper and on how the official code differs:
[`docs/`](docs/).

## Acknowledgements

Datasets and cached 2023 LLM responses are from the authors' release at
https://github.com/stanford-futuredata/FrugalGPT.
