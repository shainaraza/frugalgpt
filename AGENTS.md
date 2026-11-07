# FrugalGPT — Re-implementation

A clean, from-scratch implementation of **FrugalGPT** (Chen, Zaharia, Zou — TMLR 12/2024):
a budget-aware **LLM cascade** that calls cheap LLMs first, scores each answer with a small learned
model, and escalates to more expensive LLMs only when the score is below a threshold.

- Paper: [`paper.pdf`](paper.pdf) · https://openreview.net/forum?id=cSimKw5p6R
- Paper notes: [`docs/paper_summary.md`](docs/paper_summary.md)
- Where the official code differs from the paper: [`docs/paper_vs_code.md`](docs/paper_vs_code.md)
- Official code (read-only reference, not imported): `reference/FrugalGPT/` (git-ignored; clone of
  https://github.com/stanford-futuredata/FrugalGPT)
- Progress: [`ROADMAP.md`](ROADMAP.md)

## How we work

- **One step at a time.** Build only the current step in `ROADMAP.md`. Propose the next step and wait for
  approval before starting it.
- Every step ends with: tests passing, the roadmap updated, and a short explanation of the design choices
  and trade-offs.
- **Never call a paid API** (OpenAI) without explicit approval and a cost estimate first. All experiments
  run offline on cached responses by default.
- Don't copy code from `reference/`. Read it to understand behaviour, then write our own implementation.

## Core design: offline-first

1. **Collect** each LLM's answer and cost for every query once, and save them to a cache.
2. Turn each dataset into three `n_queries × n_llms` tables: `correct`, `cost`, `score`.
3. Scorer training, MPI, cascade simulation, optimization, baselines and plots all operate on those
   tables using numpy. Only `src/frugalgpt/llm/` talks to the network.

## Decisions (defaults match the official code, which produced the paper's numbers)

| Topic | Default | Alternative (config flag) |
|---|---|---|
| Scorer output | P(correct) = softmax[1] | `max_softmax` (paper text) |
| Accept rule | accept LLM `i`'s answer if `score > τ_i`; last LLM always accepted | — |
| Chain search | exhaustive over ordered tuples of length `L` | `mpi_pruned` with top-`m` (paper Eq. 2) |
| Threshold search | quantile-parameterized grid + Nelder-Mead refine | `grid_qp` (paper Sec. 4) |
| Budget | **average cost per query** (USD) | — |
| Chain length | `L = 3` | — |
| Split | 50% train / 50% test | — |
| Reward | exact match after SQuAD-style normalization | — |
| Scorer | `distilbert-base-uncased` + 2-way head, 8 epochs, batch 8, warmup 500, wd 0.01 | ALBERT / BERT backbones |

We always store and report thresholds as **score thresholds** (`score > τ`), never as `1 − τ`.

## Environment

- macOS, no CUDA. Torch device order: `mps` → `cpu`.
- Python 3.12 managed with **uv**. Run everything through `uv run …`. Add a dependency only in the step that first needs it (`uv add <pkg>`; dev tools with `uv add --dev`).
- Only LLM provider available: **OpenAI** (`OPENAI_API_KEY` from the environment; never commit keys).

## Conventions

- Package lives in `src/frugalgpt/`; scripts in `scripts/`; tests in `tests/`; configs in `configs/` (YAML → dataclasses).
- Small, single-purpose modules. Each public function has a docstring that names the paper section /
  equation it implements (e.g. `"""Eq. 2: MPI pruning."""`).
- Prefer plain, readable code over clever abstractions. Use a plain PyTorch training loop, not HF `Trainer`.
- Type hints on public functions. Vectorized numpy over Python loops where it stays readable.
- Tests: `uv run pytest`. Use small synthetic tables where the right answer can be worked out by hand.
- Lint/format: `uv run ruff check` and `uv run ruff format`.
- Fixed seeds everywhere (default `2024`) so results are reproducible.
- Large data, caches, checkpoints and outputs go in `data/`, `cache/`, `checkpoints/`, `outputs/` (git-ignored).

## Planned layout

```
src/frugalgpt/
  data.py        # datasets + cached responses → correct/cost tables
  metrics.py     # normalization + exact match (reward r)
  pricing.py     # cost model c_i(p) and price tables
  llm/           # OpenAI client + on-disk response cache
  scorer.py      # generation scorer g_i(q, a)
  cascade.py     # Cascade(order, thresholds): simulate on tables / run live
  mpi.py         # MPI matrix + pruning (Eq. 2)
  optimizer.py   # chain + threshold search under a budget (Eq. 1)
  baselines.py   # best single LLM, log-prob threshold baseline
scripts/         # download_data, collect_responses, train_scorers, run_tradeoff
tests/
configs/
```
