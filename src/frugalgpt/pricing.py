"""Cost model ``c_i(p)`` and the LLM price tables (Sec. 3; Tables 1 and 3).

Sec. 3 defines the cost of calling LLM ``i`` on prompt ``p`` as

    c_i(p) = c̃_{i,2} · ‖f_i(p)‖  +  c̃_{i,1} · ‖p‖  +  c̃_{i,0}
             (output tokens)        (input tokens)    (fixed per request)

Prices here are stored **per token** (or per character, see CoHere below) in USD, as in the
official ``config/serviceinfo.json``. The paper's tables quote them per 1M tokens.

Token counting (``COUNTING_2023``) reproduces how the cached ``cost`` column of the release CSVs
was computed, so we can check this module against it (``scripts/check_pricing.py``). The official
code read token counts from each API's response; we do not have those, so we re-tokenize the
cached prompt and answer with ``tiktoken``. That is exact for OpenAI and GPT-J, and an
approximation for models whose own tokenizer is not GPT-2's (GPT-Neo; AI21 J1 on a few queries).

These recomputed costs are only a check on the cost model. Later steps (cascade, optimizer) use the
cached per-call ``cost`` table, which is what produced the paper's numbers.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class Price:
    """Sec. 3 per-unit prices in USD: ``c̃_1`` input, ``c̃_2`` output, ``c̃_0`` per request.

    ``unit`` says what ``input`` / ``output`` are priced per: a token, or (CoHere) a character.
    """

    input: float
    output: float
    request: float = 0.0
    unit: Literal["token", "char"] = "token"


def per_million(input_usd: float, output_usd: float, request_usd: float = 0.0) -> Price:
    """Build a :class:`Price` from the per-1M-token prices quoted in Tables 1 and 3."""
    return Price(input=input_usd / 1e6, output=output_usd / 1e6, request=request_usd)


def compute_cost(
    price: Price, n_input: float | np.ndarray, n_output: float | np.ndarray
) -> float | np.ndarray:
    """Sec. 3 cost model: ``c̃_2 · n_output + c̃_1 · n_input + c̃_0`` (USD) for single calls.

    Works on scalars or numpy arrays (broadcast elementwise). Counts are in ``price.unit`` and must
    be finite and non-negative. The official code also has a ``fixed_size`` of free output
    tokens; it is 0 for every model, so it is left out.
    """
    n_input = np.asarray(n_input, dtype=np.float64)
    n_output = np.asarray(n_output, dtype=np.float64)
    if not (np.isfinite(n_input).all() and np.isfinite(n_output).all()):
        raise ValueError("token counts must be finite")
    if (n_input < 0).any() or (n_output < 0).any():
        raise ValueError("token counts must be non-negative")
    cost = price.output * n_output + price.input * n_input + price.request
    return float(cost) if cost.ndim == 0 else cost


# Table 1 (prices from March 2023), keyed by the model ids used in the release CSVs.
# CoHere is billed per *character* in the official code: $0.0025 per 1000 characters, which the
# paper reports as roughly $10 per 1M tokens.
PRICES_2023: dict[str, Price] = {
    "openai/text-curie-001": per_million(2.0, 2.0),
    "openaichat/gpt-3.5-turbo": per_million(2.0, 2.0),
    "openai/text-davinci-002": per_million(20.0, 20.0),
    "openaichat/gpt-4": per_million(30.0, 60.0),
    "ai21/j1-large": per_million(0.0, 30.0, 0.0003),
    "ai21/j1-grande": per_million(0.0, 80.0, 0.0008),
    "ai21/j1-jumbo": per_million(0.0, 250.0, 0.005),
    "cohere/xlarge": Price(input=2.5e-6, output=2.5e-6, unit="char"),
    "cohere/medium": Price(input=2.5e-6, output=2.5e-6, unit="char"),
    "textsynth/gptj_6B": per_million(0.2, 5.0),
    "textsynth/fairseq_gpt_13B": per_million(0.6, 15.0),
    "textsynth/gptneox_20B": per_million(1.4, 35.0),
}

# Table 3 (prices from November 2024). Model versions as named in the Table 3 caption.
PRICES_2024: dict[str, Price] = {
    "ai21/jamba-1.5-large": per_million(2.0, 8.0),
    "anthropic/claude-3-5-sonnet-20240620": per_million(3.0, 15.0),
    "google/gemini-1.5-pro": per_million(1.25, 5.0),
    "google/gemini-1.5-flash": per_million(0.075, 0.30),
    "google/gemini-1.5-flash-8b": per_million(0.0375, 0.15),
    "openaichat/gpt-4o-2024-05-13": per_million(5.0, 15.0),
    "openaichat/gpt-4-turbo": per_million(10.0, 30.0),
    "openaichat/gpt-4o-mini": per_million(0.15, 0.60),
    "togetherai/meta-llama/Meta-Llama-3-70B-Instruct-Turbo": per_million(0.88, 0.88),
    "togetherai/google/gemma-2-9b-it": per_million(0.30, 0.30),
}


@dataclass(frozen=True)
class Counting:
    """How a provider counts the units it bills for one call.

    ``encoding`` is a ``tiktoken`` encoding name, or ``None`` to count characters.
    ``input_extra`` / ``output_extra`` are tokens billed on top of the text itself (chat-format
    overhead, stop token). ``input_per_blank_line`` is added for each ``"\n\n"`` in the prompt.
    ``max_output`` caps the output count (the request's ``max_tokens``: an answer cut off at the
    limit has no stop token). ``output_fixed`` replaces the output count by a constant.
    """

    encoding: str | None
    input_extra: int = 0
    input_per_blank_line: int = 0
    output_extra: int = 0
    max_output: int | None = None
    output_fixed: int | None = None


# ``max_tokens`` of every cached request (release CSV column; docs/paper_vs_code.md).
MAX_TOKENS = 50

# Fitted to the cached ``cost`` column of HEADLINES (docs/paper_vs_code.md, "Cost recomputation"):
# - chat models: OpenAI's chat format adds 7 prompt tokens for one user message;
#   gpt-3.5-turbo's cached answers are one token shorter than what was billed.
# - TextSynth and AI21 bill the stop token ("\n"), which is stripped from the cached answer.
# - FAIRSEQ's tokenizer turns each blank line "\n\n" into one token fewer than GPT-2's.
# - Billed output never exceeds max_tokens.
# - CoHere: the official code bills len(prompt) characters plus len(completion_dict) == 2
#   "characters" of output, whatever the answer is.
_M = MAX_TOKENS
COUNTING_2023: dict[str, Counting] = {
    "openai/text-curie-001": Counting("r50k_base", max_output=_M),
    "openai/text-davinci-002": Counting("p50k_base", max_output=_M),
    "openaichat/gpt-3.5-turbo": Counting(
        "cl100k_base", input_extra=7, output_extra=1, max_output=_M
    ),
    "openaichat/gpt-4": Counting("cl100k_base", input_extra=7, max_output=_M),
    # AI21's tokenizer differs from GPT-2's: exact on ~99.9% of queries.
    "ai21/j1-large": Counting("r50k_base", output_extra=1, max_output=_M),
    "ai21/j1-grande": Counting("r50k_base", output_extra=1, max_output=_M),
    "ai21/j1-jumbo": Counting("r50k_base", output_extra=1, max_output=_M),
    "cohere/xlarge": Counting(None, output_fixed=2),
    "cohere/medium": Counting(None, output_fixed=2),
    "textsynth/gptj_6B": Counting("r50k_base", output_extra=1, max_output=_M),
    "textsynth/fairseq_gpt_13B": Counting(
        "r50k_base", input_per_blank_line=-1, output_extra=1, max_output=_M
    ),
    # GPT-NeoX has its own BPE: approximation (total within ~0.4%).
    "textsynth/gptneox_20B": Counting("r50k_base", output_extra=1, max_output=_M),
}


@cache
def _encoder(name: str):
    import tiktoken  # dev dependency; only needed to re-tokenize cached text

    return tiktoken.get_encoding(name)


def count_units(texts: Sequence[str], encoding: str | None) -> np.ndarray:
    """Number of tokens (or characters if ``encoding`` is ``None``) in each text."""
    if encoding is None:
        return np.array([len(t) for t in texts], dtype=np.int64)
    enc = _encoder(encoding)
    return np.array([len(ids) for ids in enc.encode_ordinary_batch(list(texts))], dtype=np.int64)


def count_io(
    prompts: Sequence[str], answers: Sequence[str], counting: Counting
) -> tuple[np.ndarray, np.ndarray]:
    """Billed (input, output) unit counts for each (prompt, answer) pair."""
    if len(prompts) != len(answers):
        raise ValueError("prompts and answers must have the same length")
    n_in = count_units(prompts, counting.encoding) + counting.input_extra
    if counting.input_per_blank_line:
        blank = np.array([p.count("\n\n") for p in prompts], dtype=np.int64)
        n_in = n_in + counting.input_per_blank_line * blank
    if counting.output_fixed is not None:
        n_out = np.full(len(answers), counting.output_fixed, dtype=np.int64)
    else:
        n_out = count_units(answers, counting.encoding) + counting.output_extra
        if counting.max_output is not None:
            n_out = np.minimum(n_out, counting.max_output)
    return n_in, n_out


def recompute_costs(
    queries: np.ndarray,
    answers: np.ndarray,
    llms: Sequence[str],
    prices: dict[str, Price] = PRICES_2023,
    counting: dict[str, Counting] = COUNTING_2023,
) -> np.ndarray:
    """Sec. 3: ``c_k(p_n)`` for every query ``n`` (rows) and LLM ``k`` (columns), in USD.

    ``answers`` is the ``n_queries × n_llms`` table of cached answers. Each entry is the cost of
    one call, not cumulative: a cascade sums it over every LLM it calls (Sec. 4).
    """
    answers = np.asarray(answers)
    if answers.shape != (len(queries), len(llms)):
        raise ValueError(f"answers shape {answers.shape} != ({len(queries)}, {len(llms)})")
    missing = [m for m in llms if m not in prices or m not in counting]
    if missing:
        raise KeyError(f"no price or counting rule for {missing}")
    for m in llms:
        if (prices[m].unit == "char") != (counting[m].encoding is None):
            raise ValueError(f"{m}: price unit {prices[m].unit!r} does not match its counting rule")
    # Cost model: c = c̃_input · #input tokens + c̃_output · #output tokens + c̃_request.
    # Input (the prompt with few-shot examples) dominates for short answers, which is why
    # FrugalGPT's prompt-adaptation ideas target the prompt length.
    cost = np.empty(answers.shape, dtype=np.float64)
    for k, model in enumerate(llms):
        n_in, n_out = count_io(list(queries), list(answers[:, k]), counting[model])
        cost[:, k] = compute_cost(prices[model], n_in, n_out)
    return cost
