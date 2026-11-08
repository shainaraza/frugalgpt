from pathlib import Path

import numpy as np
import pytest

from frugalgpt.data import SHORT_NAMES, load_tables
from frugalgpt.pricing import (
    COUNTING_2023,
    PRICES_2023,
    PRICES_2024,
    Counting,
    Price,
    compute_cost,
    count_io,
    count_units,
    per_million,
    recompute_costs,
)

TABLES_DIR = Path(__file__).parents[1] / "data" / "tables"


def test_per_million() -> None:
    p = per_million(30.0, 60.0, 0.005)
    assert p == Price(input=30e-6, output=60e-6, request=0.005)


def test_compute_cost_by_hand() -> None:
    # GPT-4: 100 in × $30/1M + 2 out × $60/1M = 0.003 + 0.00012
    assert compute_cost(PRICES_2023["openaichat/gpt-4"], 100, 2) == pytest.approx(0.00312)
    # J1-Large: input free, $0.0003 per request + 2 out × $30/1M
    assert compute_cost(PRICES_2023["ai21/j1-large"], 500, 2) == pytest.approx(0.00036)


def test_compute_cost_zero_tokens_is_request_fee() -> None:
    assert compute_cost(Price(1.0, 1.0, request=0.5), 0, 0) == 0.5
    assert compute_cost(Price(1.0, 1.0), 0, 0) == 0.0


def test_compute_cost_returns_float_for_scalars() -> None:
    assert isinstance(compute_cost(Price(1.0, 2.0), 1, 1), float)


def test_compute_cost_vectorized() -> None:
    out = compute_cost(Price(1.0, 10.0, 100.0), np.array([0, 1, 2]), np.array([3, 0, 1]))
    np.testing.assert_allclose(out, [130.0, 101.0, 112.0])


def test_compute_cost_rejects_negative_counts() -> None:
    with pytest.raises(ValueError):
        compute_cost(Price(1.0, 1.0), -1, 0)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_compute_cost_rejects_non_finite_counts(bad: float) -> None:
    with pytest.raises(ValueError):
        compute_cost(Price(1.0, 1.0), np.array([1.0, bad]), 0)


def test_price_tables_cover_paper_models() -> None:
    assert set(PRICES_2023) == set(SHORT_NAMES)  # the 12 LLMs in the release CSVs
    assert set(COUNTING_2023) == set(PRICES_2023)
    assert len(PRICES_2024) == 10  # Table 3
    # Table 1 headline numbers: GPT-J $0.2 / GPT-4 $30 per 1M input tokens.
    assert PRICES_2023["textsynth/gptj_6B"].input * 1e6 == pytest.approx(0.2)
    assert PRICES_2023["openaichat/gpt-4"].input * 1e6 == pytest.approx(30.0)
    # Intro: GPT-4 Turbo is 200× Gemini Flash 8B on output.
    ratio = (
        PRICES_2024["openaichat/gpt-4-turbo"].output
        / PRICES_2024["google/gemini-1.5-flash-8b"].output
    )
    assert ratio == pytest.approx(200.0)


def test_count_units_characters() -> None:
    np.testing.assert_array_equal(count_units(["abc", "", "a b"], None), [3, 0, 3])


def test_count_units_tokens() -> None:
    # " up" is a single GPT-2 token; "" is zero tokens.
    np.testing.assert_array_equal(count_units([" up", ""], "r50k_base"), [1, 0])


def test_count_units_ignores_special_token_text() -> None:
    # Text that looks like a special token must be counted as plain text, not raise.
    assert count_units(["<|endoftext|>"], "r50k_base")[0] > 1


def test_count_io_extras_and_fixed_output() -> None:
    n_in, n_out = count_io(["abcd"], ["xyz"], Counting(None, input_extra=7, output_extra=1))
    assert (n_in[0], n_out[0]) == (11, 4)
    n_in, n_out = count_io(["abcd", "ab"], ["long answer", ""], Counting(None, output_fixed=2))
    np.testing.assert_array_equal(n_out, [2, 2])


def test_count_io_caps_output_at_max_tokens() -> None:
    # A 6-char answer + stop token would be 7, but an answer cut off at max_tokens has no stop.
    _, n_out = count_io(["q", "q"], ["abcdef", "ab"], Counting(None, output_extra=1, max_output=5))
    np.testing.assert_array_equal(n_out, [5, 3])


def test_count_io_blank_line_adjustment() -> None:
    n_in, _ = count_io(["a\n\nb\n\nc"], [""], Counting(None, input_per_blank_line=-1))
    assert n_in[0] == 7 - 2


def test_count_io_length_mismatch() -> None:
    with pytest.raises(ValueError):
        count_io(["a"], [], Counting(None))


def test_recompute_costs_by_hand() -> None:
    prices = {"a/char": Price(1.0, 10.0, 100.0, unit="char")}
    counting = {"a/char": Counting(None)}
    cost = recompute_costs(
        np.array(["ab", "abcd"]), np.array([["x"], [""]]), ["a/char"], prices, counting
    )
    np.testing.assert_allclose(cost, [[2 + 10 + 100], [4 + 0 + 100]])


def test_recompute_costs_shape_and_unknown_model() -> None:
    with pytest.raises(ValueError):
        recompute_costs(np.array(["q"]), np.array([["a", "b"]]), ["openaichat/gpt-4"])
    with pytest.raises(KeyError):
        recompute_costs(np.array(["q"]), np.array([["a"]]), ["nope/model"])


def test_recompute_costs_rejects_unit_mismatch() -> None:
    # A per-character price with token counting would be ~4x off.
    with pytest.raises(ValueError):
        recompute_costs(
            np.array(["q"]),
            np.array([["a"]]),
            ["x/y"],
            {"x/y": Price(1.0, 1.0, unit="char")},
            {"x/y": Counting("r50k_base")},
        )


def test_cohere_priced_per_character() -> None:
    assert PRICES_2023["cohere/xlarge"].unit == "char"
    assert PRICES_2023["openaichat/gpt-4"].unit == "token"


# Exact for models whose tokenizer we have; approximate (GPT-2 BPE stand-in) for the others.
EXACT = {"openai/text-curie-001", "openai/text-davinci-002", "openaichat/gpt-3.5-turbo",
         "openaichat/gpt-4", "textsynth/gptj_6B", "textsynth/fairseq_gpt_13B",
         "cohere/xlarge", "cohere/medium"}  # fmt: skip


@pytest.mark.parametrize("split", ["train", "test"])
def test_recomputed_costs_match_cache(split: str) -> None:
    path = TABLES_DIR / f"HEADLINES_{split}.npz"
    if not path.exists():
        pytest.skip("run scripts/prepare_data.py first")
    t = load_tables(path)
    ours = recompute_costs(t.queries, t.answers, t.llms)
    for k, model in enumerate(t.llms):
        if model in EXACT:
            np.testing.assert_allclose(ours[:, k], t.cost[:, k], rtol=1e-9, err_msg=model)
        elif model.startswith("ai21/"):  # GPT-2 tokens + stop: exact on ≥ 99.9% of queries
            assert np.isclose(ours[:, k], t.cost[:, k]).mean() >= 0.999, model
        else:  # GPT-Neo: own tokenizer; total within 1%
            assert abs(ours[:, k].sum() / t.cost[:, k].sum() - 1) < 0.01, model
