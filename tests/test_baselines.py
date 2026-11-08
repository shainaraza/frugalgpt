from pathlib import Path

import numpy as np
import pytest

from frugalgpt.baselines import (
    best_single_llm,
    cheapest_at_accuracy,
    finite_scores,
    mean_token_logprob,
    single_llm_points,
    threshold_curve,
)
from frugalgpt.data import load_tables

# 4 queries × 3 LLMs: cheap (60%-ish), mid, expensive.
CORRECT = np.array(
    [
        [True, False, True],
        [False, True, True],
        [False, False, True],
        [True, True, False],
    ]
)
COST = np.tile([1.0, 10.0, 100.0], (4, 1))

TABLE_DIR = Path(__file__).parents[1] / "data" / "tables"


def test_single_llm_points_by_hand() -> None:
    acc, cost = single_llm_points(CORRECT, COST)
    np.testing.assert_allclose(acc, [0.5, 0.5, 0.75])
    np.testing.assert_allclose(cost, [1.0, 10.0, 100.0])


def test_best_single_llm() -> None:
    assert best_single_llm(CORRECT, COST) == 2  # most accurate overall
    assert best_single_llm(CORRECT, COST, budget=50.0) == 0  # tie 0/1 at 0.5 → cheaper
    assert best_single_llm(CORRECT, COST, budget=100.0) == 2  # budget exactly at cost is OK
    assert best_single_llm(CORRECT, COST, budget=0.5) is None


def test_best_single_llm_tie_same_cost_lower_index() -> None:
    assert best_single_llm(CORRECT[:, :2], np.ones((4, 2))) == 0
    # three-way accuracy tie (cols 0, 1, and a copy of 0) at equal cost → lowest index
    c = np.column_stack([CORRECT[:, 1], CORRECT[:, 0], CORRECT[:, 0]])
    assert best_single_llm(c, np.ones((4, 3))) == 0


def test_best_single_llm_float_boundary() -> None:
    # mean([0.1, 0.2, 0.3]) = 0.20000000000000004: same tolerance as the optimizer.
    cost = np.array([[0.1], [0.2], [0.3]])
    assert best_single_llm(np.ones((3, 1), dtype=bool), cost, budget=0.2) == 0


def test_best_single_llm_int_table_and_zero_budget() -> None:
    correct = CORRECT.astype(int)
    cost = COST.copy()
    cost[:, 1] = 0.0
    assert best_single_llm(correct, cost, budget=0.0) == 1


@pytest.mark.parametrize("budget", [np.nan, np.inf, -1.0])
def test_best_single_llm_invalid_budget(budget: float) -> None:
    with pytest.raises(ValueError):
        best_single_llm(CORRECT, COST, budget=budget)


def test_mean_token_logprob() -> None:
    assert mean_token_logprob([-0.5, -1.5]) == -1.0
    assert mean_token_logprob([]) == float("-inf")
    with pytest.raises(ValueError):
        mean_token_logprob([None, -1.0])
    with pytest.raises(ValueError):
        mean_token_logprob([np.nan, -1.0])


def test_finite_scores() -> None:
    s = finite_scores(np.array([-0.2, -np.inf, -1.0]))
    np.testing.assert_allclose(s, [-0.2, -2.0, -1.0])
    np.testing.assert_allclose(finite_scores(np.array([-np.inf, -np.inf])), [-1.0, -1.0])
    np.testing.assert_allclose(finite_scores(np.array([0.3])), [0.3])
    with pytest.raises(ValueError):
        finite_scores(np.array([0.1, np.nan]))
    with pytest.raises(ValueError):
        finite_scores(np.array([np.inf]))
    with pytest.raises(ValueError):  # floor must stay below real scores
        finite_scores(np.array([-1.0, -np.inf]), floor=-0.5)


def test_threshold_curve_by_hand() -> None:
    # Cheap LLM 0's scores: high on the queries it gets right (q0, q3).
    score = np.array([-0.1, -2.0, -1.5, -0.3])
    curve = threshold_curve(score, CORRECT, COST, cheap=0, expensive=2, thresholds=[-3, -1, 0])
    # τ = −3: everyone keeps the cheap answer → acc 0.5, cost 1.
    # τ = −1: q0, q3 keep cheap (both right); q1, q2 → expensive (both right) → acc 1.
    #         cost = (1 + 101 + 101 + 1) / 4 = 51.
    # τ = 0: everyone escalates → acc 0.75 (LLM 2), cost 101.
    np.testing.assert_allclose(curve.accuracy, [0.5, 1.0, 0.75])
    np.testing.assert_allclose(curve.mean_cost, [1.0, 51.0, 101.0])
    np.testing.assert_allclose(curve.escalated, [0.0, 0.5, 1.0])


def test_threshold_curve_neg_inf_never_accepted() -> None:
    score = np.array([-np.inf, -0.1, -0.1, -0.1])
    # Even a threshold far below every real score must send the -inf query on.
    curve = threshold_curve(score, CORRECT, COST, cheap=0, expensive=2, thresholds=[-1e9, -1.05])
    np.testing.assert_allclose(curve.escalated, [0.25, 0.25])
    # Decision for the -inf query must not depend on the other queries' scores.
    other = np.array([-np.inf, -10.0, -0.1, -0.1])
    curve = threshold_curve(other, CORRECT, COST, cheap=0, expensive=2, thresholds=[-20.0])
    assert curve.escalated[0] == 0.25


def test_threshold_curve_all_neg_inf() -> None:
    score = np.full(4, -np.inf)
    curve = threshold_curve(score, CORRECT, COST, cheap=0, expensive=2, thresholds=[-1e9])
    assert curve.escalated[0] == 1.0


def test_threshold_curve_tie_escalates() -> None:
    # score == τ is not "larger than", so the query escalates.
    score = np.array([-1.0, -1.0, -1.0, -1.0])
    curve = threshold_curve(score, CORRECT, COST, cheap=0, expensive=2, thresholds=[-1.0])
    assert curve.escalated[0] == 1.0


def test_threshold_curve_total_cost() -> None:
    score = np.array([-0.1, -2.0, -1.5, -0.3])
    curve = threshold_curve(score, CORRECT, COST, cheap=0, expensive=2, thresholds=[-1])
    assert curve.n_queries == 4
    np.testing.assert_allclose(curve.total_cost, [204.0])


def test_threshold_curve_invalid() -> None:
    s = np.zeros(4)
    with pytest.raises(ValueError):
        threshold_curve(s, CORRECT, COST, cheap=0, expensive=0, thresholds=[0.0])
    with pytest.raises(ValueError):
        threshold_curve(np.zeros(3), CORRECT, COST, cheap=0, expensive=2, thresholds=[0.0])
    with pytest.raises(ValueError):
        threshold_curve(s, CORRECT, COST, cheap=0, expensive=2, thresholds=[])
    with pytest.raises(ValueError):
        threshold_curve(s, CORRECT, COST, cheap=0, expensive=2, thresholds=[np.nan])
    with pytest.raises(ValueError):  # out of range: ValueError, not IndexError
        threshold_curve(s, CORRECT, COST, cheap=5, expensive=0, thresholds=[0.0])
    with pytest.raises(ValueError):
        threshold_curve(s, CORRECT, COST, cheap=-1, expensive=0, thresholds=[0.0])


def test_cheapest_at_accuracy() -> None:
    acc = np.array([0.5, 1.0, 0.75])
    cost = np.array([1.0, 51.0, 101.0])
    assert cheapest_at_accuracy(acc, cost, 0.75) == (1, 51.0)
    assert cheapest_at_accuracy(acc, cost, 0.5) == (0, 1.0)
    assert cheapest_at_accuracy(acc, cost, 1.01) is None
    assert cheapest_at_accuracy(np.array([0.8, 0.9]), np.array([5.0, 5.0]), 0.8) == (0, 5.0)
    with pytest.raises(ValueError):
        cheapest_at_accuracy(np.array([np.nan, 0.5]), np.array([1.0, 2.0]), 0.5)


@pytest.mark.parametrize("split", ["train", "test"])
def test_headlines_best_single_llm_is_gpt4(split: str) -> None:
    path = TABLE_DIR / f"HEADLINES_{split}.npz"
    if not path.exists():
        pytest.skip("run scripts/prepare_data.py first")
    t = load_tables(path)
    best = best_single_llm(t.correct, t.cost)
    assert t.llms[best] == "openaichat/gpt-4"
    if split == "test":  # Table 2: GPT-4 costs $33.1 over the test set
        assert abs(t.cost[:, best].sum() - 33.1) / 33.1 < 0.01
