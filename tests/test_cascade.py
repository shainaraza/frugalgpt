import numpy as np
import pytest

from frugalgpt.cascade import Cascade, simulate, stop_index

# 4 queries × 3 LLMs. Column 0 = cheap, 1 = medium, 2 = expensive.
CORRECT = np.array(
    [
        [True, False, True],
        [False, True, True],
        [False, False, True],
        [True, True, False],
    ]
)
COST = np.array(
    [
        [1.0, 10.0, 100.0],
        [1.0, 10.0, 100.0],
        [2.0, 20.0, 200.0],
        [1.0, 10.0, 100.0],
    ]
)
SCORE = np.array(
    [
        [0.9, 0.1, 0.5],  # cheap is confident → stop at 0
        [0.2, 0.8, 0.5],  # medium is confident → stop at 1
        [0.1, 0.1, 0.5],  # nobody confident → last
        [0.5, 0.3, 0.9],  # cheap score == τ_0 → reject; medium 0.3 < 0.6 → last
    ]
)


def test_simulate_by_hand() -> None:
    r = simulate(Cascade((0, 1, 2), (0.5, 0.6)), CORRECT, COST, SCORE)
    np.testing.assert_array_equal(r.stop, [0, 1, 2, 2])
    np.testing.assert_array_equal(r.llm, [0, 1, 2, 2])
    np.testing.assert_array_equal(r.correct, [True, True, True, False])
    # Cost is cumulative over every LLM called.
    np.testing.assert_allclose(r.cost, [1.0, 11.0, 222.0, 111.0])
    assert r.accuracy == 0.75
    assert r.mean_cost == pytest.approx(345.0 / 4)
    assert r.total_cost == pytest.approx(345.0)
    np.testing.assert_allclose(r.reach_share(), [1.0, 0.75, 0.5])


def test_order_other_than_column_order() -> None:
    # Chain 2 → 0: expensive first. Query 3 has score 0.9 > 0.85 at LLM 2 → stop.
    r = simulate(Cascade((2, 0), (0.85,)), CORRECT, COST, SCORE)
    np.testing.assert_array_equal(r.llm, [0, 0, 0, 2])
    np.testing.assert_allclose(r.cost, [101.0, 101.0, 202.0, 100.0])
    np.testing.assert_array_equal(r.correct, [True, False, False, False])


def test_single_llm_chain() -> None:
    r = simulate(Cascade((2,), ()), CORRECT, COST, SCORE)
    np.testing.assert_array_equal(r.llm, [2, 2, 2, 2])
    np.testing.assert_allclose(r.cost, COST[:, 2])
    assert r.accuracy == CORRECT[:, 2].mean()


def test_all_stop_at_first() -> None:
    # Threshold below every score → everyone accepts the first LLM.
    r = simulate(Cascade((0, 1, 2), (-1.0, -1.0)), CORRECT, COST, SCORE)
    np.testing.assert_array_equal(r.stop, [0, 0, 0, 0])
    np.testing.assert_allclose(r.cost, COST[:, 0])


def test_all_reach_last() -> None:
    # Threshold at the max score is never exceeded (strict >) → everyone reaches the last LLM.
    r = simulate(Cascade((0, 1, 2), (1.0, 1.0)), CORRECT, COST, SCORE)
    np.testing.assert_array_equal(r.stop, [2, 2, 2, 2])
    np.testing.assert_allclose(r.cost, COST.sum(axis=1))


def test_last_llm_always_accepted_even_with_zero_score() -> None:
    score = SCORE.copy()
    score[:, 2] = 0.0  # the official code (score > 0 at the last step) would drop these queries
    r = simulate(Cascade((0, 1, 2), (1.0, 1.0)), CORRECT, COST, score)
    np.testing.assert_array_equal(r.stop, [2, 2, 2, 2])
    np.testing.assert_array_equal(r.correct, CORRECT[:, 2])


def test_last_llm_score_may_be_nan() -> None:
    # The last LLM's score is never read.
    score = SCORE.copy()
    score[:, 2] = np.nan
    r = simulate(Cascade((0, 1, 2), (0.5, 0.6)), CORRECT, COST, score)
    np.testing.assert_array_equal(r.stop, [0, 1, 2, 2])


def test_stop_index_tie_is_rejected() -> None:
    np.testing.assert_array_equal(stop_index(np.array([[0.5, 0.0]]), (0.5,)), [1])
    np.testing.assert_array_equal(stop_index(np.array([[0.50001, 0.0]]), (0.5,)), [0])


def test_stop_index_threshold_count() -> None:
    with pytest.raises(ValueError):
        stop_index(np.zeros((2, 3)), (0.5,))


def test_empty_query_set_rejected() -> None:
    with pytest.raises(ValueError):
        simulate(
            Cascade((0, 1), (0.5,)), np.zeros((0, 2), bool), np.zeros((0, 2)), np.zeros((0, 2))
        )


def test_correct_as_0_1_ints() -> None:
    r = simulate(Cascade((0, 1, 2), (0.5, 0.6)), CORRECT.astype(int), COST, SCORE)
    np.testing.assert_array_equal(r.correct, [True, True, True, False])


@pytest.mark.parametrize("bad", [np.nan, 0.3, 2.0])
def test_correct_must_be_binary(bad: float) -> None:
    correct = CORRECT.astype(float)
    correct[0, 0] = bad
    with pytest.raises(ValueError):
        simulate(Cascade((0, 1, 2), (0.5, 0.6)), correct, COST, SCORE)


def _official_f(L_mat, C_mat, d_mat, delta):  # noqa: N803 — names as in optimizer.py
    """Inline restatement of the official objective: stop at first j with d < δ_j."""
    n, m = d_mat.shape
    acc = cost = 0.0
    for i in range(n):
        for j in range(m):
            if d_mat[i, j] < delta[j]:
                acc += L_mat[i, j]
                cost += C_mat[i, j]
                break
    return acc, cost


def test_matches_official_formulation_on_random_tables() -> None:
    rng = np.random.default_rng(2024)
    n, k = 300, 5
    correct = rng.random((n, k)) < 0.6
    cost = rng.random((n, k))
    score = rng.uniform(1e-6, 1.0, (n, k))  # official assumption: scores in (0, 1]
    for _ in range(20):
        order = tuple(int(x) for x in rng.choice(k, size=3, replace=False))
        taus = tuple(float(x) for x in rng.random(2))
        r = simulate(Cascade(order, taus), correct, cost, score)
        o = list(order)
        delta = [1 - t for t in taus] + [1.0]
        acc, tot = _official_f(correct[:, o], np.cumsum(cost[:, o], axis=1), 1 - score[:, o], delta)
        assert r.correct.sum() == acc
        assert r.total_cost == pytest.approx(tot)


@pytest.mark.parametrize(
    "order, thresholds",
    [
        ((), ()),  # empty chain
        ((0, 0), (0.5,)),  # repeated LLM
        ((-1, 0), (0.5,)),  # negative index
        ((0, 1, 2), (0.5,)),  # too few thresholds
        ((0, 1), (0.5, 0.5)),  # too many thresholds
        ((0, 1), (np.nan,)),  # NaN threshold
        ((0.0, 1.0), (0.5,)),  # float indices
        ((True, False), (0.5,)),  # bool indices
    ],
)
def test_invalid_cascade(order: tuple, thresholds: tuple) -> None:
    with pytest.raises(ValueError):
        Cascade(order, thresholds)


def test_invalid_tables() -> None:
    c = Cascade((0, 1), (0.5,))
    with pytest.raises(ValueError):  # shape mismatch
        simulate(c, CORRECT, COST[:, :2], SCORE)
    with pytest.raises(ValueError):  # order out of range
        simulate(Cascade((0, 3), (0.5,)), CORRECT, COST, SCORE)
    bad = SCORE.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError):  # NaN in a score that is used
        simulate(c, CORRECT, COST, bad)
    neg = COST.copy()
    neg[0, 1] = -1.0
    with pytest.raises(ValueError):  # negative cost
        simulate(c, CORRECT, neg, SCORE)
