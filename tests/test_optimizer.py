import itertools

import numpy as np
import pytest

from frugalgpt.cascade import Cascade, simulate
from frugalgpt.optimizer import (
    PENALTY,
    ChainData,
    _grid_objective,
    _objective,
    candidate_chains,
    evaluate,
    optimize,
    over_budget,
    quantiles_to_thresholds,
    search_thresholds,
)


def _random_tables(n: int = 200, k: int = 4, seed: int = 2024):
    rng = np.random.default_rng(seed)
    correct = rng.random((n, k)) < np.linspace(0.4, 0.9, k)
    cost = np.tile(np.geomspace(1.0, 100.0, k), (n, 1)) * rng.uniform(0.9, 1.1, (n, k))
    # Informative but noisy scores: higher when correct.
    score = np.clip(0.5 * correct + rng.normal(0.25, 0.2, (n, k)), 0.0, 1.0)
    return correct, cost, score


def _perfect_scorer_tables():
    """Cheap LLM right on half the queries, expensive LLM always right, perfect scorer."""
    n = 10
    correct = np.zeros((n, 2), dtype=bool)
    correct[:5, 0] = True
    correct[:, 1] = True
    cost = np.tile([1.0, 10.0], (n, 1))
    score = np.where(correct, 0.9, 0.1)
    score[:, 0] += np.linspace(0, 0.01, n)  # break ties so quantiles are clean
    return correct, cost, score


def test_quantiles_to_thresholds_by_hand() -> None:
    data = ChainData.build(
        (0, 1, 2),
        np.ones((5, 3), bool),
        np.ones((5, 3)),
        np.array([[0.1, 0.5, 0], [0.2, 0.4, 0], [0.3, 0.3, 0], [0.4, 0.2, 0], [0.5, 0.1, 0]]),
    )
    # q0 = 0.5 → τ0 = median(0.1..0.5) = 0.3; queries with score ≤ 0.3 reach step 1 (rows 0-2).
    # q1 = 0.5 → τ1 = median of step-1 scores of rows 0-2 = median(0.5, 0.4, 0.3) = 0.4.
    np.testing.assert_allclose(quantiles_to_thresholds(data, np.array([0.5, 0.5])), [0.3, 0.4])
    # Quantiles are clipped to [0, 1].
    np.testing.assert_allclose(quantiles_to_thresholds(data, np.array([-1.0, 2.0]))[0], 0.1)


def test_evaluate_matches_simulate() -> None:
    correct, cost, score = _random_tables()
    data = ChainData.build((2, 0, 3), correct, cost, score)
    thresholds = quantiles_to_thresholds(data, np.array([0.3, 0.6]))
    n_correct, total = evaluate(data, thresholds)
    r = simulate(Cascade((2, 0, 3), tuple(thresholds)), correct, cost, score)
    assert n_correct == r.correct.sum()
    assert total == pytest.approx(r.total_cost)


@pytest.mark.parametrize("length", [2, 3, 4])
def test_grid_objective_matches_pointwise(length: int) -> None:
    correct, cost, score = _random_tables(n=120, k=5)
    data = ChainData.build(tuple(range(length)), correct, cost, score)
    grid = np.linspace(1e-5, 1 - 1e-5, 7)
    budget = float(np.median(data.cumcost[:, -1]) * 0.6)
    fast = _grid_objective(data, grid, budget)
    for idx in itertools.product(range(len(grid)), repeat=length - 1):
        slow = _objective(data, grid[list(idx)], budget)
        assert fast[idx] == pytest.approx(slow), idx


def test_perfect_scorer_finds_known_optimum() -> None:
    correct, cost, score = _perfect_scorer_tables()
    # Optimum: cheap first, send only its 5 wrong answers on. Accuracy 1, mean cost 1 + 5·10/10 = 6.
    res = optimize(correct, cost, score, budget=6.0, length=2)
    assert res is not None
    assert res.best.cascade.order == (0, 1)
    assert res.best.accuracy == 1.0
    assert res.best.mean_cost == pytest.approx(6.0)
    # Check against simulate on the returned cascade.
    r = simulate(res.best.cascade, correct, cost, score)
    assert r.accuracy == 1.0 and r.mean_cost == pytest.approx(6.0)


def test_budget_too_small_returns_none() -> None:
    correct, cost, score = _perfect_scorer_tables()
    assert optimize(correct, cost, score, budget=0.5, length=2) is None


def test_huge_budget_reaches_best_possible() -> None:
    correct, cost, score = _random_tables()
    res = optimize(correct, cost, score, budget=1e9, length=2)
    assert res is not None and res.n_feasible == res.n_chains == 12
    # With no budget limit, accuracy is at least that of the best single LLM.
    assert res.best.accuracy >= correct.mean(axis=0).max()


def test_respects_budget_and_beats_brute_force_grid() -> None:
    correct, cost, score = _random_tables(n=150, k=4)
    budget = 20.0
    res = optimize(correct, cost, score, budget=budget, length=3)
    assert res is not None
    assert res.best.mean_cost <= budget + 1e-12
    r = simulate(res.best.cascade, correct, cost, score)
    assert r.accuracy == pytest.approx(res.best.accuracy)
    # Plain brute force over a fine threshold grid (not quantiles) on every chain.
    best = 0.0
    taus = np.linspace(0, 1, 21)
    for order in itertools.permutations(range(4), 3):
        data = ChainData.build(order, correct, cost, score)
        for t in itertools.product(taus, repeat=2):
            n_correct, total = evaluate(data, np.array(t))
            if total <= budget * len(correct):
                best = max(best, n_correct / len(correct))
    # The quantile search should be close to (usually at least as good as) a coarse grid.
    assert res.best.accuracy >= best - 0.02


def test_single_llm_chains() -> None:
    correct, cost, score = _random_tables()
    res = optimize(correct, cost, score, budget=1e9, length=1)
    assert res is not None
    assert res.best.cascade.order == (int(correct.mean(axis=0).argmax()),)
    assert res.best.cascade.thresholds == ()


def test_first_llm_too_expensive_is_skipped() -> None:
    correct, cost, score = _perfect_scorer_tables()
    data = ChainData.build((1, 0), correct, cost, score)
    assert search_thresholds(data, budget=9.0) is None


def test_decreasing_quantiles_rejected() -> None:
    correct, cost, score = _random_tables()
    data = ChainData.build((0, 1, 2), correct, cost, score)
    assert _objective(data, np.array([0.6, 0.4]), 1e9) == PENALTY
    assert _objective(data, np.array([0.4, 0.6]), 1e9) < 0


def test_tie_keeps_first_chain() -> None:
    # Two identical LLMs: every chain scores the same → first chain in permutation order wins.
    correct = np.array([[True, True], [False, False]])
    cost = np.ones((2, 2))
    score = np.full((2, 2), 0.5)
    res = optimize(correct, cost, score, budget=10.0, length=2)
    assert res is not None and res.best.cascade.order == (0, 1)


def test_mpi_pruned_matches_exhaustive_when_m_is_large() -> None:
    correct, cost, score = _random_tables(k=4)
    a = optimize(correct, cost, score, budget=20.0, length=2)
    b = optimize(correct, cost, score, budget=20.0, length=2, chain_search="mpi_pruned", m=3)
    assert a is not None and b is not None
    assert b.n_chains == 12
    assert a.best.accuracy == b.best.accuracy


def test_candidate_chains_and_bad_inputs() -> None:
    correct, cost, score = _random_tables(k=4)
    assert len(list(candidate_chains(correct, 3))) == 24
    assert len(list(candidate_chains(correct, 3, "mpi_pruned", m=1))) == 4
    with pytest.raises(ValueError):
        list(candidate_chains(correct, 3, "nope"))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        optimize(correct, cost, score, budget=1.0, length=5)
    with pytest.raises(ValueError):
        optimize(correct[:0], cost[:0], score[:0], budget=1.0)
    bad = score.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        optimize(correct, cost, bad, budget=1.0)


def _official_objective(data: ChainData, quantiles: np.ndarray, budget: float) -> float:
    """Our own restatement of the official ``g``/``f`` on distances d = 1 − score."""
    if np.any(np.diff(quantiles) < 0):
        return PENALTY
    d = 1.0 - data.score
    n, length = d.shape
    delta = np.ones(length)
    reach = np.ones(n, dtype=bool)
    for t, q in enumerate(quantiles):
        delta[t] = np.quantile(d[reach, t], 1 - min(max(q, 0.0), 1.0))
        reach &= d[:, t] >= delta[t]
    acc = cost = 0.0
    for i in range(n):
        for j in range(length):
            if d[i, j] < delta[j]:
                acc += data.correct[i, j]
                cost += data.cumcost[i, j]
                break
    return PENALTY if cost > budget * n else -acc


@pytest.mark.parametrize("decimals", [1, 2])
def test_matches_official_objective_with_tied_scores(decimals: int) -> None:
    correct, cost, score = _random_tables(n=60, k=4, seed=7)
    score = np.round(score, decimals)  # many ties
    grid = np.linspace(1e-5, 1 - 1e-5, 9)
    for order in [(0, 1, 2), (3, 1, 0)]:
        data = ChainData.build(order, correct, cost, score)
        budget = float(np.median(data.cumcost[:, -1]) * 0.7)
        fast = _grid_objective(data, grid, budget)
        for idx in itertools.product(range(len(grid)), repeat=2):
            q = grid[list(idx)]
            official = _official_objective(data, q, budget)
            assert _objective(data, q, budget) == official, (order, idx)
            assert fast[idx] == official, (order, idx)


def test_budget_exactly_at_achieved_cost_is_feasible() -> None:
    correct, cost, score = _random_tables(n=23, k=3, seed=3)
    cost = np.round(cost, 1)
    data = ChainData.build((0, 1, 2), correct, cost, score)
    grid = np.linspace(1e-5, 1 - 1e-5, 40)
    for q in [(0.05, 0.9), (0.3, 0.3), (0.5, 0.7)]:
        n_correct, total = evaluate(data, quantiles_to_thresholds(data, np.array(q)))
        budget = total / data.n
        assert _objective(data, np.array(q), budget) == -n_correct
    # Grid and point-wise agree at a budget that is exactly one grid point's cost.
    g = _grid_objective(data, grid, 1e9)
    i = np.unravel_index(np.argmin(g), g.shape)
    _, total = evaluate(data, quantiles_to_thresholds(data, grid[list(i)]))
    exact = _grid_objective(data, grid, total / data.n)
    assert exact[i] == _objective(data, grid[list(i)], total / data.n) < 0
    assert not over_budget(total, total / data.n, data.n)


def test_budget_equal_to_first_llm_cost() -> None:
    # The lowest-scoring query always goes on, so a 2-LLM chain costs more than LLM 0 alone.
    correct, cost, score = _perfect_scorer_tables()
    assert optimize(correct, cost, score, budget=1.0, length=2) is None
    res = optimize(correct, cost, score, budget=1.0, length=1)
    assert res is not None and res.best.cascade.order == (0,)


def test_all_scores_equal_sends_everyone_on() -> None:
    correct, cost, _ = _random_tables(n=30, k=3)
    data = ChainData.build((0, 1), correct, cost, np.full((30, 3), 0.5))
    tau = quantiles_to_thresholds(data, np.array([0.3]))
    assert tau[0] == 0.5
    _, total = evaluate(data, tau)
    assert total == pytest.approx(data.cumcost[:, 1].sum())


def test_single_query() -> None:
    correct = np.array([[False, True]])
    cost = np.array([[1.0, 5.0]])
    score = np.array([[0.2, 0.9]])
    res = optimize(correct, cost, score, budget=6.0, length=2)
    assert res is not None and res.best.accuracy == 1.0


@pytest.mark.parametrize("budget", [np.nan, np.inf, -1.0])
def test_invalid_budget(budget: float) -> None:
    correct, cost, score = _random_tables()
    with pytest.raises(ValueError):
        optimize(correct, cost, score, budget=budget, length=2)
