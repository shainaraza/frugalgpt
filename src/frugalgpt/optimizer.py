"""Budget-constrained search for the LLM chain and its thresholds (Sec. 4, Eq. 1).

Eq. 1: maximize E[r(q, â(s, q))] subject to E[c(s, q)] ≤ b, where b is the **average cost per
query** (USD). Defaults follow the official code (``llmchain.py: train``, ``optimizer.py``):

1. **Chain search.** Every ordered chain of ``length`` LLMs (``exhaustive``), or only the chains
   allowed by MPI pruning (``mpi_pruned``, Eq. 2). A chain is skipped if its first LLM's average
   cost alone is over the budget.
2. **Threshold search.** Thresholds are parameterized by quantiles: ``q_t`` is the share of the
   queries reaching step ``t`` whose score is at or below ``τ_t`` (they are passed on). That is
   ``τ_t = quantile(scores of the queries reaching step t, q_t)``. The official code
   rejects quantile vectors that are not non-decreasing. Search = a 40-point grid per axis on
   ``(1e-5, 1 − 1e-5)`` (as ``scipy.optimize.brute``), then Nelder-Mead (``scipy.optimize.fmin``)
   from the best grid point. Over-budget points get a flat penalty.
3. The chain with the highest training accuracy wins; on a tie the first one found is kept.

The official code works on distances ``d = 1 − score``. We work on scores directly: its
``(1 − q)``-quantile of ``d`` is our ``q``-quantile of the score, and its "``d ≥ δ`` continues"
is our "``score ≤ τ`` continues".
"""

import itertools
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

import numpy as np
import scipy.optimize

from frugalgpt.cascade import Cascade
from frugalgpt.mpi import mpi_matrix, pruned_chains

GRID_POINTS = 40  # official: scipy.optimize.brute(..., Ns=40)
Q_LOW, Q_HIGH = 1e-5, 1 - 1e-5  # official quantile range
PENALTY = 10000.0  # official objective value for infeasible / rejected points
BUDGET_RTOL = 1e-12  # float slack so a cost exactly at the budget counts as feasible


def over_budget(total_cost: float | np.ndarray, budget: float, n: int) -> bool | np.ndarray:
    """Eq. 1 constraint, violated: mean cost ``total_cost / n`` above ``budget``.

    One shared check with a tiny relative tolerance, so summing the same costs in a different
    order cannot flip a point that sits exactly on the budget.
    """
    return total_cost > budget * n * (1 + BUDGET_RTOL)


@dataclass(frozen=True)
class ChainData:
    """Per-chain columns, built once and reused for every threshold tried (Sec. 4).

    Column ``t`` is the ``t``-th LLM of ``order``. ``cumcost[:, t]`` is the cost of calling LLMs
    ``0..t`` of the chain (official ``C_mat``).
    """

    order: tuple[int, ...]
    correct: np.ndarray  # (n, L) float 0/1
    cumcost: np.ndarray  # (n, L) float
    score: np.ndarray  # (n, L) float

    @classmethod
    def build(
        cls, order: tuple[int, ...], correct: np.ndarray, cost: np.ndarray, score: np.ndarray
    ) -> "ChainData":
        """Slice the chain's columns out of full ``n_queries × n_llms`` tables."""
        o = list(order)
        return cls(
            order=tuple(order),
            correct=np.asarray(correct, dtype=np.float64)[:, o],
            cumcost=np.cumsum(np.asarray(cost, dtype=np.float64)[:, o], axis=1),
            score=np.asarray(score, dtype=np.float64)[:, o],
        )

    @property
    def n(self) -> int:
        """Number of queries."""
        return self.correct.shape[0]

    @property
    def length(self) -> int:
        """Chain length L."""
        return self.correct.shape[1]


def quantiles_to_thresholds(data: ChainData, quantiles: np.ndarray) -> np.ndarray:
    """Official ``quatile2thres_batch``, on scores: ``τ_t`` = ``q_t``-quantile among reachers.

    ``quantiles`` has ``L − 1`` entries; each is clipped to ``[0, 1]`` as in the official code.
    """
    thresholds = np.empty(data.length - 1)
    reach = np.ones(data.n, dtype=bool)
    for t, q in enumerate(quantiles):
        s = data.score[reach, t]
        thresholds[t] = np.quantile(s, min(max(float(q), 0.0), 1.0))
        reach &= data.score[:, t] <= thresholds[t]
    return thresholds


def evaluate(data: ChainData, thresholds: np.ndarray) -> tuple[float, float]:
    """Sec. 4: (number of correct answers, total cost) of the chain with these thresholds."""
    n, length = data.correct.shape
    accept = np.ones((n, length), dtype=bool)
    accept[:, :-1] = data.score[:, :-1] > np.asarray(thresholds)
    stop = accept.argmax(axis=1)
    rows = np.arange(n)
    return float(data.correct[rows, stop].sum()), float(data.cumcost[rows, stop].sum())


def _objective(data: ChainData, quantiles: np.ndarray, budget: float) -> float:
    """Official ``g``: −(#correct) if the quantiles are non-decreasing and within budget."""
    if np.any(np.diff(quantiles) < 0):
        return PENALTY
    n_correct, total_cost = evaluate(data, quantiles_to_thresholds(data, quantiles))
    if over_budget(total_cost, budget, data.n):
        return PENALTY
    return -n_correct


def _grid_objective(data: ChainData, grid: np.ndarray, budget: float) -> np.ndarray:
    """``_objective`` at every point of the product grid ``grid^(L-1)``, shape ``(G,)*(L-1)``.

    Same values as calling ``_objective`` point by point (up to float rounding of sums, absorbed
    by :func:`over_budget`), but fast. Prefixes are walked
    recursively, and for the last scored step all grid values are handled at once on the queries
    sorted by score.
    """
    length, g = data.length, len(grid)
    n_correct = np.zeros((g,) * (length - 1))
    total_cost = np.zeros((g,) * (length - 1))

    def last_step(t: int, reach: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        idx = np.flatnonzero(reach)
        s = data.score[idx, t]
        order = np.argsort(s, kind="stable")
        s_sorted = s[order]
        tau = np.quantile(s, grid)
        # Queries with score ≤ τ continue to the last LLM; the rest stop at step t.
        k = np.searchsorted(s_sorted, tau, side="right")
        stop_c = np.concatenate([[0.0], np.cumsum(data.correct[idx[order], t])])
        last_c = np.concatenate([[0.0], np.cumsum(data.correct[idx[order], length - 1])])
        stop_k = np.concatenate([[0.0], np.cumsum(data.cumcost[idx[order], t])])
        last_k = np.concatenate([[0.0], np.cumsum(data.cumcost[idx[order], length - 1])])
        acc = (stop_c[-1] - stop_c[k]) + last_c[k]
        cost = (stop_k[-1] - stop_k[k]) + last_k[k]
        return acc, cost

    def walk(t: int, reach: np.ndarray, prefix: tuple[int, ...], acc0: float, cost0: float):
        if t == length - 2:
            acc, cost = last_step(t, reach)
            n_correct[prefix] = acc0 + acc
            total_cost[prefix] = cost0 + cost
            return
        s = data.score[reach, t]
        for i, tau in enumerate(np.quantile(s, grid)):
            stops = reach & (data.score[:, t] > tau)
            walk(
                t + 1,
                reach & ~stops,
                (*prefix, i),
                acc0 + data.correct[stops, t].sum(),
                cost0 + data.cumcost[stops, t].sum(),
            )

    walk(0, np.ones(data.n, dtype=bool), (), 0.0, 0.0)
    values = np.where(over_budget(total_cost, budget, data.n), PENALTY, -n_correct)
    # Reject grid points whose quantiles decrease along any axis (index order = value order).
    for a in range(length - 2):
        ia = np.arange(g).reshape([-1 if d == a else 1 for d in range(length - 1)])
        ib = np.arange(g).reshape([-1 if d == a + 1 else 1 for d in range(length - 1)])
        values = np.where(ib < ia, PENALTY, values)
    return values


@dataclass(frozen=True)
class ChainResult:
    """Best thresholds found for one chain on the training tables."""

    cascade: Cascade
    quantiles: tuple[float, ...]
    accuracy: float
    mean_cost: float


def search_thresholds(data: ChainData, budget: float) -> ChainResult | None:
    """Sec. 4 threshold search for one chain (official ``optimizer.optimize``).

    Returns ``None`` if the chain cannot meet the budget: its first LLM alone is too expensive,
    or no quantile vector is within budget.
    """
    if over_budget(data.cumcost[:, 0].sum(), budget, data.n):
        return None
    if data.length == 1:
        n_correct, total_cost = evaluate(data, np.empty(0))
        return ChainResult(Cascade(data.order, ()), (), n_correct / data.n, total_cost / data.n)

    grid = np.linspace(Q_LOW, Q_HIGH, GRID_POINTS)
    values = _grid_objective(data, grid, budget)
    best = np.unravel_index(np.argmin(values), values.shape)  # first minimum, C order (= brute)
    x0 = grid[list(best)]
    x, fun, *_ = scipy.optimize.fmin(
        lambda q: _objective(data, q, budget), x0, full_output=True, disp=False
    )
    if fun >= PENALTY:
        return None
    x = np.atleast_1d(x)
    thresholds = quantiles_to_thresholds(data, x)
    n_correct, total_cost = evaluate(data, thresholds)
    cascade = Cascade(data.order, tuple(float(t) for t in thresholds))
    return ChainResult(cascade, tuple(float(v) for v in x), n_correct / data.n, total_cost / data.n)


def candidate_chains(
    correct: np.ndarray,
    length: int,
    chain_search: Literal["exhaustive", "mpi_pruned"] = "exhaustive",
    m: int = 3,
) -> Iterable[tuple[int, ...]]:
    """Chains to try: all ordered tuples (official default) or MPI-pruned ones (Eq. 2)."""
    k = np.asarray(correct).shape[1]
    if chain_search == "exhaustive":
        return itertools.permutations(range(k), length)
    if chain_search == "mpi_pruned":
        return pruned_chains(mpi_matrix(correct), length, m)
    raise ValueError(f"unknown chain_search {chain_search!r}")


@dataclass(frozen=True)
class OptimizeResult:
    """Best strategy for one budget, with the training-set numbers it reached."""

    best: ChainResult
    budget: float
    n_chains: int  # chains tried
    n_feasible: int  # chains that met the budget


def optimize(
    correct: np.ndarray,
    cost: np.ndarray,
    score: np.ndarray,
    budget: float,
    length: int = 3,
    chain_search: Literal["exhaustive", "mpi_pruned"] = "exhaustive",
    m: int = 3,
) -> OptimizeResult | None:
    """Eq. 1: the chain and thresholds with the best training accuracy at mean cost ≤ ``budget``.

    Returns ``None`` if no chain meets the budget.
    """
    correct = np.asarray(correct)
    if correct.ndim != 2 or correct.shape != np.shape(cost) or correct.shape != np.shape(score):
        raise ValueError("correct, cost and score must share one 2-D shape")
    if correct.shape[0] == 0:
        raise ValueError("no queries")
    if not 1 <= length <= correct.shape[1]:
        raise ValueError(f"length must be in 1..{correct.shape[1]}, got {length}")
    if not np.isfinite(score).all() or not np.isfinite(cost).all():
        raise ValueError("score and cost must be finite")
    if not (np.isfinite(budget) and budget >= 0):
        raise ValueError(f"budget must be a finite non-negative number, got {budget}")

    best: ChainResult | None = None
    n_chains = n_feasible = 0
    for order in candidate_chains(correct, length, chain_search, m):
        n_chains += 1
        result = search_thresholds(ChainData.build(order, correct, cost, score), budget)
        if result is None:
            continue
        n_feasible += 1
        if best is None or result.accuracy > best.accuracy:
            best = result
    if best is None:
        return None
    return OptimizeResult(best=best, budget=budget, n_chains=n_chains, n_feasible=n_feasible)
