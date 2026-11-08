"""LLM cascade simulated on precomputed tables (Sec. 4).

For one query the cascade calls the LLMs of ``order`` one after another:

    for t = 1 .. L:
        k = order[t]                       # pay cost[n, k]
        if score[n, k] > τ_t or t == L:    # stop judger
            return answer of k

The cost of a query is the sum over **every** LLM called up to and including the accepted one.
All inputs are ``n_queries × n_llms`` tables (``correct``, ``cost``, ``score``), so a cascade is
evaluated for all queries at once with numpy. Thresholds are score thresholds (``score > τ``).

The last LLM is always accepted, whatever its score. This matches the official inference code
(``llmchain.py: predict_one``). The official *training* objective (``optimizer.py: f``) instead
drops a query whose last score is exactly 0 (see docs/paper_vs_code.md and
``test_last_llm_always_accepted_even_with_zero_score``).
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Cascade:
    """Sec. 4 strategy: LLM chain ``order`` (column indices) and stop thresholds.

    ``thresholds[t]`` applies to ``order[t]``. There are ``len(order) - 1`` of them: the last LLM's
    answer is always accepted, so it has no threshold.
    """

    order: tuple[int, ...]
    thresholds: tuple[float, ...]

    def __post_init__(self) -> None:
        if not all(isinstance(k, int | np.integer) and not isinstance(k, bool) for k in self.order):
            raise ValueError(f"order must contain integer column indices: {self.order}")
        if len(self.order) == 0:
            raise ValueError("order must contain at least one LLM")
        if len(set(self.order)) != len(self.order):
            raise ValueError(f"order has a repeated LLM: {self.order}")
        if any(k < 0 for k in self.order):
            raise ValueError(f"order has a negative index: {self.order}")
        if len(self.thresholds) != len(self.order) - 1:
            raise ValueError(
                f"need {len(self.order) - 1} thresholds for {len(self.order)} LLMs, "
                f"got {len(self.thresholds)}"
            )
        if not np.isfinite(self.thresholds).all():
            raise ValueError("thresholds must be finite")


@dataclass(frozen=True)
class CascadeResult:
    """Per-query outcome of a cascade, plus the summary numbers used in Sec. 5."""

    stop: np.ndarray  # (n,) int — position in the chain where the query stopped (0-based)
    llm: np.ndarray  # (n,) int — column index of the LLM whose answer was accepted
    correct: np.ndarray  # (n,) bool — reward r of the accepted answer
    cost: np.ndarray  # (n,) float — USD, sum over every LLM called
    chain_length: int  # L = len(order)

    @property
    def accuracy(self) -> float:
        """Expected reward E[r] (Eq. 1 objective)."""
        return float(self.correct.mean())

    @property
    def mean_cost(self) -> float:
        """Average cost per query in USD (Eq. 1 budget unit)."""
        return float(self.cost.mean())

    @property
    def total_cost(self) -> float:
        """Total cost over all queries in USD (what Table 2 and Fig. 3 report)."""
        return float(self.cost.sum())

    def reach_share(self) -> np.ndarray:
        """Share of queries that call the ``t``-th LLM of the chain, ``t = 0..L-1`` (Fig. 2e)."""
        return np.array([(self.stop >= t).mean() for t in range(self.chain_length)])


def stop_index(chain_scores: np.ndarray, thresholds: tuple[float, ...]) -> np.ndarray:
    """Sec. 4 stop judger: first chain position ``t`` with ``score > τ_t``, else the last one.

    ``chain_scores`` is ``n × L`` (scores of the chain's LLMs, in chain order).
    """
    n, length = chain_scores.shape
    if len(thresholds) != length - 1:
        raise ValueError(f"need {length - 1} thresholds, got {len(thresholds)}")
    # Build an (n, L) "would accept here" mask; the first True per row is where the query stops.
    # The last column is forced True so every query gets an answer (no abstention).
    accept = np.ones((n, length), dtype=bool)  # last column: always accept
    accept[:, :-1] = chain_scores[:, :-1] > np.asarray(thresholds, dtype=np.float64)
    return accept.argmax(axis=1)  # argmax returns the first True


def simulate(
    cascade: Cascade, correct: np.ndarray, cost: np.ndarray, score: np.ndarray
) -> CascadeResult:
    """Sec. 4: run ``cascade`` on every query of the ``correct`` / ``cost`` / ``score`` tables.

    ``correct`` must be bool or 0/1. There must be at least one query.
    """
    correct = np.asarray(correct)
    if correct.dtype != bool:
        if not np.isin(correct, (0, 1)).all():
            raise ValueError("correct must contain only True/False or 0/1")
        correct = correct.astype(bool)
    cost = np.asarray(cost, dtype=np.float64)
    score = np.asarray(score, dtype=np.float64)
    if correct.ndim != 2 or correct.shape != cost.shape or correct.shape != score.shape:
        raise ValueError(
            f"tables must share one 2-D shape, got {correct.shape}, {cost.shape}, {score.shape}"
        )
    if correct.shape[0] == 0:
        raise ValueError("no queries")
    if max(cascade.order) >= correct.shape[1]:
        raise ValueError(f"order {cascade.order} out of range for {correct.shape[1]} LLMs")

    order = np.array(cascade.order)
    # The last LLM's score is never used, so only the scored positions must be valid.
    chain_scores = score[:, order]
    if not np.isfinite(chain_scores[:, :-1]).all():
        raise ValueError("scores must be finite")
    if not np.isfinite(cost[:, order]).all() or (cost[:, order] < 0).any():
        raise ValueError("costs must be finite and non-negative")

    # Vectorized cascade: instead of looping over queries, find for every query the chain
    # position where it stops, then gather that position's correctness and cumulative cost.
    stop = stop_index(chain_scores, cascade.thresholds)
    rows = np.arange(len(stop))
    # Cumulative, not per-LLM: a query that stops at step t has paid for steps 0..t, because
    # every LLM before it was called and its answer rejected. This is why a bad scorer on a cheap
    # LLM is expensive: rejected answers still cost money.
    cumulative_cost = np.cumsum(cost[:, order], axis=1)
    return CascadeResult(
        stop=stop,
        llm=order[stop],
        correct=correct[:, order][rows, stop],
        cost=cumulative_cost[rows, stop],
        chain_length=len(order),
    )
