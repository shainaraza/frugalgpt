"""Baselines FrugalGPT is compared with (Sec. 5: Table 2, Fig. 3, Fig. 4).

- **Best single LLM**: always call one LLM. Its (cost, accuracy) points are the dots in Fig. 3,
  and the most accurate one is the reference that Table 2's FrugalGPT has to match.
- **Simple threshold** (Fig. 4): call a cheap LLM, keep its answer if a confidence score is above a
  threshold, otherwise call an expensive LLM. In the paper the score is GPT-J's mean token
  log-prob and the expensive LLM is GPT-4. The authors did not release those log-probs, so
  :func:`threshold_curve` takes any per-query score (see docs/paper_vs_code.md).
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from frugalgpt.cascade import Cascade, simulate
from frugalgpt.optimizer import over_budget


def _as_bool_table(correct: np.ndarray) -> np.ndarray:
    c = np.asarray(correct)
    if c.ndim != 2 or c.shape[0] == 0:
        raise ValueError(f"correct must be a non-empty 2-D table, got shape {c.shape}")
    if c.dtype != bool:
        if not np.isin(c, (0, 1)).all():
            raise ValueError("correct must contain only True/False or 0/1")
        c = c.astype(bool)
    return c


def single_llm_points(correct: np.ndarray, cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fig. 3 dots: (accuracy, **mean** cost per query in USD) of each LLM used alone.

    Table 2 and Fig. 3 plot totals over the test set: multiply the cost by ``n_queries``.
    """
    c = _as_bool_table(correct)
    cost = np.asarray(cost, dtype=np.float64)
    if cost.shape != c.shape:
        raise ValueError(f"cost shape {cost.shape} != correct shape {c.shape}")
    return c.mean(axis=0), cost.mean(axis=0)


def best_single_llm(
    correct: np.ndarray, cost: np.ndarray, budget: float | None = None
) -> int | None:
    """Table 2 reference: the most accurate LLM, among those with mean cost ≤ ``budget``.

    ``budget=None`` means no limit. Ties go to the cheaper LLM, then the lower column index.
    Returns ``None`` if no LLM fits the budget.
    """
    acc, mean_cost = single_llm_points(correct, cost)
    if budget is not None and not (np.isfinite(budget) and budget >= 0):
        raise ValueError(f"budget must be a finite non-negative number, got {budget}")
    n = np.asarray(correct).shape[0]
    ok = np.ones(len(acc), dtype=bool) if budget is None else ~over_budget(mean_cost * n, budget, n)
    if not ok.any():
        return None
    candidates = [k for k in range(len(acc)) if ok[k]]
    return min(candidates, key=lambda k: (-acc[k], mean_cost[k], k))


def mean_token_logprob(token_logprobs: Sequence[float]) -> float:
    """Fig. 4 score: average log-probability of the generated tokens (``-inf`` if none).

    Raises on ``None`` (some APIs return it for the first token) or NaN, so bad inputs fail here
    instead of deep inside the cascade.
    """
    if len(token_logprobs) == 0:
        return float("-inf")
    if any(x is None for x in token_logprobs):
        raise ValueError("token_logprobs contains None; drop or fill those tokens first")
    lp = np.asarray(token_logprobs, dtype=np.float64)
    if np.isnan(lp).any() or np.isposinf(lp).any():
        raise ValueError("token_logprobs must not contain NaN or +inf")
    return float(lp.mean())


def finite_scores(score: np.ndarray, floor: float | None = None) -> np.ndarray:
    """Replace ``-inf`` (e.g. an empty answer's log-prob) by a finite value below every other score.

    The cascade only accepts finite scores. ``-inf`` becomes ``floor`` (default: the smallest
    finite score minus 1). It is "never accepted" only for thresholds ``τ ≥ floor``, so callers
    that sweep thresholds must pick a floor below every threshold too (as :func:`threshold_curve`
    does). NaN and ``+inf`` are rejected.
    """
    s = np.asarray(score, dtype=np.float64)
    if np.isnan(s).any() or np.isposinf(s).any():
        raise ValueError("scores must not be NaN or +inf")
    neg = np.isneginf(s)
    if not neg.any():
        return s
    finite = s[~neg]
    if floor is None:
        floor = (finite.min() if finite.size else 0.0) - 1.0
    elif finite.size and floor >= finite.min():
        raise ValueError(f"floor {floor} must be below the smallest finite score {finite.min()}")
    return np.where(neg, floor, s)


@dataclass(frozen=True)
class ThresholdCurve:
    """Fig. 4: one point per threshold of the two-LLM ``cheap → expensive`` rule."""

    cheap: int
    expensive: int
    thresholds: np.ndarray  # (T,)
    accuracy: np.ndarray  # (T,)
    mean_cost: np.ndarray  # (T,) USD per query
    escalated: np.ndarray  # (T,) share of queries sent on to the expensive LLM
    n_queries: int

    @property
    def total_cost(self) -> np.ndarray:
        """(T,) total USD over all queries: the Fig. 4 x-axis."""
        return self.mean_cost * self.n_queries


def threshold_curve(
    score: np.ndarray,
    correct: np.ndarray,
    cost: np.ndarray,
    cheap: int,
    expensive: int,
    thresholds: Sequence[float],
) -> ThresholdCurve:
    """Fig. 4 simple threshold: keep ``cheap``'s answer if ``score > τ``, else call ``expensive``.

    ``score`` is one value per query for the cheap LLM's answer (e.g. its mean token log-prob;
    ``-inf`` allowed, e.g. an empty answer). A ``-inf`` query is always sent on, whatever ``τ``,
    since ``-inf > τ`` is false for every finite ``τ``. Each threshold is evaluated with the
    Sec. 4 cascade simulator, so the cost of an escalated query is cheap + expensive.
    """
    c = _as_bool_table(correct)
    n_llms = c.shape[1]
    for name, k in (("cheap", cheap), ("expensive", expensive)):
        if not isinstance(k, (int, np.integer)) or not 0 <= k < n_llms:
            raise ValueError(f"{name} must be a column index in [0, {n_llms}), got {k!r}")
    if cheap == expensive:
        raise ValueError("cheap and expensive must be different LLMs")
    taus = np.asarray(thresholds, dtype=np.float64)
    if taus.ndim != 1 or taus.size == 0:
        raise ValueError("thresholds must be a non-empty 1-D sequence")
    if not np.isfinite(taus).all():
        raise ValueError("thresholds must be finite")
    raw = np.asarray(score, dtype=np.float64)
    if raw.shape != (c.shape[0],):
        raise ValueError(f"score must have one value per query ({c.shape[0]}), got {raw.shape}")
    finite = raw[np.isfinite(raw)]
    floor = min(finite.min() if finite.size else 0.0, taus.min()) - 1.0
    s = finite_scores(raw, floor=floor)  # below every score and every τ → never accepted

    table = np.full(c.shape, np.nan)  # only the cheap LLM's score is read
    table[:, cheap] = s
    acc, mean_cost, escalated = [], [], []
    for tau in taus:
        r = simulate(Cascade((cheap, expensive), (float(tau),)), c, cost, table)
        acc.append(r.accuracy)
        mean_cost.append(r.mean_cost)
        escalated.append(float((r.stop == 1).mean()))
    return ThresholdCurve(
        cheap=cheap,
        expensive=expensive,
        thresholds=taus,
        accuracy=np.array(acc),
        mean_cost=np.array(mean_cost),
        escalated=np.array(escalated),
        n_queries=c.shape[0],
    )


def cheapest_at_accuracy(
    accuracy: np.ndarray, mean_cost: np.ndarray, target: float
) -> tuple[int, float] | None:
    """Table 2 / Fig. 4 read-out: the cheapest point with accuracy ≥ ``target``.

    Returns ``(index, mean_cost)``, or ``None`` if no point reaches the target. Ties in cost go
    to the lower index.
    """
    acc = np.asarray(accuracy, dtype=np.float64)
    cost = np.asarray(mean_cost, dtype=np.float64)
    if acc.shape != cost.shape or acc.ndim != 1:
        raise ValueError("accuracy and mean_cost must be 1-D arrays of the same length")
    if np.isnan(acc).any() or np.isnan(cost).any():
        raise ValueError("accuracy and mean_cost must not contain NaN")
    ok = np.flatnonzero(acc >= target)
    if ok.size == 0:
        return None
    i = int(ok[np.argmin(cost[ok])])
    return i, float(cost[i])
