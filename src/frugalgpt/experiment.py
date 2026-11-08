"""Sec. 5 experiments on tables: trade-off curve (Fig. 3), Table 2 read-out, Fig. 2 case study.

Protocol (official ``FrugalGPT_gen_tradeoff_HEADLINES.ipynb``, which runs it on the 2024 model
pool; we apply it to the paper's 2023 pool): for each budget, learn the strategy (chain +
thresholds, Eq. 1) on the **train** split, then simulate it on the **test** split.
Budgets are average cost per query; the paper's tables and figures report **totals over the
test set**, so every row carries both.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from frugalgpt.cascade import Cascade, simulate
from frugalgpt.mpi import mpi_matrix
from frugalgpt.optimizer import optimize


@dataclass(frozen=True)
class Split:
    """One split's ``n_queries × n_llms`` tables (Sec. 4 offline-first design)."""

    correct: np.ndarray  # (n, k) bool
    cost: np.ndarray  # (n, k) USD
    score: np.ndarray  # (n, k) scorer output g_i(q, a)

    def __post_init__(self) -> None:
        shapes = {np.shape(self.correct), np.shape(self.cost), np.shape(self.score)}
        if len(shapes) != 1 or len(next(iter(shapes))) != 2:
            raise ValueError(f"correct, cost and score must share one 2-D shape, got {shapes}")

    @property
    def n(self) -> int:
        """Number of queries."""
        return self.correct.shape[0]


def tradeoff(
    train: Split,
    test: Split,
    budgets: Sequence[float],
    llms: Sequence[str] | None = None,
    length: int = 3,
    chain_search: Literal["exhaustive", "mpi_pruned"] = "exhaustive",
    m: int = 3,
) -> pd.DataFrame:
    """Fig. 3: one row per budget — strategy learned on ``train`` (Eq. 1), evaluated on ``test``.

    Budgets with no feasible chain give a row with ``chain = None`` and NaN metrics.
    """
    if train.correct.shape[1] != test.correct.shape[1]:
        raise ValueError("train and test must have the same LLM columns")
    k = train.correct.shape[1]
    names = list(llms) if llms is not None else [str(i) for i in range(k)]
    if len(names) != k:
        raise ValueError(f"{len(names)} LLM names for {k} columns")

    # Protocol: the strategy is *learned* on train and *reported* on test, so test numbers are
    # an honest estimate. Each budget is one independent optimization → one point of the curve.
    rows = []
    for budget in budgets:  # an empty list gives an empty frame with the usual columns
        res = optimize(train.correct, train.cost, train.score, budget, length, chain_search, m)
        row: dict = {"budget": float(budget)}
        if res is None:
            row.update(chain=None, order=None, thresholds=None)
            row.update({c: np.nan for c in _METRICS})
            rows.append(row)
            continue
        cascade = res.best.cascade
        tr = simulate(cascade, train.correct, train.cost, train.score)
        te = simulate(cascade, test.correct, test.cost, test.score)
        row.update(
            chain=" → ".join(names[i] for i in cascade.order),
            order=tuple(cascade.order),
            thresholds=tuple(cascade.thresholds),
            train_acc=tr.accuracy,
            train_cost=tr.mean_cost,
            test_acc=te.accuracy,
            test_cost=te.mean_cost,
            test_total_cost=te.total_cost,
            n_feasible=res.n_feasible,
        )
        rows.append(row)
    return pd.DataFrame(rows, columns=_COLUMNS)


_METRICS = ("train_acc", "train_cost", "test_acc", "test_cost", "test_total_cost", "n_feasible")
_COLUMNS = ["budget", "chain", "order", "thresholds", *_METRICS]


def cost_to_match(curve: pd.DataFrame, target_acc: float) -> pd.Series | None:
    """Table 2: the cheapest test point whose test accuracy is ≥ ``target_acc`` (``None`` if none).

    Ties in cost go to the higher accuracy, then the smaller budget. The point is selected on
    noisy test accuracy, so it is optimistic by selection (a point that matches by a few queries
    by chance wins); see ``robust_cost_to_match`` for a margin-robust read-out.
    """
    if curve.empty:
        return None
    ok = curve[curve["test_acc"] >= target_acc]
    if ok.empty:
        return None
    ok = ok.sort_values(["test_cost", "test_acc", "budget"], ascending=[True, False, True])
    return ok.iloc[0]


def match_support(
    cascade_correct: np.ndarray,
    ref_correct: np.ndarray,
    n_boot: int = 2000,
    seed: int = 2024,
) -> float:
    """Table 2 robustness: share of paired bootstrap resamples where the cascade ≥ the reference.

    Both inputs are per-query 0/1 correctness on the same test queries. Resampling queries with
    replacement keeps the pairing, so the share reflects how often "matches the best LLM" holds
    beyond the noise of this particular test set.
    """
    a = np.asarray(cascade_correct, dtype=float)
    b = np.asarray(ref_correct, dtype=float)
    if a.shape != b.shape or a.ndim != 1 or a.size == 0:
        raise ValueError("need two non-empty 1-D arrays of the same length")
    if n_boot < 1:
        raise ValueError("n_boot must be ≥ 1")
    # Paired: resample *queries*, keeping each query's (cascade, reference) outcome together.
    # Only queries where they disagree matter; a match "by 2 queries" fails in many resamples.
    diff = a - b
    idx = np.random.default_rng(seed).integers(0, diff.size, size=(n_boot, diff.size))
    return float((diff[idx].sum(axis=1) >= 0).mean())


@dataclass(frozen=True)
class RobustMatch:
    """Table 2, margin-robust: the chosen curve row and its bootstrap support."""

    row: pd.Series
    support: float  # share of paired bootstrap resamples with cascade accuracy ≥ reference
    margin_queries: int  # (cascade correct − reference correct) on the test set


def robust_cost_to_match(
    curve: pd.DataFrame,
    test: Split,
    ref: int,
    level: float = 0.9,
    n_boot: int = 2000,
    seed: int = 2024,
) -> RobustMatch | None:
    """Table 2 read-out that resists test noise: cheapest point matching LLM ``ref`` robustly.

    Among rows with test accuracy ≥ the reference's (cheapest first, same ties as
    ``cost_to_match``), returns the first whose ``match_support`` is ≥ ``level``.
    """
    ref_correct = np.asarray(test.correct)[:, ref].astype(bool)
    ok = curve[curve["test_acc"] >= ref_correct.mean()]
    ok = ok.sort_values(["test_cost", "test_acc", "budget"], ascending=[True, False, True])
    for _, row in ok.iterrows():
        res = simulate(
            Cascade(tuple(row["order"]), tuple(row["thresholds"])),
            test.correct,
            test.cost,
            test.score,
        )
        support = match_support(res.correct, ref_correct, n_boot, seed)
        if support >= level:
            margin = int(res.correct.sum()) - int(ref_correct.sum())
            return RobustMatch(row=row, support=support, margin_queries=margin)
    return None


@dataclass(frozen=True)
class CaseStudy:
    """Fig. 2: one budget's learned strategy, its test numbers and the reference LLM's."""

    budget: float
    cascade: Cascade
    chain: tuple[str, ...]
    test_acc: float
    test_total_cost: float
    reach_share: tuple[float, ...]  # Fig. 2e: share of test queries that call each chain LLM
    ref_llm: str
    ref_acc: float
    ref_total_cost: float
    mpi: np.ndarray  # Fig. 2d (test split; row = the LLM that is wrong)

    @property
    def cost_saving(self) -> float:
        """Fig. 2c: relative cost reduction vs. the reference LLM (``nan`` if it costs nothing)."""
        if self.ref_total_cost == 0:
            return float("nan")
        return 1.0 - self.test_total_cost / self.ref_total_cost

    @property
    def acc_gain(self) -> float:
        """Fig. 2c: accuracy difference vs. the reference LLM (fraction, not points)."""
        return self.test_acc - self.ref_acc


def case_study(
    train: Split,
    test: Split,
    budget: float,
    llms: Sequence[str],
    ref: int,
    length: int = 3,
    chain_search: Literal["exhaustive", "mpi_pruned"] = "exhaustive",
    m: int = 3,
) -> CaseStudy | None:
    """Fig. 2: learn at ``budget`` on train, report test accuracy / cost against LLM ``ref``."""
    res = optimize(train.correct, train.cost, train.score, budget, length, chain_search, m)
    if res is None:
        return None
    cascade = res.best.cascade
    te = simulate(cascade, test.correct, test.cost, test.score)
    return CaseStudy(
        budget=float(budget),
        cascade=cascade,
        chain=tuple(llms[i] for i in cascade.order),
        test_acc=te.accuracy,
        test_total_cost=te.total_cost,
        reach_share=tuple(float(x) for x in te.reach_share()),
        ref_llm=llms[ref],
        ref_acc=float(np.asarray(test.correct)[:, ref].mean()),
        ref_total_cost=float(np.asarray(test.cost)[:, ref].sum()),
        mpi=mpi_matrix(test.correct).T,  # Fig. 2d orientation: row = wrong, column = right
    )


def budget_grid(
    official: Sequence[float], low: float, high: float, n_log: int, extra: Sequence[float] = ()
) -> np.ndarray:
    """Sec. 5 budgets: official grid ∪ ``n_log`` log-spaced points in ``[low, high]`` ∪ extras.

    Sorted and de-duplicated (to 1e-12 relative).
    """
    pts = np.concatenate([np.asarray(official, float), np.geomspace(low, high, n_log), extra])
    pts = np.sort(pts)
    keep = np.concatenate([[True], np.diff(pts) > 1e-12 * pts[1:]])
    return pts[keep]
