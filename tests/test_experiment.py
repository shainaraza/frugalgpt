import numpy as np
import pandas as pd
import pytest

from frugalgpt.experiment import Split, budget_grid, case_study, cost_to_match, tradeoff
from frugalgpt.mpi import mpi_matrix

# 8 queries × 3 LLMs. LLM 0 (cost 1) is right on half the queries and its scorer knows exactly
# which; LLM 2 (cost 100) is always right; LLM 1 (cost 10) is never right.
CORRECT = np.zeros((8, 3), dtype=bool)
CORRECT[:4, 0] = True
CORRECT[:, 2] = True
COST = np.tile([1.0, 10.0, 100.0], (8, 1))
SCORE = np.zeros((8, 3))
SCORE[:4, 0] = 0.9
SCORE[4:, 0] = 0.1
SCORE[:, 1] = 0.5
SCORE[:, 2] = 0.5
SPLIT = Split(CORRECT, COST, SCORE)
NAMES = ["cheap", "useless", "big"]


def test_split_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError):
        Split(CORRECT, COST[:, :2], SCORE)


def test_tradeoff_by_hand() -> None:
    # Perfect cascade "cheap → (useless) → big": half the queries stop at cheap (cost 1), the
    # other half go to big. With L = 2: mean cost = (4·1 + 4·101) / 8 = 51, accuracy 1.
    curve = tradeoff(SPLIT, SPLIT, [0.5, 51.0, 200.0], NAMES, length=2)
    assert list(curve.columns[:4]) == ["budget", "chain", "order", "thresholds"]
    # budget 0.5: even the cheapest LLM alone is too expensive → infeasible row
    assert pd.isna(curve.loc[0, "chain"]) and np.isnan(curve.loc[0, "test_acc"])
    # budget 51: exactly the perfect cascade (budget boundary is feasible)
    assert curve.loc[1, "chain"] == "cheap → big"
    assert curve.loc[1, "test_acc"] == 1.0
    assert curve.loc[1, "test_cost"] == pytest.approx(51.0)
    assert curve.loc[1, "test_total_cost"] == pytest.approx(51.0 * 8)
    # more budget cannot beat accuracy 1; the first chain found with that accuracy is kept
    assert curve.loc[2, "test_acc"] == 1.0


def test_tradeoff_budget_equal_to_cheapest_llm_is_infeasible() -> None:
    # Official quantile range starts at 1e-5, so the lowest-scored queries (here 4 tied at 0.1)
    # are always passed on: "accept the cheap LLM always" (mean cost 1) cannot be represented.
    curve = tradeoff(SPLIT, SPLIT, [1.0], NAMES, length=2)
    assert pd.isna(curve.loc[0, "chain"])
    # Enough for those 4 queries to escalate to the cheapest next LLM ("useless", +10 each).
    curve = tradeoff(SPLIT, SPLIT, [1.0 + 4 * 10 / 8], NAMES, length=2)
    assert curve.loc[0, "order"] == (0, 1)
    assert curve.loc[0, "test_acc"] == 0.5


def test_tradeoff_evaluates_on_test_not_train() -> None:
    # Same strategy, but on "test" the cheap LLM is wrong where its scorer is confident.
    test_correct = CORRECT.copy()
    test_correct[:4, 0] = False
    test = Split(test_correct, COST, SCORE)
    curve = tradeoff(SPLIT, test, [51.0], NAMES, length=2)
    assert curve.loc[0, "train_acc"] == 1.0
    assert curve.loc[0, "test_acc"] == 0.5


def test_tradeoff_validates() -> None:
    with pytest.raises(ValueError):
        tradeoff(SPLIT, Split(CORRECT[:, :2], COST[:, :2], SCORE[:, :2]), [1.0])
    with pytest.raises(ValueError):
        tradeoff(SPLIT, SPLIT, [1.0], ["a", "b"])


def test_cost_to_match() -> None:
    curve = pd.DataFrame(
        {
            "budget": [1, 2, 3, 4],
            "test_acc": [0.5, 0.9, 0.95, np.nan],
            "test_cost": [1.0, 3.0, 3.0, 0.1],
        }
    )
    row = cost_to_match(curve, 0.9)
    assert row["budget"] == 3  # same cost as budget 2, higher accuracy
    assert cost_to_match(curve, 0.5)["budget"] == 1
    assert cost_to_match(curve, 0.99) is None  # NaN rows never match


def test_case_study_by_hand() -> None:
    cs = case_study(SPLIT, SPLIT, 51.0, NAMES, ref=2, length=2)
    assert cs.chain == ("cheap", "big")
    assert cs.test_acc == 1.0 and cs.ref_acc == 1.0
    assert cs.test_total_cost == pytest.approx(408.0)
    assert cs.ref_total_cost == pytest.approx(800.0)
    assert cs.cost_saving == pytest.approx(0.49)
    assert cs.acc_gain == 0.0
    assert cs.reach_share == (1.0, 0.5)
    # Fig. 2d orientation: row = LLM that is wrong, column = LLM that is right
    np.testing.assert_allclose(cs.mpi, mpi_matrix(CORRECT).T)
    assert cs.mpi[1, 2] == 1.0  # "useless" wrong while "big" right on every query
    assert cs.mpi[2, 1] == 0.0


def test_case_study_infeasible() -> None:
    assert case_study(SPLIT, SPLIT, 0.5, NAMES, ref=2) is None


def test_budget_grid() -> None:
    g = budget_grid([1.0, 2.0], 0.1, 1.0, 3, extra=[2.0, 0.5])
    np.testing.assert_allclose(g, [0.1, np.sqrt(0.1), 0.5, 1.0, 2.0])
    assert (np.diff(g) > 0).all()


def test_tradeoff_empty_budgets() -> None:
    curve = tradeoff(SPLIT, SPLIT, [], NAMES)
    assert curve.empty and "test_acc" in curve.columns
    assert cost_to_match(curve, 0.5) is None


def test_case_study_zero_cost_reference() -> None:
    cost = COST.copy()
    cost[:, 1] = 0.0
    cs = case_study(
        Split(CORRECT, cost, SCORE), Split(CORRECT, cost, SCORE), 51.0, NAMES, ref=1, length=2
    )
    assert np.isnan(cs.cost_saving)


# --- step 10: Table 2 robustness -------------------------------------------------------------


def test_match_support_by_hand() -> None:
    from frugalgpt.experiment import match_support

    ref = np.array([1, 1, 0, 0], dtype=bool)
    # always at least as good (cascade ⊇ reference) → every resample matches
    assert match_support(np.array([1, 1, 1, 0], dtype=bool), ref) == 1.0
    # always worse on the one query where they differ → only resamples avoiding it match
    worse = np.array([1, 0, 0, 0], dtype=bool)
    # P(query 1 never drawn in 4 draws) = (3/4)^4 ≈ 0.316
    assert match_support(worse, ref, n_boot=20000) == pytest.approx(0.75**4, abs=0.02)
    # identical → always a tie, which counts as a match
    assert match_support(ref, ref) == 1.0
    with pytest.raises(ValueError):
        match_support(ref[:3], ref)
    with pytest.raises(ValueError):
        match_support(np.array([], dtype=bool), np.array([], dtype=bool))


def test_match_support_is_seeded() -> None:
    from frugalgpt.experiment import match_support

    rng = np.random.default_rng(0)
    a, b = rng.random(200) < 0.8, rng.random(200) < 0.8
    assert match_support(a, b, seed=2024) == match_support(a, b, seed=2024)


def test_cost_to_match_skips_infeasible_nan_rows() -> None:
    curve = pd.DataFrame(
        {
            "budget": [1.0, 2.0, 3.0],
            "test_acc": [np.nan, 0.9, 0.95],
            "test_cost": [np.nan, 2.0, 3.0],
        }
    )
    assert cost_to_match(curve, 0.9)["budget"] == 2.0
    assert cost_to_match(curve.iloc[:1], 0.0) is None  # NaN ≥ x is False → no match


def test_robust_cost_to_match_skips_a_lucky_point() -> None:
    from frugalgpt.experiment import robust_cost_to_match

    # 40 queries; reference LLM 1 is right on 30. LLM 0 is cheap, right on a different 30 (so a
    # point accepting it everywhere ties the reference but wins only ~half of resamples).
    n = 40
    correct = np.zeros((n, 2), dtype=bool)
    correct[:30, 1] = True
    correct[10:, 0] = True
    cost = np.tile([1.0, 10.0], (n, 1))
    score = np.zeros((n, 2))
    score[:, 0] = np.where(correct[:, 0], 0.9, 0.1)  # perfect scorer on LLM 0
    test = Split(correct, cost, score)
    curve = pd.DataFrame(
        {
            "budget": [1.0, 6.0],
            "chain": ["0 → 1 (accept all)", "0 → 1 (perfect)"],
            "order": [(0, 1), (0, 1)],
            "thresholds": [(0.0,), (0.5,)],
            "test_acc": [0.75, 1.0],
            "test_cost": [1.0, 1.0 + 10 * 10 / n],
        }
    )
    assert cost_to_match(curve, 0.75)["budget"] == 1.0  # plain read-out takes the tie
    m = robust_cost_to_match(curve, test, ref=1, level=0.9)
    assert m is not None and m.row["budget"] == 6.0
    assert m.margin_queries == 10 and m.support == 1.0
    assert robust_cost_to_match(curve.iloc[:1], test, ref=1, level=0.9) is None


def test_table2_reference_is_chosen_on_train() -> None:
    # OVERRULING-like: LLM 1 best on train, LLM 0 best on test → reference = LLM 1.
    from frugalgpt.baselines import best_single_llm

    train = np.array([[1, 1], [0, 1], [0, 1], [1, 0]], dtype=bool)
    test = np.array([[1, 0], [1, 1], [1, 0], [0, 1]], dtype=bool)
    cost = np.ones((4, 2))
    assert best_single_llm(train, cost) == 1
    assert best_single_llm(test, cost) == 0
