import numpy as np
import pytest

from frugalgpt.mpi import mpi_matrix, pruned_chains, top_m_next

# 4 queries × 3 LLMs.
CORRECT = np.array(
    [
        [True, False, True],
        [False, True, True],
        [False, False, True],
        [True, True, False],
    ]
)


def test_mpi_by_hand() -> None:
    mpi = mpi_matrix(CORRECT)
    # MPI[a, b] = share of queries where a is right and b is wrong.
    # Row 0: 0 right & 1 wrong on q0 → 1; 0 right & 2 wrong on q3 → 1.
    # Row 1: 1 right & 0 wrong on q1 → 1; 1 right & 2 wrong on q3 → 1.
    # Row 2: 2 right & 0 wrong on q1, q2 → 2; 2 right & 1 wrong on q0, q2 → 2.
    expected = np.array([[0, 1, 1], [1, 0, 1], [2, 2, 0]])
    np.testing.assert_allclose(mpi, expected / 4)
    np.testing.assert_allclose(np.diag(mpi), 0)


def test_mpi_accepts_0_1_and_rejects_other_values() -> None:
    np.testing.assert_allclose(mpi_matrix(CORRECT.astype(int)), mpi_matrix(CORRECT))
    with pytest.raises(ValueError):
        mpi_matrix(np.array([[0.5, 1.0]]))
    with pytest.raises(ValueError):
        mpi_matrix(np.zeros((0, 3), dtype=bool))


def test_top_m_next_ties_and_used() -> None:
    mpi = np.array(
        [
            [0.0, 0.3, 0.1],
            [0.2, 0.0, 0.1],
            [0.2, 0.3, 0.0],
            [0.2, 0.1, 0.1],
        ]
    )
    mpi = np.pad(mpi, ((0, 0), (0, 1)))  # 4 × 4
    # Column 0 (prev = 0): values 0, .2, .2, .2 → ties broken by lower index.
    assert top_m_next(mpi, prev=0, used=(0,), m=2) == [1, 2]
    assert top_m_next(mpi, prev=1, used=(1,), m=2) == [0, 2]  # 0.3, 0.3 tie → 0 then 2
    assert top_m_next(mpi, prev=1, used=(1, 0), m=5) == [2, 3]  # fewer left than m


def test_pruned_chains_count_and_validity() -> None:
    rng = np.random.default_rng(2024)
    mpi = mpi_matrix(rng.random((50, 6)) < 0.5)
    chains = list(pruned_chains(mpi, length=3, m=2))
    assert len(chains) == 6 * 2 * 2
    assert all(len(set(c)) == 3 for c in chains)
    assert len(set(chains)) == len(chains)
    # With m ≥ K − 1, pruning keeps every ordered chain.
    assert len(list(pruned_chains(mpi, length=3, m=5))) == 6 * 5 * 4


def test_pruned_chains_follow_top_m() -> None:
    mpi = mpi_matrix(CORRECT)
    for chain in pruned_chains(mpi, length=2, m=1):
        assert chain[1] == top_m_next(mpi, chain[0], (chain[0],), 1)[0]


@pytest.mark.parametrize("length, m", [(0, 1), (4, 1), (2, 0)])
def test_pruned_chains_invalid(length: int, m: int) -> None:
    with pytest.raises(ValueError):
        list(pruned_chains(mpi_matrix(CORRECT), length, m))


def test_top_m_next_ranks_by_column_prev() -> None:
    # Asymmetric: MPI[k', prev] (column prev) decides, not MPI[prev, k'] (row prev).
    mpi = np.array(
        [
            [0.0, 0.9, 0.1],
            [0.1, 0.0, 0.1],
            [0.5, 0.1, 0.0],
        ]
    )
    # prev = 0: column 0 = [0, 0.1, 0.5] → LLM 2 first. Row 0 would wrongly give LLM 1.
    assert top_m_next(mpi, prev=0, used=(0,), m=1) == [2]


def test_pruned_chains_length_4() -> None:
    rng = np.random.default_rng(2024)
    mpi = mpi_matrix(rng.random((40, 6)) < 0.5)
    chains = list(pruned_chains(mpi, length=4, m=2))
    assert len(chains) == 6 * 2 * 2 * 2
    assert all(len(set(c)) == 4 for c in chains)
