"""Maximum performance improvement (MPI) and chain pruning (Sec. 4, Eq. 2; Fig. 2d)."""

from collections.abc import Iterator

import numpy as np


def mpi_matrix(correct: np.ndarray) -> np.ndarray:
    """Sec. 4 MPI: ``MPI[a, b] = Pr[r(f_a) > r(f_b)]`` = Pr[LLM a correct and LLM b wrong].

    ``correct`` is ``n_queries × n_llms`` (bool or 0/1). The diagonal is 0. Fig. 2d plots the
    **transpose** (row = the LLM that makes the mistake, column = the one that is correct).
    """
    c = np.asarray(correct)
    if c.ndim != 2 or c.shape[0] == 0:
        raise ValueError(f"correct must be a non-empty 2-D table, got shape {c.shape}")
    if c.dtype != bool:
        if not np.isin(c, (0, 1)).all():
            raise ValueError("correct must contain only True/False or 0/1")
        c = c.astype(bool)
    right = c.astype(np.float64)
    # One matrix product counts all pairs at once: entry [a, b] sums over queries of
    # right[q, a] · wrong[q, b], i.e. "a got it and b missed it". Dividing by n gives a rate.
    # Intuition: MPI[a, b] is the most accuracy that adding LLM a *after* LLM b could ever buy
    # (with a perfect scorer). An LLM that fails on the same queries as b adds nothing.
    return right.T @ (1.0 - right) / c.shape[0]


def top_m_next(mpi: np.ndarray, prev: int, used: tuple[int, ...], m: int) -> list[int]:
    """Eq. 2: the ``m`` LLMs ``k'`` with the largest ``MPI[k', prev]``, skipping LLMs in ``used``.

    The paper does not say how to break ties; we keep the lower column index first.
    """
    candidates = [k for k in range(mpi.shape[0]) if k not in used]
    candidates.sort(key=lambda k: (-mpi[k, prev], k))
    return candidates[:m]


def pruned_chains(mpi: np.ndarray, length: int, m: int) -> Iterator[tuple[int, ...]]:
    """Eq. 2: ordered chains of ``length`` LLMs where each next LLM is a top-``m`` MPI choice.

    Any LLM may come first. This gives at most ``K · m^(length-1)`` chains instead of
    ``K! / (K - length)!``.
    """
    k = mpi.shape[0]
    if mpi.shape != (k, k):
        raise ValueError(f"mpi must be square, got {mpi.shape}")
    if not 1 <= length <= k:
        raise ValueError(f"length must be in 1..{k}, got {length}")
    if m < 1:
        raise ValueError(f"m must be >= 1, got {m}")

    # Depth-first: from the current last LLM, branch only to the m LLMs that best fix its
    # mistakes. Cuts K=12, L=3 from 1320 ordered chains to at most 12·3·3 = 108.
    def extend(chain: tuple[int, ...]) -> Iterator[tuple[int, ...]]:
        if len(chain) == length:
            yield chain
            return
        for nxt in top_m_next(mpi, chain[-1], chain, m):
            yield from extend((*chain, nxt))

    for first in range(k):
        yield from extend((first,))
