"""Reward r(a, a*) for FrugalGPT: exact match after SQuAD-style normalization.

The paper (Sec. 3) leaves the reward abstract. The official code uses exact match after the standard
SQuAD normalization (lowercase, strip punctuation, drop English articles, collapse whitespace).
Every dataset in the official code (HEADLINES, OVERRULING, AGNEWS, COQA, SCIQ) uses this plain exact
match; its multiple-choice (``mc``) and F1 variants are side metrics, never the reward.
"""

import re
import string

import numpy as np

_PUNCTUATION = frozenset(string.punctuation)
_ARTICLES = re.compile(r"\b(a|an|the)\b")


def normalize_answer(text: str) -> str:
    """Sec. 3 (reward r): SQuAD-style normalization.

    Steps, in this order: lowercase → remove ASCII punctuation → replace the words a/an/the with a
    space → collapse runs of whitespace and strip the ends.
    """
    # Normalization makes formatting noise not count as a wrong answer: "The Up." == "up".
    # The order matters: punctuation is removed before articles, so "the." still loses "the".
    text = text.lower()
    text = "".join(ch for ch in text if ch not in _PUNCTUATION)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def exact_match(prediction: str, reference: str) -> bool:
    """Sec. 3 (reward r): 1 if the normalized prediction equals the normalized reference."""
    return normalize_answer(prediction) == normalize_answer(reference)


def exact_match_array(predictions: np.ndarray, references: np.ndarray) -> np.ndarray:
    """Sec. 3 (reward r), element-wise: the ``correct`` table.

    ``predictions`` is ``(n_queries, n_llms)`` and ``references`` must have the same number of
    dimensions, typically ``(n_queries, 1)``. A 1-D ``references`` is rejected because numpy would
    broadcast it along the LLM axis, which silently gives wrong results on square tables.
    Inputs must be strings; ``None``/NaN raise, as in the official code.
    """
    predictions = np.asarray(predictions, dtype=object)
    references = np.asarray(references, dtype=object)
    if references.ndim != predictions.ndim:
        raise ValueError(
            f"references must have {predictions.ndim} dims (e.g. shape (n_queries, 1)), "
            f"got shape {references.shape}"
        )
    predictions, references = np.broadcast_arrays(predictions, references)
    out = np.fromiter(
        (exact_match(p, r) for p, r in zip(predictions.ravel(), references.ravel(), strict=True)),
        dtype=bool,
        count=predictions.size,
    )
    return out.reshape(predictions.shape)
