import numpy as np
import pytest

from frugalgpt.metrics import exact_match, exact_match_array, normalize_answer


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (" down", "down"),  # cached answers usually start with a space
        ("Down.", "down"),
        ("  UP!!  ", "up"),
        ("The answer is a dog", "answer is dog"),
        ("an apple", "apple"),
        ("theory", "theory"),  # 'the' inside a word is kept
        ("anthem", "anthem"),
        ("a\tb\nc", "b c"),  # 'a' removed, all whitespace collapsed
        ("", ""),
        ("...", ""),
        ("$1,601.50", "160150"),
        ("café", "café"),  # non-ASCII letters untouched
        ("THE Up", "up"),  # lowercased before articles are removed
        ("A", ""),
        ("up-down", "updown"),  # punctuation is deleted, not replaced by a space
        ("don't", "dont"),
        ("a.the", "athe"),  # punctuation removed first, so articles glued by '.' survive
        ("“up”", "“up”"),  # unicode quotes are not ASCII punctuation: kept (matches official code)
        ("down…", "down…"),
        ("(B)", "b"),  # no special multiple-choice handling by default
        ("up ", "up"),  # non-breaking space counts as whitespace
    ],
)
def test_normalize_answer(raw: str, expected: str) -> None:
    assert normalize_answer(raw) == expected


def test_exact_match() -> None:
    assert exact_match(" down", "down")
    assert exact_match("Neutral.", "neutral")
    assert not exact_match(" down ", "up")
    assert not exact_match("down down", "down")
    assert exact_match("", "the")  # both normalize to the empty string
    assert not exact_match("none", "")


def test_exact_match_array_broadcasts_reference_column() -> None:
    answers = np.array([[" up", "down"], ["None", " none."]])
    refs = np.array(["up", "none"])[:, None]
    got = exact_match_array(answers, refs)
    assert got.dtype == bool
    np.testing.assert_array_equal(got, [[True, False], [True, True]])


def test_exact_match_array_rejects_1d_references() -> None:
    # Square table: a 1-D reference list would broadcast along the LLM axis and silently be wrong.
    answers = np.array([["up", "down"], ["x", "y"]])
    with pytest.raises(ValueError, match="dims"):
        exact_match_array(answers, np.array(["up", "x"]))
    got = exact_match_array(answers, np.array(["up", "x"])[:, None])
    np.testing.assert_array_equal(got, [[True, False], [True, False]])


@pytest.mark.parametrize(
    ("prediction", "reference"),
    [
        # Real reference answers from the cached datasets, checked against the official normalizer.
        (" AD 60", "AD\xa060"),  # COQA: non-breaking space in the reference
        (" yes", "Yes."),
        ("Unknown", "unknown"),
        (" neutral", "neutral"),
    ],
)
def test_exact_match_on_real_dataset_answers(prediction: str, reference: str) -> None:
    assert exact_match(prediction, reference)


def test_non_string_input_raises() -> None:
    with pytest.raises(AttributeError):
        normalize_answer(None)  # type: ignore[arg-type]


def test_exact_match_array_empty() -> None:
    assert exact_match_array(np.empty((0, 3), dtype=str), np.empty((0, 1), dtype=str)).shape == (
        0,
        3,
    )
