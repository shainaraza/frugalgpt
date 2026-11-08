from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from frugalgpt.data import (
    Tables,
    build_tables,
    load_responses,
    load_tables,
    release_url,
    save_tables,
    summarize,
)


def _long_df() -> pd.DataFrame:
    """3 queries × 2 LLMs, in the release's long format, rows deliberately shuffled."""
    rows = [
        # query, ref, model, answer, cost
        ("q1", "up", "a/cheap", " up", 0.1),
        ("q2", "down", "b/big", " down", 2.0),
        ("q1", "up", "b/big", " Up.", 1.0),
        ("q3", "none", "a/cheap", " None", 0.3),
        ("q2", "down", "a/cheap", " up", 0.2),
        ("q3", "none", "b/big", " neutral", 3.0),
    ]
    return pd.DataFrame(rows, columns=["query", "ref_answer", "model name", "answer", "cost"])


def test_build_tables_by_hand() -> None:
    t = build_tables(_long_df())
    assert t.llms == ("a/cheap", "b/big")  # order of first appearance
    np.testing.assert_array_equal(t.queries, ["q1", "q2", "q3"])
    np.testing.assert_array_equal(t.ref_answers, ["up", "down", "none"])
    np.testing.assert_array_equal(
        t.answers, [[" up", " Up."], [" up", " down"], [" None", " neutral"]]
    )
    np.testing.assert_array_equal(t.correct, [[True, True], [False, True], [True, False]])
    np.testing.assert_allclose(t.cost, [[0.1, 1.0], [0.2, 2.0], [0.3, 3.0]])
    assert (t.n_queries, t.n_llms) == (3, 2)


def test_summarize_by_hand() -> None:
    s = summarize(build_tables(_long_df()))
    np.testing.assert_allclose(s["accuracy"], [2 / 3, 2 / 3])
    np.testing.assert_allclose(s["mean_cost"], [0.2, 2.0])
    np.testing.assert_allclose(s["total_cost"], [0.6, 6.0])


def test_build_tables_rejects_missing_pair() -> None:
    with pytest.raises(ValueError, match="missing"):
        build_tables(_long_df().iloc[:-1])


def test_build_tables_keeps_repeated_prompts_as_separate_queries() -> None:
    # COQA-style: the same prompt twice per LLM, with different reference answers; each LLM
    # answers both copies identically (responses were cached by prompt).
    df = _long_df()
    extra = pd.DataFrame(
        [("q1", "down", "a/cheap", " up", 0.1), ("q1", "down", "b/big", " Up.", 1.0)],
        columns=df.columns,
    )
    t = build_tables(pd.concat([df, extra], ignore_index=True))
    np.testing.assert_array_equal(t.queries, ["q1", "q2", "q3", "q1"])
    np.testing.assert_array_equal(t.ref_answers, ["up", "down", "none", "down"])
    np.testing.assert_array_equal(t.correct[3], [False, False])
    np.testing.assert_allclose(t.cost[3], [0.1, 1.0])


def test_build_tables_rejects_unmatched_copy() -> None:
    df = _long_df()  # a second copy of q1 for one LLM only
    with pytest.raises(ValueError, match="missing"):
        build_tables(pd.concat([df, df.iloc[:1]]))


def test_build_tables_rejects_conflicting_reference() -> None:
    df = _long_df()
    df.loc[0, "ref_answer"] = "down"
    with pytest.raises(ValueError, match="reference"):
        build_tables(df)


def test_build_tables_rejects_empty() -> None:
    with pytest.raises(ValueError, match="no responses"):
        build_tables(_long_df().iloc[:0])


def test_load_responses_keeps_none_like_strings(tmp_path: Path) -> None:
    df = _long_df()
    df.loc[3, "answer"] = "None"
    df.loc[5, "answer"] = ""
    path = tmp_path / "x.csv"
    df.to_csv(path)  # writes an index column like the release files
    back = load_responses(path)
    assert back.loc[3, "answer"] == "None"
    assert back.loc[5, "answer"] == ""
    assert build_tables(back).correct[2].tolist() == [True, False]


def test_load_responses_rejects_missing_column(tmp_path: Path) -> None:
    path = tmp_path / "x.csv"
    _long_df().drop(columns="cost").to_csv(path)
    with pytest.raises(ValueError, match="cost"):
        load_responses(path)


def test_save_load_round_trip(tmp_path: Path) -> None:
    t = build_tables(_long_df())
    save_tables(t, tmp_path / "sub" / "t.npz")
    back = load_tables(tmp_path / "sub" / "t.npz")
    assert isinstance(back, Tables)
    assert back.llms == t.llms
    for field in ("queries", "ref_answers", "answers", "correct", "cost"):
        np.testing.assert_array_equal(getattr(back, field), getattr(t, field))


def test_release_url() -> None:
    assert release_url("HEADLINES", "test").endswith("/0.0.1/HEADLINES_test.csv")
    with pytest.raises(ValueError):
        release_url("NOPE", "test")
    with pytest.raises(ValueError):
        release_url("HEADLINES", "dev")


def test_load_responses_keeps_numeric_text_columns_as_strings(tmp_path) -> None:
    from frugalgpt.data import load_responses

    path = tmp_path / "num.csv"
    path.write_text(
        ",query,ref_answer,model name,answer,cost\n0,q1,2,m,2,0.1\n1,q2,3.50,m,3.50,0.2\n"
    )
    df = load_responses(path)
    assert df["ref_answer"].tolist() == ["2", "3.50"]
    assert df["answer"].tolist() == ["2", "3.50"]
    assert df["cost"].dtype == float
