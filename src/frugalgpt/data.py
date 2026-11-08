"""Cached LLM responses → ``correct`` / ``cost`` tables (Sec. 5 experimental setup).

The authors released every LLM's answer and cost for each query of each dataset as CSVs on the
FrugalGPT GitHub release ``0.0.1``. Each CSV is in *long* format: one row per (query, LLM) pair,
with columns ``query, ref_answer, model name, answer, cost, …``.

We pivot that into *wide* ``n_queries × n_llms`` tables. All later steps (cascade simulation,
optimizer, baselines) work on these tables with numpy only.
"""

import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from frugalgpt.metrics import exact_match_array

RELEASE_URL = "https://github.com/stanford-futuredata/FrugalGPT/releases/download/0.0.1"
DATASETS = ("HEADLINES", "OVERRULING", "COQA", "AGNEWS", "SCIQ")
SPLITS = ("train", "test")

# Provider/model id in the release CSVs → short name used in the paper's figures (Table 1).
SHORT_NAMES = {
    "textsynth/gptj_6B": "GPT-J",
    "textsynth/fairseq_gpt_13B": "GPT-FSQ",
    "textsynth/gptneox_20B": "GPT-Neo",
    "ai21/j1-large": "J1-L",
    "ai21/j1-grande": "J1-G",
    "ai21/j1-jumbo": "J1",
    "cohere/medium": "CoHere-M",
    "cohere/xlarge": "CoHere",
    "openai/text-curie-001": "GPT-C",
    "openai/text-davinci-002": "GPT-3",
    "openaichat/gpt-3.5-turbo": "ChatGPT",
    "openaichat/gpt-4": "GPT-4",
}


def release_url(dataset: str, split: str) -> str:
    """URL of the authors' cached-response CSV for one dataset split (Sec. 5 data)."""
    _check_name(dataset, split)
    return f"{RELEASE_URL}/{dataset}_{split}.csv"


def download_dataset(
    dataset: str, root: Path = Path("data/raw"), splits: tuple[str, ...] = SPLITS
) -> list[Path]:
    """Download the cached-response CSVs into ``root/<dataset>/``. Existing files are kept."""
    paths = []
    for split in splits:
        path = Path(root) / dataset / f"{dataset}_{split}.csv"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".part")
            urllib.request.urlretrieve(release_url(dataset, split), tmp)
            tmp.rename(path)
        paths.append(path)
    return paths


def load_responses(path: Path) -> pd.DataFrame:
    """Read one long-format CSV of cached responses.

    ``keep_default_na=False`` keeps answers such as ``"None"`` or ``""`` as strings instead of
    NaN, and ``dtype=str`` on the text columns stops pandas reading ``"2"`` as ``2.0``.
    """
    text_cols = {"query": str, "ref_answer": str, "model name": str, "answer": str}
    df = pd.read_csv(path, keep_default_na=False, index_col=0, dtype=text_cols)
    missing = {"query", "ref_answer", "model name", "answer", "cost"} - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    return df


@dataclass(frozen=True)
class Tables:
    """Wide per-dataset tables. Row ``n`` = query, column ``k`` = LLM ``llms[k]``.

    ``correct[n, k]`` is the reward r (exact match); ``cost[n, k]`` is ``c_k(p_n)`` in USD (Sec. 3).
    """

    queries: np.ndarray  # (n,) str — full prompt sent to the LLM
    ref_answers: np.ndarray  # (n,) str
    llms: tuple[str, ...]  # (k,)
    answers: np.ndarray  # (n, k) str
    correct: np.ndarray  # (n, k) bool
    cost: np.ndarray  # (n, k) float64

    @property
    def n_queries(self) -> int:
        return len(self.queries)

    @property
    def n_llms(self) -> int:
        return len(self.llms)


def build_tables(df: pd.DataFrame) -> Tables:
    """Pivot long-format responses into ``n_queries × n_llms`` tables (Sec. 5 setup).

    Queries and LLMs keep their order of first appearance. A prompt that appears several times
    for the same LLM (COQA, SCIQ: distinct dataset items with identical text, sometimes with
    different reference answers) is kept as several queries: its ``j``-th copy for one LLM is
    matched with its ``j``-th copy for every other LLM, as the official code evaluates each row
    separately. Every (query, copy) must appear once per LLM with a single reference answer.
    """
    if df.empty:
        raise ValueError("no responses")
    # Long → wide: the CSV has one row per (query, LLM). The "copy" counter (0, 1, 2 … per
    # repeated prompt) makes (query, copy) a unique key, so identical prompt texts that are
    # different dataset items stay separate rows instead of colliding in the pivot.
    df = df.assign(copy=df.groupby(["query", "model name"]).cumcount())
    refs = df.groupby(["query", "copy"], sort=False)["ref_answer"].unique()
    if (refs.map(len) != 1).any():
        raise ValueError("a query has more than one reference answer")

    keys = refs.index
    llms = tuple(df["model name"].unique())
    wide = df.pivot(index=["query", "copy"], columns="model name", values=["answer", "cost"])
    wide = wide.reindex(index=keys)
    answers = wide["answer"][list(llms)]
    if answers.isna().to_numpy().any():
        raise ValueError("some (query, model) pairs are missing")

    queries = keys.get_level_values("query").to_numpy(dtype=str)
    answers = answers.to_numpy(dtype=str)
    ref_answers = refs.map(lambda r: r[0]).to_numpy(dtype=str)
    return Tables(
        queries=queries,
        ref_answers=ref_answers,
        llms=llms,
        answers=answers,
        correct=exact_match_array(answers, ref_answers[:, None]),
        cost=wide["cost"][list(llms)].to_numpy(dtype=np.float64),
    )


def save_tables(tables: Tables, path: Path) -> None:
    """Save tables to a single ``.npz`` file (no pickling)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        queries=tables.queries,
        ref_answers=tables.ref_answers,
        llms=np.array(json.dumps(list(tables.llms))),
        answers=tables.answers,
        correct=tables.correct,
        cost=tables.cost,
    )


def load_tables(path: Path) -> Tables:
    """Load tables written by :func:`save_tables`."""
    with np.load(path, allow_pickle=False) as f:
        return Tables(
            queries=f["queries"],
            ref_answers=f["ref_answers"],
            llms=tuple(json.loads(str(f["llms"]))),
            answers=f["answers"],
            correct=f["correct"],
            cost=f["cost"],
        )


def summarize(tables: Tables) -> pd.DataFrame:
    """Per-LLM accuracy, mean cost per query and total cost (USD): single-LLM points of Fig. 3."""
    return pd.DataFrame(
        {
            "llm": tables.llms,
            "name": [SHORT_NAMES.get(m, m) for m in tables.llms],
            "accuracy": tables.correct.mean(axis=0),
            "mean_cost": tables.cost.mean(axis=0),
            "total_cost": tables.cost.sum(axis=0),
        }
    )


def _check_name(dataset: str, split: str) -> None:
    if dataset not in DATASETS:
        raise ValueError(f"unknown dataset {dataset!r}; expected one of {DATASETS}")
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
