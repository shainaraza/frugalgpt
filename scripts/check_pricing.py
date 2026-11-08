"""Recompute every cached cost from token counts and compare with the release's ``cost`` column.

Usage:
    uv run python scripts/check_pricing.py --dataset HEADLINES

Needs ``data/tables/<DATASET>_<split>.npz`` (from ``scripts/prepare_data.py``). No API is called.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from frugalgpt.data import SHORT_NAMES, SPLITS, load_tables
from frugalgpt.pricing import recompute_costs


def compare(cached: np.ndarray, ours: np.ndarray, llms: tuple[str, ...]) -> pd.DataFrame:
    """Per-LLM share of exact matches and relative error of our costs vs. the cached ones."""
    rel = np.abs(ours - cached) / cached
    return pd.DataFrame(
        {
            "name": [SHORT_NAMES.get(m, m) for m in llms],
            "exact_share": np.isclose(ours, cached, rtol=1e-9, atol=1e-12).mean(axis=0),
            "median_rel_err": np.median(rel, axis=0),
            "max_rel_err": rel.max(axis=0),
            "total_rel_diff": ours.sum(axis=0) / cached.sum(axis=0) - 1,
        },
        index=list(llms),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="HEADLINES")
    parser.add_argument("--tables", type=Path, default=Path("data/tables"))
    args = parser.parse_args()

    pd.set_option("display.width", 140)
    for split in SPLITS:
        t = load_tables(args.tables / f"{args.dataset}_{split}.npz")
        ours = recompute_costs(t.queries, t.answers, t.llms)
        print(f"\n{args.dataset} {split} ({t.n_queries} queries)")
        print(compare(t.cost, ours, t.llms).to_string(float_format=lambda x: f"{x:.5f}"))


if __name__ == "__main__":
    main()
