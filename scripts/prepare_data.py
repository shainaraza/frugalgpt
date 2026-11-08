"""Download a dataset's cached LLM responses, build the tables, and check them against the authors.

Usage:
    uv run python scripts/prepare_data.py --dataset HEADLINES

Writes ``data/tables/<DATASET>_{train,test}.npz`` and prints per-LLM accuracy / cost. If the
official per-model summary is available (``--official``), it also prints the difference to the
authors' numbers.
No paid API is called: the files are public GitHub release assets.
"""

import argparse
from pathlib import Path

import pandas as pd

from frugalgpt.data import (
    SPLITS,
    build_tables,
    download_dataset,
    load_responses,
    save_tables,
    summarize,
)

OFFICIAL = {
    "HEADLINES": "reference/FrugalGPT/summary/headlines_e8_perf_seed_9_2024updatedmodel.csv",
}
# Short names in the official summary that differ from ours.
OFFICIAL_ALIASES = {"CHATGPT": "ChatGPT"}


def compare_to_official(train_summary: pd.DataFrame, official_csv: Path) -> pd.DataFrame:
    """Join our train-split numbers with the authors' ``Train_acc`` / ``Train_cost`` (total USD)."""
    off = pd.read_csv(official_csv)
    off["name"] = off["Method"].replace(OFFICIAL_ALIASES)
    off = off.drop_duplicates("name").set_index("name")[["Train_acc", "Train_cost"]]
    out = train_summary.set_index("name")[["accuracy", "total_cost"]].join(off, how="inner")
    out["acc_diff"] = out["accuracy"] - out["Train_acc"]
    out["cost_rel_diff"] = out["total_cost"] / out["Train_cost"] - 1
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default="HEADLINES")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/tables"))
    parser.add_argument("--official", type=Path, default=None)
    args = parser.parse_args()

    paths = download_dataset(args.dataset, root=args.raw_dir)
    summaries = {}
    for split, path in zip(SPLITS, paths, strict=True):
        tables = build_tables(load_responses(path))
        save_tables(tables, args.out_dir / f"{args.dataset}_{split}.npz")
        summaries[split] = summarize(tables)
        print(f"\n{args.dataset} {split}: {tables.n_queries} queries × {tables.n_llms} LLMs")
        print(summaries[split].to_string(index=False, float_format="%.6g"))

    official = args.official or (Path(OFFICIAL[args.dataset]) if args.dataset in OFFICIAL else None)
    if official is not None and official.exists():
        cmp = compare_to_official(summaries["train"], official)
        print(f"\nTrain split vs official summary ({official}):")
        print(cmp.to_string(float_format="%.5g"))
        print(
            f"max |acc diff| = {cmp['acc_diff'].abs().max():.4f}, "
            f"max |cost rel diff| = {cmp['cost_rel_diff'].abs().max():.4%}"
        )


if __name__ == "__main__":
    main()
