"""Pre-training checks for one dataset (step 10): size, LLM pool, prompt format, truncation.

    uv run python scripts/check_dataset.py --dataset COQA

Reports, per split: number of queries and LLMs, the share of prompts containing ``"Q:"`` (the
scorer keeps only the text after the last one), what the scorer text looks like, and the share of
scorer texts longer than the scorer's ``max_length`` in DistilBERT tokens.
"""

import argparse
from pathlib import Path

import numpy as np

from frugalgpt.data import SHORT_NAMES, load_tables
from frugalgpt.scorer import load_scorer_config, prompt_prefix, scorer_text

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--sample", type=int, default=2000, help="queries per split to tokenize")
    args = ap.parse_args()
    cfg = load_scorer_config(ROOT / "configs" / "scorer.yaml")

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg.backbone)
    rng = np.random.default_rng(cfg.seed)
    prefix = prompt_prefix(
        load_tables(ROOT / "data" / "tables" / f"{args.dataset}_train.npz").queries
    )
    print(f"shared few-shot prefix (from train prompts): {len(prefix)} chars")
    train_t = load_tables(ROOT / "data" / "tables" / f"{args.dataset}_train.npz")
    k_best = int(np.argmax(train_t.correct.mean(axis=0)))
    for split in ("train", "test"):
        t = load_tables(ROOT / "data" / "tables" / f"{args.dataset}_{split}.npz")
        print(f"\n== {args.dataset} {split}: {t.n_queries} queries × {t.n_llms} LLMs")
        print("LLMs:", ", ".join(SHORT_NAMES.get(m, m) for m in t.llms))
        has_q = np.array(["Q:" in q for q in t.queries])
        print(f"prompts containing 'Q:': {has_q.mean():.1%}")
        if prefix:
            share = np.mean([q.startswith(prefix) for q in t.queries])
            print(f"prompts starting with the prefix: {share:.1%}")
        idx = rng.choice(t.n_queries, min(args.sample, t.n_queries), replace=False)
        k = k_best  # most accurate LLM on train (no choice made by looking at test)
        texts = [scorer_text(t.queries[i], t.answers[i, k], prefix) for i in idx]
        lens = np.array([len(tok(x)["input_ids"]) for x in texts])
        full = np.array([len(tok(t.queries[i])["input_ids"]) for i in idx[:200]])
        print(
            f"scorer text tokens: median {np.median(lens):.0f}, p95 {np.percentile(lens, 95):.0f}, "
            f"max {lens.max()}; over {cfg.max_length}: {(lens > cfg.max_length).mean():.1%}"
        )
        print(f"full prompt tokens (200 sampled): median {np.median(full):.0f}")
        if split == "train":
            print("example scorer text:", repr(texts[0][:400]))
            print("reference answer:   ", repr(t.ref_answers[idx[0]]))


if __name__ == "__main__":
    main()
