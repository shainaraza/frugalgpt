"""Train one Sec. 4 scorer per LLM on a dataset's train split; score both splits.

Writes one file per LLM to ``data/scores/<DATASET>/<llm>.npz`` (so an interrupted run resumes),
then combines them into ``data/scores/<DATASET>_{train,test}.npz`` with an ``(n, k)`` ``score``
table in the same column order as the data tables. Prints per-LLM AUC on both splits.

    uv run python scripts/train_scorers.py --dataset HEADLINES
    uv run python scripts/train_scorers.py --dataset HEADLINES --llms textsynth/gptj_6B
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from frugalgpt.data import load_tables
from frugalgpt.scorer import (
    combine_scores,
    fresh_copy,
    load_backbone,
    load_scorer_config,
    pick_device,
    prefix_coverage,
    prompt_prefix,
    queries_hash,
    roc_auc,
    score_texts,
    scorer_text,
    scores_fingerprint,
    train_scorer,
)

ROOT = Path(__file__).resolve().parents[1]


def _safe(name: str) -> str:
    return name.replace("/", "__")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--dataset", default="HEADLINES")
    ap.add_argument("--config", default=str(ROOT / "configs" / "scorer.yaml"))
    ap.add_argument("--llms", nargs="*", help="subset of LLM names (default: all)")
    ap.add_argument("--out", default=str(ROOT / "data" / "scores"))
    ap.add_argument("--save-checkpoints", action="store_true", help="≈260 MB per LLM")
    ap.add_argument("--force", action="store_true", help="retrain LLMs that already have scores")
    args = ap.parse_args()

    cfg = load_scorer_config(args.config)
    tables = {
        s: load_tables(ROOT / "data" / "tables" / f"{args.dataset}_{s}.npz")
        for s in ("train", "test")
    }
    train, test = tables["train"], tables["test"]
    assert train.llms == test.llms
    llms = args.llms or list(train.llms)
    unknown = set(llms) - set(train.llms)
    if unknown:
        raise SystemExit(f"unknown LLMs: {sorted(unknown)}")

    out_dir = Path(args.out) / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    print(f"device={device} config={cfg}")
    queries = {s: t.queries for s, t in tables.items()}

    prefix = prompt_prefix(train.queries)  # few-shot block, stripped before scoring
    for s, t in tables.items():
        cover = prefix_coverage(t.queries, prefix)
        if cover < 1.0:
            raise SystemExit(f"only {cover:.1%} of {s} prompts start with the few-shot prefix")

    def fingerprint(k: int) -> str:
        answers = np.concatenate([train.answers[:, k], test.answers[:, k]])
        return scores_fingerprint(queries, answers, cfg, prefix)

    def is_fresh(path: Path, k: int) -> bool:
        return path.exists() and str(np.load(path).get("fingerprint", "")) == fingerprint(k)

    base_model, tokenizer = load_backbone(cfg)

    for name in llms:
        path = out_dir / f"{_safe(name)}.npz"
        k = train.llms.index(name)
        if is_fresh(path, k) and not args.force:
            print(f"[skip] {name}: {path.name} matches the current tables and config")
            continue
        if path.exists():
            print(f"[stale] {name}: retraining ({path.name} is from other tables or config)")
        texts = {
            s: [scorer_text(q, a, prefix) for q, a in zip(t.queries, t.answers[:, k], strict=True)]
            for s, t in tables.items()
        }
        t0 = time.time()
        model = fresh_copy(base_model)
        result = train_scorer(
            model,
            tokenizer,
            texts["train"],
            train.correct[:, k],
            cfg,
            device,
            log=lambda e, n=name: print(
                f"  {n} epoch {e.epoch}: train_loss={e.train_loss:.4f} "
                f"val_loss={e.val_loss:.4f} val_acc={e.val_accuracy:.4f}",
                flush=True,
            ),
        )
        scores = {s: score_texts(model, tokenizer, texts[s], cfg, device) for s in tables}
        auc = {s: roc_auc(tables[s].correct[:, k], scores[s]) for s in tables}
        secs = time.time() - t0
        np.savez(
            path,
            llm=name,
            fingerprint=fingerprint(k),
            train=scores["train"],
            test=scores["test"],
            info=json.dumps(
                {**result.as_dict(), "auc": auc, "seconds": secs, "config": cfg.__dict__}
            ),
        )
        print(
            f"[done] {name}: best_epoch={result.best_epoch} AUC train={auc['train']:.3f} "
            f"test={auc['test']:.3f} ({secs / 60:.1f} min)",
            flush=True,
        )
        if args.save_checkpoints:
            ckpt = ROOT / "checkpoints" / args.dataset / _safe(name)
            model.save_pretrained(ckpt)
            tokenizer.save_pretrained(ckpt)
        del model

    # Combine every up-to-date per-LLM file into (n, k) tables; missing / stale LLMs stay NaN.
    fresh = {
        name: out_dir / f"{_safe(name)}.npz"
        for k, name in enumerate(train.llms)
        if is_fresh(out_dir / f"{_safe(name)}.npz", k)
    }
    for s, t in tables.items():
        table = combine_scores({n: np.load(p)[s] for n, p in fresh.items()}, t.llms, t.n_queries)
        np.savez(
            Path(args.out) / f"{args.dataset}_{s}.npz",
            llms=np.array(t.llms),
            score=table,
            queries_hash=queries_hash(t.queries),
        )

    print(f"\n{'LLM':40s} {'acc(test)':>9s} {'AUC train':>9s} {'AUC test':>9s}")
    for k, name in enumerate(train.llms):
        if name in fresh:
            info = json.loads(str(np.load(fresh[name])["info"]))
            print(
                f"{name:40s} {test.correct[:, k].mean():9.4f} "
                f"{info['auc']['train']:9.3f} {info['auc']['test']:9.3f}"
            )


if __name__ == "__main__":
    main()
