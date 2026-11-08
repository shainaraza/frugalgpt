"""Sec. 4: the generation scorer ``g_i(q, a)``, one small classifier per LLM.

A DistilBERT encoder with a 2-way head reads ``query + answer`` and predicts whether the answer
is correct. The cascade accepts LLM ``i``'s answer when its score is above ``τ_i``.

Matches the official ``scoring.py`` / ``llmcascade.py`` (see docs/paper_vs_code.md §1):
- input text = last ``Q: … A:`` block of the prompt + ``" " + answer`` (few-shot examples dropped);
- 55% of the scorer's data held out for validation; the epoch with the lowest validation loss is
  kept (HF ``load_best_model_at_end`` without ``metric_for_best_model``);
- AdamW, lr 5e-5, linear warmup (500 steps) then linear decay, weight decay 0.01 except on bias /
  LayerNorm, gradient clipping at 1.0, 8 epochs, batch 8, max length 512;
- score = ``softmax[1]`` = P(correct); the paper text's ``max(softmax)`` is
  ``output="max_softmax"``.

We use a plain PyTorch loop instead of HF ``Trainer``, and pad per batch rather than to the
longest text in the whole dataset (same outputs thanks to the attention mask).
"""

import copy
import hashlib
import json
import math
import os
from collections.abc import Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
import yaml

ScoreOutput = Literal["p_correct", "max_softmax"]


@dataclass(frozen=True)
class ScorerConfig:
    """Scorer hyperparameters (``configs/scorer.yaml``)."""

    backbone: str = "distilbert-base-uncased"
    output: ScoreOutput = "p_correct"
    epochs: int = 8
    batch_size: int = 8
    eval_batch_size: int = 64
    learning_rate: float = 5e-5
    warmup_steps: int = 500
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    max_length: int = 512
    val_size: float = 0.55
    seed: int = 2024

    def __post_init__(self) -> None:
        if self.output not in ("p_correct", "max_softmax"):
            raise ValueError(f"output must be 'p_correct' or 'max_softmax', got {self.output!r}")
        if not 0.0 <= self.val_size < 1.0:
            raise ValueError(f"val_size must be in [0, 1), got {self.val_size}")
        if self.epochs < 1 or self.batch_size < 1 or self.eval_batch_size < 1:
            raise ValueError("epochs and batch sizes must be ≥ 1")


def load_scorer_config(path: Path | str) -> ScorerConfig:
    """Sec. 4 scorer settings from YAML (``configs/scorer.yaml``); unknown keys are an error."""
    raw = yaml.safe_load(Path(path).read_text()) or {}
    known = {f.name for f in fields(ScorerConfig)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"unknown scorer config keys: {sorted(unknown)}")
    return ScorerConfig(**raw)


def scorer_text(query: str, answer: str, prefix: str = "") -> str:
    """Sec. 4 scorer input: the last ``Q: … A:`` block of the prompt followed by the answer.

    Official rule (``llmcascade.py: scorer_text``): join ``query + " " + answer``, drop ``prefix``,
    keep the text after the last ``"Q:"`` and put ``"Q:"`` back in front. A prompt without any
    ``"Q:"`` is kept whole, still with ``"Q:"`` prepended. ``prefix`` is the dataset's few-shot
    block (:func:`prompt_prefix`), which the official notebooks strip this way; it matters for
    datasets whose prompts do not use ``"Q:"`` (OVERRULING, COQA, SCIQ).
    """
    # Why strip the few-shot block: it is identical for every query, so it carries no signal,
    # and it eats the 512-token window. On COQA / SCIQ the block alone exceeds 512 tokens, so
    # without stripping every truncated input would be the same string and the scorer useless.
    text = (query + " " + answer).removeprefix(prefix)
    return "Q:" + text.split("Q:")[-1]


def prompt_prefix(queries: Sequence[str]) -> str:
    """Sec. 4 scorer input: the few-shot block shared by every prompt of a dataset.

    Longest common start of ``queries``, cut back to its last blank line (``"\n\n"``), so the
    query's own block is never eaten. The official notebooks strip a prefix read from
    ``config/prompt/<DATASET>/prefix_eN.txt``; those 2024 files differ slightly from the 2023
    cached prompts (AGNEWS, COQA, SCIQ), so we derive the exact block from the prompts instead.
    It equals the official file on HEADLINES and OVERRULING. Empty if there is no shared block.
    """
    qs = list(queries)
    if not qs:
        return ""
    # Character-level common start can end mid-way into the query's own text (e.g. two queries
    # that both begin "Q: What"), so cut back to the last blank line, which ends a few-shot block.
    common = os.path.commonprefix(qs)
    cut = common.rfind("\n\n")
    return common[: cut + 2] if cut >= 0 else ""


def pick_device() -> torch.device:
    """Device for scorer training / inference (Sec. 4); order from AGENTS.md: ``mps`` → ``cpu``."""
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def split_indices(n: int, val_size: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Sec. 4 scorer train / validation split of ``range(n)``; ``ceil(val_size · n)`` held out.

    Same sizes as sklearn's ``train_test_split(test_size=val_size)``. The official split is
    deterministic (global ``set_seed(2024)``) but depends on call order and row order, so it
    cannot be reproduced bit-for-bit; ours is seeded directly and identical for every LLM.
    """
    if n < 1:
        raise ValueError("need at least one example")
    n_val = math.ceil(val_size * n)
    if n_val >= n:
        raise ValueError(f"val_size {val_size} leaves no training examples out of {n}")
    perm = np.random.default_rng(seed).permutation(n)
    return np.sort(perm[n_val:]), np.sort(perm[:n_val])


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Scorer quality (Sec. 4, reported per LLM): area under the ROC curve.

    Mann-Whitney U with average ranks for ties.

    Probability that a random correct answer scores higher than a random wrong one (ties count
    half). ``nan`` when only one class is present.
    """
    y = np.asarray(labels).astype(bool)
    s = np.asarray(scores, dtype=np.float64)
    if y.shape != s.shape or y.ndim != 1:
        raise ValueError("labels and scores must be 1-D arrays of the same length")
    if np.isnan(s).any():
        raise ValueError("scores must not contain NaN")
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    # AUC = how well the score *ranks* right answers above wrong ones (threshold-free). This is
    # what the cascade needs: it only compares scores to a threshold, never uses their scale.
    order = np.argsort(s, kind="mergesort")
    sorted_s = s[order]
    ranks = np.empty(len(s), dtype=np.float64)
    # average rank (1-based) within each block of tied scores
    _, start, counts = np.unique(sorted_s, return_index=True, return_counts=True)
    avg = start + (counts + 1) / 2.0
    ranks[order] = np.repeat(avg, counts)
    u = ranks[y].sum() - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def load_backbone(cfg: ScorerConfig):
    """Sec. 4 scorer backbone: pretrained encoder + fresh 2-way head, and its tokenizer.

    Downloads the weights on first use (``~/.cache/huggingface``).
    """
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.manual_seed(cfg.seed)  # the new head's initial weights
    tokenizer = AutoTokenizer.from_pretrained(cfg.backbone)
    model = AutoModelForSequenceClassification.from_pretrained(cfg.backbone, num_labels=2)
    return model, tokenizer


def _batches(n: int, size: int, perm: np.ndarray | None = None):
    idx = np.arange(n) if perm is None else perm
    for start in range(0, n, size):
        yield idx[start : start + size]


def _encode(tokenizer, texts: Sequence[str], max_length: int, device: torch.device) -> dict:
    enc = tokenizer(
        list(texts), truncation=True, padding=True, max_length=max_length, return_tensors="pt"
    )
    return {k: v.to(device) for k, v in enc.items()}


def _param_groups(model: torch.nn.Module, weight_decay: float) -> list[dict]:
    """HF Trainer rule: no weight decay on biases and LayerNorm weights."""
    norm_params = {
        id(p)
        for m in model.modules()
        if isinstance(m, torch.nn.LayerNorm)
        for p in m.parameters(recurse=False)
    }
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if name.endswith("bias") or id(p) in norm_params else decay).append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def make_optimizer(model: torch.nn.Module, cfg: ScorerConfig, total_steps: int):
    """Sec. 4 scorer optimizer (HF Trainer defaults): AdamW + linear warmup then decay to 0."""
    from transformers import get_linear_schedule_with_warmup

    optimizer = torch.optim.AdamW(
        _param_groups(model, cfg.weight_decay), lr=cfg.learning_rate, betas=(0.9, 0.999), eps=1e-8
    )
    scheduler = get_linear_schedule_with_warmup(optimizer, cfg.warmup_steps, total_steps)
    return optimizer, scheduler


@torch.no_grad()
def predict_logits(
    model: torch.nn.Module,
    tokenizer,
    texts: Sequence[str],
    batch_size: int = 64,
    max_length: int = 512,
    device: torch.device | None = None,
) -> np.ndarray:
    """Sec. 4 scorer forward pass: raw 2-way logits ``(n, 2)`` per text, in eval mode.

    Texts longer than ``max_length`` tokens are truncated on the right (as in the official
    code), so on long prompts the answer itself can be cut off.
    """
    device = device or next(model.parameters()).device
    model.eval()
    out = []
    for b in _batches(len(texts), batch_size):
        enc = _encode(tokenizer, [texts[i] for i in b], max_length, device)
        out.append(model(**enc).logits.float().cpu())
    if not out:
        return np.zeros((0, 2), dtype=np.float32)
    return torch.cat(out).numpy()


def logits_to_score(logits: np.ndarray, output: ScoreOutput = "p_correct") -> np.ndarray:
    """Sec. 4 score from logits: ``softmax[1]`` (official) or ``max(softmax)`` (paper text)."""
    # p_correct = P(label 1 = "answer is correct"): monotone in confidence that the answer is
    # right. max_softmax (paper wording) is high for confident *wrong* predictions too, so it
    # cannot tell "surely right" from "surely wrong"; the official code uses p_correct.
    probs = torch.softmax(torch.as_tensor(logits, dtype=torch.float64), dim=-1).numpy()
    if output == "p_correct":
        return probs[:, 1]
    if output == "max_softmax":
        return probs.max(axis=1)
    raise ValueError(f"unknown output {output!r}")


def score_texts(
    model: torch.nn.Module,
    tokenizer,
    texts: Sequence[str],
    cfg: ScorerConfig,
    device: torch.device | None = None,
) -> np.ndarray:
    """Sec. 4: ``g_i(q, a)`` for each prepared scorer text, shape ``(n,)``."""
    logits = predict_logits(model, tokenizer, texts, cfg.eval_batch_size, cfg.max_length, device)
    return logits_to_score(logits, cfg.output)


@dataclass
class EpochLog:
    """Validation metrics after one epoch (val_* are ``nan`` without a validation set)."""

    epoch: int
    train_loss: float
    val_loss: float
    val_accuracy: float


@dataclass
class TrainResult:
    """Training history and the kept epoch (1-based)."""

    history: list[EpochLog]
    best_epoch: int
    n_train: int
    n_val: int

    def as_dict(self) -> dict:
        """JSON-ready copy (stored next to the scores)."""
        return {**asdict(self), "history": [asdict(h) for h in self.history]}


def train_scorer(
    model: torch.nn.Module,
    tokenizer,
    texts: Sequence[str],
    labels: np.ndarray,
    cfg: ScorerConfig,
    device: torch.device | None = None,
    log=None,
) -> TrainResult:
    """Sec. 4: fine-tune one scorer on (scorer text, is-correct) pairs; keeps the best epoch.

    ``model`` is trained in place and ends with the weights of the epoch with the lowest
    validation loss (first one on ties). With ``val_size = 0`` the last epoch is kept.
    """
    y = np.asarray(labels)
    if y.shape != (len(texts),):
        raise ValueError(f"labels must have shape ({len(texts)},), got {y.shape}")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("labels must be 0/1 or bool")
    y = y.astype(np.int64)

    device = device or pick_device()
    model.to(device)
    tr_idx, va_idx = (
        split_indices(len(texts), cfg.val_size, cfg.seed)
        if cfg.val_size > 0
        else (np.arange(len(texts)), np.array([], dtype=int))
    )
    tr_texts = [texts[i] for i in tr_idx]
    va_texts = [texts[i] for i in va_idx]

    steps_per_epoch = math.ceil(len(tr_idx) / cfg.batch_size)
    optimizer, scheduler = make_optimizer(model, cfg, steps_per_epoch * cfg.epochs)
    gen = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)  # dropout

    # Early stopping by checkpoint: train all epochs, remember the weights of the epoch with the
    # lowest validation loss, restore them at the end. The scorer overfits fast (best epoch is
    # usually 2-3 of 8), so this matters.
    history: list[EpochLog] = []
    best_loss, best_epoch, best_state = math.inf, cfg.epochs, None
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total, seen = 0.0, 0
        for b in _batches(len(tr_idx), cfg.batch_size, gen.permutation(len(tr_idx))):
            enc = _encode(tokenizer, [tr_texts[i] for i in b], cfg.max_length, device)
            target = torch.as_tensor(y[tr_idx[b]], device=device)
            loss = F.cross_entropy(model(**enc).logits, target)
            optimizer.zero_grad()
            loss.backward()
            # Standard BERT fine-tuning recipe: clip gradients, then step the warmup/decay LR.
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()
            scheduler.step()
            total += loss.item() * len(b)
            seen += len(b)

        val_loss = val_acc = float("nan")
        if len(va_idx):
            logits = predict_logits(
                model, tokenizer, va_texts, cfg.eval_batch_size, cfg.max_length, device
            )
            t = torch.as_tensor(y[va_idx])
            val_loss = float(F.cross_entropy(torch.as_tensor(logits), t))
            val_acc = float((logits.argmax(axis=1) == y[va_idx]).mean())
            if val_loss < best_loss:
                best_loss, best_epoch = val_loss, epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        history.append(EpochLog(epoch, total / max(seen, 1), val_loss, val_acc))
        if log:
            log(history[-1])

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return TrainResult(history, best_epoch, len(tr_idx), len(va_idx))


def fresh_copy(model: torch.nn.Module) -> torch.nn.Module:
    """Sec. 4, one scorer per LLM: deep copy of the untrained backbone, so all start alike."""
    return copy.deepcopy(model)


def scores_fingerprint(
    queries: dict[str, Sequence[str]], answers: Sequence[str], cfg: ScorerConfig, prefix: str = ""
) -> str:
    """Identity of one LLM's saved scores: queries per split, its answers, config, prompt prefix.

    Saved scores are reused only if this matches, so regenerated tables (other split, other row
    order) or a changed config can never be combined with stale scores.
    """
    h = hashlib.sha256()
    for split in sorted(queries):
        h.update(split.encode())
        for q in queries[split]:
            h.update(q.encode() + b"\x00")
    for a in answers:
        h.update(str(a).encode() + b"\x00")
    h.update(json.dumps(asdict(cfg), sort_keys=True).encode())
    h.update(b"prefix\x00" + prefix.encode())
    return h.hexdigest()


def queries_hash(queries: Sequence[str]) -> str:
    """Row identity of one split's ``(n, k)`` score table (Sec. 4 offline-first design).

    Stored next to combined scores and checked on load, so scores can never be paired with
    the rows of rebuilt tables (other row count, order or text).
    """
    h = hashlib.sha256()
    for q in queries:
        h.update(q.encode() + b"\x00")
    return h.hexdigest()


def prefix_coverage(queries: Sequence[str], prefix: str) -> float:
    """Sec. 4 scorer input check: share of ``queries`` that start with the stripped ``prefix``.

    ``scorer_text`` silently keeps a prompt that does not start with the prefix, so callers
    should require 1.0. An empty prefix covers everything.
    """
    qs = list(queries)
    if not qs or not prefix:
        return 1.0
    return float(np.mean([q.startswith(prefix) for q in qs]))


def combine_scores(per_llm: dict[str, np.ndarray], llms: Sequence[str], n: int) -> np.ndarray:
    """Stack per-LLM score vectors into the ``(n, k)`` table used by the cascade (Sec. 4).

    Columns follow ``llms`` (the data-table order); LLMs without scores stay NaN.
    """
    table = np.full((n, len(llms)), np.nan)
    for name, v in per_llm.items():
        if name not in llms:
            raise ValueError(f"unknown LLM {name!r}")
        v = np.asarray(v, dtype=np.float64)
        if v.shape != (n,):
            raise ValueError(f"scores for {name!r} have shape {v.shape}, expected ({n},)")
        table[:, list(llms).index(name)] = v
    return table
