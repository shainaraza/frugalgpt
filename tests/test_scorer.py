from pathlib import Path

import numpy as np
import pytest
import torch

from frugalgpt.scorer import (
    ScorerConfig,
    _param_groups,
    fresh_copy,
    load_scorer_config,
    logits_to_score,
    roc_auc,
    score_texts,
    scorer_text,
    split_indices,
    train_scorer,
)

CONFIG = Path(__file__).parents[1] / "configs" / "scorer.yaml"


# --- scorer text -------------------------------------------------------------------------------


def test_scorer_text_keeps_last_block() -> None:
    q = "Determine the direction.\n\nQ: gold up\nA: up\n\nQ: gold down\nA: down\n\nQ: gold flat\nA:"
    assert scorer_text(q, " neutral") == "Q: gold flat\nA:  neutral"


def test_scorer_text_without_q() -> None:
    assert scorer_text("no marker", "x") == "Q:no marker x"


def test_scorer_text_q_in_answer_and_prefix() -> None:
    # The official rule splits after joining, so a "Q:" inside the answer wins.
    assert scorer_text("Q: a\nA:", "b Q: c") == "Q: c"
    assert scorer_text("PRE Q: a\nA:", "b", prefix="PRE ") == "Q: a\nA: b"


# --- config ------------------------------------------------------------------------------------


def test_config_file_matches_defaults() -> None:
    assert load_scorer_config(CONFIG) == ScorerConfig()


def test_config_rejects_bad_values(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        ScorerConfig(output="logit")
    with pytest.raises(ValueError):
        ScorerConfig(val_size=1.0)
    p = tmp_path / "c.yaml"
    p.write_text("epochs: 2\nlr: 0.1\n")
    with pytest.raises(ValueError, match="unknown"):
        load_scorer_config(p)


# --- split, AUC, score output -----------------------------------------------------------------


def test_split_indices_sizes_and_seed() -> None:
    tr, va = split_indices(10, 0.55, 2024)
    assert len(va) == 6 and len(tr) == 4  # ceil(5.5) = 6, like sklearn
    assert sorted(np.concatenate([tr, va]).tolist()) == list(range(10))
    tr2, va2 = split_indices(10, 0.55, 2024)
    assert (tr == tr2).all() and (va == va2).all()
    with pytest.raises(ValueError):
        split_indices(1, 0.55, 0)


def test_roc_auc_by_hand() -> None:
    assert roc_auc(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.3, 0.4])) == 1.0
    assert roc_auc(np.array([1, 1, 0, 0]), np.array([0.1, 0.2, 0.3, 0.4])) == 0.0
    # pairs (pos, neg): (0.35>0.1) (0.35>0.4 no) (0.8>0.1) (0.8>0.4) → 3/4
    assert roc_auc(np.array([0, 1, 0, 1]), np.array([0.1, 0.35, 0.4, 0.8])) == 0.75
    # all tied → 0.5
    assert roc_auc(np.array([0, 1, 0, 1]), np.zeros(4)) == 0.5
    # one tie between a pos and a neg counts half: pairs → 1, 0.5, 1, 1 → 0.875
    assert roc_auc(np.array([0, 1, 0, 1]), np.array([0.1, 0.4, 0.4, 0.9])) == 0.875
    assert np.isnan(roc_auc(np.array([1, 1]), np.array([0.1, 0.2])))
    with pytest.raises(ValueError):
        roc_auc(np.array([0, 1]), np.array([np.nan, 0.1]))


def test_logits_to_score() -> None:
    logits = np.array([[0.0, 0.0], [0.0, np.log(3.0)], [np.log(3.0), 0.0]])
    np.testing.assert_allclose(logits_to_score(logits, "p_correct"), [0.5, 0.75, 0.25])
    np.testing.assert_allclose(logits_to_score(logits, "max_softmax"), [0.5, 0.75, 0.75])
    with pytest.raises(ValueError):
        logits_to_score(logits, "other")


# --- training on a tiny local model ------------------------------------------------------------

WORDS = ["q", "a", "up", "down", "gold", "good", "bad", ":"]


@pytest.fixture(scope="module")
def tiny():
    from transformers import (
        DistilBertConfig,
        DistilBertForSequenceClassification,
        DistilBertTokenizerFast,
    )

    words = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *WORDS]
    tok = DistilBertTokenizerFast(vocab={w: i for i, w in enumerate(words)})
    assert tok.unk_token_id not in tok("Q: gold A: good").input_ids
    torch.manual_seed(0)
    cfg = DistilBertConfig(
        vocab_size=len(WORDS) + 5,
        dim=16,
        hidden_dim=32,
        n_layers=1,
        n_heads=2,
        max_position_embeddings=32,
        num_labels=2,
    )
    return DistilBertForSequenceClassification(cfg), tok


def _toy_data(n: int = 40):
    texts = [("Q: gold A: good" if i % 2 == 0 else "Q: gold A: bad") for i in range(n)]
    labels = np.array([i % 2 == 0 for i in range(n)])
    return texts, labels


TINY_CFG = ScorerConfig(
    epochs=40,
    batch_size=8,
    eval_batch_size=16,
    learning_rate=5e-3,
    warmup_steps=5,
    max_length=16,
    val_size=0.25,
)


def test_train_scorer_learns_separable_toy(tiny) -> None:
    base, tok = tiny
    model = fresh_copy(base)
    texts, labels = _toy_data()
    res = train_scorer(model, tok, texts, labels, TINY_CFG, device=torch.device("cpu"))
    assert res.n_val == 10 and res.n_train == 30
    assert len(res.history) == TINY_CFG.epochs
    # best epoch = lowest validation loss (first on ties)
    losses = [h.val_loss for h in res.history]
    assert res.best_epoch == int(np.argmin(losses)) + 1
    s = score_texts(model, tok, texts, TINY_CFG, torch.device("cpu"))
    assert s.shape == (40,) and ((s >= 0) & (s <= 1)).all()
    assert roc_auc(labels, s) == 1.0
    assert s[labels].min() > 0.5 > s[~labels].max()


def test_train_scorer_keeps_best_epoch_weights(tiny) -> None:
    base, tok = tiny
    texts, labels = _toy_data()
    model = fresh_copy(base)
    res = train_scorer(model, tok, texts, labels, TINY_CFG, device=torch.device("cpu"))
    # Re-evaluating the final model on the validation split reproduces the best epoch's loss.
    from frugalgpt.scorer import predict_logits

    _, va = split_indices(len(texts), TINY_CFG.val_size, TINY_CFG.seed)
    logits = predict_logits(model, tok, [texts[i] for i in va], 16, 16, torch.device("cpu"))
    loss = torch.nn.functional.cross_entropy(
        torch.as_tensor(logits), torch.as_tensor(labels[va].astype(np.int64))
    )
    assert float(loss) == pytest.approx(res.history[res.best_epoch - 1].val_loss, rel=1e-5)


def test_train_scorer_is_reproducible(tiny) -> None:
    base, tok = tiny
    texts, labels = _toy_data()
    cfg = ScorerConfig(**{**TINY_CFG.__dict__, "epochs": 2})
    outs = []
    for _ in range(2):
        m = fresh_copy(base)
        train_scorer(m, tok, texts, labels, cfg, device=torch.device("cpu"))
        outs.append(score_texts(m, tok, texts, cfg, torch.device("cpu")))
    np.testing.assert_allclose(outs[0], outs[1])


def test_train_scorer_no_validation_keeps_last(tiny) -> None:
    base, tok = tiny
    texts, labels = _toy_data(16)
    cfg = ScorerConfig(**{**TINY_CFG.__dict__, "epochs": 2, "val_size": 0.0})
    res = train_scorer(fresh_copy(base), tok, texts, labels, cfg, device=torch.device("cpu"))
    assert res.best_epoch == 2 and res.n_val == 0
    assert np.isnan(res.history[-1].val_loss)


def test_train_scorer_bad_labels(tiny) -> None:
    base, tok = tiny
    texts, _ = _toy_data(8)
    with pytest.raises(ValueError):
        train_scorer(fresh_copy(base), tok, texts, np.full(8, 2), TINY_CFG)
    with pytest.raises(ValueError):
        train_scorer(fresh_copy(base), tok, texts, np.zeros(7), TINY_CFG)


def test_param_groups_no_decay_on_bias_and_layernorm(tiny) -> None:
    base, _ = tiny
    decay, no_decay = _param_groups(base, 0.01)
    names = {id(p): n for n, p in base.named_parameters()}
    assert decay["weight_decay"] == 0.01 and no_decay["weight_decay"] == 0.0
    nd = [names[id(p)] for p in no_decay["params"]]
    norms = {
        n
        for mn, m in base.named_modules()
        if isinstance(m, torch.nn.LayerNorm)
        for n in (f"{mn}.weight", f"{mn}.bias")
    }
    assert nd and all(n.endswith("bias") or n in norms for n in nd)
    assert norms <= set(nd)
    d = [names[id(p)] for p in decay["params"]]
    assert not any(n.endswith("bias") for n in d)
    assert len(nd) + len(d) == len(list(base.parameters()))


# --- review additions --------------------------------------------------------------------------


def test_lr_schedule_warmup_then_linear_decay(tiny) -> None:
    from frugalgpt.scorer import make_optimizer

    base, _ = tiny
    cfg = ScorerConfig(learning_rate=1e-3, warmup_steps=4)
    opt, sched = make_optimizer(fresh_copy(base), cfg, total_steps=12)
    assert opt.defaults["betas"] == (0.9, 0.999) and opt.defaults["eps"] == 1e-8
    lrs = []
    for _ in range(12):
        lrs.append(sched.get_last_lr()[0])
        opt.step()
        sched.step()
    lrs.append(sched.get_last_lr()[0])
    # 0 → peak at step 4 → 0 at step 12, linear on both sides
    np.testing.assert_allclose(lrs[:5], [0, 2.5e-4, 5e-4, 7.5e-4, 1e-3])
    np.testing.assert_allclose(lrs[4:], np.linspace(1e-3, 0, 9), atol=1e-12)


def test_train_scorer_single_class_labels(tiny) -> None:
    base, tok = tiny
    texts, _ = _toy_data(16)
    cfg = ScorerConfig(**{**TINY_CFG.__dict__, "epochs": 2})
    labels = np.ones(16, dtype=bool)  # an LLM that is always right
    model = fresh_copy(base)
    train_scorer(model, tok, texts, labels, cfg, device=torch.device("cpu"))
    s = score_texts(model, tok, texts, cfg, torch.device("cpu"))
    assert np.isfinite(s).all()
    assert np.isnan(roc_auc(labels, s))


def test_scoring_is_deterministic_without_reseeding(tiny) -> None:
    # Dropout must be off at scoring time.
    base, tok = tiny
    model = fresh_copy(base)
    model.config.dropout = 0.5
    texts, _ = _toy_data(8)
    a = score_texts(model, tok, texts, TINY_CFG, torch.device("cpu"))
    model.train()  # score_texts must switch to eval mode itself
    b = score_texts(model, tok, texts, TINY_CFG, torch.device("cpu"))
    np.testing.assert_array_equal(a, b)


def test_long_text_is_truncated_and_scored(tiny) -> None:
    base, tok = tiny
    long = "Q: " + "gold " * 200 + "A: good"  # far beyond max_length=16 and the 32 positions
    s = score_texts(fresh_copy(base), tok, [long, "Q: gold A: bad"], TINY_CFG, torch.device("cpu"))
    assert s.shape == (2,) and np.isfinite(s).all()


def test_scores_fingerprint() -> None:
    from frugalgpt.scorer import scores_fingerprint

    cfg = ScorerConfig()
    q = {"train": ["a", "b"], "test": ["c"]}
    f = scores_fingerprint(q, ["x", "y", "z"], cfg)
    assert f == scores_fingerprint(q, ["x", "y", "z"], cfg)
    assert f != scores_fingerprint({"train": ["b", "a"], "test": ["c"]}, ["x", "y", "z"], cfg)
    assert f != scores_fingerprint(q, ["x", "y", "w"], cfg)
    assert f != scores_fingerprint(q, ["x", "y", "z"], ScorerConfig(epochs=7))
    # no ambiguity from joining strings
    assert scores_fingerprint({"t": ["ab", "c"]}, [], cfg) != scores_fingerprint(
        {"t": ["a", "bc"]}, [], cfg
    )


def test_combine_scores() -> None:
    from frugalgpt.scorer import combine_scores

    llms = ("m0", "m1", "m2")
    t = combine_scores({"m2": [0.1, 0.2], "m0": [0.3, 0.4]}, llms, 2)
    np.testing.assert_allclose(t[:, 0], [0.3, 0.4])
    np.testing.assert_allclose(t[:, 2], [0.1, 0.2])
    assert np.isnan(t[:, 1]).all()  # missing LLM stays NaN
    with pytest.raises(ValueError):
        combine_scores({"zz": [0.1, 0.2]}, llms, 2)
    with pytest.raises(ValueError):
        combine_scores({"m0": [0.1]}, llms, 2)


def test_scores_fingerprint_includes_prefix() -> None:
    from frugalgpt.scorer import scores_fingerprint

    q = {"train": ["a"]}
    cfg = ScorerConfig()
    assert scores_fingerprint(q, ["x"], cfg) == scores_fingerprint(q, ["x"], cfg, "")
    assert scores_fingerprint(q, ["x"], cfg, "PRE") != scores_fingerprint(q, ["x"], cfg)


def test_scorer_text_overruling_style_prefix() -> None:
    # Prompts without "Q:" keep everything after the stripped few-shot prefix.
    pre = "Context: a\nQuestion: Is it overruling?\nAnswer: Yes\n\n"
    q = pre + "Context: b\nQuestion: Is it overruling?\nAnswer:"
    assert scorer_text(q, "No", pre) == "Q:Context: b\nQuestion: Is it overruling?\nAnswer: No"


def test_prompt_prefix() -> None:
    from frugalgpt.scorer import prompt_prefix

    shot = "Context: a\nAnswer: Yes\n\nContext: b\nAnswer: No\n\n"
    qs = [shot + "Context: cat\nAnswer:", shot + "Context: car\nAnswer:"]
    # common start goes into "Context: ca", but the cut stops at the last blank line
    assert prompt_prefix(qs) == shot
    assert prompt_prefix([]) == ""
    assert prompt_prefix(["Q: x", "Q: y"]) == ""  # no shared block
    assert prompt_prefix(["A\n\nQ: x"]) == "A\n\n"  # single prompt: up to its last blank line


def test_queries_hash_tracks_rows() -> None:
    from frugalgpt.scorer import queries_hash

    assert queries_hash(["a", "b"]) == queries_hash(["a", "b"])
    assert queries_hash(["a", "b"]) != queries_hash(["b", "a"])  # row order
    assert queries_hash(["a", "b"]) != queries_hash(["a", "b", "b"])  # row count
    assert queries_hash(["ab"]) != queries_hash(["a", "b"])  # separator


def test_prefix_coverage() -> None:
    from frugalgpt.scorer import prefix_coverage

    prefix = "Example 1\n\n"
    qs = ["Example 1\n\nQ: x\nA:", "Example 1\n\nQ: y\nA:"]
    assert prefix_coverage(qs, prefix) == 1.0
    assert prefix_coverage([*qs, "Other\n\nQ: z\nA:"], prefix) == pytest.approx(2 / 3)
    assert prefix_coverage(qs, "") == 1.0
    assert prefix_coverage([], prefix) == 1.0


def test_prompt_prefix_identical_prompts_keeps_own_block() -> None:
    from frugalgpt.scorer import prompt_prefix

    # every prompt identical: the common start is the whole prompt, cut at the last blank line,
    # so the final block (the query's own) survives.
    q = "shot 1\n\nshot 2\n\nQ: x\nA:"
    assert prompt_prefix([q, q]) == "shot 1\n\nshot 2\n\n"
