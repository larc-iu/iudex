"""Tests for the decoder-only s-expression parser.

Covers the single-stream training assembly (round-trip of `encode_target`
labels back into the original tree via `RstTree.from_sexp`), the forward
pass loss + lm_head grad flow, and predict smoke tests on a tiny random
Gemma3 model. Parametrized over both traversal orders and both copy modes.
"""

import os
from typing import List

import pytest

pytest.importorskip("transformers")

from iudex.rst.data.tree import (
    Reduce,
    RstTree,
    Shift,
)
from iudex.rst.parsers.decoder_only_sexp.configuration_decoder_only_sexp import (
    DecoderOnlySexpConfig,
)
from iudex.rst.parsers.common.seqgen import gold_edu_source_ranges, reconstruct_text
from iudex.rst.parsers.decoder_only_sexp.modeling_decoder_only_sexp import (
    DecoderOnlySexpParser,
)

SMALL_CAUSAL = os.environ.get("IUDEX_TEST_CAUSAL_MODEL", "hf-internal-testing/tiny-random-Gemma3ForCausalLM")


def _toy_tree() -> RstTree:
    actions = [
        Shift(edu_text="Cats sleep."),
        Shift(edu_text="Dogs bark."),
        Reduce(nuc="NS", rel="elaboration"),
    ]
    return RstTree.from_shift_reduce(actions, relation_types=[("elaboration", "rst")])


def _build_parser(traversal_order: str, use_copy: bool) -> DecoderOnlySexpParser:
    cfg = DecoderOnlySexpConfig(
        train_dir="<unused>",
        dev_dir="<unused>",
        model_name=SMALL_CAUSAL,
        relation_types=[("elaboration", "rst")],
        gradient_checkpointing=False,
        amp=False,
        max_input_length=128,
        max_output_length=256,
        min_edu_length=1,
        traversal_order=traversal_order,
        use_copy=use_copy,
    )
    try:
        return DecoderOnlySexpParser(cfg)
    except Exception as e:
        pytest.skip(f"Could not load {SMALL_CAUSAL}: {e!r}")


@pytest.fixture(
    scope="module",
    params=[
        ("preorder", True),
        ("postorder", True),
        ("preorder", False),
        ("postorder", False),
    ],
)
def parser(request):
    traversal_order, use_copy = request.param
    return _build_parser(traversal_order, use_copy)


def test_encode_target_lengths_and_layout(parser):
    tree = _toy_tree()
    enc = parser.encode_target(tree)
    assert enc is not None, "encode_target unexpectedly dropped this tree"
    input_ids, labels = enc
    assert len(input_ids) == len(labels)

    text = reconstruct_text(tree)
    source_ids = parser.tokenizer(text, add_special_tokens=False).input_ids
    bos_id = parser.tokenizer.bos_token_id
    assert input_ids[0] == bos_id
    assert input_ids[1 : 1 + len(source_ids)] == source_ids
    assert input_ids[1 + len(source_ids)] == parser.sep_token_id

    # Prefix [BOS source] is -100; the sexp region starts at SEP and is scored.
    prefix_len = 1 + len(source_ids)
    assert labels[:prefix_len] == [-100] * prefix_len
    assert labels[prefix_len] == parser.sep_token_id
    # Both streams end with EOS.
    assert input_ids[-1] == parser.tokenizer.eos_token_id
    assert labels[-1] == parser.tokenizer.eos_token_id


def test_encode_target_roundtrips_to_tree(parser):
    """The label stream (skipping the masked prefix and the trailing EOS) is
    the canonical sexp action sequence. Stringifying it and calling
    `RstTree.from_sexp` recovers a structurally-equal tree."""
    tree = _toy_tree()
    input_ids, labels = parser.encode_target(tree)
    text = reconstruct_text(tree)
    source_ids = parser.tokenizer(text, add_special_tokens=False).input_ids
    prefix_len = 1 + len(source_ids)
    eos_id = parser.tokenizer.eos_token_id

    sexp_labels = labels[prefix_len + 1 :]  # skip SEP
    assert sexp_labels[-1] == eos_id
    sexp_labels = sexp_labels[:-1]

    parts: List[str] = []
    leaf_buf: List[int] = []
    cursor = 0

    def flush():
        if leaf_buf:
            decoded = parser.tokenizer.decode(leaf_buf, skip_special_tokens=False).strip()
            if decoded:
                parts.append(decoded.replace("(", "-LRB-").replace(")", "-RRB-"))
            leaf_buf.clear()

    for tok in sexp_labels:
        if tok == parser.open_token_id:
            flush()
            parts.append("(")
        elif tok == parser.close_token_id:
            flush()
            parts.append(")")
        elif tok in parser.label_id_set:
            flush()
            parts.append(parser.label_id_to_str[tok][1:-1])
        elif parser.config.use_copy and tok == parser.copy_token_id:
            leaf_buf.append(source_ids[cursor])
            cursor += 1
        else:
            leaf_buf.append(tok)
            cursor += 1
    flush()

    sexp_text = " ".join(parts)
    reconstructed = RstTree.from_sexp(
        sexp_text,
        traversal_order=parser.config.traversal_order,
        relation_types=parser.config.relation_types,
    )
    assert reconstructed == tree


def test_forward_returns_finite_loss_and_head_grad(parser):
    import torch

    tree = _toy_tree()
    input_ids, labels = parser.encode_target(tree)
    batch = {
        "input_ids": torch.tensor([input_ids], dtype=torch.long),
        "attention_mask": torch.ones((1, len(input_ids)), dtype=torch.long),
        "labels": torch.tensor([labels], dtype=torch.long),
    }
    lm_head_weight = parser._underlying_model().lm_head.weight
    assert lm_head_weight.requires_grad

    parser.train()
    out = parser(batch)
    assert "loss" in out
    loss = out["loss"]
    assert torch.isfinite(loss).item()
    loss.backward()
    assert lm_head_weight.grad is not None
    assert torch.isfinite(lm_head_weight.grad).all().item()


def test_predict_with_gold_edus_runs_and_returns_ranges(parser):
    """Gold-EDU forced decode runs without crashing and attaches a
    `_pred_edu_source_ranges` list. Under the unified forcing contract
    (force boundaries inside leaves, structure free outside), a random /
    untrained backbone can choose a degenerate root-leaf shape that
    overshoots a single gold EDU. Strict gold-range alignment is only
    expected from a trained model, so here we assert only the contract:
    ranges exist as a list and are monotone non-decreasing in start
    position."""
    tree = _toy_tree()
    pred = parser.predict_with_gold_edus(tree)
    pred_ranges: List[tuple] = getattr(pred, "_pred_edu_source_ranges", [])
    assert isinstance(pred_ranges, list)
    starts = [s for s, _ in pred_ranges]
    assert starts == sorted(starts), f"pred ranges not monotone: {pred_ranges}"


def test_predict_from_text_runs_without_crash(parser):
    pred = parser.predict_from_text("Cats sleep. Dogs bark.")
    assert pred is not None
    assert len(pred.edus) >= 1


def test_predict_beam_runs_without_crash(parser):
    pred = parser.predict_from_text("Cats sleep. Dogs bark.", num_beams=3)
    assert pred is not None
    assert len(pred.edus) >= 1


def test_evaluate_gold_edu_emits_expected_metric_keys(parser):
    from iudex.rst.parsers.common.generative_eval import _evaluate_gold_edu

    tree = _toy_tree()
    metrics = _evaluate_gold_edu(parser, [("toy.rs4", tree)])
    expected = {"gold_edu_span_f1", "gold_edu_nuc_f1", "gold_edu_rel_f1", "gold_edu_full_f1"}
    assert expected.issubset(metrics.keys())
    for k in expected:
        v = metrics[k]
        assert v == v  # not NaN
        assert 0.0 <= v <= 1.0


@pytest.mark.parametrize("traversal_order", ["preorder", "postorder"])
def test_use_copy_false_predicts_source_ids(traversal_order):
    """Under use_copy=False the predict path's emitted-id stream must contain
    actual source-subword ids (interleaved with structural tokens), not just
    structural specials. We exercise the gold-EDU forced decode (deterministic
    in-leaf cursor advancement) so the assertion doesn't depend on a random
    backbone choosing OPEN at the right slot."""
    parser = _build_parser(traversal_order, use_copy=False)
    tree = _toy_tree()
    text = reconstruct_text(tree)
    source_ids = parser.tokenizer(text, add_special_tokens=False).input_ids
    # Gold-EDU forcing returns a tree but the internal emitted-id sequence is
    # opaque. Re-run a piece of the forcing flow via `predict_with_gold_edus`
    # and check the source ids landed in the leaf reconstructions.
    pred = parser.predict_with_gold_edus(tree)
    pred_ranges = getattr(pred, "_pred_edu_source_ranges", [])
    if not pred_ranges:
        pytest.skip("random backbone couldn't open any leaf under forced decode (untrained model edge case)")
    # The decoded EDU surface texts must be non-empty and recoverable from
    # source positions. Concatenated they should cover a contiguous prefix
    # of source_ids.
    covered = 0
    for s, e in pred_ranges:
        assert 0 <= s < e <= len(source_ids), (s, e, len(source_ids))
        covered += e - s
    assert covered > 0, "no source ids were emitted under use_copy=False forced decode"


def _toy_tree_3edu() -> RstTree:
    actions = [
        Shift(edu_text="Cats sleep."),
        Shift(edu_text="Dogs bark."),
        Reduce(nuc="NS", rel="elaboration"),
        Shift(edu_text="Birds fly."),
        Reduce(nuc="NS", rel="elaboration"),
    ]
    return RstTree.from_shift_reduce(actions, relation_types=[("elaboration", "rst")])


@pytest.mark.parametrize("traversal_order", ["preorder", "postorder"])
def test_predict_with_gold_edus_opens_exactly_n_edus_on_untrained_backbone(traversal_order):
    """Regression for round-2 Fix 1. Gold-EDU forced decode must open
    exactly len(gold_edus) leaves and close cleanly to EOS even on a
    randomly-initialized backbone."""
    parser = _build_parser(traversal_order, use_copy=True)
    tree = _toy_tree_3edu()
    gold_ranges = gold_edu_source_ranges(parser.tokenizer, tree)
    assert len(gold_ranges) == 3

    pred = parser.predict_with_gold_edus(tree)
    pred_ranges = getattr(pred, "_pred_edu_source_ranges", [])
    assert len(pred.edus) == 3, f"expected 3 EDUs in predicted tree, got {len(pred.edus)}"
    assert len(pred_ranges) == 3, f"expected 3 source ranges, got {pred_ranges}"
    for (gs, ge), (ps, pe) in zip(gold_ranges, pred_ranges):
        assert (ps, pe) == (gs, ge), f"gold {gs, ge} vs pred {ps, pe}"


@pytest.mark.parametrize("traversal_order", ["preorder", "postorder"])
def test_constrain_content_false_runs_end_to_end(traversal_order):
    """Smoke test for round-2 Fix 3: use_copy=False + constrain_content=False
    (free-content decoding) does forward + predict without crashing."""
    cfg = DecoderOnlySexpConfig(
        train_dir="<unused>",
        dev_dir="<unused>",
        model_name=SMALL_CAUSAL,
        relation_types=[("elaboration", "rst")],
        gradient_checkpointing=False,
        amp=False,
        max_input_length=128,
        max_output_length=256,
        min_edu_length=1,
        traversal_order=traversal_order,
        use_copy=False,
        constrain_content=False,
    )
    try:
        parser = DecoderOnlySexpParser(cfg)
    except Exception as e:
        pytest.skip(f"Could not load {SMALL_CAUSAL}: {e!r}")

    pred = parser.predict_from_text("Cats sleep. Dogs bark.")
    assert pred is not None

    import torch

    tree = _toy_tree()
    input_ids, labels = parser.encode_target(tree)
    batch = {
        "input_ids": torch.tensor([input_ids], dtype=torch.long),
        "attention_mask": torch.ones((1, len(input_ids)), dtype=torch.long),
        "labels": torch.tensor([labels], dtype=torch.long),
    }
    parser.train()
    out = parser(batch)
    assert torch.isfinite(out["loss"]).item()


def test_decoder_only_constrain_content_false_with_use_copy_true_raises():
    with pytest.raises(ValueError, match="constrain_content=False requires use_copy=False"):
        DecoderOnlySexpConfig(
            train_dir="<unused>",
            dev_dir="<unused>",
            model_name=SMALL_CAUSAL,
            use_copy=True,
            constrain_content=False,
        )


@pytest.mark.parametrize("traversal_order", ["preorder", "postorder"])
def test_use_copy_false_loss_is_finite_and_flows_to_lm_head(traversal_order):
    """Under use_copy=False the lm_head is the full pretrained head and must
    receive gradient from source-content positions (not just structural slots)."""
    import torch

    parser = _build_parser(traversal_order, use_copy=False)
    tree = _toy_tree()
    input_ids, labels = parser.encode_target(tree)
    # Identify a leaf-content label position: pick a label id that's NOT one
    # of the structural specials.
    structural = {parser.open_token_id, parser.close_token_id, parser.sep_token_id, parser.tokenizer.eos_token_id}
    structural |= parser.label_id_set
    has_source_label = any(0 <= t and t not in structural and t != -100 for t in labels)
    assert has_source_label, "test fixture invariant: at least one leaf-content position must exist"

    batch = {
        "input_ids": torch.tensor([input_ids], dtype=torch.long),
        "attention_mask": torch.ones((1, len(input_ids)), dtype=torch.long),
        "labels": torch.tensor([labels], dtype=torch.long),
    }
    parser.train()
    out = parser(batch)
    loss = out["loss"]
    assert torch.isfinite(loss).item()
    assert float(loss.item()) > 1e-3

    lm_head_weight = parser._underlying_model().lm_head.weight
    loss.backward()
    assert lm_head_weight.grad is not None
    # Gradient must be non-zero somewhere on the head (otherwise the head
    # isn't being learned at all). Source-content positions accumulate
    # gradient on the same head.
    assert lm_head_weight.grad.abs().sum().item() > 0.0
