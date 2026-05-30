"""Tests for the decoder-only sibling of `seq2seq_sr`.

Covers the single-stream training assembly (round-trip of `encode_target`
labels back into the original action sequence), the validity-mask logic
on the gold-EDU forced decode path, and a forward+predict smoke test on
a tiny random Gemma3 model.
"""

import os
from typing import List

import pytest

pytest.importorskip("transformers")

from iudex.rst.data.tree import (
    Reduce,
    RstTree,
    Shift,
    strings_to_actions,
)
from iudex.rst.parsers.decoder_only_sr.configuration_decoder_only_sr import (
    DecoderOnlySRConfig,
)
from iudex.rst.parsers.common.seqgen import gold_edu_source_ranges, reconstruct_text
from iudex.rst.parsers.decoder_only_sr.modeling_decoder_only_sr import (
    DecoderOnlySRParser,
)

# Tiny random Gemma3 covers the relevant code paths (SentencePiece
# tokenizer with offset mapping, `model.lm_head`, `model.embed_tokens`,
# DynamicCache for KV reordering) without a real-model download.
SMALL_CAUSAL = os.environ.get("IUDEX_TEST_CAUSAL_MODEL", "hf-internal-testing/tiny-random-Gemma3ForCausalLM")


def _toy_tree() -> RstTree:
    actions = [
        Shift(edu_text="Cats sleep."),
        Shift(edu_text="Dogs bark."),
        Reduce(nuc="NS", rel="elaboration"),
    ]
    return RstTree.from_shift_reduce(actions, relation_types=[("elaboration", "rst")])


def _build_parser():
    cfg = DecoderOnlySRConfig(
        train_dir="<unused>",
        dev_dir="<unused>",
        model_name=SMALL_CAUSAL,
        relation_types=[("elaboration", "rst")],
        gradient_checkpointing=False,
        amp=False,
        max_input_length=128,
        max_output_length=128,
        min_edu_length=1,
    )
    try:
        return DecoderOnlySRParser(cfg)
    except Exception as e:  # network failure, missing cache, etc.
        pytest.skip(f"Could not load {SMALL_CAUSAL}: {e!r}")


@pytest.fixture(scope="module")
def parser():
    return _build_parser()


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

    # Prefix [BOS source SEP] is all -100; the action region starts after.
    prefix_len = 1 + len(source_ids) + 1
    assert labels[:prefix_len] == [-100] * prefix_len
    assert all(lbl != -100 for lbl in labels[prefix_len:])
    # EOS terminates the action region in both streams.
    assert input_ids[-1] == parser.tokenizer.eos_token_id
    assert labels[-1] == parser.tokenizer.eos_token_id


def test_encode_target_action_labels_roundtrip(parser):
    """The label stream (filtered to non-prefix non-EOS positions) decodes
    back to the same action sequence via `strings_to_actions`. Mirrors
    `test_shift_reduce_roundtrip.test_string_roundtrip` but exercises the
    decoder-only encode path."""
    tree = _toy_tree()
    input_ids, labels = parser.encode_target(tree)
    text = reconstruct_text(tree)
    source_ids = parser.tokenizer(text, add_special_tokens=False).input_ids
    prefix_len = 1 + len(source_ids) + 1
    eos_id = parser.tokenizer.eos_token_id

    # Walk the label stream just like the inference reconstruction does:
    # COPY -> next source subword, SHIFT/REDUCE -> token string.
    strings: list[str] = []
    src_buf: list[int] = []
    cursor = 0

    def _flush():
        if src_buf:
            decoded = parser.tokenizer.decode(src_buf, skip_special_tokens=False)
            strings.extend(decoded.split())
            src_buf.clear()

    for lbl in labels[prefix_len:]:
        if lbl == eos_id:
            _flush()
            break
        if lbl == parser.copy_token_id:
            src_buf.append(source_ids[cursor])
            cursor += 1
        elif lbl == parser.shift_token_id:
            _flush()
            strings.append(Shift().to_token())
        elif lbl in parser.reduce_token_ids:
            _flush()
            strings.append(parser.tokenizer.convert_ids_to_tokens(lbl))
    _flush()

    actions = strings_to_actions(strings, parser.reduce_token_map)
    assert sum(1 for a in actions if isinstance(a, Shift)) == len(tree.edus)
    assert sum(1 for a in actions if isinstance(a, Reduce)) == len(tree.edus) - 1
    # And reconstructs into a structurally-equal tree.
    reconstructed = RstTree.from_shift_reduce(actions, relation_types=parser.config.relation_types)
    assert reconstructed == tree


def test_forward_returns_finite_loss(parser):
    import torch

    tree = _toy_tree()
    input_ids, labels = parser.encode_target(tree)
    batch = {
        "input_ids": torch.tensor([input_ids], dtype=torch.long),
        "attention_mask": torch.ones((1, len(input_ids)), dtype=torch.long),
        "labels": torch.tensor([labels], dtype=torch.long),
    }
    # The replacement lm_head must be trainable. PEFT freezes the base
    # before head replacement, but the new Linear is constructed afterward
    # so its parameters default to requires_grad=True. Asserting directly
    # so a regression (e.g. accidentally calling .requires_grad_(False) on
    # the head) gets caught even when the grad-flow check below would also
    # pass via stale state.
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


def test_predict_with_gold_edus_emits_one_shift_per_gold_edu(parser):
    """The forced-segmentation decode reproduces gold EDU ranges exactly:
    copies are masked-in inside each EDU and a shift is masked-in at every
    boundary. Identical contract to the seq2seq_sr gold-EDU test."""
    tree = _toy_tree()
    gold_ranges = gold_edu_source_ranges(parser.tokenizer, tree)
    pred = parser.predict_with_gold_edus(tree)
    pred_ranges: List[tuple] = getattr(pred, "_pred_edu_source_ranges", [])
    assert pred_ranges == gold_ranges, f"pred {pred_ranges} != gold {gold_ranges}"
    assert len(pred.edus) == len(tree.edus)


def test_predict_from_text_runs_without_crash(parser):
    """Smoke test for the full greedy decode (constraints + KV cache + tree
    reconstruction). The tiny random model produces nonsense actions; we
    only care that the pipeline completes and returns an RstTree."""
    pred = parser.predict_from_text("Cats sleep. Dogs bark.")
    assert pred is not None
    assert len(pred.edus) >= 1


def test_predict_beam_runs_without_crash(parser):
    """Beam search runs end-to-end with K>1 (exercises the KV-cache reorder
    path under DynamicCache)."""
    pred = parser.predict_from_text("Cats sleep. Dogs bark.", num_beams=3)
    assert pred is not None
    assert len(pred.edus) >= 1


def test_evaluate_gold_edu_emits_expected_metric_keys(parser):
    from iudex.rst.parsers.decoder_only_sr.train_decoder_only_sr import _evaluate_gold_edu

    tree = _toy_tree()
    metrics = _evaluate_gold_edu(parser, [("toy.rs4", tree)])
    expected = {"gold_edu_span_f1", "gold_edu_nuc_f1", "gold_edu_rel_f1", "gold_edu_full_f1"}
    assert expected.issubset(metrics.keys())
    for k in expected:
        v = metrics[k]
        assert v == v  # not NaN
        assert 0.0 <= v <= 1.0
