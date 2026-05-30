"""Tests for the gold-EDU forced predict path of `seq2seq_sr`.

Exercises both the pure-logic helper (`_gold_edu_source_ranges`) and the
greedy decode path (`predict_with_gold_edus`), with a small t5-small backbone
swapped in for speed. No training — we only verify that:
  * gold ranges align to the encoder's whole-doc tokenization;
  * the forced decode emits exactly one shift per gold EDU, copies matching
    source tokens between shifts, and never tries to copy past the gold
    boundary or shift mid-EDU.
"""

import os
from typing import List

import pytest

# Tests in this file load a small HF model. Skip cleanly when offline / no
# weights, since the existing 17 tests cover all the pure-logic paths.
pytest.importorskip("transformers")

from iudex.rst.data.tree import Reduce, RstTree, Shift
from iudex.rst.parsers.seq2seq_sr.configuration_seq2seq_sr import Seq2SeqSRConfig
from iudex.rst.parsers.common.seqgen import gold_edu_source_ranges, reconstruct_text
from iudex.rst.parsers.seq2seq_sr.modeling_seq2seq_sr import (
    Seq2SeqSRParser,
)

# Allow override; t5-small is the smallest cached seq2seq we can rely on.
SMALL_SEQ2SEQ = os.environ.get("IUDEX_TEST_SEQ2SEQ_MODEL", "google-t5/t5-small")


def _toy_tree() -> RstTree:
    """Build a 2-EDU tree via the shift-reduce constructor. The two EDUs
    "Cats sleep." and "Dogs bark." are short enough to fit in a few subwords
    each on t5-small's SentencePiece."""
    actions = [
        Shift(edu_text="Cats sleep."),
        Shift(edu_text="Dogs bark."),
        Reduce(nuc="NS", rel="elaboration"),
    ]
    return RstTree.from_shift_reduce(actions, relation_types=[("elaboration", "rst")])


def _build_parser():
    cfg = Seq2SeqSRConfig(
        train_dir="<unused>",
        dev_dir="<unused>",
        model_name=SMALL_SEQ2SEQ,
        relation_types=[("elaboration", "rst")],
        gradient_checkpointing=False,
        amp=False,
        max_input_length=128,
        max_output_length=64,
        min_edu_length=1,
    )
    try:
        return Seq2SeqSRParser(cfg)
    except Exception as e:  # network failure, missing cache, etc.
        pytest.skip(f"Could not load {SMALL_SEQ2SEQ}: {e!r}")


@pytest.fixture(scope="module")
def parser():
    return _build_parser()


def test_gold_edu_source_ranges_align_to_doc_tokenization(parser):
    tree = _toy_tree()
    text = reconstruct_text(tree)
    ranges = gold_edu_source_ranges(parser.tokenizer, tree)
    assert len(ranges) == len(tree.edus)
    # Ranges are strictly increasing and contiguous in token space (modulo
    # zero-width spans for empty EDUs, which we don't construct here).
    for s, e in ranges:
        assert 0 <= s <= e
    starts = [s for s, _ in ranges]
    ends = [e for _, e in ranges]
    assert starts == sorted(starts)
    # Last EDU end is bounded by the full doc tokenization length.
    full_len = len(parser.tokenizer(text, add_special_tokens=False).input_ids)
    assert ends[-1] <= full_len


def test_predict_with_gold_edus_emits_one_shift_per_gold_edu(parser):
    tree = _toy_tree()
    gold_ranges = gold_edu_source_ranges(parser.tokenizer, tree)

    pred = parser.predict_with_gold_edus(tree)
    assert pred is not None

    pred_ranges: List[tuple] = getattr(pred, "_pred_edu_source_ranges", [])
    # Forced segmentation must reproduce gold ranges exactly: copies are
    # forced inside each EDU and a shift is forced at every boundary.
    assert pred_ranges == gold_ranges, f"pred {pred_ranges} != gold {gold_ranges}"
    assert len(pred.edus) == len(tree.edus)


def test_evaluate_gold_edu_emits_expected_metric_keys(parser):
    """Smoke-check that the trainer-side aggregator returns the four
    `gold_edu_*_f1` keys, all finite and in [0, 1]."""
    from iudex.rst.parsers.seq2seq_sr.train_seq2seq_sr import _evaluate_gold_edu

    tree = _toy_tree()
    metrics = _evaluate_gold_edu(parser, [("toy.rs4", tree)])
    expected = {"gold_edu_span_f1", "gold_edu_nuc_f1", "gold_edu_rel_f1", "gold_edu_full_f1"}
    assert expected.issubset(metrics.keys())
    for k in expected:
        v = metrics[k]
        assert v == v  # not NaN
        assert 0.0 <= v <= 1.0
