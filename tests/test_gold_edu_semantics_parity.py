"""Parity test for the unified gold-EDU forcing contract across the four
parsers (seq2seq_sr, decoder_only_sr, seq2seq_sexp, decoder_only_sexp).

Sexp contract (Round-2 Fix 1): force EXACTLY `n_edus_target` leaves via
a `GoldEduForcer` (right-spine planner) so even untrained backbones
terminate cleanly. Inside a leaf, force content / close. Outside a leaf,
force OPEN until all leaves are planted, then force CLOSE to root + EOS.
Tree shape is fixed by the forcer. Only label slots and use_copy/
constrain_content content choices remain free.

For the sexp pair we assert the docstrings mention the boundary / leaf
contract and that each parser's `_predict_one_gold_edu` runs end-to-end
on a toy tree without exceptions.
"""

import os
from typing import List

import pytest

pytest.importorskip("transformers")

from iudex.rst.data.tree import Reduce, RstTree, Shift


SMALL_SEQ2SEQ = os.environ.get("IUDEX_TEST_SEQ2SEQ_MODEL", "google-t5/t5-small")
SMALL_CAUSAL = os.environ.get("IUDEX_TEST_CAUSAL_MODEL", "hf-internal-testing/tiny-random-Gemma3ForCausalLM")


def _toy_tree() -> RstTree:
    actions = [
        Shift(edu_text="Cats sleep."),
        Shift(edu_text="Dogs bark."),
        Reduce(nuc="NS", rel="elaboration"),
    ]
    return RstTree.from_shift_reduce(actions, relation_types=[("elaboration", "rst")])


def test_seq2seq_sexp_gold_edu_docstring_mentions_contract():
    from iudex.rst.parsers.seq2seq_sexp.modeling_seq2seq_sexp import Seq2SeqSexpParser

    doc = (Seq2SeqSexpParser._predict_one_gold_edu.__doc__ or "").lower()
    assert "boundaries" in doc or "gold" in doc
    assert "leaf" in doc or "structure" in doc


def test_decoder_only_sexp_gold_edu_docstring_mentions_contract():
    from iudex.rst.parsers.decoder_only_sexp.modeling_decoder_only_sexp import DecoderOnlySexpParser

    doc = (DecoderOnlySexpParser._predict_one_gold_edu.__doc__ or "").lower()
    assert "boundaries" in doc or "gold" in doc
    assert "leaf" in doc or "structure" in doc


def _build_seq2seq_sexp():
    from iudex.rst.parsers.seq2seq_sexp.configuration_seq2seq_sexp import Seq2SeqSexpConfig
    from iudex.rst.parsers.seq2seq_sexp.modeling_seq2seq_sexp import Seq2SeqSexpParser

    cfg = Seq2SeqSexpConfig(
        train_dir="<unused>",
        dev_dir="<unused>",
        model_name=SMALL_SEQ2SEQ,
        relation_types=[("elaboration", "rst")],
        gradient_checkpointing=False,
        amp=False,
        max_input_length=128,
        max_output_length=128,
        min_edu_length=1,
        traversal_order="postorder",
        use_copy=True,
    )
    try:
        return Seq2SeqSexpParser(cfg)
    except Exception as e:
        pytest.skip(f"Could not load {SMALL_SEQ2SEQ}: {e!r}")


def _build_decoder_only_sexp():
    from iudex.rst.parsers.decoder_only_sexp.configuration_decoder_only_sexp import DecoderOnlySexpConfig
    from iudex.rst.parsers.decoder_only_sexp.modeling_decoder_only_sexp import DecoderOnlySexpParser

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
        traversal_order="postorder",
        use_copy=True,
    )
    try:
        return DecoderOnlySexpParser(cfg)
    except Exception as e:
        pytest.skip(f"Could not load {SMALL_CAUSAL}: {e!r}")


@pytest.mark.parametrize("build", [_build_seq2seq_sexp, _build_decoder_only_sexp])
def test_gold_edu_runs_and_ranges_are_monotone(build):
    """Both sexp parsers run gold-EDU forced decode on a toy tree without
    exceptions, and any emitted ranges are monotone non-decreasing in
    start position (the shared contract; strict gold alignment requires
    a trained model)."""
    parser = build()
    tree = _toy_tree()
    pred = parser.predict_with_gold_edus(tree)
    pred_ranges: List[tuple] = getattr(pred, "_pred_edu_source_ranges", [])
    assert isinstance(pred_ranges, list)
    starts = [s for s, _ in pred_ranges]
    assert starts == sorted(starts), f"pred ranges not monotone: {pred_ranges}"


def test_sexp_parsers_use_same_gold_edu_strategy_keywords():
    """Both sexp parsers' `_predict_one_gold_edu` source uses the shared
    `GoldEduForcer` planner and drives off clamped gold ranges. Catches
    accidental drift back to a per-parser ad hoc loop."""
    import inspect

    from iudex.rst.parsers.decoder_only_sexp.modeling_decoder_only_sexp import DecoderOnlySexpParser
    from iudex.rst.parsers.seq2seq_sexp.modeling_seq2seq_sexp import Seq2SeqSexpParser

    for cls in (Seq2SeqSexpParser, DecoderOnlySexpParser):
        src = inspect.getsource(cls._predict_one_gold_edu)
        assert "GoldEduForcer" in src, f"{cls.__name__}._predict_one_gold_edu doesn't use GoldEduForcer"
        assert "clamped_ranges" in src, f"{cls.__name__}._predict_one_gold_edu lost the clamped-ranges drive"
        assert '"LABEL"' not in src, f"{cls.__name__} still uses LABEL sentinel"
