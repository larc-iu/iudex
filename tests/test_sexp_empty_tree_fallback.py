"""Fix 4 regression: when `_tree_from_emitted` falls back to `_empty_tree`
(because `RstTree.from_sexp` raised), the
predict path must null out `_pred_edu_source_ranges` so downstream
gold-EDU eval doesn't see action-derived ranges that disagree with the
single-EDU fallback (which would silently filter the doc out of the
parseval aggregator)."""

import os

import pytest

pytest.importorskip("transformers")


SMALL_CAUSAL = os.environ.get("IUDEX_TEST_CAUSAL_MODEL", "hf-internal-testing/tiny-random-Gemma3ForCausalLM")
SMALL_SEQ2SEQ = os.environ.get("IUDEX_TEST_SEQ2SEQ_MODEL", "google-t5/t5-small")


def test_decoder_only_sexp_fallback_marks_failure():
    """Calling `_tree_from_emitted` with a malformed action stream falls
    back to the empty tree and marks it with `_from_sexp_failed=True`."""
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
        parser = DecoderOnlySexpParser(cfg)
    except Exception as e:
        pytest.skip(f"Could not load {SMALL_CAUSAL}: {e!r}")
    # An empty emission yields an empty sexp string, which RstTree.from_sexp
    # rejects -> empty-tree fallback path runs.
    tree = parser._tree_from_emitted([], source_ids=[0, 1, 2])
    assert getattr(tree, "_from_sexp_failed", False) is True


def test_seq2seq_sexp_fallback_attaches_marker():
    """The seq2seq_sexp parser's `_actions_to_sexp_string` always produces
    a parseable sexp by design (best-effort closing + degenerate-leaf
    fallback), so triggering the post-`from_sexp` fallback in isolation
    is awkward. We verify the source-level invariant: the exception
    handler sets `_from_sexp_failed = True` on the empty tree before
    returning it, AND the marker handling is centralized in `_finalize_tree`
    (which nulls out `_pred_edu_source_ranges` on a marked tree), AND the
    three predict paths all funnel through `_finalize_tree`."""
    import inspect

    from iudex.rst.parsers.seq2seq_sexp.modeling_seq2seq_sexp import Seq2SeqSexpParser

    src_make = inspect.getsource(Seq2SeqSexpParser._tree_from_emitted)
    assert "_from_sexp_failed = True" in src_make
    src_finalize = inspect.getsource(Seq2SeqSexpParser._finalize_tree)
    assert "_from_sexp_failed" in src_finalize, "_finalize_tree doesn't honor _from_sexp_failed"
    for name in ("_predict_one_greedy", "_predict_one_beam", "_predict_one_gold_edu"):
        src = inspect.getsource(getattr(Seq2SeqSexpParser, name))
        assert "_finalize_tree" in src, f"{name} doesn't funnel through _finalize_tree"


def test_decoder_only_sexp_predict_paths_honor_marker():
    """Same source-level invariant for decoder_only_sexp."""
    import inspect

    from iudex.rst.parsers.decoder_only_sexp.modeling_decoder_only_sexp import DecoderOnlySexpParser

    src_make = inspect.getsource(DecoderOnlySexpParser._tree_from_emitted)
    assert "_from_sexp_failed = True" in src_make
    src_finalize = inspect.getsource(DecoderOnlySexpParser._finalize_tree)
    assert "_from_sexp_failed" in src_finalize, "_finalize_tree doesn't honor _from_sexp_failed"
    for name in ("_predict_one_greedy", "_predict_one_beam", "_predict_one_gold_edu"):
        src = inspect.getsource(getattr(DecoderOnlySexpParser, name))
        assert "_finalize_tree" in src, f"{name} doesn't funnel through _finalize_tree"


def test_use_copy_false_is_constructible():
    """`use_copy=False` is the no-COPY mode (Hu and Wan 2023 mirror). Both
    configs should accept it without raising. The full-vocab head and
    source-id in-stream emission are wired up in their respective parsers."""
    from iudex.rst.parsers.decoder_only_sexp.configuration_decoder_only_sexp import DecoderOnlySexpConfig
    from iudex.rst.parsers.seq2seq_sexp.configuration_seq2seq_sexp import Seq2SeqSexpConfig

    Seq2SeqSexpConfig(
        train_dir="<unused>",
        dev_dir="<unused>",
        relation_types=[("elaboration", "rst")],
        use_copy=False,
    )
    DecoderOnlySexpConfig(
        train_dir="<unused>",
        dev_dir="<unused>",
        relation_types=[("elaboration", "rst")],
        use_copy=False,
    )
