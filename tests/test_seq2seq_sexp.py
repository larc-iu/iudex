"""Tests for `seq2seq_sexp` (s-expression seq2seq RST parser).

Covers encode_target round-trip across both traversal orders x both copy
modes, forward+backward grad flow, and predict smoke tests on a tiny
t5-small backbone. Mirrors `tests/test_decoder_only_sr.py` structure.
"""

import os
from typing import List

import pytest

pytest.importorskip("transformers")

from iudex.rst.data.tree import Reduce, RstTree, Shift
from iudex.rst.parsers.seq2seq_sexp.configuration_seq2seq_sexp import Seq2SeqSexpConfig
from iudex.rst.parsers.seq2seq_sexp.modeling_seq2seq_sexp import (
    Seq2SeqSexpParser,
    _gold_edu_source_ranges,
    _reconstruct_text,
    _reduce_token_to_label,
)

SMALL_SEQ2SEQ = os.environ.get("IUDEX_TEST_SEQ2SEQ_MODEL", "google-t5/t5-small")


def _toy_tree() -> RstTree:
    actions = [
        Shift(edu_text="Cats sleep."),
        Shift(edu_text="Dogs bark."),
        Reduce(nuc="NS", rel="elaboration"),
    ]
    return RstTree.from_shift_reduce(actions, relation_types=[("elaboration", "rst")])


def _build_parser(*, traversal_order: str = "postorder", use_copy: bool = True) -> Seq2SeqSexpParser:
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
        traversal_order=traversal_order,
        use_copy=use_copy,
    )
    try:
        return Seq2SeqSexpParser(cfg)
    except Exception as e:
        pytest.skip(f"Could not load {SMALL_SEQ2SEQ}: {e!r}")


@pytest.fixture(
    scope="module",
    params=[
        ("postorder", True),
        ("postorder", False),
        ("preorder", True),
        ("preorder", False),
    ],
)
def parser(request):
    traversal_order, use_copy = request.param
    return _build_parser(traversal_order=traversal_order, use_copy=use_copy)


# Cached secondary parsers (preorder, no-copy variants) for variant-coverage
# tests. Built lazily per request and cached for the module.
_PARSER_CACHE: dict[tuple[str, bool], Seq2SeqSexpParser] = {}


def _variant_parser(traversal_order: str, use_copy: bool) -> Seq2SeqSexpParser:
    key = (traversal_order, use_copy)
    if key not in _PARSER_CACHE:
        _PARSER_CACHE[key] = _build_parser(traversal_order=traversal_order, use_copy=use_copy)
    return _PARSER_CACHE[key]


def _reconstruct_sexp_from_labels(parser: Seq2SeqSexpParser, labels: list[int], source_ids: list[int]) -> str:
    """Mirror what `_actions_to_sexp_string` does but on a clean LABEL stream
    (no truncation, no falling back to placeholder collapse). We build the
    sexp text in two parses: collect all reduce labels' positions and EDU
    placeholders, then render with `<edu>` and `NUC:rel` tokens."""
    eos_id = parser.tokenizer.eos_token_id
    # Tokenize via a depth-tracked scan; transform leaves into `<edu>`.
    pieces: list[str] = []
    depth = 0
    cursor = 0
    # Per-depth kind: None / "leaf" / "internal".
    kinds: list[str | None] = []
    for tok in labels:
        if tok == eos_id:
            break
        if tok == parser.open_token_id:
            pieces.append("(")
            kinds.append(None)
            depth += 1
        elif tok == parser.close_token_id:
            if kinds and kinds[-1] == "leaf":
                # Collapse this leaf's open into a `<edu>` placeholder.
                for k in range(len(pieces) - 1, -1, -1):
                    if pieces[k] == "(":
                        pieces[k] = "<edu>"
                        break
            else:
                pieces.append(")")
            if kinds:
                kinds.pop()
            depth -= 1
        elif tok in parser.label_id_set:
            token_str = parser.tokenizer.convert_ids_to_tokens(tok)
            pieces.append(_reduce_token_to_label(token_str))
            if kinds:
                kinds[-1] = "internal"
        elif parser.config.use_copy and tok == parser.copy_token_id:
            if cursor < len(source_ids):
                cursor += 1
            if kinds and kinds[-1] is None:
                kinds[-1] = "leaf"
        else:
            # use_copy=False: this is a source subword id.
            if cursor < len(source_ids) and tok == source_ids[cursor]:
                cursor += 1
            if kinds and kinds[-1] is None:
                kinds[-1] = "leaf"
    return " ".join(pieces)


def _labels_action_only(parser: Seq2SeqSexpParser, labels: list[int]) -> list[str]:
    """Drop content tokens (source ids / `<copy>`) to leave only structural
    pieces. Used for shape assertions."""
    out: list[str] = []
    for tok in labels:
        if tok == parser.open_token_id:
            out.append(parser.OPEN_TOKEN)
        elif tok == parser.close_token_id:
            out.append(parser.CLOSE_TOKEN)
        elif tok in parser.label_id_set:
            out.append(parser.tokenizer.convert_ids_to_tokens(tok))
        elif tok == parser.tokenizer.eos_token_id:
            out.append("</s>")
        elif parser.config.use_copy and tok == parser.copy_token_id:
            out.append(parser.COPY_TOKEN)
        # source ids in use_copy=False mode: drop
    return out


@pytest.mark.parametrize("traversal_order", ["preorder", "postorder"])
@pytest.mark.parametrize("use_copy", [True, False])
def test_encode_target_roundtrip(traversal_order, use_copy):
    """The label stream encodes the tree's sexp serialization. Walk the
    labels, reconstruct a sexp string, run `RstTree.from_sexp`, and check
    the reconstructed tree is structurally equal to the original."""
    parser = _variant_parser(traversal_order, use_copy)
    tree = _toy_tree()
    encoded = parser.encode_target(tree)
    assert encoded is not None, "encode_target unexpectedly dropped this tree"
    labels, decoder_input_ids = encoded
    assert len(labels) == len(decoder_input_ids)

    text = _reconstruct_text(tree)
    enc = parser.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    source_ids = enc["input_ids"]
    sexp_str = _reconstruct_sexp_from_labels(parser, labels, source_ids)
    edus = [edu.text for edu in tree.edus]
    reconstructed = RstTree.from_sexp(
        sexp_str,
        traversal_order=traversal_order,
        edus=edus,
        relation_types=parser.config.relation_types,
    )
    assert reconstructed == tree


@pytest.mark.parametrize("traversal_order", ["preorder", "postorder"])
@pytest.mark.parametrize("use_copy", [True, False])
def test_encode_target_label_shape(traversal_order, use_copy):
    """Structural-action-only view of the label stream is the canonical
    sexp shape: one `<sexp_open>`/`<sexp_close>` pair per leaf and one
    per internal node, with the label slot in the configured position."""
    parser = _variant_parser(traversal_order, use_copy)
    tree = _toy_tree()
    labels, _ = parser.encode_target(tree)
    seq = _labels_action_only(parser, labels)
    # 2 EDUs + 1 internal node => 3 opens and 3 closes (+ EOS at the end).
    assert seq.count(parser.OPEN_TOKEN) == 3
    assert seq.count(parser.CLOSE_TOKEN) == 3
    assert seq[-1] == "</s>"
    label_str = Reduce(nuc="NS", rel="elaboration").to_token()
    assert seq.count(label_str) == 1
    if traversal_order == "preorder":
        # First open is root; next token is the label.
        assert seq[0] == parser.OPEN_TOKEN
        assert seq[1] == label_str
    else:
        # Label appears just before the root close.
        # Root close is the second-to-last structural token (EOS is last).
        assert seq[-2] == parser.CLOSE_TOKEN
        assert seq[-3] == label_str


def test_forward_returns_finite_loss(parser):
    import torch

    tree = _toy_tree()
    labels, decoder_input_ids = parser.encode_target(tree)
    text = _reconstruct_text(tree)
    enc = parser.encode_input(text)
    batch = {
        "input_ids": torch.tensor([enc["input_ids"]], dtype=torch.long),
        "attention_mask": torch.tensor([enc["attention_mask"]], dtype=torch.long),
        "labels": torch.tensor([labels], dtype=torch.long),
        "decoder_input_ids": torch.tensor([decoder_input_ids], dtype=torch.long),
    }
    # The replacement lm_head must be trainable. Catching regressions
    # where the head accidentally gets frozen by PEFT/freeze-all.
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


def test_predict_from_text_runs_without_crash(parser):
    """Smoke test for the full greedy decode (constraints + KV cache +
    tree reconstruction)."""
    pred = parser.predict_from_text("Cats sleep. Dogs bark.")
    assert pred is not None
    assert len(pred.edus) >= 1


def test_predict_beam_runs_without_crash(parser):
    """Beam search runs end-to-end with K>1 (exercises the KV-cache
    reorder path)."""
    pred = parser.predict_from_text("Cats sleep. Dogs bark.", num_beams=3)
    assert pred is not None
    assert len(pred.edus) >= 1


def test_gold_edu_source_ranges_align_to_doc_tokenization(parser):
    tree = _toy_tree()
    text = _reconstruct_text(tree)
    ranges = _gold_edu_source_ranges(parser.tokenizer, tree)
    assert len(ranges) == len(tree.edus)
    for s, e in ranges:
        assert 0 <= s <= e
    starts = [s for s, _ in ranges]
    assert starts == sorted(starts)
    full_len = len(parser.tokenizer(text, add_special_tokens=False).input_ids)
    assert ranges[-1][1] <= full_len


def test_predict_with_gold_edus_aligns_to_gold_ranges(parser):
    """The forced-segmentation decode attaches a `_pred_edu_source_ranges`
    list. Forcing contract (shared across all four parsers): leave structure
    free, force boundaries. A random / untrained backbone can choose a
    degenerate root-leaf shape that overshoots a single gold EDU (the root-
    close constraint blocks a mid-source leaf-close, so the leaf just keeps
    eating source tokens). Strict gold-range alignment is only expected
    from a trained model, so here we assert only the contract: ranges
    exist as a list and starts are monotone non-decreasing."""
    tree = _toy_tree()
    pred = parser.predict_with_gold_edus(tree)
    pred_ranges: List[tuple] = getattr(pred, "_pred_edu_source_ranges", [])
    assert isinstance(pred_ranges, list)
    starts = [s for s, _ in pred_ranges]
    assert starts == sorted(starts), f"pred ranges not monotone: {pred_ranges}"


def test_evaluate_gold_edu_emits_expected_metric_keys(parser):
    from iudex.rst.parsers.seq2seq_sexp.train_seq2seq_sexp import _evaluate_gold_edu

    tree = _toy_tree()
    metrics = _evaluate_gold_edu(parser, [("toy.rs4", tree)])
    expected = {"gold_edu_span_f1", "gold_edu_nuc_f1", "gold_edu_rel_f1", "gold_edu_full_f1"}
    assert expected.issubset(metrics.keys())
    for k in expected:
        v = metrics[k]
        assert v == v
        assert 0.0 <= v <= 1.0


@pytest.mark.parametrize("traversal_order", ["preorder", "postorder"])
def test_use_copy_false_predicts_source_ids(traversal_order):
    """Under use_copy=False the predict path's emitted-id stream must
    interleave actual source-subword ids inside leaf frames. The forced
    gold-EDU decode is deterministic for in-leaf positions, so use it to
    bypass the untrained-backbone choice of OPEN slot."""
    parser = _variant_parser(traversal_order, use_copy=False)
    tree = _toy_tree()
    text = _reconstruct_text(tree)
    enc = parser.tokenizer(text, add_special_tokens=False)
    source_ids = enc["input_ids"]
    pred = parser.predict_with_gold_edus(tree)
    pred_ranges: List[tuple] = getattr(pred, "_pred_edu_source_ranges", [])
    if not pred_ranges:
        pytest.skip("random backbone couldn't open any leaf under forced decode (untrained model edge case)")
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
    randomly-initialized backbone, with each predicted EDU's source range
    matching the corresponding gold range. Previously the postorder loop
    could spin OPEN until max_output_length on untrained models."""
    parser = _variant_parser(traversal_order, use_copy=True)
    tree = _toy_tree_3edu()
    gold_ranges = _gold_edu_source_ranges(parser.tokenizer, tree)
    assert len(gold_ranges) == 3

    pred = parser.predict_with_gold_edus(tree)
    pred_ranges = getattr(pred, "_pred_edu_source_ranges", [])
    # Force contract: exactly n EDUs opened and closed, ranges match gold.
    assert len(pred.edus) == 3, f"expected 3 EDUs in predicted tree, got {len(pred.edus)}"
    # _pred_edu_source_ranges may be empty if the sexp parser fell back, but
    # under the new forcing logic it shouldn't.
    assert len(pred_ranges) == 3, f"expected 3 source ranges, got {pred_ranges}"
    # Ranges should match gold exactly (with possible end clamp for
    # truncation, but text is short enough to fit).
    for (gs, ge), (ps, pe) in zip(gold_ranges, pred_ranges):
        assert (ps, pe) == (gs, ge), f"gold {gs, ge} vs pred {ps, pe}"


@pytest.mark.parametrize("traversal_order", ["preorder", "postorder"])
def test_use_copy_false_loss_is_finite_and_flows_to_lm_head(traversal_order):
    """Under use_copy=False the full pretrained lm_head must receive gradient
    from source-content positions (not just from structural slots)."""
    import torch

    parser = _variant_parser(traversal_order, use_copy=False)
    tree = _toy_tree()
    labels, decoder_input_ids = parser.encode_target(tree)
    text = _reconstruct_text(tree)
    inp = parser.encode_input(text)
    # Confirm the label stream contains at least one non-structural id (a
    # source subword). Without that the test would degenerate.
    structural = {parser.open_token_id, parser.close_token_id, parser.tokenizer.eos_token_id} | parser.label_id_set
    has_source_label = any(t not in structural for t in labels)
    assert has_source_label, "test fixture invariant: at least one leaf-content position must exist"

    batch = {
        "input_ids": torch.tensor([inp["input_ids"]], dtype=torch.long),
        "attention_mask": torch.tensor([inp["attention_mask"]], dtype=torch.long),
        "labels": torch.tensor([labels], dtype=torch.long),
        "decoder_input_ids": torch.tensor([decoder_input_ids], dtype=torch.long),
    }
    parser.train()
    out = parser(batch)
    loss = out["loss"]
    assert torch.isfinite(loss).item()
    assert float(loss.item()) > 1e-3
    lm_head_weight = parser._underlying_model().lm_head.weight
    loss.backward()
    assert lm_head_weight.grad is not None
    assert lm_head_weight.grad.abs().sum().item() > 0.0


@pytest.mark.parametrize("traversal_order", ["preorder", "postorder"])
def test_constrain_content_false_runs_end_to_end(traversal_order):
    """Smoke test for round-2 Fix 3: use_copy=False + constrain_content=False
    (free-content decoding) does forward + predict without crashing."""
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
        traversal_order=traversal_order,
        use_copy=False,
        constrain_content=False,
    )
    try:
        parser = Seq2SeqSexpParser(cfg)
    except Exception as e:
        pytest.skip(f"Could not load {SMALL_SEQ2SEQ}: {e!r}")

    tree = _toy_tree()
    pred = parser.predict_from_text("Cats sleep. Dogs bark.")
    assert pred is not None

    import torch

    labels, decoder_input_ids = parser.encode_target(tree)
    text = _reconstruct_text(tree)
    inp = parser.encode_input(text)
    batch = {
        "input_ids": torch.tensor([inp["input_ids"]], dtype=torch.long),
        "attention_mask": torch.tensor([inp["attention_mask"]], dtype=torch.long),
        "labels": torch.tensor([labels], dtype=torch.long),
        "decoder_input_ids": torch.tensor([decoder_input_ids], dtype=torch.long),
    }
    parser.train()
    out = parser(batch)
    assert torch.isfinite(out["loss"]).item()


def test_constrain_content_true_with_use_copy_true_is_default():
    cfg = Seq2SeqSexpConfig(
        train_dir="<unused>",
        dev_dir="<unused>",
        model_name="google-t5/t5-small",
        use_copy=True,
    )
    assert cfg.constrain_content is True


def test_constrain_content_false_with_use_copy_true_raises():
    with pytest.raises(ValueError, match="constrain_content=False requires use_copy=False"):
        Seq2SeqSexpConfig(
            train_dir="<unused>",
            dev_dir="<unused>",
            model_name="google-t5/t5-small",
            use_copy=True,
            constrain_content=False,
        )


# ---------------------------------------------------------------------------
# Round-3 critical fixes
# ---------------------------------------------------------------------------


def _smallest_gum_train_tree():
    """Smallest tree in the GUM train split. Used by Critical-1 regression
    so we exercise `encode_target` on a real binarized tree rather than the
    synthetic 3-EDU fixture that happens to round-trip under both DFS orders.
    """
    import glob
    import os

    candidate_dirs = ["data/gum_12.1.0_notok/train"]
    paths = []
    for d in candidate_dirs:
        if os.path.isdir(d):
            paths.extend(glob.glob(os.path.join(d, "*.rs4")))
            break
    if not paths:
        pytest.skip("GUM train data not present at expected path")
    from iudex.rst.data.reader import read_rst_file

    smallest = None
    smallest_n = 10**9
    for p in paths:
        try:
            tree = read_rst_file(p)
        except Exception:
            continue
        n = len(tree.edus)
        if n < smallest_n and n >= 5:
            smallest_n = n
            smallest = tree
    if smallest is None:
        pytest.skip("no usable GUM train tree found")
    return smallest


@pytest.mark.parametrize("traversal_order", ["preorder", "postorder"])
@pytest.mark.parametrize("use_copy", [True, False])
def test_encode_target_on_real_gum_tree_does_not_raise(traversal_order, use_copy):
    """Critical-1 regression. The pre-fix `_spans_from_parsing_actions`
    replay assumed `parsing_actions` was left-first DFS, but it's actually
    right-first DFS, so on any non-trivial real tree the recursion popped
    an empty list and raised IndexError. We now walk `_build_binary_tree`
    directly, so a real GUM tree (14 EDUs) round-trips through
    `encode_target` without exception and produces structurally valid output."""
    tree = _smallest_gum_train_tree()
    from iudex.rst.data.reader import infer_relation_types

    rels = infer_relation_types(["data/gum_12.1.0_notok/train"])
    cfg = Seq2SeqSexpConfig(
        train_dir="<unused>",
        dev_dir="<unused>",
        model_name=SMALL_SEQ2SEQ,
        relation_types=rels,
        gradient_checkpointing=False,
        amp=False,
        max_input_length=2048,
        max_output_length=8192,
        min_edu_length=1,
        traversal_order=traversal_order,
        use_copy=use_copy,
    )
    try:
        parser = Seq2SeqSexpParser(cfg)
    except Exception as e:
        pytest.skip(f"Could not load {SMALL_SEQ2SEQ}: {e!r}")
    # Some relations from infer_relation_types may not exist on the toy tree.
    # We just need encode_target to not raise.
    encoded = parser.encode_target(tree)
    assert encoded is not None, "encode_target dropped a real GUM tree"
    labels, decoder_input_ids = encoded
    assert len(labels) == len(decoder_input_ids)
    # Structural sanity: one open and one close per EDU and per internal node.
    n_edus = len(tree.edus)
    n_internal = n_edus - 1
    n_open = sum(1 for t in labels if t == parser.open_token_id)
    n_close = sum(1 for t in labels if t == parser.close_token_id)
    assert n_open == n_edus + n_internal
    assert n_close == n_edus + n_internal


def test_constrain_content_false_emits_leaf_text_into_predicted_tree():
    """Critical-2 regression. Under `use_copy=False + constrain_content=False`
    the model emits free (non-source-aligned) subwords inside leaves. The
    pre-fix `_actions_to_sexp_string` only buffered tokens when they
    matched `source_ids[cursor]`, silently dropping all free content and
    yielding empty leaves. We assert that an action stream containing tokens
    that differ from `source_ids[cursor]` still produces non-empty leaf text."""
    parser = _variant_parser("postorder", use_copy=False)
    parser.config.constrain_content = False
    # Source has tokens [10, 11, ...] but action stream emits tokens 100 and
    # 200 (real subword ids in the t5 vocab, deliberately != source_ids[cursor]).
    # Pre-fix code would drop both because they don't match source_ids[0]=10.
    source_ids = [10, 11, 12, 13, 14, 15]
    action_ids = [
        parser.open_token_id,
        100,
        200,
        parser.close_token_id,
        parser.tokenizer.eos_token_id,
    ]
    sexp, edu_texts = parser._actions_to_sexp_string(action_ids, source_ids)
    assert len(edu_texts) == 1, edu_texts
    # The leaf must have decoded SOMETHING; pre-fix it would be empty since
    # neither 100 nor 200 == 10 (source_ids[0]).
    assert edu_texts[0] != "", "free content was dropped; leaf came back empty"
    # And the sexp string should mark the leaf as `<edu>` (placeholder
    # because the leaf was collapsed to a placeholder with text out-of-band).
    assert "<edu>" in sexp


def test_constrain_content_false_buffers_unconditional_in_real_decode():
    """Companion to test_constrain_content_false_emits_leaf_text. Run an
    end-to-end predict under `constrain_content=False` on a tiny model and
    assert the predicted tree at least has non-empty leaves (we can't
    predict tokens, just that they exist)."""
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
        use_copy=False,
        constrain_content=False,
    )
    try:
        parser = Seq2SeqSexpParser(cfg)
    except Exception as e:
        pytest.skip(f"Could not load {SMALL_SEQ2SEQ}: {e!r}")
    tree = _toy_tree()
    pred = parser.predict_with_gold_edus(tree)
    assert pred is not None
    # At least one predicted EDU should have non-empty text (the model
    # emits something at each leaf-content position).
    has_text = any(getattr(edu, "text", "") for edu in pred.edus)
    assert has_text or len(pred.edus) <= 1, (
        f"predicted tree has empty leaves under constrain_content=False: edus={pred.edus}"
    )


def test_gold_edu_forcer_blocks_close_before_target_end_under_cc_false():
    """Important-3 regression. Under `constrain_content=False` the
    GoldEduForcer must prevent leaf-close when the cursor has not reached
    `target_end`, so an undertrained model cannot argmax leaf-close mid-EDU.

    The forcer now signals this with the FORCE_CONTENT sentinel (not a
    CLOSE-excluding frozenset): the caller builds a content-wildcard mask
    that zeroes out all structural ids including CLOSE, so CLOSE is
    unreachable mid-leaf. See `sexp_constraints.narrowed_legal`."""
    from iudex.rst.parsers.common.sexp_constraints import (
        FORCE_CONTENT,
        GoldEduForcer,
        make_initial_state,
    )

    OPEN, CLOSE, EOS = 1, 2, 3
    LABEL_NS = 100
    label_ids = frozenset({LABEL_NS})
    # 4-token source, two 2-token EDUs.
    src = [10, 11, 12, 13]
    state = make_initial_state(
        source_len=4,
        traversal_order="postorder",
        use_copy=False,
        open_id=OPEN,
        close_id=CLOSE,
        eos_id=EOS,
        label_ids=label_ids,
        source_ids=src,
        min_edu_length=1,
        constrain_content=False,
    )
    forcer = GoldEduForcer(2, [(0, 2), (2, 4)])
    # Drive state into the first leaf: open root, open leaf, emit one
    # content token. Should land at cursor=1 mid-leaf, target_end=2.
    state = state.step(OPEN)  # root opens
    forcer.observe(
        make_initial_state(  # before
            source_len=4,
            traversal_order="postorder",
            use_copy=False,
            open_id=OPEN,
            close_id=CLOSE,
            eos_id=EOS,
            label_ids=label_ids,
            source_ids=src,
            min_edu_length=1,
            constrain_content=False,
        ),
        state,
        OPEN,
    )
    before = state
    state = state.step(OPEN)  # leaf opens
    forcer.observe(before, state, OPEN)
    before = state
    state = state.step(10)  # emit one content token, cursor=1
    forcer.observe(before, state, 10)
    assert state.in_edu_leaf
    assert state.cursor == 1
    # target_end for the first leaf is 2; we're mid-leaf. The forcer returns
    # FORCE_CONTENT, whose mask (all ids minus structural_ids()) excludes
    # CLOSE, so leaf-close is unreachable mid-EDU.
    narrowed = forcer.narrowed_legal(state)
    assert narrowed is FORCE_CONTENT
    assert CLOSE in state.structural_ids(), (
        "CLOSE must be a structural id so the FORCE_CONTENT mask zeroes it out mid-EDU"
    )


def test_structural_ids_includes_tokenizer_specials():
    """Minor-5 regression. `structural_ids()` must union in the tokenizer's
    PAD/BOS/UNK/decoder-start ids when the caller provided them, so the
    `constrain_content=False` wildcard masking knocks them out and they
    can't leak into EDU surface text."""
    from iudex.rst.parsers.common.sexp_constraints import SexpDecodingState

    state = SexpDecodingState(
        source_len=3,
        traversal_order="postorder",
        use_copy=False,
        open_id=1,
        close_id=2,
        eos_id=3,
        label_ids=frozenset({100}),
        source_ids=(10, 11, 12),
        tokenizer_special_ids=frozenset({0, 99, 7}),
    )
    sids = state.structural_ids()
    assert {1, 2, 3, 100, 0, 99, 7} <= sids
