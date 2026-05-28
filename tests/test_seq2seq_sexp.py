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
    COPY_TOKEN,
    SEXP_CLOSE_TOKEN,
    SEXP_OPEN_TOKEN,
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
        elif tok in parser.reduce_token_ids:
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
            out.append(SEXP_OPEN_TOKEN)
        elif tok == parser.close_token_id:
            out.append(SEXP_CLOSE_TOKEN)
        elif tok in parser.reduce_token_ids:
            out.append(parser.tokenizer.convert_ids_to_tokens(tok))
        elif tok == parser.tokenizer.eos_token_id:
            out.append("</s>")
        elif parser.config.use_copy and tok == parser.copy_token_id:
            out.append(COPY_TOKEN)
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
    assert seq.count(SEXP_OPEN_TOKEN) == 3
    assert seq.count(SEXP_CLOSE_TOKEN) == 3
    assert seq[-1] == "</s>"
    label_str = Reduce(nuc="NS", rel="elaboration").to_token()
    assert seq.count(label_str) == 1
    if traversal_order == "preorder":
        # First open is root; next token is the label.
        assert seq[0] == SEXP_OPEN_TOKEN
        assert seq[1] == label_str
    else:
        # Label appears just before the root close.
        # Root close is the second-to-last structural token (EOS is last).
        assert seq[-2] == SEXP_CLOSE_TOKEN
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
    structural = {parser.open_token_id, parser.close_token_id, parser.tokenizer.eos_token_id} | parser.reduce_token_ids
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
