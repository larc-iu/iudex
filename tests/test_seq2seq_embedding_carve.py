"""Tests for the new-token embedding carve-out shared by the four seq2seq /
decoder-only RST parsers.

The carve-out replaces the old PEFT `modules_to_save=['embed_tokens']` scheme
(which kept a frozen copy plus a full trainable copy of the vocab x hidden
matrix) with a frozen base embedding plus a small trainable
`new_token_embeddings` Parameter holding only the newly-added action-token
rows. These tests assert, per parser:
  * trainable embedding params == n_new x hidden, not full vocab x hidden,
    and the frozen base has requires_grad=False (test b);
  * a forward+backward flows gradient to `new_token_embeddings` and the
    frozen base weight's `.grad` stays None (test c);
  * the checkpoint round-trip (state_dict save -> fresh Parser(cfg) ->
    load_state_dict(strict=True)) reproduces identical logits (test a).

Tiny t5-small / tiny-random-Gemma3 backbones, CPU only, no training.
"""

import os

import pytest

pytest.importorskip("transformers")

import torch

from iudex.rst.data.tree import Reduce, RstTree, Shift
from iudex.rst.parsers.decoder_only_sexp.configuration_decoder_only_sexp import DecoderOnlySexpConfig
from iudex.rst.parsers.decoder_only_sexp.modeling_decoder_only_sexp import DecoderOnlySexpParser
from iudex.rst.parsers.decoder_only_sr.configuration_decoder_only_sr import DecoderOnlySRConfig
from iudex.rst.parsers.decoder_only_sr.modeling_decoder_only_sr import DecoderOnlySRParser
from iudex.rst.parsers.seq2seq_sexp.configuration_seq2seq_sexp import Seq2SeqSexpConfig
from iudex.rst.parsers.seq2seq_sexp.modeling_seq2seq_sexp import Seq2SeqSexpParser
from iudex.rst.parsers.seq2seq_sr.configuration_seq2seq_sr import Seq2SeqSRConfig
from iudex.rst.parsers.seq2seq_sr.modeling_seq2seq_sr import Seq2SeqSRParser

SMALL_SEQ2SEQ = os.environ.get("IUDEX_TEST_SEQ2SEQ_MODEL", "google-t5/t5-small")
SMALL_CAUSAL = os.environ.get("IUDEX_TEST_CAUSAL_MODEL", "hf-internal-testing/tiny-random-Gemma3ForCausalLM")

RELATION_TYPES = [("elaboration", "rst")]


def _toy_tree() -> RstTree:
    actions = [
        Shift(edu_text="Cats sleep."),
        Shift(edu_text="Dogs bark."),
        Reduce(nuc="NS", rel="elaboration"),
    ]
    return RstTree.from_shift_reduce(actions, relation_types=RELATION_TYPES)


def _seq2seq_sr_cfg():
    return Seq2SeqSRConfig(
        train_dir="<unused>",
        dev_dir="<unused>",
        model_name=SMALL_SEQ2SEQ,
        relation_types=RELATION_TYPES,
        gradient_checkpointing=False,
        amp=False,
        max_input_length=128,
        max_output_length=64,
        min_edu_length=1,
    )


def _decoder_only_sr_cfg():
    return DecoderOnlySRConfig(
        train_dir="<unused>",
        dev_dir="<unused>",
        model_name=SMALL_CAUSAL,
        relation_types=RELATION_TYPES,
        gradient_checkpointing=False,
        amp=False,
        max_input_length=128,
        max_output_length=128,
        min_edu_length=1,
    )


def _seq2seq_sexp_cfg():
    return Seq2SeqSexpConfig(
        train_dir="<unused>",
        dev_dir="<unused>",
        model_name=SMALL_SEQ2SEQ,
        relation_types=RELATION_TYPES,
        gradient_checkpointing=False,
        amp=False,
        max_input_length=128,
        max_output_length=128,
        use_copy=True,
    )


def _decoder_only_sexp_cfg():
    return DecoderOnlySexpConfig(
        train_dir="<unused>",
        dev_dir="<unused>",
        model_name=SMALL_CAUSAL,
        relation_types=RELATION_TYPES,
        gradient_checkpointing=False,
        amp=False,
        max_input_length=128,
        max_output_length=128,
        use_copy=True,
    )


PARSERS = {
    "seq2seq_sr": (Seq2SeqSRParser, _seq2seq_sr_cfg),
    "decoder_only_sr": (DecoderOnlySRParser, _decoder_only_sr_cfg),
    "seq2seq_sexp": (Seq2SeqSexpParser, _seq2seq_sexp_cfg),
    "decoder_only_sexp": (DecoderOnlySexpParser, _decoder_only_sexp_cfg),
}


def _build(parser_cls, cfg_fn):
    try:
        return parser_cls(cfg_fn())
    except Exception as e:  # offline / missing weights
        pytest.skip(f"Could not build {parser_cls.__name__}: {e!r}")


def _make_batch(parser):
    """A teacher-forced batch for any of the four parsers. SR/decoder-only-SR
    and the sexp pair share the `encode_target -> (a, b)` shape; only the
    decoder-side key name differs (decoder_input_ids for seq2seq, input_ids
    for decoder-only)."""
    enc = parser.encode_target(_toy_tree())
    assert enc is not None
    a, b = enc
    # seq2seq parsers return (labels, decoder_input_ids); decoder-only return
    # (input_ids, labels). Disambiguate by class.
    is_seq2seq = parser.__class__.__name__.startswith("Seq2Seq")
    if is_seq2seq:
        labels, decoder_input_ids = a, b
        text = _reconstruct(parser)
        src = parser.tokenizer(text, add_special_tokens=False).input_ids
        enc_input = parser.tokenizer(text, add_special_tokens=True).input_ids
        return {
            "input_ids": torch.tensor([enc_input], dtype=torch.long),
            "attention_mask": torch.ones((1, len(enc_input)), dtype=torch.long),
            "decoder_input_ids": torch.tensor([decoder_input_ids], dtype=torch.long),
            "labels": torch.tensor([labels], dtype=torch.long),
        }
    input_ids, labels = a, b
    return {
        "input_ids": torch.tensor([input_ids], dtype=torch.long),
        "attention_mask": torch.ones((1, len(input_ids)), dtype=torch.long),
        "labels": torch.tensor([labels], dtype=torch.long),
    }


def _reconstruct(parser):
    from iudex.rst.parsers.seq2seq_sr.modeling_seq2seq_sr import _reconstruct_text

    return _reconstruct_text(_toy_tree())


@pytest.mark.parametrize("name", list(PARSERS))
def test_trainable_embedding_is_only_new_rows(name):
    parser_cls, cfg_fn = PARSERS[name]
    parser = _build(parser_cls, cfg_fn)

    assert hasattr(parser, "new_token_embeddings")
    new = parser.new_token_embeddings
    n_old = parser._embed_n_old
    hidden = parser._underlying_model().get_input_embeddings().weight.shape[1]
    n_total = len(parser.tokenizer)
    assert new.requires_grad
    assert tuple(new.shape) == (n_total - n_old, hidden)

    # The base embedding matrix is frozen.
    base_weight = parser._underlying_model().get_input_embeddings().weight
    assert base_weight.requires_grad is False
    assert base_weight.shape[0] == n_total
    # The only trainable params whose name carries embed/shared are the
    # carved new-rows Parameter, never the full vocab x hidden matrix.
    embed_trainable = [
        (pn, p)
        for pn, p in parser.named_parameters()
        if p.requires_grad and ("embed" in pn.lower() or "shared" in pn.lower())
    ]
    assert all(p.numel() == (n_total - n_old) * hidden for _, p in embed_trainable), [
        (pn, tuple(p.shape)) for pn, p in embed_trainable
    ]


@pytest.mark.parametrize("name", list(PARSERS))
def test_grad_flows_to_new_rows_only(name):
    parser_cls, cfg_fn = PARSERS[name]
    parser = _build(parser_cls, cfg_fn)
    parser.train()
    parser.zero_grad(set_to_none=True)

    batch = _make_batch(parser)
    out = parser(batch)
    loss = out["loss"]
    assert torch.isfinite(loss).item()
    loss.backward()

    assert parser.new_token_embeddings.grad is not None
    assert torch.isfinite(parser.new_token_embeddings.grad).all().item()
    # The frozen base embedding never accumulates gradient.
    base_weight = parser._underlying_model().get_input_embeddings().weight
    assert base_weight.grad is None


@pytest.mark.parametrize("name", list(PARSERS))
def test_checkpoint_roundtrip_identical_logits(name):
    parser_cls, cfg_fn = PARSERS[name]
    cfg = cfg_fn()
    try:
        parser = parser_cls(cfg)
    except Exception as e:
        pytest.skip(f"Could not build {parser_cls.__name__}: {e!r}")
    parser.eval()

    batch = _make_batch(parser)
    is_seq2seq = parser.__class__.__name__.startswith("Seq2Seq")
    with torch.no_grad():
        if is_seq2seq:
            ref = parser.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                decoder_input_ids=batch["decoder_input_ids"],
                return_dict=True,
            ).logits
        else:
            ref = parser.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                return_dict=True,
            ).logits

    state = parser.state_dict()
    assert "new_token_embeddings" in state

    fresh = parser_cls(cfg)
    fresh.load_state_dict(state, strict=True)
    fresh.eval()
    with torch.no_grad():
        if is_seq2seq:
            got = fresh.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                decoder_input_ids=batch["decoder_input_ids"],
                return_dict=True,
            ).logits
        else:
            got = fresh.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                return_dict=True,
            ).logits

    assert torch.equal(ref, got), f"{name}: logits diverged after state_dict round-trip"
