"""Regression tests for two generative-parser paths the runtime smoke sweep
missed: (1) checkpoint save -> reload-from-disk -> predict (strict
`load_state_dict` of the carved embedding Parameter, the small action head, and
the resized action vocab), and (2) seq2seq_sr batched greedy decode with >=2
ragged documents (the `ShiftReduceDecodeState` batched path).

Tiny models, CPU. Skips gracefully if a model can't be fetched.
"""

from __future__ import annotations

import dataclasses
import os

import pytest
import torch

pytest.importorskip("transformers")

from iudex.common.training import save_checkpoint  # noqa: E402
from iudex.rst.parsers.common.inference import load_parser_from_checkpoint  # noqa: E402

T5 = os.environ.get("IUDEX_TEST_SEQ2SEQ_MODEL", "google-t5/t5-small")
CAUSAL = os.environ.get("IUDEX_TEST_CAUSAL_MODEL", "hf-internal-testing/tiny-random-Gemma3ForCausalLM")


def _imports(parser_kind: str):
    if parser_kind == "seq2seq_sr":
        from iudex.rst.parsers.seq2seq_sr.configuration_seq2seq_sr import Seq2SeqSRConfig
        from iudex.rst.parsers.seq2seq_sr.modeling_seq2seq_sr import Seq2SeqSRParser

        return Seq2SeqSRConfig, Seq2SeqSRParser
    if parser_kind == "decoder_only_sr":
        from iudex.rst.parsers.decoder_only_sr.configuration_decoder_only_sr import DecoderOnlySRConfig
        from iudex.rst.parsers.decoder_only_sr.modeling_decoder_only_sr import DecoderOnlySRParser

        return DecoderOnlySRConfig, DecoderOnlySRParser
    if parser_kind == "seq2seq_sexp":
        from iudex.rst.parsers.seq2seq_sexp.configuration_seq2seq_sexp import Seq2SeqSexpConfig
        from iudex.rst.parsers.seq2seq_sexp.modeling_seq2seq_sexp import Seq2SeqSexpParser

        return Seq2SeqSexpConfig, Seq2SeqSexpParser
    from iudex.rst.parsers.decoder_only_sexp.configuration_decoder_only_sexp import DecoderOnlySexpConfig
    from iudex.rst.parsers.decoder_only_sexp.modeling_decoder_only_sexp import DecoderOnlySexpParser

    return DecoderOnlySexpConfig, DecoderOnlySexpParser


def _make_parser(parser_kind: str, model: str, extra: dict):
    cfg_cls, parser_cls = _imports(parser_kind)
    cfg = cfg_cls.from_dict(
        {
            "train_dir": "<unused>",
            "dev_dir": "<unused>",
            "model_name": model,
            "relation_types": [["elaboration", "rst"], ["joint", "multinuc"]],
            "amp": False,
            "max_input_length": 256,
            "max_output_length": 512,
            "num_beams": 1,
            **extra,
        }
    )
    try:
        return parser_cls(cfg)
    except Exception as e:  # network / gated model / arch mismatch on this host
        pytest.skip(f"Could not construct {parser_kind} with {model}: {e!r}")


# (id, parser_kind, model, extra) -- covers carve+small-head (use_copy) and the
# full-head + modules_to_save path (use_copy=False), both backbones.
ROUNDTRIP_CASES = [
    ("seq2seq_sr", "seq2seq_sr", T5, {}),
    ("decoder_only_sr", "decoder_only_sr", CAUSAL, {}),
    (
        "seq2seq_sexp_copy",
        "seq2seq_sexp",
        T5,
        {"traversal_order": "postorder", "use_copy": True, "constrain_content": True},
    ),
    (
        "seq2seq_sexp_nocopy",
        "seq2seq_sexp",
        T5,
        {"traversal_order": "postorder", "use_copy": False, "constrain_content": True},
    ),
    (
        "decoder_only_sexp_copy",
        "decoder_only_sexp",
        CAUSAL,
        {"traversal_order": "preorder", "use_copy": True, "constrain_content": True},
    ),
]


@pytest.mark.parametrize("name,parser_kind,model,extra", ROUNDTRIP_CASES, ids=[c[0] for c in ROUNDTRIP_CASES])
def test_checkpoint_roundtrip_then_predict(tmp_path, name, parser_kind, model, extra):
    """Save a freshly-built parser, reload it strictly from disk, and predict.
    Catches state_dict drift in the carved embedding / small head / resized vocab."""
    cfg_cls, parser_cls = _imports(parser_kind)
    parser = _make_parser(parser_kind, model, extra)
    opt = torch.optim.SGD([p for p in parser.parameters() if p.requires_grad], lr=0.1)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda _s: 1.0)
    ckpt = str(tmp_path / "best_model.pt")
    save_checkpoint(
        ckpt,
        parser,
        opt,
        sched,
        config=dataclasses.asdict(parser.config),
        config_hash="test",
        global_step=0,
        epoch=0,
        best_val=0.0,
        parser_kind=parser_kind,
    )
    # Strict reload from disk (the real from_pretrained path).
    loaded = load_parser_from_checkpoint(ckpt, torch.device("cpu"), cfg_cls, parser_cls)
    tree = loaded.predict_from_text("The plan was clear. The result was not.")
    assert len(tree.edus) >= 1


def test_seq2seq_sr_batched_greedy_ragged():
    """Batched greedy over 2 ragged documents exercises the per-row
    ShiftReduceDecodeState path (padding + per-row done tracking)."""
    parser = _make_parser("seq2seq_sr", T5, {})
    texts = ["Short doc here.", "A noticeably longer document with several more tokens to force ragged lengths."]
    trees = parser.predict_batch_from_texts(texts, num_beams=1)
    assert len(trees) == 2
    assert all(len(t.edus) >= 1 for t in trees)
