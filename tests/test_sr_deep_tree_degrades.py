"""Regression: an undertrained generative SR model can emit a *valid* action
sequence that builds a pathologically deep tree (e.g. a long doc decoded as
hundreds of single-token EDUs, then a linear reduce chain). `from_shift_reduce`
-> `binarize_tree` -> `compute_edu_yields` recurses once per node and blows
Python's recursion limit. Real trees are shallow (GUM maxes ~235 EDUs), so this
only fires on untrusted model output, where degrading to a single-EDU tree is
correct (mirrors the sexp parsers' `_tree_from_emitted`).

Caught live by the full-GUM small-model regime: seq2seq_sr crashed at the
epoch-1 dev eval before this guard existed.
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("transformers")

T5 = os.environ.get("IUDEX_TEST_SEQ2SEQ_MODEL", "google-t5/t5-small")
CAUSAL = os.environ.get("IUDEX_TEST_CAUSAL_MODEL", "hf-internal-testing/tiny-random-Gemma3ForCausalLM")


def _build(parser_kind: str, model: str):
    if parser_kind == "seq2seq_sr":
        from iudex.rst.parsers.seq2seq_sr.configuration_seq2seq_sr import Seq2SeqSRConfig
        from iudex.rst.parsers.seq2seq_sr.modeling_seq2seq_sr import Seq2SeqSRParser

        cfg_cls, parser_cls = Seq2SeqSRConfig, Seq2SeqSRParser
    else:
        from iudex.rst.parsers.decoder_only_sr.configuration_decoder_only_sr import DecoderOnlySRConfig
        from iudex.rst.parsers.decoder_only_sr.modeling_decoder_only_sr import DecoderOnlySRParser

        cfg_cls, parser_cls = DecoderOnlySRConfig, DecoderOnlySRParser
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
        }
    )
    try:
        return parser_cls(cfg)
    except Exception as e:  # network / gated / arch mismatch on this host
        pytest.skip(f"Could not construct {parser_kind} with {model}: {e!r}")


@pytest.mark.parametrize("parser_kind,model", [("seq2seq_sr", T5), ("decoder_only_sr", CAUSAL)])
def test_deep_action_sequence_degrades_not_crashes(parser_kind, model):
    parser = _build(parser_kind, model)
    # ~1300 single-token EDUs then a linear reduce chain => ~1300-deep tree,
    # past CPython's default 1000-frame limit.
    src = parser.tokenizer("word " * 1300, add_special_tokens=False).input_ids[:1300]
    if len(src) < 1200:
        pytest.skip("tokenizer produced too few ids to force deep recursion")
    action_ids: list[int] = []
    for s in src:
        action_ids += [s, parser.shift_token_id]
    action_ids += [sorted(parser.reduce_token_ids)[0]] * (len(src) - 1)
    tree = parser._tree_from_action_sequence(action_ids, src)  # must not raise RecursionError
    assert len(tree.edus) >= 1
