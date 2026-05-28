"""Regression test for the dead-beam NaN-poisoning bug across all four
beam-search decoders (seq2seq_sr, decoder_only_sr, seq2seq_sexp,
decoder_only_sexp).

The bug: after a beam finished, `beam_scores[j]=-inf` was set, but the
beam's masked-logits row stayed all `-inf`. `F.log_softmax(all -inf)`
returns NaN. `cum = -inf + NaN = NaN`. `topk` ranks NaN above any finite
score, so the dead beam's children would crowd out live beams.

Fix: replace NaN in `cum` with `-inf` before `topk`.
"""

import torch
import torch.nn.functional as F


def test_dead_beam_nan_repro_without_fix():
    """The minimal failing reproducer (reviewer A): without the NaN -> -inf
    sanitization, parents collapse onto the dead beam."""
    K, V = 3, 5
    masked = torch.full((K, V), float("-inf"))
    masked[1, 2] = -1.0
    masked[2, 3] = -0.5
    beam_scores = torch.tensor([float("-inf"), -2.0, -1.0])
    cum = beam_scores.unsqueeze(1) + F.log_softmax(masked.float(), dim=-1)
    parents = (cum.view(-1).topk(K).indices // V).tolist()
    assert parents == [0, 0, 0], "Repro premise broken: dead beam no longer wins"


def test_dead_beam_nan_fix():
    """With NaN -> -inf, top-K parents come exclusively from live beams."""
    K, V = 3, 5
    masked = torch.full((K, V), float("-inf"))
    masked[1, 2] = -1.0
    masked[2, 3] = -0.5
    beam_scores = torch.tensor([float("-inf"), -2.0, -1.0])
    cum = beam_scores.unsqueeze(1) + F.log_softmax(masked.float(), dim=-1)
    cum = torch.where(torch.isnan(cum), torch.full_like(cum, float("-inf")), cum)
    parents = (cum.view(-1).topk(K).indices // V).tolist()
    assert 0 not in parents, f"dead beam (0) leaked into selected parents: {parents}"
    assert set(parents).issubset({1, 2}), parents


def test_all_four_parsers_have_nan_guard():
    """All four parser modules contain the `torch.isnan(cum)` sanitization."""
    paths = [
        "iudex/rst/parsers/seq2seq_sr/modeling_seq2seq_sr.py",
        "iudex/rst/parsers/decoder_only_sr/modeling_decoder_only_sr.py",
        "iudex/rst/parsers/seq2seq_sexp/modeling_seq2seq_sexp.py",
        "iudex/rst/parsers/decoder_only_sexp/modeling_decoder_only_sexp.py",
    ]
    import os

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for rel in paths:
        full = os.path.join(repo_root, rel)
        with open(full, encoding="utf-8") as f:
            text = f.read()
        assert "torch.isnan(cum)" in text, f"missing NaN guard in {rel}"
