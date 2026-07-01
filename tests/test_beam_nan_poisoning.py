"""Regression tests for the shared beam-search scoring helpers
(`common/seqgen.py`), used by all four generative parsers' beam decoders.

Covers two historical bugs:

1. Dead-beam NaN poisoning: after a beam finished, `beam_scores[j]=-inf` was
   set, but the beam's logits row stayed fully masked. `F.log_softmax(all
   -inf)` returns NaN, `cum = -inf + NaN = NaN`, and `topk` ranks NaN above
   any finite score, so the dead beam's children crowded out live beams.
   Fix: replace NaN in `cum` with `-inf` before `topk` (lives in
   `beam_topk_step`).

2. Renormalized constrained scoring: `log_softmax` over PRE-masked logits
   renormalizes the model distribution over the legal set, making
   heavily-constrained steps nearly free. Under the sexp constraints the
   in-leaf legal set is ~2 ids, so skipping an EDU boundary cost ~0 and beam
   search collapsed documents to a few giant EDUs. Fix: `beam_topk_step`
   scores with the raw full-vocab log-probs and masks AFTER the softmax.
"""

import torch
import torch.nn.functional as F

from iudex.rst.parsers.common.seqgen import beam_topk_step, select_best_beam


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


def test_dead_beam_excluded_by_beam_topk_step():
    """`beam_topk_step` never selects children of a dead beam (all-False mask
    row + -inf score)."""
    K, V = 3, 5
    logits = torch.zeros((K, V))
    legal = torch.zeros((K, V), dtype=torch.bool)
    legal[1, 2] = True
    legal[2, 3] = True
    beam_scores = torch.tensor([float("-inf"), -2.0, -1.0])
    _, parents, _ = beam_topk_step(beam_scores, logits, legal, K)
    finite_children = [
        p for p, s in zip(parents, beam_topk_step(beam_scores, logits, legal, K)[0].tolist()) if s > float("-inf")
    ]
    assert 0 not in finite_children, f"dead beam (0) leaked into finite-score parents: {parents}"


def test_scoring_is_not_renormalized_over_legal_set():
    """A constrained step must cost the TRUE model probability of the chosen
    action, not the probability renormalized over the legal set. With one
    legal action carrying tiny raw probability, the renormalized score would
    be 0.0 (free); the raw score must be very negative."""
    K, V = 1, 10
    logits = torch.zeros((K, V))
    logits[0, 0] = 20.0  # model overwhelmingly prefers an ILLEGAL action
    legal = torch.zeros((K, V), dtype=torch.bool)
    legal[0, 1] = True  # the only legal action has ~e^-20 raw probability
    beam_scores = torch.zeros(K)
    top_scores, _, actions = beam_topk_step(beam_scores, logits, legal, K)
    assert actions[0] == 1
    assert top_scores[0].item() < -15.0, (
        f"constrained step scored {top_scores[0].item():.3f}; ~0 means the "
        "log-softmax renormalized over the legal set again"
    )


def test_select_best_beam_prefers_finished():
    """An unfinished (truncated) candidate must not outrank a finished parse,
    even with a better length-normalized score; unfinished is fallback-only."""
    finished = {"score": -50.0, "length": 100, "finished": True}
    truncated = {"score": -1.0, "length": 100, "finished": False}
    assert select_best_beam([truncated, finished]) is finished
    assert select_best_beam([truncated]) is truncated


def test_all_four_parsers_use_shared_beam_step():
    """All four parsers route beam expansion through `beam_topk_step` (where
    the NaN guard and raw-scoring fix live)."""
    import os

    paths = [
        "iudex/rst/parsers/seq2seq_sr/modeling_seq2seq_sr.py",
        "iudex/rst/parsers/decoder_only_sr/modeling_decoder_only_sr.py",
        "iudex/rst/parsers/seq2seq_sexp/modeling_seq2seq_sexp.py",
        "iudex/rst/parsers/decoder_only_sexp/modeling_decoder_only_sexp.py",
    ]
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for rel in paths:
        with open(os.path.join(repo_root, rel), encoding="utf-8") as f:
            text = f.read()
        assert "beam_topk_step(beam_scores, logits, legal, K)" in text, f"{rel} not using the shared beam step"
    with open(os.path.join(repo_root, "iudex/rst/parsers/common/seqgen.py"), encoding="utf-8") as f:
        seqgen = f.read()
    assert "torch.isnan(cum)" in seqgen, "NaN guard missing from beam_topk_step"
