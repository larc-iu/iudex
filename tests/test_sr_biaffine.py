"""Tests for the transition-based shift-reduce parser `sr_biaffine`.

Covers the teacher-forced forward (scalar loss with gradient), the greedy
legality-masked decode (well-formed, gold-EDU-count-preserving tree),
single-EDU edge cases, and a short overfit that confirms the oracle replay
and the decoder agree (the parser can recover a tree it was trained on).

Uses a small BERT-style encoder; override with IUDEX_TEST_ENCODER. The test
skips if the encoder cannot be loaded (no network / not cached), matching the
other parser tests.
"""

import os

import pytest

pytest.importorskip("transformers")

import torch

from iudex.rst.data.tree import Reduce, RstTree, Shift
from iudex.rst.parsers.sr_biaffine.configuration_sr_biaffine import SRBiaffineConfig
from iudex.rst.parsers.sr_biaffine.modeling_sr_biaffine import SRBiaffineParser

# Any BERT-style encoder with CLS/SEP works. Default to a tiny random model;
# set IUDEX_TEST_ENCODER=bert-base-uncased (or similar) to run against a cache.
SMALL_ENCODER = os.environ.get("IUDEX_TEST_ENCODER", "hf-internal-testing/tiny-random-BertModel")

RELS = [("cause", "rst"), ("circumstance", "rst"), ("joint", "multinuc")]


def _toy_tree() -> RstTree:
    # ((e0 e1)NS:cause (e2 e3)SN:circumstance)NN:joint
    actions = [
        Shift(edu_text="Cats sleep a lot"),
        Shift(edu_text="because they are lazy"),
        Reduce(nuc="NS", rel="cause"),
        Shift(edu_text="Dogs bark loudly"),
        Shift(edu_text="when strangers approach"),
        Reduce(nuc="SN", rel="circumstance"),
        Reduce(nuc="NN", rel="joint"),
    ]
    return RstTree.from_shift_reduce(actions, relation_types=RELS)


def _build_parser() -> SRBiaffineParser:
    cfg = SRBiaffineConfig(
        train_dir="<unused>",
        dev_dir="<unused>",
        model_name=SMALL_ENCODER,
        relation_types=RELS,
        amp=False,
        ffn_hidden_size=32,
        action_ffn_hidden_size=32,
        dropout=0.0,
    )
    try:
        return SRBiaffineParser(cfg)
    except Exception as e:  # network failure, missing cache, etc.
        pytest.skip(f"Could not load {SMALL_ENCODER}: {e!r}")


@pytest.fixture(scope="module")
def parser() -> SRBiaffineParser:
    return _build_parser()


def test_forward_scalar_loss_with_grad(parser):
    parser.train()
    out = parser(_toy_tree())
    assert out["loss"].ndim == 0
    assert out["loss"].requires_grad
    out["loss"].backward()
    grad_sum = sum(p.grad.abs().sum().item() for p in parser.parameters() if p.grad is not None)
    assert grad_sum > 0


def test_predict_is_well_formed(parser):
    tree = _toy_tree()
    pred = parser.predict(tree)
    assert isinstance(pred, RstTree)
    assert pred.is_binary
    # Gold EDU segmentation is preserved (this parser does not segment).
    assert len(pred.edus) == len(tree.edus)
    assert pred.edu_strings == tree.edu_strings


def test_single_edu_tree_is_a_noop(parser):
    solo = RstTree.from_shift_reduce([Shift(edu_text="Only one.")], relation_types=RELS)
    loss = parser(solo)["loss"]
    assert float(loss.detach()) == 0.0
    assert loss.requires_grad
    assert len(parser.predict(solo).edus) == 1


def test_overfits_single_tree(parser):
    """The oracle replay (training) and the greedy decoder must agree: after
    overfitting one tree, predict should recover its exact span set."""
    tree = _toy_tree()
    opt = torch.optim.Adam(parser.parameters(), lr=2e-4)
    parser.train()
    for _ in range(250):
        opt.zero_grad()
        loss = parser(tree)["loss"]
        loss.backward()
        opt.step()
    pred = parser.predict(tree)
    assert sorted(pred.spans_with_ranges()) == sorted(tree.spans_with_ranges())
