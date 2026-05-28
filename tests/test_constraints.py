"""Unit tests for the shift-reduce decoding validity constraints.

Covers `_replay` directly (cursor / stack / edu_has_content) and the
`__call__` legal-mask behaviour over synthetic vocabularies.
"""

import pytest
import torch

from iudex.rst.parsers.seq2seq_sr.constraints import E2EShiftReduceValidityProcessor


# Toy vocabulary. Five source-copy IDs (0..4), one shift, three reduces, eos, pad.
SRC_IDS = [10, 11, 12, 13, 14]  # input subword IDs (arbitrary values)
SHIFT = 100
REDUCE_NS = 200
REDUCE_SN = 201
REDUCE_NN = 202
EOS = 300
PAD = 301
DECODER_START = 302
VOCAB_SIZE = 400


def _make_processor(source_ids=SRC_IDS):
    return E2EShiftReduceValidityProcessor(
        per_row_source_ids=[source_ids],
        shift_id=SHIFT,
        reduce_ids=[REDUCE_NS, REDUCE_SN, REDUCE_NN],
        eos_id=EOS,
        pad_id=PAD,
        decoder_start_id=DECODER_START,
        num_beams=1,
    )


# ---------------------------------------------------------------------------
# _replay
# ---------------------------------------------------------------------------


def test_replay_empty_prefix():
    p = _make_processor()
    cursor, stack, has_content = p._replay([])
    assert cursor == 0 and stack == 0 and has_content is False


def test_replay_skips_decoder_start_and_pad():
    p = _make_processor()
    cursor, stack, has_content = p._replay([DECODER_START, PAD, PAD])
    assert cursor == 0 and stack == 0 and has_content is False


def test_replay_after_one_source_copy():
    p = _make_processor()
    cursor, stack, has_content = p._replay([DECODER_START, 10])
    assert cursor == 1
    assert stack == 0
    assert has_content is True


def test_replay_after_shift():
    p = _make_processor()
    # Emit two source tokens, then SHIFT to commit the first EDU.
    cursor, stack, has_content = p._replay([DECODER_START, 10, 11, SHIFT])
    assert cursor == 2
    assert stack == 1
    assert has_content is False  # SHIFT just fired; current EDU is empty


def test_replay_two_shifts_then_reduce():
    p = _make_processor()
    cursor, stack, has_content = p._replay([DECODER_START, 10, SHIFT, 11, SHIFT, REDUCE_NS])
    assert cursor == 2
    assert stack == 1  # 2 - 1 from the reduce
    assert has_content is False


def test_replay_stops_at_eos():
    p = _make_processor()
    cursor, stack, has_content = p._replay([DECODER_START, 10, SHIFT, EOS, 11, SHIFT])
    assert cursor == 1
    assert stack == 1
    assert has_content is False


# ---------------------------------------------------------------------------
# Legal-mask behaviour at key inflection points
# ---------------------------------------------------------------------------


def _legal_ids(processor, prefix_ids: list[int]) -> set[int]:
    """Return the set of token IDs that survive the processor's mask."""
    input_ids = torch.tensor([prefix_ids], dtype=torch.long)
    scores = torch.zeros((1, VOCAB_SIZE))  # zero so we can detect '-inf' clearly
    out = processor(input_ids, scores)
    finite_mask = torch.isfinite(out[0])
    return set(int(i) for i in finite_mask.nonzero(as_tuple=True)[0].tolist())


def test_legal_at_start_only_first_source():
    p = _make_processor()
    legal = _legal_ids(p, [DECODER_START])
    # No content yet, stack empty, cursor at 0: only the first source token.
    assert legal == {SRC_IDS[0]}


def test_legal_after_one_token_shift_or_continue():
    p = _make_processor()
    # Emitted one source token. SHIFT is legal now, plus next source.
    legal = _legal_ids(p, [DECODER_START, SRC_IDS[0]])
    assert legal == {SRC_IDS[1], SHIFT}


def test_legal_after_two_shifts_reduce_allowed():
    p = _make_processor()
    # Two EDUs committed; stack=2 -> all reduces legal. Cursor at 2 -> next source still legal.
    prefix = [DECODER_START, SRC_IDS[0], SHIFT, SRC_IDS[1], SHIFT]
    legal = _legal_ids(p, prefix)
    # No content yet in the third EDU (just shifted), so SHIFT not legal.
    assert legal == {SRC_IDS[2], REDUCE_NS, REDUCE_SN, REDUCE_NN}


def test_legal_at_end_with_singleton_stack_only_eos():
    p = _make_processor(source_ids=[SRC_IDS[0], SRC_IDS[1]])
    # n=2 doc: copy first src, shift, copy second src, shift, reduce -> stack=1, cursor=2.
    prefix = [DECODER_START, SRC_IDS[0], SHIFT, SRC_IDS[1], SHIFT, REDUCE_NS]
    legal = _legal_ids(p, prefix)
    assert legal == {EOS}


def test_legal_at_end_with_stack_gt1_only_reduces():
    p = _make_processor(source_ids=[SRC_IDS[0], SRC_IDS[1], SRC_IDS[2]])
    # n=3 doc, all shifted but no reduces yet. Cursor=3, stack=3.
    prefix = [DECODER_START, SRC_IDS[0], SHIFT, SRC_IDS[1], SHIFT, SRC_IDS[2], SHIFT]
    legal = _legal_ids(p, prefix)
    assert legal == {REDUCE_NS, REDUCE_SN, REDUCE_NN}


# ---------------------------------------------------------------------------
# Sanity: a full constrained walk on n=5 EDUs in two tree shapes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tree_shape",
    [
        # Left-spine: ((((1,2),3),4),5) -> shifts and reduces alternate after first 2 shifts
        [
            SRC_IDS[0],
            SHIFT,
            SRC_IDS[1],
            SHIFT,
            REDUCE_NS,
            SRC_IDS[2],
            SHIFT,
            REDUCE_NS,
            SRC_IDS[3],
            SHIFT,
            REDUCE_NS,
            SRC_IDS[4],
            SHIFT,
            REDUCE_NS,
            EOS,
        ],
        # Right-spine: (1,(2,(3,(4,5)))) -> shift all then reduce all
        [
            SRC_IDS[0],
            SHIFT,
            SRC_IDS[1],
            SHIFT,
            SRC_IDS[2],
            SHIFT,
            SRC_IDS[3],
            SHIFT,
            SRC_IDS[4],
            SHIFT,
            REDUCE_NS,
            REDUCE_NS,
            REDUCE_NS,
            REDUCE_NS,
            EOS,
        ],
    ],
)
def test_full_walk_each_step_legal(tree_shape):
    """Every token in a valid trajectory must be legal at the step it appears."""
    p = _make_processor()
    prefix = [DECODER_START]
    for next_tok in tree_shape:
        legal = _legal_ids(p, prefix)
        assert next_tok in legal, f"Token {next_tok} illegal at step {len(prefix)}; legal set {legal}, prefix {prefix}"
        prefix.append(next_tok)
