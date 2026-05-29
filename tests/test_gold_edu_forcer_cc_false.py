"""Tests for `GoldEduForcer` under `constrain_content=False` (Bug C2) and the
M6 zero-width / non-monotonic gold-range guard.

These drive `GoldEduForcer` + `SexpDecodingState` purely at the id level, no
model. We emulate exactly what the modeling consumer must do per the
`narrowed_legal` contract (see `sexp_constraints.narrowed_legal`):

  * None -> argmax over the full legal set (here: an adversarial fake model).
  * frozenset -> whitelist; intersect with legal and argmax.
  * FORCE_CONTENT -> build a mask admitting ONLY the content wildcard
    (all ids minus `state.structural_ids()`), then argmax.

The fake "model" is deliberately adversarial: at every step it ranks
structural tokens (OPEN, then labels, then CLOSE, then EOS) ABOVE content,
so any leaf the forcer fails to force would be turned into an internal node
(the exact C2 failure mode). A correct forcer overrides this and still lands
a tree with exactly the gold leaf count.

Action-id convention matches tests/test_sexp_constraints.py:
    OPEN=1 CLOSE=2 EOS=3   LABEL_NS=100 LABEL_SN=101 LABEL_NN=102
Content (source) ids start at 10. cc=False content is "free", so the fake
model may emit ANY non-structural id as content. We also reserve a couple of
fake tokenizer-special ids (200, 201) to prove FORCE_CONTENT masks them out.
"""

from __future__ import annotations

import pytest

from iudex.rst.parsers.common.sexp_constraints import (
    FORCE_CONTENT,
    GoldEduForcer,
    SexpDecodingState,
    make_initial_state,
)


OPEN_ID = 1
CLOSE_ID = 2
EOS_ID = 3
LABEL_NS = 100
LABEL_SN = 101
LABEL_NN = 102
LABEL_IDS = frozenset({LABEL_NS, LABEL_SN, LABEL_NN})
SPECIAL_IDS = frozenset({200, 201})  # fake PAD/BOS, must never leak as content

# A fake full vocab. Content tokens are everything in this range not used as a
# structural id. The adversarial model picks the lowest-ranked id available.
VOCAB_SIZE = 256
# A "content" id the adversarial model would emit if forced to emit content.
CONTENT_PICK = 42


def _make_state(source_len: int, traversal_order: str) -> SexpDecodingState:
    # cc=False: source_ids still required (len == source_len) for state
    # construction, but content is free, the ids are placeholders.
    source_ids = [10 + i for i in range(source_len)]
    return make_initial_state(
        source_len=source_len,
        traversal_order=traversal_order,
        use_copy=False,
        open_id=OPEN_ID,
        close_id=CLOSE_ID,
        eos_id=EOS_ID,
        label_ids=LABEL_IDS,
        source_ids=source_ids,
        min_edu_length=1,
        constrain_content=False,
        tokenizer_special_ids=SPECIAL_IDS,
    )


def _legal_mask(state: SexpDecodingState) -> list[bool]:
    """Mirror modeling's `_full_ids_to_head_mask` for use_copy=False:
    legal ids on, plus the content wildcard expansion when active."""
    mask = [False] * VOCAB_SIZE
    legal = state.legal_actions()
    for fid in legal:
        if 0 <= fid < VOCAB_SIZE:
            mask[fid] = True
    if state.content_is_wildcard():
        mask = [True] * VOCAB_SIZE
        for fid in state.structural_ids():
            if 0 <= fid < VOCAB_SIZE:
                mask[fid] = False
        for fid in legal:
            if 0 <= fid < VOCAB_SIZE:
                mask[fid] = True
    return mask


def _force_content_mask(state: SexpDecodingState) -> list[bool]:
    """The FORCE_CONTENT mask per the contract: admit ONLY the content
    wildcard. Start all-True, zero out every structural id. (This is the
    `content_is_wildcard()` branch of the legal mask WITHOUT re-enabling the
    legal structural ids, i.e. CLOSE is masked out so the leaf can't close
    before the gold target_end.)"""
    mask = [True] * VOCAB_SIZE
    for fid in state.structural_ids():
        if 0 <= fid < VOCAB_SIZE:
            mask[fid] = False
    return mask


def _adversarial_pick(mask: list[bool]) -> int:
    """An adversarial model: prefer structural tokens over content so any
    un-forced leaf becomes an internal node. Preference order: OPEN, labels,
    CLOSE, EOS, then content (prefer specials first to prove they're masked,
    then CONTENT_PICK)."""
    order = [OPEN_ID, LABEL_NS, LABEL_SN, LABEL_NN, CLOSE_ID, EOS_ID, 200, 201, CONTENT_PICK]
    for fid in order:
        if 0 <= fid < VOCAB_SIZE and mask[fid]:
            return fid
    # Fall back to first allowed id (some content token).
    for fid, ok in enumerate(mask):
        if ok:
            return fid
    raise AssertionError("No legal action in mask (all -inf).")


def _run_forced_decode(state: SexpDecodingState, forcer: GoldEduForcer, max_steps: int):
    """Drive the forced decode the way the modeling consumer does. Returns
    (final_state, action_seq, n_steps)."""
    action_seq: list[int] = []
    for _ in range(max_steps):
        if state.is_terminal():
            break
        narrowed = forcer.narrowed_legal(state)
        if narrowed is FORCE_CONTENT:
            mask = _force_content_mask(state)
        elif narrowed is None:
            mask = _legal_mask(state)
        elif len(narrowed) == 1:
            chosen = next(iter(narrowed))
            before = state
            state = state.step(chosen)
            forcer.observe(before, state, chosen)
            action_seq.append(chosen)
            continue
        else:
            # Multi-element whitelist: intersect with the legal mask.
            base = _legal_mask(state)
            mask = [base[i] and (i in narrowed) for i in range(VOCAB_SIZE)]
        chosen = _adversarial_pick(mask)
        before = state
        state = state.step(chosen)
        forcer.observe(before, state, chosen)
        action_seq.append(chosen)
    return state, action_seq, len(action_seq)


# ---------------------------------------------------------------------------
# C2: cc=False forced decode reaches a terminal state with exactly the gold
# leaf count, in a bounded number of steps, in both traversal orders.
# ---------------------------------------------------------------------------


def _gold_ranges_dense(n_leaves: int, tokens_per_leaf: int = 2):
    """Contiguous non-overlapping ranges, `tokens_per_leaf` wide each."""
    ranges = []
    cur = 0
    for _ in range(n_leaves):
        ranges.append((cur, cur + tokens_per_leaf))
        cur += tokens_per_leaf
    return ranges, cur  # (ranges, source_len)


@pytest.mark.parametrize("traversal_order", ["preorder", "postorder"])
@pytest.mark.parametrize("n_leaves", [1, 2, 3, 4])
def test_cc_false_forced_decode_hits_exact_leaf_count(traversal_order, n_leaves):
    gold_ranges, source_len = _gold_ranges_dense(n_leaves, tokens_per_leaf=2)
    state = _make_state(source_len, traversal_order)
    forcer = GoldEduForcer(n_leaves, gold_ranges)

    # Generous but finite bound: a binary tree over n leaves has <= 2n-1 nodes,
    # each contributing OPEN/CLOSE/label plus content tokens, plus EOS. Use a
    # comfortable multiple to catch any runaway while still failing on spin.
    expected_actions = source_len + (2 * n_leaves - 1) * 3 + 1
    max_steps = expected_actions * 4

    final, actions, n_steps = _run_forced_decode(state, forcer, max_steps)

    assert final.is_terminal(), (
        f"decode did not terminate in {n_steps} steps (order={traversal_order}, n_leaves={n_leaves}); actions={actions}"
    )
    assert forcer.closed_leaves == n_leaves, f"closed {forcer.closed_leaves} leaves, expected {n_leaves}"
    assert final.cursor == source_len
    assert n_steps <= expected_actions * 2, f"decode used {n_steps} steps, exceeds sane bound {expected_actions * 2}"
    # No tokenizer special leaked as a content token.
    assert 200 not in actions and 201 not in actions
    # The adversarial model would emit OPEN/labels everywhere it could, but the
    # forcer must have produced exactly n_leaves leaves, so it emitted content.
    assert CONTENT_PICK in actions


@pytest.mark.parametrize("traversal_order", ["preorder", "postorder"])
def test_cc_false_single_leaf_no_internal_node(traversal_order):
    """A 1-leaf gold layout must NOT become an internal node. The adversarial
    model wants OPEN; the forcer must force content instead."""
    gold_ranges, source_len = _gold_ranges_dense(1, tokens_per_leaf=3)
    state = _make_state(source_len, traversal_order)
    forcer = GoldEduForcer(1, gold_ranges)
    final, actions, _ = _run_forced_decode(state, forcer, max_steps=64)
    assert final.is_terminal()
    assert forcer.closed_leaves == 1
    # No internal label emitted: a single leaf tree has no internal node.
    assert not (LABEL_NS in actions or LABEL_SN in actions or LABEL_NN in actions)


def test_cc_false_force_content_returned_at_fresh_leaf_frame():
    """White-box: at a fresh leaf-start frame under cc=False the forcer
    returns the FORCE_CONTENT sentinel (not None, not an empty set)."""
    gold_ranges, source_len = _gold_ranges_dense(1, tokens_per_leaf=2)
    state = _make_state(source_len, "preorder")
    forcer = GoldEduForcer(1, gold_ranges)
    # Pre-root forces OPEN.
    assert forcer.narrowed_legal(state) == frozenset({OPEN_ID})
    before = state
    state = state.step(OPEN_ID)
    forcer.observe(before, state, OPEN_ID)
    # Now at a fresh frame whose subtree target is 1 -> force content to start
    # the leaf (NOT defer, which would let the model emit a label).
    assert forcer.narrowed_legal(state) is FORCE_CONTENT


def test_cc_false_force_content_returned_mid_leaf_below_target():
    """White-box: mid-leaf, below the gold target_end, returns FORCE_CONTENT
    rather than an empty whitelist."""
    gold_ranges = [(0, 3)]  # one 3-wide leaf
    state = _make_state(3, "preorder")
    forcer = GoldEduForcer(1, gold_ranges)
    before = state
    state = state.step(OPEN_ID)
    forcer.observe(before, state, OPEN_ID)
    # Force content to start the leaf, emit one content token.
    assert forcer.narrowed_legal(state) is FORCE_CONTENT
    before = state
    state = state.step(CONTENT_PICK)
    forcer.observe(before, state, CONTENT_PICK)
    assert state.in_edu_leaf
    assert state.cursor == 1  # below target_end=3
    # Still below target -> FORCE_CONTENT, not the empty set.
    assert forcer.narrowed_legal(state) is FORCE_CONTENT


def test_cc_false_close_forced_at_target_end():
    """Once the cursor reaches the gold target_end, the forcer forces CLOSE."""
    gold_ranges = [(0, 2)]
    state = _make_state(2, "preorder")
    forcer = GoldEduForcer(1, gold_ranges)
    for a in (OPEN_ID, CONTENT_PICK, CONTENT_PICK):
        before = state
        state = state.step(a)
        forcer.observe(before, state, a)
    assert state.cursor == 2  # == target_end
    assert forcer.narrowed_legal(state) == frozenset({CLOSE_ID})


# ---------------------------------------------------------------------------
# M6: zero-width / non-monotonic gold ranges no longer cause OPEN runaway.
# ---------------------------------------------------------------------------


def test_m6_zero_width_range_dropped():
    """A `(s, s)` zero-width range is dropped, so the forcer targets only the
    real leaves and terminates instead of spinning OPEN forever."""
    # Layout: a real 2-wide leaf, then a degenerate zero-width range, then
    # another real leaf. source_len covers the two real leaves.
    gold_ranges = [(0, 2), (2, 2), (2, 4)]
    source_len = 4
    state = _make_state(source_len, "preorder")
    forcer = GoldEduForcer(len(gold_ranges), gold_ranges)
    # The zero-width range is sanitized away.
    assert forcer.n_edus_target == 2
    assert (2, 2) not in forcer.gold_ranges

    expected = source_len + (2 * 2 - 1) * 3 + 1
    final, actions, n_steps = _run_forced_decode(state, forcer, max_steps=200)
    assert final.is_terminal(), f"runaway: {n_steps} steps, actions={actions}"
    assert forcer.closed_leaves == 2
    assert n_steps <= expected * 2


def test_m6_non_monotonic_start_clamped():
    """A backward (non-monotonic) range start is clamped to the running floor,
    so it can't force OPEN on a frame whose leaf can never receive content."""
    # Second range starts BEFORE the first ended (backward anchor).
    gold_ranges = [(0, 3), (1, 4)]  # second start 1 < floor 3
    source_len = 4
    state = _make_state(source_len, "preorder")
    forcer = GoldEduForcer(len(gold_ranges), gold_ranges)
    # Starts are clamped non-decreasing; second range start >= 3.
    starts = [s for s, _ in forcer.gold_ranges]
    assert starts == sorted(starts)
    assert forcer.gold_ranges[1][0] >= forcer.gold_ranges[0][1]

    final, actions, n_steps = _run_forced_decode(state, forcer, max_steps=200)
    assert final.is_terminal(), f"runaway: {n_steps} steps, actions={actions}"
    assert final.cursor == source_len


def test_m6_overlapping_range_dropped_not_fabricated():
    """A later range whose room is fully consumed by the monotonic floor is
    DROPPED, not fabricated into a phantom leaf past source_len. Regression for
    the `end = max(e, start+1)` bug that re-triggered the OPEN-runaway."""
    # Second range overlaps the first; once clamped its start hits source_len.
    gold_ranges = [(0, 4), (3, 4)]  # floor after first = 4; second clamps to start 4, e=4 -> no room
    source_len = 4
    state = _make_state(source_len, "preorder")
    forcer = GoldEduForcer(len(gold_ranges), gold_ranges)
    assert forcer.gold_ranges == [(0, 4)]  # phantom (4, 5) leaf never fabricated
    assert forcer.n_edus_target == 1
    assert all(e <= source_len for _, e in forcer.gold_ranges)

    final, actions, n_steps = _run_forced_decode(state, forcer, max_steps=200)
    assert final.is_terminal(), f"runaway: {n_steps} steps, actions={actions}"
    assert final.cursor == source_len


def test_m6_all_zero_width_yields_empty_target():
    """If every gold range is zero-width, the forcer targets zero leaves
    (n_edus_target == 0). At pre-root it forces EOS (empty tree)."""
    gold_ranges = [(0, 0), (1, 1)]
    state = _make_state(2, "preorder")
    forcer = GoldEduForcer(len(gold_ranges), gold_ranges)
    assert forcer.n_edus_target == 0
    # source_len=2 but no leaves: pre-root with n_edus_target==0 wants EOS,
    # which is only legal once root_emitted AND cursor==source_len. With no
    # tree emitted and source unexhausted, EOS isn't legal yet -> forcer
    # returns None (defer). This is an acknowledged degenerate input; the
    # guard's job is only to prevent the OPEN runaway, which it does (no spin).
    narrowed = forcer.narrowed_legal(state)
    assert narrowed is None or narrowed == frozenset({EOS_ID})
