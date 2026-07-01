"""Unit tests for `SexpDecodingState` (PDA validity constraints for the
s-expression seq2seq decoder).

The state machine is a pure function of the action-id prefix. These tests
build small synthetic vocabularies and walk known-valid / known-invalid
action sequences against it.

Action-id convention used throughout (no overlap with any source id):
    OPEN_ID  = 1
    CLOSE_ID = 2
    EOS_ID   = 3
    COPY_ID  = 4
    EDU_ID   = 5  (the `<edu>` placeholder, used only in include_text=False mode)
    LABEL_NS = 100
    LABEL_SN = 101
    LABEL_NN = 102
Source-token ids start at 10 and increase, so they're disjoint from the above.
"""

from __future__ import annotations

import pytest

from iudex.rst.parsers.common.sexp_constraints import (
    SexpDecodingState,
    make_initial_state,
)


OPEN_ID = 1
CLOSE_ID = 2
EOS_ID = 3
COPY_ID = 4
EDU_ID = 5
LABEL_NS = 100
LABEL_SN = 101
LABEL_NN = 102
LABEL_IDS = frozenset({LABEL_NS, LABEL_SN, LABEL_NN})


def _make(
    source_len: int,
    traversal_order: str = "preorder",
    use_copy: bool = False,
    *,
    source_ids=None,
    edu_placeholder_id=None,
    min_edu_length: int = 1,
) -> SexpDecodingState:
    if source_ids is None and not use_copy:
        # Default: 10, 11, 12, ... one per source position.
        source_ids = [10 + i for i in range(source_len)]
    return make_initial_state(
        source_len=source_len,
        traversal_order=traversal_order,
        use_copy=use_copy,
        open_id=OPEN_ID,
        close_id=CLOSE_ID,
        eos_id=EOS_ID,
        label_ids=LABEL_IDS,
        copy_id=COPY_ID if use_copy else None,
        source_ids=source_ids,
        edu_placeholder_id=edu_placeholder_id,
        min_edu_length=min_edu_length,
    )


def _replay(state: SexpDecodingState, actions):
    """Step through `actions`, asserting each is legal at its position.
    Returns the final state."""
    for i, a in enumerate(actions):
        legal = state.legal_actions()
        assert a in legal, (
            f"step {i}: action {a} not in legal set {sorted(legal)} "
            f"(cursor={state.cursor}, depth={state.depth}, "
            f"stack={[fr.kind for fr in state.stack]})"
        )
        state = state.step(a)
    return state


# ---------------------------------------------------------------------------
# Initialization / empty prefix
# ---------------------------------------------------------------------------


def test_empty_prefix_only_open_legal_preorder():
    s = _make(source_len=2)
    assert s.legal_actions() == frozenset({OPEN_ID})


def test_empty_prefix_only_open_legal_postorder():
    s = _make(source_len=2, traversal_order="postorder")
    assert s.legal_actions() == frozenset({OPEN_ID})


def test_empty_prefix_admits_edu_placeholder_when_configured():
    s = _make(source_len=1, use_copy=True, edu_placeholder_id=EDU_ID)
    assert s.legal_actions() == frozenset({OPEN_ID, EDU_ID})


def test_make_initial_state_validates_source_ids_length():
    with pytest.raises(ValueError):
        make_initial_state(
            source_len=3,
            traversal_order="preorder",
            use_copy=False,
            open_id=OPEN_ID,
            close_id=CLOSE_ID,
            eos_id=EOS_ID,
            label_ids=LABEL_IDS,
            source_ids=[10, 11],  # short by one
        )


def test_make_initial_state_validates_copy_id_when_use_copy():
    with pytest.raises(ValueError):
        make_initial_state(
            source_len=2,
            traversal_order="preorder",
            use_copy=True,
            open_id=OPEN_ID,
            close_id=CLOSE_ID,
            eos_id=EOS_ID,
            label_ids=LABEL_IDS,
            copy_id=None,
        )


def test_unknown_traversal_order_rejected():
    with pytest.raises(ValueError):
        make_initial_state(
            source_len=1,
            traversal_order="inorder",
            use_copy=True,
            open_id=OPEN_ID,
            close_id=CLOSE_ID,
            eos_id=EOS_ID,
            label_ids=LABEL_IDS,
            copy_id=COPY_ID,
        )


# ---------------------------------------------------------------------------
# 2-EDU trees: both traversal orders x both copy modes
# ---------------------------------------------------------------------------


def test_2edu_preorder_use_copy_false_full_walk():
    """`(NS:r (a b) (c d))` preorder with explicit source-token emission."""
    s = _make(source_len=4)
    # source_ids = [10, 11, 12, 13]
    seq = [
        OPEN_ID,  # open root
        LABEL_NS,  # root label
        OPEN_ID,
        10,
        11,
        CLOSE_ID,  # left EDU
        OPEN_ID,
        12,
        13,
        CLOSE_ID,  # right EDU
        CLOSE_ID,  # close root
        EOS_ID,
    ]
    final = _replay(s, seq)
    assert final.is_terminal()
    assert final.cursor == 4
    assert final.depth == 0


def test_2edu_postorder_use_copy_false_full_walk():
    """`((a b) (c d) NS:r)` postorder."""
    s = _make(source_len=4, traversal_order="postorder")
    seq = [
        OPEN_ID,  # open root (kind undetermined)
        OPEN_ID,
        10,
        11,
        CLOSE_ID,  # left child (leaf)
        OPEN_ID,
        12,
        13,
        CLOSE_ID,  # right child (leaf)
        LABEL_NS,  # post-order label slot
        CLOSE_ID,  # close root
        EOS_ID,
    ]
    final = _replay(s, seq)
    assert final.is_terminal()
    assert final.cursor == 4


def test_2edu_preorder_use_copy_true_full_walk():
    """`(NS:r (<copy> <copy>) (<copy> <copy>))` preorder, copy mode."""
    s = _make(source_len=4, use_copy=True)
    seq = [
        OPEN_ID,
        LABEL_NS,
        OPEN_ID,
        COPY_ID,
        COPY_ID,
        CLOSE_ID,
        OPEN_ID,
        COPY_ID,
        COPY_ID,
        CLOSE_ID,
        CLOSE_ID,
        EOS_ID,
    ]
    final = _replay(s, seq)
    assert final.is_terminal()


def test_2edu_postorder_use_copy_true_full_walk():
    s = _make(source_len=4, traversal_order="postorder", use_copy=True)
    seq = [
        OPEN_ID,
        OPEN_ID,
        COPY_ID,
        COPY_ID,
        CLOSE_ID,
        OPEN_ID,
        COPY_ID,
        COPY_ID,
        CLOSE_ID,
        LABEL_NS,
        CLOSE_ID,
        EOS_ID,
    ]
    final = _replay(s, seq)
    assert final.is_terminal()


def test_2edu_premature_root_close_rejected():
    """After both EDUs emitted but BEFORE we close the root, EOS is illegal
    (we still need the root's `)`). And before the second EDU has been
    consumed, root-close itself is illegal."""
    s = _make(source_len=2)  # one source token per EDU
    s = s.step(OPEN_ID).step(LABEL_NS).step(OPEN_ID).step(10).step(CLOSE_ID)
    # We just closed the first EDU. cursor=1 < source_len=2, depth=1. Root-close
    # would close the whole tree with the source unexhausted.
    assert CLOSE_ID not in s.legal_actions()
    with pytest.raises(ValueError):
        s.step(CLOSE_ID)


def test_2edu_close_empty_leaf_rejected():
    """An EDU leaf with zero content tokens cannot be closed."""
    s = _make(source_len=2)
    s = s.step(OPEN_ID).step(LABEL_NS).step(OPEN_ID)  # opened an empty leaf
    assert CLOSE_ID not in s.legal_actions()
    with pytest.raises(ValueError):
        s.step(CLOSE_ID)


def test_2edu_use_copy_false_wrong_source_token_rejected():
    """In use_copy=False, only the token at the cursor position is legal."""
    s = _make(source_len=2)  # source_ids = [10, 11]
    s = s.step(OPEN_ID).step(LABEL_NS).step(OPEN_ID)
    # cursor=0, expected source-id is 10.
    assert 10 in s.legal_actions()
    assert 11 not in s.legal_actions()
    with pytest.raises(ValueError):
        s.step(11)


# ---------------------------------------------------------------------------
# 3-EDU trees: both binarizations, both traversal orders
# ---------------------------------------------------------------------------


def test_3edu_left_branching_preorder():
    """Left-branching: `(NS:a (NS:b (e0) (e1)) (e2))`.
    One source token per EDU: SRC = [10, 11, 12]."""
    s = _make(source_len=3)
    seq = [
        OPEN_ID,
        LABEL_NS,
        OPEN_ID,
        LABEL_NS,
        OPEN_ID,
        10,
        CLOSE_ID,
        OPEN_ID,
        11,
        CLOSE_ID,
        CLOSE_ID,
        OPEN_ID,
        12,
        CLOSE_ID,
        CLOSE_ID,
        EOS_ID,
    ]
    final = _replay(s, seq)
    assert final.is_terminal()


def test_3edu_right_branching_preorder():
    """Right-branching: `(NS:a (e0) (NS:b (e1) (e2)))`."""
    s = _make(source_len=3)
    seq = [
        OPEN_ID,
        LABEL_NS,
        OPEN_ID,
        10,
        CLOSE_ID,
        OPEN_ID,
        LABEL_NS,
        OPEN_ID,
        11,
        CLOSE_ID,
        OPEN_ID,
        12,
        CLOSE_ID,
        CLOSE_ID,
        CLOSE_ID,
        EOS_ID,
    ]
    final = _replay(s, seq)
    assert final.is_terminal()


def test_3edu_left_branching_postorder():
    """Left-branching postorder: `(((e0) (e1) NS:b) (e2) NS:a)`."""
    s = _make(source_len=3, traversal_order="postorder")
    seq = [
        OPEN_ID,
        OPEN_ID,
        OPEN_ID,
        10,
        CLOSE_ID,
        OPEN_ID,
        11,
        CLOSE_ID,
        LABEL_NS,
        CLOSE_ID,
        OPEN_ID,
        12,
        CLOSE_ID,
        LABEL_NS,
        CLOSE_ID,
        EOS_ID,
    ]
    final = _replay(s, seq)
    assert final.is_terminal()


def test_3edu_right_branching_postorder():
    """Right-branching postorder: `((e0) ((e1) (e2) NS:b) NS:a)`."""
    s = _make(source_len=3, traversal_order="postorder")
    seq = [
        OPEN_ID,
        OPEN_ID,
        10,
        CLOSE_ID,
        OPEN_ID,
        OPEN_ID,
        11,
        CLOSE_ID,
        OPEN_ID,
        12,
        CLOSE_ID,
        LABEL_NS,
        CLOSE_ID,
        LABEL_NS,
        CLOSE_ID,
        EOS_ID,
    ]
    final = _replay(s, seq)
    assert final.is_terminal()


# ---------------------------------------------------------------------------
# n=5 EDU tree: at least one shape
# ---------------------------------------------------------------------------


def test_5edu_left_spine_preorder():
    """Left-spine: `((((e0,e1),e2),e3),e4)` preorder."""
    s = _make(source_len=5)
    seq = [
        OPEN_ID,
        LABEL_NS,
        OPEN_ID,
        LABEL_NS,
        OPEN_ID,
        LABEL_NS,
        OPEN_ID,
        LABEL_NN,
        OPEN_ID,
        10,
        CLOSE_ID,
        OPEN_ID,
        11,
        CLOSE_ID,
        CLOSE_ID,
        OPEN_ID,
        12,
        CLOSE_ID,
        CLOSE_ID,
        OPEN_ID,
        13,
        CLOSE_ID,
        CLOSE_ID,
        OPEN_ID,
        14,
        CLOSE_ID,
        CLOSE_ID,
        EOS_ID,
    ]
    final = _replay(s, seq)
    assert final.is_terminal()


def test_5edu_right_spine_postorder():
    """Right-spine postorder: `((e0) ((e1) ((e2) ((e3) (e4) NS:d) NS:c) NS:b) NS:a)`."""
    s = _make(source_len=5, traversal_order="postorder")
    seq = [
        OPEN_ID,  # root
        OPEN_ID,
        10,
        CLOSE_ID,  # e0
        OPEN_ID,  # internal-1
        OPEN_ID,
        11,
        CLOSE_ID,  # e1
        OPEN_ID,  # internal-2
        OPEN_ID,
        12,
        CLOSE_ID,  # e2
        OPEN_ID,  # internal-3
        OPEN_ID,
        13,
        CLOSE_ID,  # e3
        OPEN_ID,
        14,
        CLOSE_ID,  # e4
        LABEL_SN,  # close internal-3
        CLOSE_ID,
        LABEL_SN,  # close internal-2
        CLOSE_ID,
        LABEL_SN,  # close internal-1
        CLOSE_ID,
        LABEL_SN,  # close root
        CLOSE_ID,
        EOS_ID,
    ]
    final = _replay(s, seq)
    assert final.is_terminal()


# ---------------------------------------------------------------------------
# Edge-case asserts
# ---------------------------------------------------------------------------


def test_only_eos_legal_at_terminal_step():
    """After root-close, with cursor==source_len and depth==0, the only
    legal action is EOS."""
    s = _make(source_len=2)
    s = (
        s.step(OPEN_ID)
        .step(LABEL_NS)
        .step(OPEN_ID)
        .step(10)
        .step(CLOSE_ID)
        .step(OPEN_ID)
        .step(11)
        .step(CLOSE_ID)
        .step(CLOSE_ID)
    )
    assert s.depth == 0
    assert s.cursor == 2
    assert s.root_emitted
    assert s.legal_actions() == frozenset({EOS_ID})


def test_step_on_terminated_state_raises():
    s = _make(source_len=1)
    s = s.step(OPEN_ID).step(LABEL_NS)
    # 1-source-token document: we still need a 2-EDU minimum because the
    # grammar always pairs internal nodes. Use source_len=2 instead.
    s = _make(source_len=2)
    s = (
        s.step(OPEN_ID)
        .step(LABEL_NS)
        .step(OPEN_ID)
        .step(10)
        .step(CLOSE_ID)
        .step(OPEN_ID)
        .step(11)
        .step(CLOSE_ID)
        .step(CLOSE_ID)
        .step(EOS_ID)
    )
    assert s.is_terminal()
    assert s.legal_actions() == frozenset()
    with pytest.raises(ValueError):
        s.step(EOS_ID)


def test_root_close_with_source_unexhausted_rejected():
    """The root span cannot close until the full source has been consumed.
    This is the constraint that obsoletes the dump-into-last-EDU heuristic."""
    s = _make(source_len=3)
    # Build a valid 2-EDU subtree using only 2 of 3 source tokens, then
    # try to root-close. The internal-state path: OPEN LABEL_NS OPEN 10 CLOSE
    # OPEN 11 CLOSE leaves cursor=2 < source_len=3 with the root still open.
    s = s.step(OPEN_ID).step(LABEL_NS).step(OPEN_ID).step(10).step(CLOSE_ID).step(OPEN_ID).step(11).step(CLOSE_ID)
    assert s.depth == 1
    assert s.cursor == 2
    # Root-close illegal because cursor < source_len.
    assert CLOSE_ID not in s.legal_actions()
    with pytest.raises(ValueError):
        s.step(CLOSE_ID)


def test_source_token_emit_blocked_at_cursor_eq_source_len():
    """Once the cursor reaches `source_len` (still inside a leaf), no further
    source-token emission is legal — close is the only structural option."""
    s = _make(source_len=2)
    s = (
        s.step(OPEN_ID)
        .step(LABEL_NS)
        .step(OPEN_ID)
        .step(10)
        .step(CLOSE_ID)  # first EDU done
        .step(OPEN_ID)
        .step(11)  # second EDU has consumed all source
    )
    assert s.cursor == 2
    assert s.in_edu_leaf
    legal = s.legal_actions()
    # No source-token in legal set (cursor==source_len).
    # In use_copy=False mode we should only see CLOSE_ID.
    assert legal == frozenset({CLOSE_ID})


def test_source_token_emit_advances_cursor_use_copy_false():
    """Emitting the expected source token (use_copy=False) advances the
    cursor by exactly one. `expected_source_id()` is only defined once
    the span's kind has resolved to `leaf` (i.e. after at least one
    content token has been emitted), so we check it post-emission."""
    s = _make(source_len=3)  # source_ids = [10, 11, 12]
    s = s.step(OPEN_ID).step(LABEL_NS).step(OPEN_ID)
    assert s.cursor == 0
    # Just-opened span: kind unresolved, expected_source_id is None.
    assert s.expected_source_id() is None
    # The legal action set still contains source_ids[0] (the leaf path).
    assert 10 in s.legal_actions()
    s2 = s.step(10)
    assert s2.cursor == 1
    assert s2.in_edu_leaf
    assert s2.expected_source_id() == 11


def test_copy_token_advances_cursor_use_copy_true():
    """In use_copy=True mode, the <copy> token advances the cursor."""
    s = _make(source_len=3, use_copy=True)
    s = s.step(OPEN_ID).step(LABEL_NS).step(OPEN_ID)
    assert s.cursor == 0
    s2 = s.step(COPY_ID)
    assert s2.cursor == 1
    # The constraint state never reveals a specific source id in copy mode.
    assert s2.expected_source_id() is None


def test_label_at_wrong_slot_rejected_preorder():
    """In preorder, a label is legal only at the just-after-open slot."""
    s = _make(source_len=2)
    # Open root, emit label, open first child, emit one source token. Now we're
    # inside a leaf — a label here is illegal.
    s = s.step(OPEN_ID).step(LABEL_NS).step(OPEN_ID).step(10)
    assert LABEL_NS not in s.legal_actions()
    assert LABEL_SN not in s.legal_actions()
    with pytest.raises(ValueError):
        s.step(LABEL_NS)


def test_label_at_wrong_slot_rejected_postorder():
    """In postorder, a label is legal only at the just-before-close slot
    of an internal node (after both children have been emitted)."""
    s = _make(source_len=2, traversal_order="postorder")
    # Open root. The first action should be `(` (opening a child) or a
    # source-content token (making root a leaf). A label here is illegal.
    s = s.step(OPEN_ID)
    assert LABEL_NS not in s.legal_actions()
    with pytest.raises(ValueError):
        s.step(LABEL_NS)


def test_eos_before_root_emitted_rejected():
    """EOS is only legal once the root has been fully emitted."""
    s = _make(source_len=2)
    assert EOS_ID not in s.legal_actions()
    with pytest.raises(ValueError):
        s.step(EOS_ID)


def test_in_edu_leaf_flag_lifecycle():
    """`in_edu_leaf` is False outside any open span, False at an unresolved
    span, True only inside a resolved leaf."""
    s = _make(source_len=2)
    assert not s.in_edu_leaf
    s1 = s.step(OPEN_ID).step(LABEL_NS).step(OPEN_ID)
    # Just opened. Kind is still None, so not in_edu_leaf yet.
    assert not s1.in_edu_leaf
    s2 = s1.step(10)
    assert s2.in_edu_leaf
    s3 = s2.step(CLOSE_ID)
    # Back to the parent's internal slot.
    assert not s3.in_edu_leaf


# ---------------------------------------------------------------------------
# min_edu_length: leaf-close is gated by leaf token count, except at
# end-of-source (where the final EDU must always be allowed to commit).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("min_edu_len", [1, 2, 3])
@pytest.mark.parametrize("traversal_order", ["preorder", "postorder"])
def test_min_edu_length_blocks_short_leaf_close_except_at_eos(min_edu_len, traversal_order):
    """Build a 2-EDU tree where each EDU gets exactly `min_edu_len * 2`
    source tokens. The first EDU's close should be illegal until it has at
    least `min_edu_len` content tokens. The final EDU may close earlier
    only when the cursor reaches source_len (end-of-source exception)."""
    per_edu = max(min_edu_len, 2) + 1
    source_len = per_edu * 2
    src_ids = [10 + i for i in range(source_len)]
    s = _make(source_len=source_len, traversal_order=traversal_order, source_ids=src_ids, min_edu_length=min_edu_len)

    # Walk into the first EDU's leaf frame.
    if traversal_order == "preorder":
        s = s.step(OPEN_ID).step(LABEL_NS).step(OPEN_ID)
    else:
        s = s.step(OPEN_ID).step(OPEN_ID)

    # Emit content tokens one-by-one. Until we hit `min_edu_len`, CLOSE
    # must NOT be legal (the leaf is below the minimum and we're not at
    # end-of-source).
    for k in range(1, min_edu_len):
        s = s.step(src_ids[s.cursor])
        assert s.in_edu_leaf
        assert CLOSE_ID not in s.legal_actions(), (
            f"CLOSE legal at leaf_token_count={k} with min_edu_length={min_edu_len}"
        )
        with pytest.raises(ValueError):
            s.step(CLOSE_ID)

    # One more content token brings us to min_edu_len. Now CLOSE is legal.
    s = s.step(src_ids[s.cursor])
    assert s.in_edu_leaf
    assert CLOSE_ID in s.legal_actions()

    # Close the first EDU.
    s = s.step(CLOSE_ID)

    # Now consume all remaining source EXCEPT the last token, in a fresh leaf.
    if traversal_order == "preorder":
        s = s.step(OPEN_ID)
    else:
        s = s.step(OPEN_ID)
    while s.cursor < source_len - 1:
        s = s.step(src_ids[s.cursor])

    # We're inside the second leaf with exactly one content token short of
    # source exhaustion. Emit the final token so cursor == source_len.
    s = s.step(src_ids[s.cursor])
    assert s.cursor == source_len
    # At end-of-source: CLOSE is legal even though we may be below
    # min_edu_len (the final EDU must be allowed to commit).
    assert CLOSE_ID in s.legal_actions()


@pytest.mark.parametrize("min_edu_len", [1, 2, 3])
def test_min_edu_length_end_of_source_exception_short_final_edu(min_edu_len):
    """The final EDU is allowed to close at exactly 1 content token when
    the cursor has reached source_len, even when min_edu_length > 1."""
    if min_edu_len <= 1:
        # The "exception" is vacuous when the floor is 1.
        return
    # Source layout: min_edu_len + 1 token for the second EDU.
    source_len = min_edu_len + 1
    src_ids = [10 + i for i in range(source_len)]
    s = _make(source_len=source_len, traversal_order="preorder", source_ids=src_ids, min_edu_length=min_edu_len)
    s = s.step(OPEN_ID).step(LABEL_NS).step(OPEN_ID)
    # First EDU swallows min_edu_len source tokens.
    for _ in range(min_edu_len):
        s = s.step(src_ids[s.cursor])
    assert CLOSE_ID in s.legal_actions()
    s = s.step(CLOSE_ID).step(OPEN_ID)
    # Second EDU: emit exactly 1 token, leaving us at end-of-source.
    s = s.step(src_ids[s.cursor])
    assert s.cursor == source_len
    # Even though 1 < min_edu_len, close is legal at EOS.
    assert CLOSE_ID in s.legal_actions()
    s = s.step(CLOSE_ID).step(CLOSE_ID).step(EOS_ID)
    assert s.is_terminal()


def test_min_edu_length_default_does_not_change_behavior():
    """The default min_edu_length=1 leaves the existing 2-EDU walk valid."""
    s = _make(source_len=4)  # min_edu_length defaults to 1
    seq = [
        OPEN_ID,
        LABEL_NS,
        OPEN_ID,
        10,
        11,
        CLOSE_ID,
        OPEN_ID,
        12,
        13,
        CLOSE_ID,
        CLOSE_ID,
        EOS_ID,
    ]
    final = _replay(s, seq)
    assert final.is_terminal()


# ---------------------------------------------------------------------------
# constrain_content=False wildcard obligation gate
# ---------------------------------------------------------------------------


def _make_cc_false(source_len: int) -> SexpDecodingState:
    return make_initial_state(
        source_len=source_len,
        traversal_order="postorder",
        use_copy=False,
        open_id=OPEN_ID,
        close_id=CLOSE_ID,
        eos_id=EOS_ID,
        label_ids=LABEL_IDS,
        source_ids=[10 + i for i in range(source_len)],
        constrain_content=False,
    )


def test_cc_false_wildcard_respects_leaf_budget_gate():
    """Under constrain_content=False, `content_is_wildcard()` must honor the
    same leaf budget gate as `legal_actions` (content illegal once every
    remaining position is reserved for a future leaf start). Regression: the
    wildcard predicate skipped the gate, the mask admitted content into
    reserved positions, and the decode later deadlocked on an empty legal set
    (an internal node owed a child with the source exhausted)."""
    # OPEN root, OPEN first child, eat one content token into that child.
    # The parent is now internal with its 2nd child still owed (obl_rest=1).
    state = _make_cc_false(2).step(OPEN_ID).step(OPEN_ID).step(10)
    assert state.in_edu_leaf
    # remaining_content == 1 == obl_rest: the last position is reserved for
    # the sibling leaf, so content must NOT be offered. CLOSE must be.
    assert state.remaining_content == 1
    assert not state.content_is_wildcard()
    assert CLOSE_ID in state.legal_actions()


def test_cc_false_wildcard_open_when_budget_allows():
    """Same prefix with one spare position: content is still wildcarded."""
    state = _make_cc_false(3).step(OPEN_ID).step(OPEN_ID).step(10)
    assert state.in_edu_leaf
    assert state.remaining_content == 2  # 1 reserved for the sibling, 1 spare
    assert state.content_is_wildcard()
