"""Unit tests for `ShiftReduceDecodeState` (the shared shift-reduce decode
automaton used by seq2seq_sr / decoder_only_sr). Pure CPU, no model: drives
the state machine at the predicate/transition level and checks full traces
against known shift-reduce action sequences."""

from __future__ import annotations

from iudex.rst.parsers.common.seqgen import ShiftReduceDecodeState


def _drive(actions: list[str], source_len: int, min_edu_length: int = 1) -> ShiftReduceDecodeState:
    """Apply a sequence of action kinds, asserting each is legal under the
    validity predicates before stepping (mirrors how the parser masks)."""
    st = ShiftReduceDecodeState(source_len=source_len, min_edu_length=min_edu_length)
    for a in actions:
        if a == "copy":
            assert st.copy_ok, f"copy illegal at {st}"
            st.step_copy()
        elif a == "shift":
            assert st.shift_ok, f"shift illegal at {st}"
            st.step_shift()
        elif a == "reduce":
            assert st.reduce_ok, f"reduce illegal at {st}"
            st.step_reduce()
        elif a == "eos":
            assert st.eos_ok, f"eos illegal at {st}"
            st.step_eos()
        else:
            raise ValueError(a)
    return st


def test_predicates_initial():
    st = ShiftReduceDecodeState(source_len=3)
    assert st.copy_ok and not st.shift_ok and not st.reduce_ok and not st.eos_ok


def test_single_edu_trace():
    # One EDU spanning all 3 source tokens: COPY COPY COPY SHIFT EOS.
    st = _drive(["copy", "copy", "copy", "shift", "eos"], source_len=3)
    assert st.done and st.cursor == 3 and st.stack_size == 1
    assert st.pred_edu_ranges == [(0, 3)]


def test_two_edu_trace_records_ranges():
    # e1=[0,2), e2=[2,3): COPY COPY SHIFT COPY SHIFT REDUCE EOS.
    st = _drive(["copy", "copy", "shift", "copy", "shift", "reduce", "eos"], source_len=3)
    assert st.done and st.stack_size == 1
    assert st.pred_edu_ranges == [(0, 2), (2, 3)]


def test_shift_requires_content():
    st = ShiftReduceDecodeState(source_len=2)
    assert not st.shift_ok  # no COPY yet
    st.step_copy()
    assert st.shift_ok


def test_reduce_requires_two_on_stack():
    st = _drive(["copy", "shift"], source_len=2)  # stack 1
    assert not st.reduce_ok
    _drive_more = st
    assert _drive_more.copy_ok
    st.step_copy()
    st.step_shift()  # stack 2
    assert st.reduce_ok


def test_eos_only_at_end_singleton_no_pending():
    st = _drive(["copy", "copy", "shift"], source_len=2)  # at end, stack 1, no pending
    assert st.eos_ok
    # With pending content (mid-EDU), EOS is illegal.
    st2 = ShiftReduceDecodeState(source_len=2)
    st2.step_copy()
    assert not st2.eos_ok  # cursor<end and edu_length>0


def test_min_edu_length_gates_shift_except_at_end():
    st = ShiftReduceDecodeState(source_len=3, min_edu_length=2)
    st.step_copy()  # edu_length 1 < 2
    assert not st.shift_ok
    st.step_copy()  # edu_length 2
    assert st.shift_ok
    # End-of-source exception: a short final EDU can still shift.
    st2 = ShiftReduceDecodeState(source_len=1, min_edu_length=3)
    st2.step_copy()  # at end, edu_length 1 < 3, but at_end exception applies
    assert st2.shift_ok


def test_step_copy_bails_when_exhausted():
    st = ShiftReduceDecodeState(source_len=1)
    st.step_copy()
    assert st.cursor == 1 and not st.copy_ok
    ok = st.step_copy()  # over-copy: should bail
    assert ok is False and st.done


def test_clone_is_independent():
    st = _drive(["copy", "shift"], source_len=3)
    c = st.clone()
    c.step_copy()
    c.step_shift()
    # Original unaffected by mutations on the clone.
    assert st.pred_edu_ranges == [(0, 1)]
    assert c.pred_edu_ranges == [(0, 1), (1, 2)]
