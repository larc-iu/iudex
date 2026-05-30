"""Validity constraints for s-expression seq2seq decoding.

A pushdown automaton over decoder positions. Validity is enforced by
restricting which action ids are legal at each step. The state is immutable
under `step()` (returns a new state) so beam search can hold per-beam states
without aliasing.

Grammar (per `RstTree.to_sexp` / `from_sexp`, plan style):

  tree   ::= '(' LABEL tree tree ')'        -- pre-order internal
           | '(' tree tree LABEL ')'        -- post-order internal
           | '(' CONTENT* ')'               -- leaf with literal source content
           | '<edu>'                        -- leaf placeholder (use_copy mode)

  CONTENT ::= source token (verbatim from input)  -- when use_copy=False
            | <copy>                              -- when use_copy=True

Action vocabulary, as integer ids:

  open_id, close_id: '(' and ')'
  label_ids: set of valid internal-node labels (NS:rel, SN:rel, NN:rel)
  eos_id: end-of-sequence
  use_copy=True:  copy_id (the single `<copy>` token; advances the cursor)
                  no source_ids passed (the decoder's leaf-text is just <copy>s)
                  optionally edu_placeholder_id (the `<edu>` bare token, if the
                  caller's vocabulary uses it for include_text=False trees)
  use_copy=False: source_ids = list of input subword ids, one per cursor
                  position. The legal source token at any cursor i is exactly
                  source_ids[i]. Any other token in the source vocabulary is
                  illegal at that position.

Constraints enforced:
  * Root close legal iff cursor == source_len AND depth becomes 0 after close.
  * Cannot close an EDU leaf with zero content tokens.
  * Exactly one label per internal span. Pre-order: label legal only at the
    just-after-open slot. Post-order: label legal only at the just-before-close
    slot (after both subtrees have been emitted).
  * Source-token / <copy> emit legal iff inside an EDU leaf AND cursor < source_len.
  * EOS legal iff depth == 0 AND cursor == source_len AND a tree has been emitted.

The state intentionally does not depend on the model's hidden state or on
which specific source token was emitted (in use_copy=False mode we still
hard-mask to source_ids[cursor], but that's done by the caller using
`expected_source_id()`). It is a pure function of the action-id prefix.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import FrozenSet, List, Optional, Tuple, Union


class _ForceContent:
    """Singleton sentinel for `GoldEduForcer.narrowed_legal`'s third return
    case. See `FORCE_CONTENT`."""

    _instance: Optional["_ForceContent"] = None

    def __new__(cls) -> "_ForceContent":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "FORCE_CONTENT"


# Third `narrowed_legal` return case (cc=False only). Tells the caller to
# build a mask that admits ONLY the content wildcard (the full vocab minus
# all structural ids), forcing the next emitted token to be EDU content. The
# whitelist-intersection protocol can't express this because under the
# content wildcard `state.legal_actions()` is empty (content is not an
# enumerable id set), so a frozenset return would be the empty set and the
# caller would mask everything to -inf. The caller handles FORCE_CONTENT by
# constructing the same mask `state.content_is_wildcard()` would produce:
# start all-True, then zero out `state.structural_ids()` (OPEN, CLOSE, all
# label ids, EOS, copy, edu placeholder, and tokenizer specials). See the
# module-level note on `narrowed_legal` for the full caller contract.
FORCE_CONTENT = _ForceContent()

# Type alias for `narrowed_legal`'s return.
NarrowedLegal = Union[None, FrozenSet[int], _ForceContent]


# Per-span state pushed onto the stack each time '(' opens a span.
# `kind` is None until we know whether the span is a leaf or an internal node;
# the first non-'(' action inside the span resolves it.
@dataclass(frozen=True)
class _Frame:
    kind: Optional[str] = None  # None | 'leaf' | 'internal'
    children_emitted: int = 0  # for internal nodes (0, 1, or 2)
    leaf_token_count: int = 0  # for leaf nodes
    label_emitted: bool = False  # for internal nodes
    label_position_ok: bool = False  # has the slot where the label must go arrived?


@dataclass(frozen=True)
class SexpDecodingState:
    source_len: int
    traversal_order: str  # 'preorder' | 'postorder'
    use_copy: bool

    open_id: int
    close_id: int
    eos_id: int
    label_ids: FrozenSet[int]
    copy_id: Optional[int] = None  # required iff use_copy=True
    source_ids: Tuple[int, ...] = ()  # required iff use_copy=False; len == source_len
    edu_placeholder_id: Optional[int] = None  # `<edu>` token, when serialized include_text=False

    # Minimum content-token count required before a leaf may close. Mirrors
    # the same-name knob on the SR parsers. Inference-only (training uses
    # teacher-forced sequences). Exception: at end-of-source the leaf may
    # close even when below the threshold, since otherwise the final EDU
    # cannot commit.
    min_edu_length: int = 1

    # When True (default) AND use_copy=False, content positions are masked
    # to the single source id at `source_ids[cursor]` (COPY-via-constraint).
    # When False AND use_copy=False, content positions admit any non-
    # structural token id. This mirrors Hu and Wan 2023's apparent setup
    # (free content generation, the model must learn to copy via attention).
    # No-op when use_copy=True (the COPY token is the sole legal content
    # action regardless).
    constrain_content: bool = True

    # Tokenizer special ids (PAD, BOS, UNK, decoder_start, ...) the caller
    # wants treated as structural at content-wildcard positions. Only
    # consumed by `structural_ids()`; the parsers populate this from
    # `tokenizer.all_special_ids` at state construction so leaked specials
    # don't end up in the EDU surface text under `constrain_content=False`.
    tokenizer_special_ids: FrozenSet[int] = frozenset()

    cursor: int = 0
    depth: int = 0
    stack: Tuple[_Frame, ...] = ()  # one frame per currently open span
    root_emitted: bool = False  # set True after the root's matching ')' fires
    terminated: bool = False  # set True after EOS

    def __post_init__(self):
        if self.traversal_order not in ("preorder", "postorder"):
            raise ValueError(f"Unknown traversal_order {self.traversal_order!r}")
        if self.use_copy:
            if self.copy_id is None:
                raise ValueError("use_copy=True requires copy_id.")
        else:
            if len(self.source_ids) != self.source_len:
                raise ValueError(
                    f"use_copy=False requires source_ids of length {self.source_len}, got {len(self.source_ids)}."
                )

    @property
    def in_edu_leaf(self) -> bool:
        return bool(self.stack) and self.stack[-1].kind == "leaf"

    def is_terminal(self) -> bool:
        return self.terminated

    def expected_source_id(self) -> Optional[int]:
        """In use_copy=False mode, the legal source-token id at the current
        cursor (or None if no source token is legal right now)."""
        if self.use_copy:
            return None
        if not self.in_edu_leaf or self.cursor >= self.source_len:
            return None
        return self.source_ids[self.cursor]

    def legal_actions(self) -> FrozenSet[int]:
        if self.terminated:
            return frozenset()

        legal: List[int] = []

        # EOS: after root has been fully emitted and source exhausted.
        if self.depth == 0 and self.cursor == self.source_len and self.root_emitted:
            legal.append(self.eos_id)
            return frozenset(legal)

        # Pre-root: only '(' or '<edu>' can start the tree.
        if self.depth == 0 and not self.root_emitted:
            # An '<edu>' top-level is allowed only for a 1-EDU document and
            # only in use_copy=True+include_text=False mode. For simplicity in
            # the constraint state we just gate on edu_placeholder_id being set.
            legal.append(self.open_id)
            if self.edu_placeholder_id is not None and self.source_len > 0:
                legal.append(self.edu_placeholder_id)
            return frozenset(legal)

        # Inside an open span. Look at the innermost frame.
        top = self.stack[-1]

        if top.kind is None:
            # Just opened. Decide what's legal based on traversal_order.
            if self.traversal_order == "preorder":
                # Internal node: starts with a label.
                # Leaf: starts with a source/copy token.
                if self.cursor < self.source_len:
                    legal.extend(self._content_legal())
                legal.extend(sorted(self.label_ids))
            else:
                # Postorder. Internal: child first, which is '(' or '<edu>'.
                # Leaf: starts with a source/copy token.
                legal.append(self.open_id)
                if self.edu_placeholder_id is not None:
                    legal.append(self.edu_placeholder_id)
                if self.cursor < self.source_len:
                    legal.extend(self._content_legal())
            return frozenset(legal)

        if top.kind == "leaf":
            # In a leaf. Continue collecting content tokens, or close.
            if self.cursor < self.source_len:
                legal.extend(self._content_legal())
            if top.leaf_token_count > 0:
                # Closing this leaf is legal whenever non-empty. (Root-close
                # has the additional constraint that cursor==source_len, but
                # that's handled by the close path below.)
                if self._can_close():
                    legal.append(self.close_id)
            return frozenset(legal)

        # top.kind == 'internal'
        if self.traversal_order == "preorder":
            # Label has already been emitted (it's how we discovered we're
            # internal). Need 2 children before close.
            if top.children_emitted < 2:
                legal.append(self.open_id)
                if self.edu_placeholder_id is not None:
                    legal.append(self.edu_placeholder_id)
            else:
                if self._can_close():
                    legal.append(self.close_id)
            return frozenset(legal)
        # Postorder internal node.
        if top.children_emitted < 2:
            legal.append(self.open_id)
            if self.edu_placeholder_id is not None:
                legal.append(self.edu_placeholder_id)
            return frozenset(legal)
        if not top.label_emitted:
            legal.extend(sorted(self.label_ids))
            return frozenset(legal)
        if self._can_close():
            legal.append(self.close_id)
        return frozenset(legal)

    def content_is_wildcard(self) -> bool:
        """True iff this is a content-emit position whose legal content is the
        wildcard (any non-structural vocab id). See the `FORCE_CONTENT`
        constant. Only possible under use_copy=False and constrain_content=False.
        """
        if self.use_copy or self.constrain_content:
            return False
        if self.cursor >= self.source_len:
            return False
        if not self.stack:
            return False
        top = self.stack[-1]
        if top.kind == "leaf":
            return True
        if top.kind is None:
            # Fresh frame: content is one of the legal first actions.
            return True
        return False

    def structural_ids(self) -> FrozenSet[int]:
        """All structural token ids (open, close, labels, eos, copy, edu
        placeholder, plus tokenizer specials like PAD/BOS/UNK/decoder_start).
        Callers use this to mask the wildcard content slot in
        `constrain_content=False` mode so tokenizer specials don't leak into
        EDU surface text."""
        ids: set[int] = {self.open_id, self.close_id, self.eos_id}
        ids.update(int(x) for x in self.label_ids)
        if self.copy_id is not None:
            ids.add(int(self.copy_id))
        if self.edu_placeholder_id is not None:
            ids.add(int(self.edu_placeholder_id))
        ids.update(int(x) for x in self.tokenizer_special_ids)
        return frozenset(ids)

    def _content_legal(self) -> List[int]:
        """Source-content tokens legal *right now*.

        use_copy=True: the single `<copy>` token.
        use_copy=False, constrain_content=True (default): the one source
            subword id at `source_ids[cursor]` (COPY-via-constraint).
        use_copy=False, constrain_content=False: returns the empty list.
            Content is wildcarded. The caller checks `content_is_wildcard()`
            and admits the full vocab minus `structural_ids()`.
        """
        if self.cursor >= self.source_len:
            return []
        if self.use_copy:
            return [self.copy_id]  # type: ignore[list-item]
        if self.constrain_content:
            return [self.source_ids[self.cursor]]
        return []

    def _can_close(self) -> bool:
        """Whether closing the innermost span is legal right now (i.e. the
        span structurally permits it). Root-close additionally requires the
        cursor to have reached source_len. Leaf-close additionally requires
        the leaf to contain at least `min_edu_length` content tokens, except
        at end-of-source (where the final EDU must be allowed to commit
        regardless)."""
        if not self.stack:
            return False
        top = self.stack[-1]
        if top.kind is None:
            return False
        if top.kind == "leaf":
            if top.leaf_token_count == 0:
                return False
            min_len = max(1, int(self.min_edu_length))
            at_end = self.cursor == self.source_len
            if top.leaf_token_count < min_len and not at_end:
                return False
        if top.kind == "internal":
            if top.children_emitted != 2:
                return False
            if self.traversal_order == "postorder" and not top.label_emitted:
                return False
        # If this would close the root, require source exhausted.
        if self.depth == 1 and self.cursor != self.source_len:
            return False
        return True

    def step(self, action_id: int) -> "SexpDecodingState":
        if self.terminated:
            raise ValueError("step() called on a terminated state.")

        if action_id == self.eos_id:
            if not (self.depth == 0 and self.cursor == self.source_len and self.root_emitted):
                raise ValueError("EOS emitted in non-terminal position.")
            return replace(self, terminated=True)

        # Pre-root: opening the tree, or one-EDU placeholder root.
        if self.depth == 0 and not self.root_emitted:
            if action_id == self.open_id:
                return replace(
                    self,
                    depth=1,
                    stack=(_Frame(),),
                )
            if self.edu_placeholder_id is not None and action_id == self.edu_placeholder_id:
                # Whole tree is a single '<edu>'; valid only if there's
                # nothing else expected. Advance the cursor by source_len.
                return replace(
                    self,
                    cursor=self.source_len,
                    root_emitted=True,
                )
            raise ValueError(f"Action {action_id} illegal at the pre-root position.")

        top = self.stack[-1]

        # Action: '('
        if action_id == self.open_id:
            new_top = top
            if top.kind == "internal":
                pass  # entering a child slot; the parent's kind is fixed
            elif top.kind is None:
                if self.traversal_order != "postorder":
                    raise ValueError("Opening '(' inside a preorder unknown-kind span is illegal.")
                new_top = replace(top, kind="internal", children_emitted=0)
            else:
                raise ValueError(f"Cannot open '(' inside a {top.kind!r} span.")
            return replace(
                self,
                depth=self.depth + 1,
                stack=self.stack[:-1] + (new_top, _Frame()),
            )

        # Action: '<edu>' placeholder
        if self.edu_placeholder_id is not None and action_id == self.edu_placeholder_id:
            new_top = top
            if top.kind == "internal":
                pass
            elif top.kind is None and self.traversal_order == "postorder":
                new_top = replace(top, kind="internal", children_emitted=0)
            else:
                raise ValueError(f"'<edu>' illegal inside a {top.kind!r} span.")
            # The placeholder consumes one EDU's worth of source. We don't
            # know exact token boundaries from the constraint side, so we
            # advance the cursor only if the caller is in a mode where every
            # placeholder corresponds to one source token (rare). The safer
            # contract: include_text=False decoding doesn't emit source
            # tokens at all, so cursor advancement is None here. The
            # `source_len` should be set to the number of EDU placeholders
            # the model is expected to emit in that mode.
            advanced_top = replace(new_top, children_emitted=new_top.children_emitted + 1)
            new_stack = self.stack[:-1] + (advanced_top,)
            return replace(self, stack=new_stack, cursor=min(self.cursor + 1, self.source_len))

        # Action: ')'
        if action_id == self.close_id:
            if not self._can_close():
                raise ValueError("')' illegal at this position.")
            popped_stack = self.stack[:-1]
            new_depth = self.depth - 1
            root_now = new_depth == 0
            if popped_stack:
                parent = popped_stack[-1]
                if parent.kind is None and self.traversal_order == "postorder":
                    parent = replace(parent, kind="internal", children_emitted=1)
                else:
                    parent = replace(parent, children_emitted=parent.children_emitted + 1)
                popped_stack = popped_stack[:-1] + (parent,)
            return replace(
                self,
                depth=new_depth,
                stack=popped_stack,
                root_emitted=self.root_emitted or root_now,
            )

        # Action: label
        if action_id in self.label_ids:
            if self.traversal_order == "preorder":
                if top.kind is not None:
                    raise ValueError("Label emitted at a non-open slot in preorder.")
                new_top = replace(top, kind="internal", label_emitted=True, children_emitted=0)
            else:
                if top.kind != "internal" or top.children_emitted != 2 or top.label_emitted:
                    raise ValueError("Label emitted at an illegal slot in postorder.")
                new_top = replace(top, label_emitted=True)
            return replace(self, stack=self.stack[:-1] + (new_top,))

        # Action: content token (<copy>, source-id, or wildcard non-structural)
        is_content = False
        if self.use_copy:
            if action_id == self.copy_id:
                is_content = True
        else:
            if self.cursor < self.source_len:
                if self.constrain_content:
                    is_content = action_id == self.source_ids[self.cursor]
                else:
                    # Anything not already in the structural ids consumed above
                    # counts as content. Since we already early-returned on
                    # open / close / label / placeholder / eos / copy, just
                    # reaching here under constrain_content=False means content.
                    is_content = True
        if is_content:
            if top.kind == "internal":
                raise ValueError("Source content emitted inside an internal node slot.")
            new_top = top
            if top.kind is None:
                new_top = replace(top, kind="leaf", leaf_token_count=1)
            else:
                new_top = replace(top, leaf_token_count=top.leaf_token_count + 1)
            return replace(
                self,
                stack=self.stack[:-1] + (new_top,),
                cursor=self.cursor + 1,
            )

        raise ValueError(f"Action {action_id} is not in the legal set.")


def make_initial_state(
    source_len: int,
    traversal_order: str,
    use_copy: bool,
    *,
    open_id: int,
    close_id: int,
    eos_id: int,
    label_ids,
    copy_id: Optional[int] = None,
    source_ids: Optional[List[int]] = None,
    edu_placeholder_id: Optional[int] = None,
    min_edu_length: int = 1,
    constrain_content: bool = True,
    tokenizer_special_ids: Optional[FrozenSet[int]] = None,
) -> SexpDecodingState:
    return SexpDecodingState(
        source_len=source_len,
        traversal_order=traversal_order,
        use_copy=use_copy,
        open_id=open_id,
        close_id=close_id,
        eos_id=eos_id,
        label_ids=frozenset(int(x) for x in label_ids),
        copy_id=copy_id,
        source_ids=tuple(source_ids or ()),
        edu_placeholder_id=edu_placeholder_id,
        min_edu_length=int(min_edu_length),
        constrain_content=bool(constrain_content),
        tokenizer_special_ids=frozenset(int(x) for x in (tokenizer_special_ids or frozenset())),
    )


class GoldEduForcer:
    """Drive a `SexpDecodingState` to emit exactly `n_edus_target` leaves
    matching the gold ranges, regardless of how (un)trained the model is.

    Strategy: a right-leaning binary spine. At every kind=None frame holding
    k leaves in its subtree, force OPEN (internal) when k>=2 or force a
    content emission (leaf) when k==1. Each internal node's left child is
    the recursive subtree with k-1 leaves. Its right child is the kth leaf.

    Tree shape is fixed by this forcing strategy. Only the LABEL slot and
    the `<copy>`-vs-source content token choice are left to the model (the
    latter is moot under `use_copy=True` or `constrain_content=True`).

    Usage:
        forcer = GoldEduForcer(n_edus_target, gold_ranges)
        for step in ...:
            forced = forcer.next_forced(state)
            ... use forced if not None, else model.argmax ...
            new_state = state.step(chosen_id)
            forcer.observe(state, new_state, chosen_id)
            state = new_state

    Assumes the driven state has `min_edu_length == 1` (both consumers pin it
    for the forced state). With `min_edu_length > 1` an earlier leaf can
    overshoot and exhaust the source before a later leaf can start, deadlocking
    the forcer into an OPEN-spin to max length; honoring min_edu>1 here would
    need a force-toward-close fallback rather than deferring to the model.
    """

    def __init__(self, n_edus_target: int, gold_ranges: List[tuple]) -> None:
        if n_edus_target != len(gold_ranges):
            raise ValueError(f"n_edus_target={n_edus_target} != len(gold_ranges)={len(gold_ranges)}.")
        # M6 guard: zero-width `(s, s)` ranges (an EDU that aligned to no
        # subword and fell back to an `(anchor, anchor)` range) and backward
        # (non-monotonic) starts would otherwise make the forcer spin OPEN on
        # a frame whose leaf can never receive content and can never close.
        # Drop any range that has no room left past the running monotonic
        # floor, so every surviving range is a non-empty, non-decreasing
        # forward span keyed to a REAL gold end (never a fabricated one).
        # Clamping the start up to the floor and then keeping the true gold
        # end `e` is safe; fabricating `end = start + 1` past the floor would
        # invent a leaf with no content target and re-trigger the OPEN-runaway
        # this guard exists to prevent. The per-parser range producers already
        # emit tiling ranges (the C1 alignment helper), so this is the
        # belt-and-suspenders guard in the shared forcer.
        sanitized: List[tuple] = []
        floor = 0
        for s, e in gold_ranges:
            s, e = int(s), int(e)
            if e <= s:
                continue  # zero-width / inverted gold range
            start = max(s, floor)
            if e <= start:
                continue  # range falls entirely behind the floor: no room
            sanitized.append((start, e))
            floor = e
        self.n_edus_target = len(sanitized)
        self.gold_ranges = sanitized
        self.closed_leaves = 0
        # subtree_sizes[i] = number of leaves the i-th open frame's subtree
        # should hold. Maintained parallel to `state.stack`.
        self._subtree_sizes: List[int] = []

    @property
    def opened_leaves(self) -> int:
        return self.closed_leaves

    def _current_target_end(self) -> Optional[int]:
        if self.closed_leaves >= self.n_edus_target:
            return None
        return self.gold_ranges[self.closed_leaves][1]

    def next_forced(self, state: SexpDecodingState) -> Optional[int]:
        """Single-action force, or None to defer to `narrowed_legal()`.

        FORCE_CONTENT is NOT a single action (it forces "some content token",
        not one specific id), so it is treated like a non-singleton narrowing
        here and returns None. The caller must consult `narrowed_legal`
        directly to see FORCE_CONTENT, not rely on `next_forced`."""
        narrowed = self.narrowed_legal(state)
        if narrowed is None or narrowed is FORCE_CONTENT or len(narrowed) != 1:
            return None
        return next(iter(narrowed))

    def narrowed_legal(self, state: SexpDecodingState) -> NarrowedLegal:
        """Narrowing of `state.legal_actions()` consistent with the gold-EDU
        plan. One of three return shapes:

          * None -> no narrowing (use the model's argmax over the full legal
            set, or over a multi-element whitelist on a later call).
          * frozenset[int] of full-vocab ids -> whitelist. The caller masks
            logits to (legal & this set) and argmaxes. A singleton is a hard
            force. (In practice this is never the empty set.)
          * FORCE_CONTENT (cc=False only) -> force a content-wildcard emit.
            See the `FORCE_CONTENT` constant for the caller contract.
        """
        if state.is_terminal():
            return None
        legal = state.legal_actions()

        # Inside an active leaf: force content or CLOSE.
        if state.in_edu_leaf and self.closed_leaves < self.n_edus_target:
            target_end = self._current_target_end()
            if target_end is None:
                return None
            if state.cursor < target_end:
                if state.use_copy:
                    return frozenset({state.copy_id}) if state.copy_id in legal else None
                if state.constrain_content:
                    if state.cursor >= state.source_len:
                        return None
                    content_id = state.source_ids[state.cursor]
                    return frozenset({content_id}) if content_id in legal else None
                # Free content (cc=False): force a content token via the
                # wildcard. We can't return `frozenset(legal - {close})` (it's
                # empty under the wildcard -> masks everything to -inf), and we
                # can't return None (the model might argmax CLOSE before the
                # gold target_end, since `legal` admits close once
                # leaf_token_count >= min_edu_length). When the source is
                # already exhausted there is no content to emit, so defer and
                # let CLOSE happen.
                if state.cursor >= state.source_len:
                    return None
                return FORCE_CONTENT
            return frozenset({state.close_id}) if state.close_id in legal else None

        # Pre-root: force OPEN.
        if state.depth == 0 and not state.root_emitted:
            if self.n_edus_target == 0:
                return frozenset({state.eos_id}) if state.eos_id in legal else None
            return frozenset({state.open_id}) if state.open_id in legal else None

        if not state.stack:
            # Post-root: tree closed. Force EOS.
            if state.cursor == state.source_len and state.root_emitted:
                return frozenset({state.eos_id}) if state.eos_id in legal else None
            return None

        top = state.stack[-1]
        top_target = self._subtree_sizes[-1] if self._subtree_sizes else self.n_edus_target

        if top.kind == "internal":
            if top.children_emitted < 2:
                return frozenset({state.open_id}) if state.open_id in legal else None
            if state.traversal_order == "postorder" and not top.label_emitted:
                # Let the model pick the label, but constrain to label_ids.
                return frozenset(state.label_ids) & legal
            return frozenset({state.close_id}) if state.close_id in legal else None

        # top.kind is None: fresh frame. Decide leaf vs internal by target.
        if top_target <= 1:
            # Force first content token. At a fresh frame `legal_actions()`
            # also offers structural starters (labels / OPEN), so deferring to
            # the model here (the old cc=False behavior) lets it turn the
            # intended leaf into an internal node that never closes. We must
            # force content to start the leaf.
            if state.use_copy:
                return frozenset({state.copy_id}) if state.copy_id in legal else None
            if state.cursor >= state.source_len:
                # No source left to start a leaf with. Defer (should not arise
                # for a valid, M6-sanitized gold range).
                return None
            if state.constrain_content:
                content_id = state.source_ids[state.cursor]
                return frozenset({content_id}) if content_id in legal else None
            # cc=False: force the content wildcard to begin the leaf.
            return FORCE_CONTENT
        # top_target >= 2: this frame is internal.
        if state.traversal_order == "preorder":
            # In preorder the first action inside an internal node is the LABEL.
            return frozenset(state.label_ids) & legal
        # Postorder: first action inside an internal is OPEN of its first child.
        return frozenset({state.open_id}) if state.open_id in legal else None

    def observe(self, before: SexpDecodingState, after: SexpDecodingState, action_id: int) -> None:
        """Update the parallel subtree-size stack to mirror `after.stack`."""
        before_top = before.stack[-1] if before.stack else None
        # Leaf close detection.
        if action_id == before.close_id and before_top is not None and before_top.kind == "leaf":
            self.closed_leaves += 1

        # Sync the subtree-size stack length with the new stack.
        before_depth = len(before.stack)
        after_depth = len(after.stack)
        if after_depth > before_depth:
            # A new frame was pushed. Its target = parent_remaining - (right-leaf slot if applicable).
            # Right-spine: when a frame's subtree_size is k>=2, its first child
            # gets k-1 and its second child gets 1.
            parent_target = self._subtree_sizes[-1] if self._subtree_sizes else self.n_edus_target
            parent_top_before = before.stack[-1] if before.stack else None
            # Determine which child slot was just opened.
            if parent_top_before is None or parent_top_before.kind is None:
                # First-ever frame (pre-root) OR fresh-frame-just-became-internal-via-OPEN.
                # In the pre-root case the new frame inherits n_edus_target.
                child_size = parent_target if not self._subtree_sizes else (parent_target - 1)
            else:
                # parent_top_before.kind == "internal" with some children_emitted count.
                children_emitted_before = parent_top_before.children_emitted
                if children_emitted_before == 0:
                    child_size = parent_target - 1  # left child gets k-1 leaves
                else:
                    child_size = 1  # right child gets 1 leaf
            self._subtree_sizes.append(max(1, int(child_size)))
        elif after_depth < before_depth:
            # A frame was popped (close).
            if self._subtree_sizes:
                self._subtree_sizes.pop()
