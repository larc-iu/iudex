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
from typing import FrozenSet, List, Optional, Tuple


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

    def _content_legal(self) -> List[int]:
        """Source-content tokens that are legal *right now*. Either the single
        <copy> token (use_copy=True) or the one source subword id that matches
        source_ids[cursor] (use_copy=False)."""
        if self.cursor >= self.source_len:
            return []
        if self.use_copy:
            return [self.copy_id]  # type: ignore[list-item]
        return [self.source_ids[self.cursor]]

    def _can_close(self) -> bool:
        """Whether closing the innermost span is legal right now (i.e. the
        span structurally permits it). Root-close additionally requires the
        cursor to have reached source_len."""
        if not self.stack:
            return False
        top = self.stack[-1]
        if top.kind is None:
            return False
        if top.kind == "leaf" and top.leaf_token_count == 0:
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

        # Action: content token (<copy> or source-id)
        is_content = False
        if self.use_copy:
            if action_id == self.copy_id:
                is_content = True
        else:
            if self.cursor < self.source_len and action_id == self.source_ids[self.cursor]:
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
    )
