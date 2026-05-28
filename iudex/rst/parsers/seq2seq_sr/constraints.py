"""Validity constraints for shift-reduce seq2seq decoding.

Enforces both structural validity (stack/queue invariants for a binary
shift-reduce parse) and input-coverage validity (the emitted source-copy
sub-sequence must equal `input_subword_ids` verbatim, in order).
"""

from typing import Iterable, Sequence

import torch
from transformers import LogitsProcessor


class E2EShiftReduceValidityProcessor(LogitsProcessor):
    """Mask out illegal next-token IDs at every decoder step.

    Supports batched generation: pass a list of `source_ids` lists, one per
    row in the batch (`per_row_source_ids[i]` is the source subword sequence
    that row `i` of `input_ids` must copy verbatim). For single-document
    inference pass `[source_ids]`. For beam search with `num_beams=N` over
    `B` examples, the row layout is `[ex0_beam0, ex0_beam1, ..., ex1_beam0,
    ...]` so the constructor accepts `num_beams` and maps row → example
    accordingly.

    State carried in the decoder prefix (replayed each step):
      - cursor: next position in this row's `source_ids` that must be copied
      - stack_size: number of items currently on the shift-reduce stack
      - edu_has_content: True iff at least one source token has been copied
        since the last SHIFT (so SHIFT would commit a non-empty EDU)
    """

    def __init__(
        self,
        per_row_source_ids: Sequence[Sequence[int]],
        shift_id: int,
        reduce_ids: Iterable[int],
        eos_id: int,
        pad_id: int | None = None,
        decoder_start_id: int | None = None,
        num_beams: int = 1,
    ):
        self.per_row_source_ids = [list(s) for s in per_row_source_ids]
        self.shift_id = int(shift_id)
        self.reduce_ids = frozenset(int(r) for r in reduce_ids)
        self.eos_id = int(eos_id)
        self.num_beams = int(num_beams)
        # Tokens to skip when replaying the prefix because they're emitted by
        # the framework (not by the model's vocabulary moves):
        #   pad: greedy/beam can pad after EOS in some configurations
        #   decoder_start: T5-family starts with `<pad>` as the decoder start
        skip = set()
        if pad_id is not None:
            skip.add(int(pad_id))
        if decoder_start_id is not None:
            skip.add(int(decoder_start_id))
        self._skip_ids = frozenset(skip)
        # Cache the sorted reduce-id list as a CPU LongTensor. Reused each
        # call's mask-write step via .to(device); avoids re-sorting and
        # re-allocating per row per decode step.
        self._reduce_idx_cpu = torch.tensor(sorted(self.reduce_ids), dtype=torch.long)
        self._reduce_idx_cache: dict[torch.device, torch.Tensor] = {}
        # Incremental state cache: from the previous __call__, remember
        # (prefix_tuple -> state). On the next call we advance state by a
        # single token instead of re-walking the full prefix. Drops total
        # replay cost from O(L^2) to O(L) per row.
        self._state_cache: dict[tuple[int, ...], tuple[int, int, bool]] = {}

    def _example_idx(self, row_idx: int) -> int:
        return row_idx // self.num_beams

    def _replay(self, prefix: list[int]) -> tuple[int, int, bool]:
        cursor = 0
        stack_size = 0
        edu_has_content = False
        for tok in prefix:
            if tok in self._skip_ids:
                continue
            if tok == self.shift_id:
                stack_size += 1
                edu_has_content = False
            elif tok in self.reduce_ids:
                stack_size -= 1
            elif tok == self.eos_id:
                break
            else:
                cursor += 1
                edu_has_content = True
        return cursor, stack_size, edu_has_content

    def _advance(self, state: tuple[int, int, bool], tok: int) -> tuple[int, int, bool]:
        """One-step transition from `state` on consuming `tok`. Mirrors
        `_replay`'s per-iteration logic. EOS terminates the walk in `_replay`
        (no further state changes); here we just no-op so the cache stays
        consistent if generate keeps padding after EOS."""
        cursor, stack_size, edu_has_content = state
        if tok in self._skip_ids:
            return state
        if tok == self.shift_id:
            return cursor, stack_size + 1, False
        if tok in self.reduce_ids:
            return cursor, stack_size - 1, edu_has_content
        if tok == self.eos_id:
            return state
        return cursor + 1, stack_size, True

    def _state_for(self, prefix: list[int]) -> tuple[int, int, bool]:
        """Get state for `prefix`, advancing from a cached parent prefix if
        we have one (the common case: parent = previous step's prefix for
        this row, cached by the previous __call__ that produced this token)."""
        prefix_tuple = tuple(prefix)
        cached = self._state_cache.get(prefix_tuple)
        if cached is not None:
            return cached
        if len(prefix_tuple) > 0:
            parent = prefix_tuple[:-1]
            parent_state = self._state_cache.get(parent)
            if parent_state is not None:
                state = self._advance(parent_state, prefix_tuple[-1])
                self._state_cache[prefix_tuple] = state
                return state
        state = self._replay(prefix)
        self._state_cache[prefix_tuple] = state
        return state

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # input_ids: [batch * num_beams, prefix_len]
        # scores:    [batch * num_beams, vocab_size]
        device = scores.device
        reduce_idx = self._reduce_idx_cache.get(device)
        if reduce_idx is None:
            reduce_idx = self._reduce_idx_cpu.to(device)
            self._reduce_idx_cache[device] = reduce_idx

        prefixes = input_ids.tolist()  # one GPU→CPU sync for the whole batch
        neg_inf = torch.full_like(scores, float("-inf"))
        for row_idx, prefix in enumerate(prefixes):
            state = self._state_for(prefix)
            cursor, stack_size, edu_has_content = state
            source_ids = self.per_row_source_ids[self._example_idx(row_idx)]

            legal = neg_inf[row_idx].clone()

            # Source copy: only the exact next subword in this row's source is legal.
            if cursor < len(source_ids):
                next_src = source_ids[cursor]
                legal[next_src] = scores[row_idx, next_src]

            # SHIFT: legal iff the current EDU has at least one token.
            if edu_has_content:
                legal[self.shift_id] = scores[row_idx, self.shift_id]

            # REDUCE-*: legal iff stack has ≥2 items.
            if stack_size >= 2:
                legal[reduce_idx] = scores[row_idx, reduce_idx]

            # EOS: legal iff cursor at end, stack singleton, no pending EDU content.
            if cursor >= len(source_ids) and stack_size == 1 and not edu_has_content:
                legal[self.eos_id] = scores[row_idx, self.eos_id]

            neg_inf[row_idx] = legal

        return neg_inf
