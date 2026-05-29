"""Shared utilities for the generative (text-to-tree) RST parsers, i.e. the
ones that fine-tune a seq2seq or causal LM to emit a linearized tree
(`seq2seq_sr`, `decoder_only_sr`, `seq2seq_sexp`, `decoder_only_sexp`). These
two helpers are lifted here, rather than duplicated per parser, because both
are pure, self-contained, and costly to keep in hand-sync across four copies:

- `align_edus_to_tokens`: the EDU to subword tiling that keeps train-time COPY
  substitution in lockstep with the inference copy-every-source-token
  constraint. The tiling invariant must agree across train and predict in
  every parser.
- `reorder_past_key_values`: beam-search KV-cache reordering. Defensive
  HF-version-compat plumbing, where a future transformers bump otherwise needs
  the same fix applied in all four parsers or three of them silently rot.

The encoder-based parsers (`dmrst`, `piudotto`, `topdown_biaffine`) do not use
these; their shared token-encoding lives in `common/encoding.py`.
"""

from dataclasses import dataclass, field
from typing import Any

import torch

from iudex.common.log import warn

# GNMT length-normalization exponent for beam selection (Wu et al. 2016). Shared
# default across the four generative parsers' beam loops.
BEAM_LENGTH_PENALTY_ALPHA = 0.6


def align_edus_to_tokens(
    tokenizer: Any,
    text: str,
    edus: Any,
) -> tuple[list[int], list[tuple[int, int]]]:
    """Tokenize `text` (the reconstructed document) and partition its subword
    tokens among `edus` so the per-EDU token ranges TILE range(len(input_ids))
    exactly: no gaps, no overlaps, sum of lengths == len(input_ids).

    `edus` is a sequence of objects with `.text: str` and `.prefix: str | None`
    (default prefix " " for all but the first EDU), matching how `_reconstruct_text`
    builds `text`. Assignment is by a single monotonic forward sweep over tokens:
    each token goes to the current EDU until its character midpoint crosses into
    the next EDU's char range, and the final EDU absorbs all trailing tokens. This
    guarantees a tiling even when a token straddles a boundary or sits in
    inter-EDU whitespace. An EDU shorter than a token may receive an empty range
    (start == end), which is allowed and still tiles.

    Returns (input_ids: list[int], edu_token_spans: list[tuple[int, int]]) where
    edu_token_spans[i] = (start, end) is a half-open token-index range into
    input_ids for EDU i.
    """
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    input_ids = enc["input_ids"]
    offsets = enc["offset_mapping"]

    # Exclusive char-end per EDU, walking prefixes/text exactly like _reconstruct_text.
    char_ends: list[int] = []
    char_cursor = 0
    for i, edu in enumerate(edus):
        if i > 0:
            prefix = edu.prefix if edu.prefix is not None else " "
            char_cursor += len(prefix)
        char_cursor += len(edu.text)
        char_ends.append(char_cursor)

    n_edus = len(char_ends)
    counts = [0] * n_edus
    edu_idx = 0
    for tcs, tce in offsets:
        m = (tcs + tce) / 2
        while edu_idx < n_edus - 1 and m >= char_ends[edu_idx]:
            edu_idx += 1
        counts[edu_idx] += 1

    spans: list[tuple[int, int]] = []
    cursor = 0
    for c in counts:
        spans.append((cursor, cursor + c))
        cursor += c
    return input_ids, spans


def reorder_past_key_values(past_key_values, beam_idx: torch.Tensor, model):
    """Reorder a HF past_key_values cache along the beam dimension. Handles
    three layouts:
      1. The model exposes `_reorder_cache(pkv, beam_idx)` (T5/T5Gemma2 and most
         HF seq2seq models).
      2. `past_key_values` is a `DynamicCache`-like object with its own
         `reorder_cache` method (newer transformers).
      3. Tuple-of-tuple of Tensors (older HF), possibly with `None` entries for
         unfilled cross-attention slots.

    `model` is the underlying (PEFT-unwrapped) model that may carry the legacy
    `_reorder_cache` helper.
    """
    # Path 1: canonical HF helper on the base model. T5Gemma 2's inherited
    # `_reorder_cache` assumes the legacy tuple-of-tuple layout, and newer HF
    # versions may hand us a DynamicCache instead, which makes that call blow
    # up. Catch and fall through to the next path on type/attribute mismatches.
    reorder = getattr(model, "_reorder_cache", None)
    if callable(reorder):
        try:
            result = reorder(past_key_values, beam_idx)
            # Modern HF cache classes mutate in place and return None.
            # Blindly returning None drops the cache on the next step.
            return result if result is not None else past_key_values
        except (TypeError, AttributeError) as e:
            warn(
                f"{type(model).__name__}._reorder_cache failed on "
                f"{type(past_key_values).__name__} ({type(e).__name__}: {e}). "
                "Falling back to object/tuple cache reordering."
            )
    # Path 2: DynamicCache or similar object-style cache.
    if hasattr(past_key_values, "reorder_cache"):
        result = past_key_values.reorder_cache(beam_idx)
        return result if result is not None else past_key_values
    # Path 3: manual tuple walk, handling Nones gracefully.
    return tuple(
        tuple(t.index_select(0, beam_idx) if isinstance(t, torch.Tensor) else t for t in layer)
        for layer in past_key_values
    )


@dataclass
class ShiftReduceDecodeState:
    """Bottom-up shift-reduce decode state for the SR generative parsers
    (`seq2seq_sr`, `decoder_only_sr`), the shift-reduce analogue of the sexp
    parsers' `SexpDecodingState`. Vocab-agnostic: it tracks the source cursor,
    the constituent-stack size, and the current EDU's COPY count, exposing the
    four validity predicates and the four transitions that the greedy, beam,
    and gold-EDU loops share. The parser maps the predicates to its own action
    head indices and classifies emitted ids back into the four action kinds, so
    the vocab-specific glue stays per-parser while the automaton lives here.

    The state machine over actions {COPY, SHIFT, REDUCE, EOS}:
      COPY   advances the source cursor and extends the current EDU.
      SHIFT  commits the current EDU (records its `(start, cursor)` source-token
             range), pushes a leaf, and resets the EDU counter.
      REDUCE pops two constituents and pushes one.
      EOS    terminates.
    """

    source_len: int
    min_edu_length: int = 1
    cursor: int = 0
    stack_size: int = 0
    edu_length: int = 0
    edu_start: int = 0
    pred_edu_ranges: list[tuple[int, int]] = field(default_factory=list)
    done: bool = False

    def clone(self) -> "ShiftReduceDecodeState":
        """Deep-enough copy for beam expansion (the only mutable field is the
        ranges list)."""
        return ShiftReduceDecodeState(
            source_len=self.source_len,
            min_edu_length=self.min_edu_length,
            cursor=self.cursor,
            stack_size=self.stack_size,
            edu_length=self.edu_length,
            edu_start=self.edu_start,
            pred_edu_ranges=list(self.pred_edu_ranges),
            done=self.done,
        )

    @property
    def at_end(self) -> bool:
        return self.cursor >= self.source_len

    @property
    def copy_ok(self) -> bool:
        return not self.at_end

    @property
    def shift_ok(self) -> bool:
        # At least `min_edu_length` COPYs, or end-of-source with any content so
        # the final EDU can still be committed.
        return self.edu_length >= self.min_edu_length or (self.at_end and self.edu_length >= 1)

    @property
    def reduce_ok(self) -> bool:
        return self.stack_size >= 2

    @property
    def eos_ok(self) -> bool:
        return self.at_end and self.stack_size == 1 and self.edu_length == 0

    def step_copy(self) -> bool:
        """Consume one source token. Returns False (and marks done) if the
        source is already exhausted, which the validity mask should prevent."""
        if self.cursor >= self.source_len:
            self.done = True
            return False
        self.cursor += 1
        self.edu_length += 1
        return True

    def step_shift(self) -> None:
        self.stack_size += 1
        self.pred_edu_ranges.append((self.edu_start, self.cursor))
        self.edu_start = self.cursor
        self.edu_length = 0

    def step_reduce(self) -> None:
        self.stack_size -= 1

    def step_eos(self) -> None:
        self.done = True
