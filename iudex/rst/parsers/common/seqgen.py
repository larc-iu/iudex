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

from typing import Any

import torch


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
            import warnings

            warnings.warn(
                f"{type(model).__name__}._reorder_cache failed on "
                f"{type(past_key_values).__name__} ({type(e).__name__}: {e}). "
                "Falling back to object/tuple cache reordering.",
                stacklevel=2,
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
