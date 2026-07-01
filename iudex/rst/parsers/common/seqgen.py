"""Shared utilities for the generative (text-to-tree) RST parsers, i.e. the
ones that fine-tune a seq2seq or causal LM to emit a linearized tree
(`seq2seq_sr`, `decoder_only_sr`, `seq2seq_sexp`, `decoder_only_sexp`). These
helpers are lifted here, rather than duplicated per parser, because they are
pure, self-contained, and costly to keep in hand-sync across copies:

- `align_edus_to_tokens`: the EDU to subword tiling that keeps train-time COPY
  substitution in lockstep with the inference copy-every-source-token
  constraint. The tiling invariant must agree across train and predict in
  every parser.
- `reorder_past_key_values`: beam-search KV-cache reordering. Defensive
  HF-version-compat plumbing, where a future transformers bump otherwise needs
  the same fix applied in all four parsers or three of them silently rot.
- `beam_topk_step` / `beam_reorder_needed` / `select_best_beam`: the
  serialization-agnostic beam-search primitives (top-K expansion with the
  dead-beam NaN guard, the reorder-is-a-no-op predicate, and GNMT
  length-normalized candidate selection). Each parser's `_predict_one_beam`
  still owns its loop top-to-bottom (mask, state transition, tree
  reconstruction stay local), but these three subtle, must-stay-in-sync blocks
  live here so a fix lands once.
- `reconstruct_text` / `gold_edu_source_ranges` / `empty_tree` /
  `repair_actions` / `fallback_reduce`: the SR parsers' shared text-reconstruct,
  gold-range tiling, single-EDU fallback, and action-sequence repair logic.

The encoder-based parsers (`dmrst`, `topdown_biaffine`, `sr_biaffine`)
do not use these; their shared token-encoding lives in `common/encoding.py`.

This functional-helper layer (not inheritance) is the deliberate de-duplication
seam for the four generative parsers; see CLAUDE.md ("generative parsers") for
why there is no shared base class.
"""

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

from iudex.common.log import warn
from iudex.rst.data.tree import (
    Reduce,
    RstTree,
    Shift,
    ShiftReduceAction,
    strings_to_actions,
)

# GNMT length-normalization exponent for beam selection (Wu et al. 2016). Shared
# default across the four generative parsers' beam loops.
BEAM_LENGTH_PENALTY_ALPHA = 0.6


# -----------------------------------------------------------------
# Embedding gradients & action-head warm-init
# -----------------------------------------------------------------


def mask_old_embedding_gradients(underlying_model: Any, n_old: int) -> tuple[int, int] | None:
    """Train only the newly-added (id >= n_old) input-embedding rows: keep the
    full embedding trainable and register a backward hook zeroing the gradient
    on the pretrained rows [0, n_old). Shared by the four generative parsers,
    which all add ~100 action tokens via `resize_token_embeddings` and must
    train only those rows.

    Crucially this never overrides the embedding module's `forward`, so any
    backbone-specific behavior baked into that forward is preserved. Notably the
    Gemma family (Gemma, T5Gemma, ...) wraps the lookup in a `*ScaledWordEmbedding`
    that multiplies by sqrt(hidden). An earlier "carve" scheme monkey-patched the
    forward to splice a small trainable Parameter for the new rows, which
    silently dropped that scaling (every input embedding ~34-48x too small) and
    badly regressed quality on Gemma backbones (invisible on vanilla T5, which
    has no scaling). The cost of this approach is a dense full-vocab gradient
    (~1 GB bf16 at 1B scale, transient); Adafactor's factored optimizer state
    stays negligible.

    `underlying_model` is the PEFT-unwrapped HF model (exposing
    `get_input_embeddings()`). Encoder/decoder input embeddings are tied (one
    storage), so hooking the single weight covers both sides. Returns
    `(n_total, n_new)` for the caller to log, or None when there are no new rows.
    """
    weight = underlying_model.get_input_embeddings().weight
    n_total = weight.shape[0]
    if n_total <= n_old:
        return None
    weight.requires_grad_(True)

    def _zero_old_rows(grad: torch.Tensor) -> torch.Tensor:
        grad = grad.clone()
        grad[:n_old] = 0
        return grad

    weight.register_hook(_zero_old_rows)
    return n_total, n_total - n_old


def warm_init_head(new_linear: torch.nn.Linear, embed_weight: torch.Tensor, full_id_for_head_idx: list[int]) -> None:
    """Warm-init each row of a freshly-built small action head from the matching
    `embed_tokens` row. The original lm_head was tied to embed_tokens, so row
    `full_id` of embed_tokens is the "right" unembedding direction for token
    `full_id`; copying those rows into the small head means the model starts
    already knowing which hidden direction maps to which token, skipping the
    training that would otherwise just relearn that alignment. For action tokens
    whose embed row was freshly created by `resize_token_embeddings` the row is
    itself random, so this is no worse than an N(0, 0.02) init there (and
    strictly better for pre-existing tokens like EOS). `full_id_for_head_idx[hi]`
    is the full-vocab id seeding head row `hi`. Mutates `new_linear.weight`.

    If the tied input embeddings are the wrong width to copy into the head
    (asymmetric encoder/decoder backbones, e.g. t5gemma-9b-2b: 3584-wide encoder
    embeddings but a 2304-wide decoder lm_head), fall back to the same N(0, 0.02)
    init fresh rows would otherwise get rather than crashing on the dim mismatch.
    """
    with torch.no_grad():
        if embed_weight.shape[-1] != new_linear.weight.shape[-1]:
            new_linear.weight.normal_(mean=0.0, std=0.02)
            return
        for hi, full_id in enumerate(full_id_for_head_idx):
            src = embed_weight[full_id].to(dtype=new_linear.weight.dtype, device=new_linear.weight.device)
            new_linear.weight[hi].copy_(src)


# -----------------------------------------------------------------
# EDU <-> token alignment
# -----------------------------------------------------------------


def align_edus_to_tokens(
    tokenizer: Any,
    text: str,
    edus: Any,
) -> tuple[list[int], list[tuple[int, int]]]:
    """Tokenize `text` (the reconstructed document) and partition its subword
    tokens among `edus` so the per-EDU token ranges TILE range(len(input_ids))
    exactly: no gaps, no overlaps, sum of lengths == len(input_ids).

    Tokenizing the whole doc once and partitioning (rather than tokenizing each
    EDU separately and concatenating) is deliberate: SentencePiece is
    whitespace-sensitive, so per-EDU tokenizations drift from the encoder's
    actual whole-doc tokenization by a few subwords per doc. Tiling keeps the
    gold EDU ranges (`encode_target`, `gold_edu_source_ranges`) in the same
    token space as the pred ranges the inference loop tracks by cursor.

    `edus` is a sequence of objects with `.text: str` and `.prefix: str | None`
    (default prefix " " for all but the first EDU), matching how `reconstruct_text`
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

    # Exclusive char-end per EDU, walking prefixes/text exactly like reconstruct_text.
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


# -----------------------------------------------------------------
# Beam search
# -----------------------------------------------------------------


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
    # Path 2b: newer transformers Cache classes renamed the beam reorder to
    # `batch_select_indices` (mutates in place, returns None). Without this
    # path a future HF bump would fall through to the tuple walk below, which
    # iterates a Cache object with version-dependent layout and could silently
    # mis-reorder every beam decode.
    if hasattr(past_key_values, "batch_select_indices"):
        result = past_key_values.batch_select_indices(beam_idx)
        return result if result is not None else past_key_values
    # Path 3: manual tuple walk, handling Nones gracefully.
    return tuple(
        tuple(t.index_select(0, beam_idx) if isinstance(t, torch.Tensor) else t for t in layer)
        for layer in past_key_values
    )


def beam_topk_step(
    beam_scores: torch.Tensor,
    logits: torch.Tensor,
    legal_mask: torch.Tensor,
    k: int,
) -> tuple[torch.Tensor, list[int], list[int]]:
    """One beam-search expansion step, serialization-agnostic.

    `logits` is the [K, V] RAW (unmasked) model output; `legal_mask` is a
    [K, V] bool tensor, True where the caller's validity constraints admit the
    continuation (a done/dead beam's row should be all False). `beam_scores`
    is [K] cumulative log-prob (dead beams at -inf). Returns the top-K
    continuations as (top_scores [K], parents list[int], actions list[int]),
    where the flat top-k index `flat = parent * V + action` is decoded back
    into the parent beam and the chosen action (a column of `logits`).

    Scoring is deliberately UNrenormalized: log_softmax runs over the full raw
    vocab FIRST, and illegal entries are then dropped to -inf. Renormalizing
    over the legal set (log_softmax of pre-masked logits, the previous
    behavior) made heavily-constrained steps nearly free. Under the sexp
    constraints the in-leaf legal set is ~2 ids, so a full-vocab distribution
    renormalized to a binary choice priced skipping an EDU boundary at ~0 and
    beam search collapsed unconfident documents to a handful of giant EDUs (a
    legal, "high-scoring" parse the raw model distribution hates, e.g.
    wsj_1118: renormalized -2.5 vs raw -974 for the 3-EDU parse). Raw scoring
    matches the HF generate() default (renormalize_logits is opt-in there) and
    greedy argmax is unaffected either way. The NaN guard stays as a backstop
    for -inf raw logits. Shared verbatim across the four generative parsers'
    beam loops (it must stay in sync, the failure mode is silent)."""
    v = logits.size(-1)
    log_probs = F.log_softmax(logits.float(), dim=-1)
    log_probs = torch.where(legal_mask, log_probs, torch.full_like(log_probs, float("-inf")))
    cum = beam_scores.unsqueeze(1) + log_probs
    cum = torch.where(torch.isnan(cum), torch.full_like(cum, float("-inf")), cum)
    top_scores, top_idx = cum.view(-1).topk(k)
    parents = (top_idx // v).tolist()
    actions = (top_idx % v).tolist()
    return top_scores, parents, actions


def beam_reorder_needed(step: int, parents: list[int], k: int, past_key_values) -> bool:
    """Whether the KV cache + decoder inputs need reordering by `parents` this
    step. Skips the no-op cases: step 0 expands K identical rows from the single
    seed beam (all parents 0), and an identity permutation rearranges nothing.
    `past_key_values is None` (pre-cache) also needs no reorder."""
    if past_key_values is None:
        return False
    is_step0_uniform = step == 0 and all(p == 0 for p in parents)
    is_identity = parents == list(range(k))
    return not (is_step0_uniform or is_identity)


def select_best_beam(candidates: list[dict], alpha: float = BEAM_LENGTH_PENALTY_ALPHA) -> dict:
    """Pick the length-normalized best beam from a candidate pool. Each candidate
    is a dict carrying at least `"score"` (cumulative sum log-prob) and
    `"length"` (token count); `"finished"` (bool) marks hypotheses that
    legitimately reached EOS. Finished candidates are preferred outright: an
    unfinished hypothesis is a truncated prefix (max_output_length hit), has
    paid for fewer tokens, and under length normalization can spuriously
    outrank a complete parse, so it is only eligible when NO hypothesis
    finished (the fallback that keeps truncated documents recoverable).
    Dividing by `length**alpha` mitigates the bias toward shorter beams (every
    emitted token has log-prob <= 0, so raw sum-log-prob monotonically favors
    fewer-token trajectories). `alpha=0.6` is the GNMT default. Caller must
    ensure `candidates` is non-empty."""
    finished = [c for c in candidates if c.get("finished", False)]
    pool = finished if finished else candidates
    return max(pool, key=lambda c: c["score"] / max(c["length"], 1) ** alpha)


# -----------------------------------------------------------------
# Shift-reduce decode state
# -----------------------------------------------------------------


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


# -----------------------------------------------------------------
# SR tree reconstruction
# -----------------------------------------------------------------


def reconstruct_text(tree: RstTree) -> str:
    """Reverse the storage convention: join EDU strings with spaces (or each
    EDU's `prefix` field if populated, for detokenized corpora)."""
    parts: list[str] = []
    for i, edu in enumerate(tree.edus):
        if i == 0:
            parts.append(edu.text)
            continue
        prefix = edu.prefix if edu.prefix is not None else " "
        parts.append(prefix + edu.text)
    return "".join(parts)


def gold_edu_source_ranges(tokenizer, tree: RstTree) -> list[tuple[int, int]]:
    """Per-EDU `(start, end_exclusive)` token-position ranges in the source
    tokenizer's whole-doc tokenization space, tiling it exactly. Delegates to
    `align_edus_to_tokens` so train and predict agree on the tiling."""
    text = reconstruct_text(tree)
    _, spans = align_edus_to_tokens(tokenizer, text, tree.edus)
    return spans


def empty_tree(relation_types, text: str = "") -> RstTree:
    """Single-EDU fallback for empty / unrecoverable input. The text payload
    becomes one EDU so downstream callers (to_rs4_string, eval) work."""
    actions: list[ShiftReduceAction] = [Shift(edu_text=text or "")]
    return RstTree.from_shift_reduce(actions, relation_types=relation_types)


def fallback_reduce(reduce_token_map) -> "Reduce | None":
    """A Reduce action to close an unfinished tree. Prefers NS-elaboration if
    available, else the first reduce in the vocabulary."""
    for _token_str, (nuc, rel) in reduce_token_map.items():
        if (nuc, rel) == ("NS", "elaboration"):
            return Reduce(nuc=nuc, rel=rel)
    for _token_str, (nuc, rel) in reduce_token_map.items():
        return Reduce(nuc=nuc, rel=rel)
    return None


def repair_actions(strings: list[str], reduce_token_map) -> tuple[list[ShiftReduceAction], str | None]:
    """Try `strings_to_actions` on the raw string list. If trailing source
    tokens are present, append a closing `<shift>` and the right number of
    fallback reduces to drain the stack. Returns the action list plus a reason
    if the sequence had to be repaired, None if it parsed cleanly."""
    try:
        actions = strings_to_actions(strings, reduce_token_map)
    except ValueError:
        # Trailing source tokens: append a closing <shift>, then check
        # stack-size against the resulting Shift count and add reduces below.
        repaired = list(strings) + [Shift().to_token()]
        try:
            actions = strings_to_actions(repaired, reduce_token_map)
        except ValueError as e:
            return [], str(e)
        n_shifts = sum(1 for a in actions if isinstance(a, Shift))
        n_reduces = sum(1 for a in actions if isinstance(a, Reduce))
        needed = (n_shifts - 1) - n_reduces
        if needed < 0:
            return actions, f"too many reduces ({n_reduces}) for {n_shifts} shifts"
        if needed > 0:
            fallback = fallback_reduce(reduce_token_map)
            if fallback is None:
                return actions, "no fallback reduce token available"
            actions = list(actions) + [fallback] * needed
        return actions, "max_length hit mid-EDU, appended closing shift/reduces"
    n_shifts = sum(1 for a in actions if isinstance(a, Shift))
    n_reduces = sum(1 for a in actions if isinstance(a, Reduce))
    if n_shifts == 0:
        return actions, "no shifts in generated sequence"
    if n_reduces != n_shifts - 1:
        needed = (n_shifts - 1) - n_reduces
        if needed < 0:
            return actions, f"too many reduces ({n_reduces}) for {n_shifts} shifts"
        fallback = fallback_reduce(reduce_token_map)
        if fallback is None:
            return actions, "stack underdrained and no fallback reduce available"
        return list(actions) + [fallback] * needed, "stack underdrained, appended closing reduces"
    return actions, None
