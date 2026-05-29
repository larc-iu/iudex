"""Shared token-encoding utilities for RST parsers which rely on BERT-like encoders."""

from typing import TYPE_CHECKING, Any

import torch
from transformers import AutoModel, AutoTokenizer

if TYPE_CHECKING:
    from iudex.rst.parsers.common.detokenization import Detokenizer

# Tokenizers report `model_max_length = int(1e30)` when they have no advertised
# limit (e.g. SpanBERT). Treat anything above this sentinel as "unspecified".
_TOKENIZER_MAX_LEN_SENTINEL = 1_000_000


def load_encoder_and_tokenizer(model_name: str, peft_config: Any | None = None) -> tuple[torch.nn.Module, Any, int]:
    """Load a BERT-style HF encoder + tokenizer. Returns (encoder, tokenizer, max_length).

    Forces fp32: transformers>=5 honors the checkpoint dtype, and fp16
    checkpoints (e.g. SpanBERT) NaN immediately under AdamW. Raises if
    CLS/SEP are missing (the striding encoder needs both).

    When `peft_config` is non-null the encoder is wrapped in a LoRA `PeftModel`
    (base weights frozen, low-rank adapters trainable). The wrapper forwards
    attribute access (`.config`, `.embeddings`, `.encoder.layer`, `.forward`)
    and its `state_dict` keeps the full base weights plus adapters, so callers
    need no other changes. Because loads reconstruct the model via `Parser(cfg)`
    then `load_state_dict(strict=True)`, `peft_config` MUST live on the parser
    config so the identical wrapping is rebuilt at load time. `peft_config` is
    duck-typed: any object with `r`, `alpha`, `dropout`, `target_modules`, `bias`, `dora`.
    """
    encoder = AutoModel.from_pretrained(model_name).float()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.cls_token_id is None or tokenizer.sep_token_id is None:
        raise ValueError(
            f"Tokenizer for {model_name!r} lacks cls_token and/or sep_token; "
            f"this parser only supports BERT-style encoders."
        )

    max_length = tokenizer.model_max_length
    if max_length > _TOKENIZER_MAX_LEN_SENTINEL:
        max_length = encoder.config.max_position_embeddings

    if peft_config is not None:
        encoder = _wrap_lora(encoder, peft_config)
    return encoder, tokenizer, max_length


def _wrap_lora(encoder: torch.nn.Module, peft_config: Any) -> torch.nn.Module:
    """Wrap `encoder` in a LoRA `PeftModel` (feature-extraction task)."""
    from peft import LoraConfig, TaskType, get_peft_model

    lora_config = LoraConfig(
        r=peft_config.r,
        lora_alpha=peft_config.alpha,
        lora_dropout=peft_config.dropout,
        target_modules=peft_config.target_modules,
        bias=peft_config.bias,
        use_dora=peft_config.dora,
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    return get_peft_model(encoder, lora_config)


def tokenize_edus(
    tokenizer: Any,
    edu_strings: list[str],
    device: torch.device,
) -> tuple[torch.Tensor, list[tuple[int, int]]]:
    """Tokenize a sequence of EDUs into a flat token-id tensor + per-EDU boundaries.

    Returns:
        input_ids: [num_tokens]
        boundaries: list of (start_token, end_token_exclusive) per EDU
    """
    all_ids: list[int] = []
    boundaries: list[tuple[int, int]] = []
    for edu_text in edu_strings:
        ids = tokenizer.encode(edu_text, add_special_tokens=False)
        start = len(all_ids)
        all_ids.extend(ids)
        boundaries.append((start, len(all_ids)))
    return torch.tensor(all_ids, dtype=torch.long, device=device), boundaries


def tokenize_document(
    tokenizer: Any,
    edu_strings: list[str],
    device: torch.device,
    detokenizer: "Detokenizer | None" = None,
    prefixes: "list[str | None] | None" = None,
) -> tuple[torch.Tensor, list[tuple[int, int]]]:
    """Tokenize EDUs as one continuous document, mapping gold EDU boundaries
    onto the continuous token offsets. Same return contract as `tokenize_edus`.

    Text reconstruction picks the most faithful source available:

      - `prefixes` given (detokenized corpora, e.g. data/gum_12.1.0_notok): EDU
        text is used verbatim and joined by its exact inter-EDU `prefix` string
        (`None` -> single space, "" -> glued). This reproduces the raw document
        byte-for-byte, so it supersedes `detokenizer` (which is ignored).
      - else `detokenizer` given: each word-tokenized EDU is detokenized to
        natural text and the EDUs are joined with single spaces.
      - else: EDUs are stripped and joined with single spaces.

    All three encode the whole string once. Unlike `tokenize_edus` (which
    encodes each EDU in isolation), this matters for joint segmenters: encoding
    an EDU in isolation strips the leading-space marker (e.g. RoBERTa/ModernBert
    `Ġ`) from its first subword, so every EDU-initial token looks word-initial.
    A segmenter trained that way learns "no leading-space marker = boundary", a
    cue absent from real continuous text, making it predict zero breaks at
    inference (where `predict_from_text` tokenizes the raw string continuously).
    Encoding continuously here keeps train and inference tokenization identical.
    SentencePiece encoders (e.g. XLM-R) mark word starts the same way in both
    modes, so they were unaffected, but this is correct for them too.

    Glued (`prefix=""`) boundaries have no joining space, so an EDU boundary can
    fall mid-token; the straddling token is then assigned to the later EDU. A
    boundary landing strictly inside a single token that also spans the previous
    boundary would empty out an EDU's token range, which we reject explicitly
    rather than letting it NaN downstream.

    Requires a fast tokenizer (offset mapping).
    """
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("tokenize_document requires a fast tokenizer (offset mapping unavailable)")

    use_prefix = prefixes is not None and any(p is not None for p in prefixes)
    if use_prefix:
        edus = list(edu_strings)
        seps = ["" if i == 0 else (prefixes[i] if prefixes[i] is not None else " ") for i in range(len(edus))]
    else:
        edus = (
            [detokenizer.detokenize(e) for e in edu_strings]
            if detokenizer is not None
            else [e.strip() for e in edu_strings]
        )
        seps = ["" if i == 0 else " " for i in range(len(edus))]

    doc = "".join(seps[i] + edus[i] for i in range(len(edus)))
    enc = tokenizer(doc, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]

    # Exclusive char-end of each EDU within the reconstructed doc.
    char_ends: list[int] = []
    pos = 0
    for i, edu in enumerate(edus):
        pos += len(seps[i]) + len(edu)
        char_ends.append(pos)

    # A token belongs to EDU i iff its char-end falls within EDU i's char span.
    # A space-joined boundary always lands on a token break; a glued boundary
    # may not, in which case the straddling token (char-end past this EDU's end)
    # falls through to the next EDU.
    boundaries: list[tuple[int, int]] = []
    tok_idx, ntok = 0, len(offsets)
    for i, end_char in enumerate(char_ends):
        start = tok_idx
        while tok_idx < ntok and offsets[tok_idx][1] <= end_char:
            tok_idx += 1
        if tok_idx == start:
            raise ValueError(
                f"EDU {i} ({edus[i]!r}) maps to an empty token span: its boundary "
                f"falls inside a single token (a glued prefix with no token break)."
            )
        boundaries.append((start, tok_idx))
    # Defensive: a token overrunning the last EDU end (shouldn't happen) is
    # folded into the final EDU rather than dropped.
    if tok_idx < ntok and boundaries:
        s, _ = boundaries[-1]
        boundaries[-1] = (s, ntok)

    input_ids = torch.tensor(enc["input_ids"], dtype=torch.long, device=device)
    return input_ids, boundaries


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
    the next EDU's char range; the final EDU absorbs all trailing tokens. This
    guarantees a tiling even when a token straddles a boundary or sits in
    inter-EDU whitespace. An EDU shorter than a token may receive an empty range
    (start == end); that is allowed and still tiles.

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


def encode_tokens_strided(
    encoder: torch.nn.Module,
    tokenizer: Any,
    input_ids: torch.Tensor,
    max_length: int,
    stride: int,
) -> torch.Tensor:
    """Encode a flat token sequence with overlapping sliding windows.

    Long documents exceed the LM's positional budget, so we tile with windows
    that overlap by `stride` tokens. Overlapped positions keep the embedding
    from the *earlier* window (more left context).

    Returns: [num_tokens, hidden_size]  (1:1 with input positions).
    """
    max_content = max_length - 2  # leave room for [CLS] ... [SEP] per chunk
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    device = input_ids.device

    content_len = input_ids.shape[0]
    chunks, chunk_lens = [], []
    pos = 0
    while True:
        end = min(pos + max_content, content_len)
        chunk = torch.cat(
            [
                torch.tensor([cls_id], device=device),
                input_ids[pos:end],
                torch.tensor([sep_id], device=device),
            ]
        )
        chunks.append(chunk)
        chunk_lens.append(chunk.shape[0])
        if end >= content_len:
            break
        pos = end - stride  # next window starts `stride` tokens before this one ended

    max_chunk_len = max(chunk_lens)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    batch_ids = torch.full((len(chunks), max_chunk_len), pad_id, device=device, dtype=torch.long)
    batch_mask = torch.zeros(len(chunks), max_chunk_len, device=device, dtype=torch.long)
    for i, cids in enumerate(chunks):
        batch_ids[i, : cids.shape[0]] = cids
        batch_mask[i, : cids.shape[0]] = 1

    hidden = encoder(input_ids=batch_ids, attention_mask=batch_mask).last_hidden_state
    # hidden: [num_chunks, max_chunk_len, hidden_size]

    # Strip CLS/SEP. For chunks i > 0, also drop the first `stride` tokens
    # (which are duplicates of the previous chunk's tail).
    pieces = []
    for i, clen in enumerate(chunk_lens):
        emb = hidden[i, 1 : clen - 1]
        pieces.append(emb if i == 0 else emb[stride:])
    return torch.cat(pieces, dim=0)[:content_len]
