"""Shared token-encoding utilities for RST parsers which rely on BERT-like encoders."""

from typing import Any

import torch
from transformers import AutoModel, AutoTokenizer

# Tokenizers report `model_max_length = int(1e30)` when they have no advertised
# limit (e.g. SpanBERT). Treat anything above this sentinel as "unspecified".
_TOKENIZER_MAX_LEN_SENTINEL = 1_000_000


def load_encoder_and_tokenizer(model_name: str) -> tuple[torch.nn.Module, Any, int]:
    """Load a BERT-style HF encoder + tokenizer. Returns (encoder, tokenizer, max_length).

    Forces fp32: transformers>=5 honors the checkpoint dtype, and fp16
    checkpoints (e.g. SpanBERT) NaN immediately under AdamW. Raises if
    CLS/SEP are missing (the striding encoder needs both).
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
    return encoder, tokenizer, max_length


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
) -> tuple[torch.Tensor, list[tuple[int, int]]]:
    """Tokenize EDUs as one continuous document, mapping gold EDU boundaries
    onto the continuous token offsets. Same return contract as `tokenize_edus`.

    Unlike `tokenize_edus` (which encodes each EDU in isolation), this joins the
    EDUs with single spaces and encodes the whole string once. The difference
    matters for joint segmenters: encoding an EDU in isolation strips the
    leading-space marker (e.g. RoBERTa/ModernBert `Ġ`) from its first subword,
    so every EDU-initial token looks word-initial. A segmenter trained that way
    learns "no leading-space marker = boundary", a cue that is absent from real
    continuous text and makes it predict zero breaks at inference (where
    `predict_from_text` tokenizes the raw string continuously). Encoding
    continuously here keeps train and inference tokenization identical.

    SentencePiece encoders (e.g. XLM-R) prefix word starts with the same marker
    in both modes, so they were unaffected, but this is correct for them too.

    Requires a fast tokenizer (offset mapping).
    """
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("tokenize_document requires a fast tokenizer (offset mapping unavailable)")

    edus = [e.strip() for e in edu_strings]
    doc = " ".join(edus)
    enc = tokenizer(doc, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]

    # Exclusive char-end of each EDU within the space-joined doc.
    char_ends: list[int] = []
    pos = 0
    for i, edu in enumerate(edus):
        if i > 0:
            pos += 1  # the single joining space
        pos += len(edu)
        char_ends.append(pos)

    # A token belongs to EDU i iff its char-end falls within EDU i's char span.
    # The joining space forces a token break, so no token straddles a boundary.
    boundaries: list[tuple[int, int]] = []
    tok_idx, ntok = 0, len(offsets)
    for end_char in char_ends:
        start = tok_idx
        while tok_idx < ntok and offsets[tok_idx][1] <= end_char:
            tok_idx += 1
        boundaries.append((start, tok_idx))
    # Defensive: a token overrunning the last EDU end (shouldn't happen with
    # space-joined EDUs) is folded into the final EDU rather than dropped.
    if tok_idx < ntok and boundaries:
        s, _ = boundaries[-1]
        boundaries[-1] = (s, ntok)

    input_ids = torch.tensor(enc["input_ids"], dtype=torch.long, device=device)
    return input_ids, boundaries


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
