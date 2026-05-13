"""Top-down RST parser with biaffine scoring (after Kobayashi et al., 2022).

Architecture:
  - Striding transformer encoder over per-EDU subtoken concatenation.
  - EDU representation: mean of first and last subtoken embeddings.
  - For a span [b, e) split at candidate k:
      left_rep_k  = (first_subtoken_of_span + last_subtoken_of_EDU_{k-1}) / 2
      right_rep_k = (first_subtoken_of_EDU_k + last_subtoken_of_span) / 2
    `split_biaffine(left_rep, right_rep) -> scalar score per k`
    `label_biaffine(left_rep, right_rep) -> num_labels logits per k`

Whole-tree training: forward(tree) walks the gold parse top-down and sums
per-decision losses. Assumes gold EDU segmentation.
"""
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from iudex.rst.data.reader import determine_label_index
from iudex.rst.data.tree import RstPpTree
from iudex.rst.parsers.topdown_biaffine.configuration_topdown_biaffine import TopdownBiaffineConfig


class _FeedForward(nn.Sequential):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_p):
        super().__init__(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, output_dim),
        )


class _DeepBiAffine(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_p):
        super().__init__()
        self.W_left = _FeedForward(input_dim, hidden_dim, hidden_dim, dropout_p)
        self.W_right = _FeedForward(input_dim, hidden_dim, hidden_dim, dropout_p)
        self.W_s = nn.Bilinear(hidden_dim, hidden_dim, output_dim)
        self.V_left = nn.Linear(hidden_dim, output_dim)
        self.V_right = nn.Linear(hidden_dim, output_dim)

    def forward(self, h_left, h_right):
        h_left = self.W_left(h_left)
        h_right = self.W_right(h_right)
        return self.W_s(h_left, h_right) + self.V_left(h_left) + self.V_right(h_right)


class TopdownBiaffineParser(nn.Module):
    def __init__(self, config: TopdownBiaffineConfig):
        super().__init__()
        self.config = config
        self._relation_types = tuple(config.relation_types)
        self.label_index = determine_label_index(self._relation_types)
        self.stride = config.stride

        encoder_kwargs = {}
        if config.attn_implementation is not None:
            encoder_kwargs["attn_implementation"] = config.attn_implementation
        # transformers >=5 honors the checkpoint's saved dtype; many HF checkpoints
        # (e.g. SpanBERT) are fp16, which makes AdamW updates NaN immediately. Force fp32.
        self.encoder = AutoModel.from_pretrained(config.model_name, **encoder_kwargs).float()
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.hidden_size = self.encoder.config.hidden_size
        # HF tokenizers can report a sentinel `model_max_length` of ~1e30 when
        # unset; fall back to the encoder's actual positional-embedding budget.
        self.max_length = min(
            getattr(self.encoder.config, "max_position_embeddings", self.tokenizer.model_max_length),
            self.tokenizer.model_max_length,
        )

        self.split_biaffine = _DeepBiAffine(self.hidden_size, config.ffn_hidden_size, 1, config.dropout)
        self.label_biaffine = _DeepBiAffine(self.hidden_size, config.ffn_hidden_size, len(self.label_index), config.dropout)

    @property
    def relation_types(self):
        return self._relation_types

    @property
    def device(self):
        return next(self.parameters()).device

    def _encode_tree(self, tree: RstPpTree) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize the document EDU-by-EDU and encode. Returns (embeddings, edu_boundaries)."""
        all_ids: List[int] = []
        boundaries: List[Tuple[int, int]] = []
        for edu_text in tree.edu_strings:
            ids = self.tokenizer.encode(edu_text, add_special_tokens=False)
            start = len(all_ids)
            all_ids.extend(ids)
            boundaries.append((start, len(all_ids)))

        input_ids = torch.tensor(all_ids, dtype=torch.long, device=self.device)
        embeddings = self._encode_subtokens(input_ids).float()
        edu_boundaries = torch.tensor(boundaries, dtype=torch.long, device=self.device)
        return embeddings, edu_boundaries

    def _encode_subtokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Encode `input_ids` with overlapping windows when it exceeds the LM's
        max length. `self.stride` is the number of tokens of overlap between
        adjacent windows; overlapped tokens keep the embedding from the *earlier*
        window (where they have more left context).
        Returns [num_subtokens, hidden_size], 1:1 with input positions.
        """
        max_content = self.max_length - 2  # leave room for [CLS] ... [SEP] per chunk
        cls_id = self.tokenizer.cls_token_id
        sep_id = self.tokenizer.sep_token_id
        device = input_ids.device

        content_len = input_ids.shape[0]
        chunks, chunk_lens = [], []
        pos = 0
        while pos < content_len:
            end = min(pos + max_content, content_len)
            chunk = torch.cat([
                torch.tensor([cls_id], device=device),
                input_ids[pos:end],
                torch.tensor([sep_id], device=device),
            ])
            chunks.append(chunk)
            chunk_lens.append(chunk.shape[0])
            if end >= content_len:
                break
            pos = end - self.stride  # next window starts `stride` tokens before this one ended

        # Pad chunks to uniform length and run them through the encoder in one batch.
        max_chunk_len = max(chunk_lens)
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        batch_ids = torch.full((len(chunks), max_chunk_len), pad_id, device=device, dtype=torch.long)
        batch_mask = torch.zeros(len(chunks), max_chunk_len, device=device, dtype=torch.long)
        for i, cids in enumerate(chunks):
            batch_ids[i, :cids.shape[0]] = cids
            batch_mask[i, :cids.shape[0]] = 1

        hidden = self.encoder(input_ids=batch_ids, attention_mask=batch_mask).last_hidden_state

        # Strip CLS/SEP; for chunks i > 0, also drop the first `stride` tokens
        # (which are duplicates of the previous chunk's tail).
        pieces = []
        for i, clen in enumerate(chunk_lens):
            emb = hidden[i, 1:clen - 1]
            pieces.append(emb if i == 0 else emb[self.stride:])
        return torch.cat(pieces, dim=0)[:content_len]

    def _packed_lr(self, embeddings, edu_boundaries, b, e):
        """Build the left/right span-edge representations for every candidate
        split k in (b, e), implementing the formula in the module docstring.

        Returns (packed_l, packed_r), each `[num_splits=e-b-1, hidden_size]`.
        embeddings: `[num_subtokens, H]`; edu_boundaries: `[num_edus, 2]` rows
        of `(start_subtoken, end_subtoken_exclusive)`.
        """
        leftmost_h = embeddings[edu_boundaries[b, 0]]            # first subtoken of span [b, e)
        rightmost_h = embeddings[edu_boundaries[e - 1, 1] - 1]   # last subtoken of span [b, e)
        num_splits = e - b - 1
        # For each candidate split k: last subtoken of EDU k-1, first of EDU k.
        left_idx = torch.stack([edu_boundaries[k - 1, 1] - 1 for k in range(b + 1, e)])
        right_idx = torch.stack([edu_boundaries[k, 0] for k in range(b + 1, e)])
        packed_l = (leftmost_h.unsqueeze(0).expand(num_splits, -1) + embeddings[left_idx]) / 2
        packed_r = (embeddings[right_idx] + rightmost_h.unsqueeze(0).expand(num_splits, -1)) / 2
        return packed_l, packed_r

    def forward(self, tree: RstPpTree) -> Dict[str, torch.Tensor]:
        """Teacher-forced DFS over the gold tree. At each non-leaf span [b, e)
        we score every candidate split (split_loss) AND the (nuclearity, relation)
        at the *gold* split (label_loss), then recurse into the gold sub-spans.
        """
        num_edus = len(tree.edus)
        if num_edus < 2:
            return {"loss": torch.zeros((), device=self.device, requires_grad=True)}

        embeddings, edu_boundaries = self._encode_tree(tree)

        # Index every gold non-leaf span by its EDU range.
        gold: Dict[Tuple[int, int], Tuple[int, str]] = {}
        for (left_range, right_range), nuc, rel in tree.spans_with_ranges():
            gold[(left_range[0], right_range[1])] = (right_range[0], f"{nuc}_{rel}")

        split_losses, label_losses = [], []
        stack = [(0, num_edus)]
        while stack:
            b, e = stack.pop()
            if e - b <= 1:  # single-EDU span: no decision to make
                continue

            packed_l, packed_r = self._packed_lr(embeddings, edu_boundaries, b, e)
            split_logits = self.split_biaffine(packed_l, packed_r).squeeze(-1)  # [e-b-1]
            label_logits = self.label_biaffine(packed_l, packed_r)              # [e-b-1, num_labels]

            gold_split, gold_label_str = gold[(b, e)]
            # Absolute EDU index → candidate-split index: gold_split ∈ [b+1, e)
            # maps to [0, e-b-2].
            gold_split_idx = gold_split - b - 1
            gold_label_idx = self.label_index.index(gold_label_str)

            # 2-EDU spans have a single forced split — no choice, no split loss.
            if e - b > 2:
                tgt = torch.tensor([gold_split_idx], device=self.device)
                split_losses.append(F.cross_entropy(split_logits.unsqueeze(0), tgt))

            tgt = torch.tensor([gold_label_idx], device=self.device)
            label_losses.append(F.cross_entropy(label_logits[gold_split_idx].unsqueeze(0), tgt))

            # Recurse into the GOLD sub-spans (teacher forcing — not the predicted split).
            stack.append((gold_split, e))
            stack.append((b, gold_split))

        split_loss = (
            sum(split_losses) / len(split_losses)
            if split_losses
            else torch.zeros((), device=self.device)
        )
        label_loss = sum(label_losses) / len(label_losses)
        return {"loss": (split_loss + label_loss) / 2}

    @torch.no_grad()
    def predict(self, tree: RstPpTree) -> RstPpTree:
        self.eval()
        num_edus = len(tree.edus)
        if num_edus < 2:
            return RstPpTree.from_parsing_actions([], tree.edus, relation_types=self._relation_types)

        embeddings, edu_boundaries = self._encode_tree(tree)

        actions = []
        queue = [(0, num_edus)]
        while queue:
            b, e = queue.pop(0)
            if e - b <= 1:
                continue

            packed_l, packed_r = self._packed_lr(embeddings, edu_boundaries, b, e)
            label_logits = self.label_biaffine(packed_l, packed_r)
            if e - b == 2:
                k = 0
            else:
                split_logits = self.split_biaffine(packed_l, packed_r).squeeze(-1)
                k = split_logits.argmax().item()
            split_point = b + 1 + k
            label = self.label_index[label_logits[k].argmax().item()]
            nuc, rel = label.split("_", 1)
            actions.append((split_point, nuc, rel))
            queue.append((b, split_point))
            queue.append((split_point, e))

        return RstPpTree.from_parsing_actions(actions, tree.edus, relation_types=self._relation_types)
