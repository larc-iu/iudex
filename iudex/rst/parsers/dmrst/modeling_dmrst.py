"""DMRST parser (Liu, Shi & Chen, ACL 2022) — gold-EDU variant.

Architecture:
  - Striding transformer encoder over per-EDU subtoken concatenation.
  - Per-EDU subtoken averaging, then a document-level BiGRU (2 layers).
  - EDU representation: reduce_dim([gru_out, first_subtoken, last_subtoken]).
  - GRU decoder maintains hidden state across split decisions (DFS, left-first).
  - PointerAttention over the candidate split positions in the current span.
  - Bilinear LabelClassifier predicts the joint nuclearity+relation label.

Whole-tree training: forward(tree) walks the gold parse top-down and sums
per-decision losses, returning split_loss / label_loss separately so the
trainer can apply dynamic loss weighting. Assumes gold EDU segmentation.
"""
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from iudex.rst.data.reader import determine_label_index
from iudex.rst.data.tree import RstPpTree
from iudex.rst.parsers.dmrst.configuration_dmrst import DMRSTConfig


class _GRUDecoder(nn.Module):
    """Unidirectional GRU decoder that maintains hidden state across decisions.

    Input:
        input: [1, length, input_size]
        last_hidden: [num_layers, 1, hidden_size]
    Output:
        output: [1, length, hidden_size]
        hidden: [num_layers, 1, hidden_size]
    """

    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=(0 if num_layers == 1 else dropout),
        )

    def forward(self, input_hidden_states, last_hidden):
        return self.gru(input_hidden_states, last_hidden)


class _PointerAttention(nn.Module):
    """Pointer attention for selecting a split position within a span.

    Input:
        encoder_outputs: [length, hidden_size]  (EDU representations for the span)
        decoder_output:  [hidden_size]          (current decoder hidden state)
    Output:
        logits: [1, length]
    """

    def __init__(self, attention_type, hidden_size):
        super().__init__()
        self.attention_type = attention_type
        self.weight1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.weight2 = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, encoder_outputs, decoder_output):
        if self.attention_type == "biaffine":
            ew1 = torch.matmul(self.weight1(encoder_outputs), decoder_output).unsqueeze(1)
            ew2 = self.weight2(encoder_outputs)
            return (ew1 + ew2).permute(1, 0)
        elif self.attention_type == "dot_product":
            return torch.matmul(encoder_outputs, decoder_output).unsqueeze(0)
        else:
            raise ValueError(f"Unknown attention type: {self.attention_type}")


class _LabelClassifier(nn.Module):
    """Bilinear classifier for predicting joint nuclearity+relation labels.

    Input:
        input_left:  [1, hidden_size]
        input_right: [1, hidden_size]
    Output:
        relation_weights:     [1, num_classes]  (softmax probabilities)
        log_relation_weights: [1, num_classes]  (log softmax for NLL loss)
    """

    def __init__(self, input_size, classifier_hidden_size, num_classes, bias=True, dropout=0.5):
        super().__init__()
        self.classifier_hidden_size = classifier_hidden_size
        self.labelspace_left = nn.Linear(input_size, classifier_hidden_size, bias=False)
        self.labelspace_right = nn.Linear(input_size, classifier_hidden_size, bias=False)
        self.weight_left = nn.Linear(classifier_hidden_size, num_classes, bias=False)
        self.weight_right = nn.Linear(classifier_hidden_size, num_classes, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.weight_bilateral = nn.Bilinear(
            classifier_hidden_size, classifier_hidden_size, num_classes, bias=bias
        )

    def forward(self, input_left, input_right):
        labelspace_left = F.elu(self.labelspace_left(input_left))
        labelspace_right = F.elu(self.labelspace_right(input_right))

        union = self.dropout(torch.cat((labelspace_left, labelspace_right), 1))
        labelspace_left = union[:, : self.classifier_hidden_size]
        labelspace_right = union[:, self.classifier_hidden_size :]

        output = (
            self.weight_bilateral(labelspace_left, labelspace_right)
            + self.weight_left(labelspace_left)
            + self.weight_right(labelspace_right)
        )
        relation_weights = F.softmax(output, 1)
        log_relation_weights = F.log_softmax(output + 1e-6, 1)
        return relation_weights, log_relation_weights


class _Segmenter(nn.Module):
    """Per-subtoken EDU-boundary classifier. Trains as a binary token-tagger
    that fires at EDU END positions; class weight on the positive label is
    typically large (~10) since end-of-EDU tokens are rare. An optional second
    head can also be trained to fire at EDU START positions (paper §3.1.1
    describes a 3-class B/I/E head; upstream code uses these two binary heads
    instead and that's what we mirror).
    """

    def __init__(self, hidden_size: int, pos_weight: float, dropout: float = 0.5, start_loss: bool = False):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, 2)
        self.start_linear = nn.Linear(hidden_size, 2) if start_loss else None
        self.register_buffer("class_weight", torch.tensor([1.0, pos_weight]))

    def loss(self, embeddings: torch.Tensor, edu_end_positions: List[int]) -> torch.Tensor:
        """`embeddings`: [num_subtokens, H]. `edu_end_positions`: subtoken indices
        of EDU ends (inclusive). Returns scalar loss."""
        n = embeddings.size(0)
        device = embeddings.device
        end_target = torch.zeros(n, dtype=torch.long, device=device)
        end_target[edu_end_positions] = 1
        logits = self.linear(self.dropout(embeddings))
        loss = F.cross_entropy(logits, end_target, weight=self.class_weight)

        if self.start_linear is not None:
            start_target = torch.zeros(n, dtype=torch.long, device=device)
            start_target[0] = 1
            # Every EDU end except the last document end is followed by an EDU start.
            for end in edu_end_positions[:-1]:
                start_target[end + 1] = 1
            start_logits = self.start_linear(self.dropout(embeddings))
            loss = loss + F.cross_entropy(start_logits, start_target, weight=self.class_weight)
        return loss

    @torch.no_grad()
    def predict_breaks(self, embeddings: torch.Tensor) -> List[int]:
        """Return predicted EDU end subtoken indices. Always forces the last
        subtoken to be a break so the final EDU is closed."""
        logits = self.linear(embeddings)
        preds = logits.argmax(-1).tolist()
        breaks = [i for i, p in enumerate(preds) if p == 1]
        last = embeddings.size(0) - 1
        if not breaks or breaks[-1] != last:
            breaks.append(last)
        # Dedupe + sort: argmax could fire on `last` independently of the force-append,
        # producing a duplicate. A duplicate would yield an empty (prev, end+1) interval
        # downstream and NaN out the per-EDU mean.
        return sorted(set(breaks))


class DMRSTParser(nn.Module):
    def __init__(self, config: DMRSTConfig):
        super().__init__()
        self.config = config
        self._relation_types = tuple(config.relation_types)
        self.label_index = determine_label_index(self._relation_types)
        self.stride = config.stride

        encoder_kwargs = {}
        if config.attn_implementation is not None:
            encoder_kwargs["attn_implementation"] = config.attn_implementation
        # transformers >=5 honors the checkpoint's saved dtype; many HF checkpoints
        # are fp16, which makes AdamW updates NaN immediately. Force fp32.
        self.encoder = AutoModel.from_pretrained(config.model_name, **encoder_kwargs).float()
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.hidden_size = self.encoder.config.hidden_size

        # Freeze early encoder layers (BERT/RoBERTa/XLM-R submodule layout).
        if config.freeze_encoder_layers > 0:
            for p in self.encoder.embeddings.parameters():
                p.requires_grad = False
            for layer in self.encoder.encoder.layer[:config.freeze_encoder_layers]:
                for p in layer.parameters():
                    p.requires_grad = False
        # HF tokenizers can report a sentinel `model_max_length` of ~1e30 when
        # unset; fall back to the encoder's actual positional-embedding budget.
        self.max_length = min(
            getattr(self.encoder.config, "max_position_embeddings", self.tokenizer.model_max_length),
            self.tokenizer.model_max_length,
        )

        H = self.hidden_size
        self.layer_norm = nn.LayerNorm(H, elementwise_affine=True)
        self.encoder_dropout = nn.Dropout(config.encoder_dropout)
        self.doc_gru = nn.GRU(
            H,
            H // 2,
            num_layers=2,
            batch_first=True,
            dropout=config.doc_gru_dropout,
            bidirectional=True,
        )
        self.reduce_dim = nn.Linear(H * 3, H, bias=False)

        self.decoder = _GRUDecoder(H, H, config.num_rnn_layers, config.decoder_dropout)
        self.pointer = _PointerAttention(config.attention_type, H)
        self.label_classifier = _LabelClassifier(
            H, H, len(self.label_index),
            bias=config.classifier_use_bias, dropout=config.labeler_dropout,
        )
        self.average_edu_level = config.average_edu_level

        self.segmenter = _Segmenter(
            H,
            pos_weight=config.seg_pos_weight,
            dropout=config.encoder_dropout,
            start_loss=config.seg_start_loss,
        ) if config.joint_segmentation else None

    @property
    def relation_types(self):
        return self._relation_types

    @property
    def device(self):
        return next(self.parameters()).device

    def _tokenize_tree(self, tree: RstPpTree) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
        """Tokenize EDU-by-EDU. Returns (input_ids [N], edu_mapping list of (start, end))."""
        all_ids: List[int] = []
        boundaries: List[Tuple[int, int]] = []
        for edu_text in tree.edu_strings:
            ids = self.tokenizer.encode(edu_text, add_special_tokens=False)
            start = len(all_ids)
            all_ids.extend(ids)
            boundaries.append((start, len(all_ids)))
        input_ids = torch.tensor(all_ids, dtype=torch.long, device=self.device)
        return input_ids, boundaries

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

    def _encode(self, tree: RstPpTree) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode tree EDUs using gold EDU segmentation.
        Returns (edu_reprs [num_edus, H], decoder_init [1, 1, H], seg_loss scalar).
        `seg_loss` is a zero tensor when joint segmentation is disabled.
        """
        input_ids, edu_mapping = self._tokenize_tree(tree)
        normed = self.layer_norm(self._encode_subtokens(input_ids).float())  # [N, H]

        if self.segmenter is not None and self.training:
            # End subtoken (inclusive) of each gold EDU; segmenter operates on the
            # layer-normed embeddings BEFORE dropout (matches upstream).
            edu_ends = [end - 1 for _, end in edu_mapping]
            seg_loss = self.segmenter.loss(normed, edu_ends)
        else:
            seg_loss = torch.zeros((), device=normed.device)

        embeddings = self.encoder_dropout(normed)
        edu_reprs, decoder_init = self._build_edu_reprs(embeddings, edu_mapping)
        return edu_reprs, decoder_init, seg_loss

    def _build_edu_reprs(
        self,
        embeddings: torch.Tensor,
        edu_mapping: List[Tuple[int, int]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Shared back half of the encoder (after subtoken encoding + LayerNorm +
        dropout): per-EDU subtoken mean → BiGRU → concat(gru_out, first, last) →
        reduce. Used by both gold-EDU and predicted-EDU paths.
        Returns (edu_reprs [num_edus, H], decoder_init [1, 1, H]).
        """
        avg_edu_reprs = torch.stack([embeddings[b:e].mean(0) for b, e in edu_mapping])

        # Document-level BiGRU over averaged EDUs.
        # gru_out:    [1, num_edus, H]; gru_hidden: [4, 1, H/2] = [layers*dirs, 1, H/2]
        gru_out, gru_hidden = self.doc_gru(avg_edu_reprs.unsqueeze(0))
        # Take last layer, concat both directions -> [1, 1, H]
        gru_hidden = gru_hidden.view(2, 2, 1, self.hidden_size // 2)[-1]
        decoder_init = gru_hidden.transpose(0, 1).reshape(1, 1, -1).contiguous()

        final_reprs = []
        for i, (b, e) in enumerate(edu_mapping):
            final_reprs.append(torch.cat([gru_out[0, i], embeddings[b], embeddings[e - 1]]))
        edu_reprs = self.reduce_dim(torch.stack(final_reprs))
        return edu_reprs, decoder_init

    def _label_inputs(self, edu_reprs, b, e, split_point):
        """Left/right child representations fed to the label classifier."""
        if e - b == 2 or not self.average_edu_level:
            return edu_reprs[split_point - 1].unsqueeze(0), edu_reprs[e - 1].unsqueeze(0)
        return edu_reprs[b:split_point].mean(0, keepdim=True), edu_reprs[split_point:e].mean(0, keepdim=True)

    def forward(self, tree: RstPpTree) -> Dict[str, torch.Tensor]:
        """Teacher-forced DFS over the gold tree. Returns split/label losses
        separately so the trainer can apply dynamic loss weighting; `loss` is
        the unweighted sum, suitable for non-DLW callers.
        """
        num_edus = len(tree.edus)
        if num_edus < 2:
            zero = torch.zeros((), device=self.device, requires_grad=True)
            return {"loss": zero, "split_loss": zero, "label_loss": zero, "seg_loss": zero}

        edu_reprs, decoder_hidden, seg_loss = self._encode(tree)

        # Index every gold non-leaf span by its EDU range.
        gold: Dict[Tuple[int, int], Tuple[int, str]] = {}
        for (left_range, right_range), nuc, rel in tree.spans_with_ranges():
            gold[(left_range[0], right_range[1])] = (right_range[0], f"{nuc}_{rel}")

        split_losses, label_losses = [], []
        stack = [(0, num_edus)]
        while stack:
            b, e = stack.pop()

            # Run decoder on the span's mean representation; this maintains
            # sequential decoder state across all decisions in the tree.
            decoder_input = edu_reprs[b:e].mean(0, keepdim=True).unsqueeze(0)
            decoder_output, decoder_hidden = self.decoder(decoder_input, last_hidden=decoder_hidden)

            gold_split, gold_label_str = gold[(b, e)]
            gold_label_idx = self.label_index.index(gold_label_str)

            if e - b == 2:
                # Single forced split — no pointer loss, label_inputs use the two EDUs directly.
                input_left = edu_reprs[b].unsqueeze(0)
                input_right = edu_reprs[b + 1].unsqueeze(0)
            else:
                # Pointer attends over candidate splits (EDU indices b..e-2 inclusive).
                split_logits = self.pointer(edu_reprs[b:e - 1], decoder_output.squeeze(0).squeeze(0))
                gold_ptr_idx = torch.tensor([gold_split - b - 1], device=self.device)
                split_losses.append(F.cross_entropy(split_logits, gold_ptr_idx))

                input_left, input_right = self._label_inputs(edu_reprs, b, e, gold_split)

                # Push right then left so left is on top — DFS left-first matches the
                # order in which a sequential decoder should see decisions.
                if e - gold_split > 1:
                    stack.append((gold_split, e))
                if gold_split - b > 1:
                    stack.append((b, gold_split))

            _, log_probs = self.label_classifier(input_left, input_right)
            tgt = torch.tensor([gold_label_idx], device=self.device)
            label_losses.append(F.nll_loss(log_probs, tgt))

        split_loss = (
            sum(split_losses) / len(split_losses)
            if split_losses
            else torch.zeros((), device=self.device)
        )
        label_loss = sum(label_losses) / len(label_losses)
        return {
            "loss": split_loss + label_loss + seg_loss,
            "split_loss": split_loss,
            "label_loss": label_loss,
            "seg_loss": seg_loss,
        }

    def _decode_actions(
        self,
        edu_reprs: torch.Tensor,
        decoder_hidden: torch.Tensor,
    ) -> List[Tuple[int, str, str]]:
        """Greedy top-down decode given precomputed edu_reprs and decoder init.
        Shared between gold-EDU and predicted-EDU prediction paths.
        """
        num_edus = edu_reprs.size(0)
        actions: List[Tuple[int, str, str]] = []
        stack = [(0, num_edus)]
        while stack:
            b, e = stack.pop()

            decoder_input = edu_reprs[b:e].mean(0, keepdim=True).unsqueeze(0)
            decoder_output, decoder_hidden = self.decoder(decoder_input, last_hidden=decoder_hidden)

            if e - b == 2:
                split_point = b + 1
                input_left = edu_reprs[b].unsqueeze(0)
                input_right = edu_reprs[b + 1].unsqueeze(0)
            else:
                split_logits = self.pointer(edu_reprs[b:e - 1], decoder_output.squeeze(0).squeeze(0))
                split_point = b + split_logits.argmax(-1).item() + 1
                input_left, input_right = self._label_inputs(edu_reprs, b, e, split_point)

            probs, _ = self.label_classifier(input_left, input_right)
            pred_label = self.label_index[probs.argmax(-1).item()]
            pred_nuc, pred_rel = pred_label.split("_", 1)
            actions.append((split_point, pred_nuc, pred_rel))

            if e - split_point > 1:
                stack.append((split_point, e))
            if split_point - b > 1:
                stack.append((b, split_point))
        return actions

    @torch.no_grad()
    def predict(self, tree: RstPpTree) -> RstPpTree:
        self.eval()
        num_edus = len(tree.edus)
        if num_edus < 2:
            return RstPpTree.from_parsing_actions([], tree.edus, relation_types=self._relation_types)

        edu_reprs, decoder_hidden, _ = self._encode(tree)
        actions = self._decode_actions(edu_reprs, decoder_hidden)
        return RstPpTree.from_parsing_actions(actions, tree.edus, relation_types=self._relation_types)

    @torch.no_grad()
    def predict_from_text(self, text: str) -> RstPpTree:
        """End-to-end inference from raw document text. Requires
        `joint_segmentation=True` in the config (the model needs a trained segmenter).
        """
        if self.segmenter is None:
            raise RuntimeError("predict_from_text requires joint_segmentation=True")
        self.eval()

        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(ids) == 0:
            return RstPpTree.from_parsing_actions([], [], relation_types=self._relation_types)
        input_ids = torch.tensor(ids, dtype=torch.long, device=self.device)
        normed = self.layer_norm(self._encode_subtokens(input_ids).float())

        breaks = self.segmenter.predict_breaks(normed)
        # `breaks` are inclusive end subtoken indices. Convert to (start, end_exclusive).
        edu_mapping: List[Tuple[int, int]] = []
        prev = 0
        for end_inclusive in breaks:
            edu_mapping.append((prev, end_inclusive + 1))
            prev = end_inclusive + 1
        edu_texts = [
            self.tokenizer.decode(ids[b:e], skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()
            for b, e in edu_mapping
        ]

        if len(edu_mapping) < 2:
            return RstPpTree.from_parsing_actions([], edu_texts, relation_types=self._relation_types)

        embeddings = self.encoder_dropout(normed)  # eval mode → dropout is identity
        edu_reprs, decoder_hidden = self._build_edu_reprs(embeddings, edu_mapping)
        actions = self._decode_actions(edu_reprs, decoder_hidden)
        return RstPpTree.from_parsing_actions(actions, edu_texts, relation_types=self._relation_types)

    @torch.no_grad()
    def predict_both(self, tree: RstPpTree) -> Dict[str, Any]:
        """Single encoder pass; returns both the gold-EDU prediction and the
        end-to-end (predicted-EDU) prediction. Used by dev evaluation when joint
        segmentation is enabled, to avoid two encoder passes per dev tree.

        When `self.segmenter is None`, only the gold-EDU keys are populated;
        the e2e keys are `None`.
        """
        self.eval()
        input_ids, gold_edu_mapping = self._tokenize_tree(tree)
        normed = self.layer_norm(self._encode_subtokens(input_ids).float())
        embeddings = self.encoder_dropout(normed)  # eval mode → identity

        # Gold-EDU path.
        gold_pred: RstPpTree
        if len(gold_edu_mapping) < 2:
            gold_pred = RstPpTree.from_parsing_actions([], tree.edus, relation_types=self._relation_types)
        else:
            edu_reprs, decoder_hidden = self._build_edu_reprs(embeddings, gold_edu_mapping)
            actions = self._decode_actions(edu_reprs, decoder_hidden)
            gold_pred = RstPpTree.from_parsing_actions(actions, tree.edus, relation_types=self._relation_types)
        gold_edu_ends = [end - 1 for _, end in gold_edu_mapping]

        out: Dict[str, Any] = {
            "gold_pred": gold_pred,
            "gold_edu_mapping": gold_edu_mapping,
            "gold_edu_ends": gold_edu_ends,
            "e2e_pred": None,
            "pred_edu_mapping": None,
            "pred_edu_ends": None,
        }

        if self.segmenter is None:
            return out

        # End-to-end path (predicted segmentation).
        pred_ends = self.segmenter.predict_breaks(normed)
        pred_edu_mapping: List[Tuple[int, int]] = []
        prev = 0
        for end_inclusive in pred_ends:
            pred_edu_mapping.append((prev, end_inclusive + 1))
            prev = end_inclusive + 1
        ids_list = input_ids.tolist()
        pred_edu_texts = [
            self.tokenizer.decode(ids_list[b:e], skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()
            for b, e in pred_edu_mapping
        ]

        e2e_pred: RstPpTree
        if len(pred_edu_mapping) < 2:
            e2e_pred = RstPpTree.from_parsing_actions([], pred_edu_texts, relation_types=self._relation_types)
        else:
            edu_reprs, decoder_hidden = self._build_edu_reprs(embeddings, pred_edu_mapping)
            actions = self._decode_actions(edu_reprs, decoder_hidden)
            e2e_pred = RstPpTree.from_parsing_actions(actions, pred_edu_texts, relation_types=self._relation_types)

        out["e2e_pred"] = e2e_pred
        out["pred_edu_mapping"] = pred_edu_mapping
        out["pred_edu_ends"] = pred_ends
        return out
