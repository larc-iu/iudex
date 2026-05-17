from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from iudex.rst.data.reader import determine_label_index
from iudex.rst.data.tree import RstTree
from iudex.rst.parsers.common.encoding import (
    encode_tokens_strided,
    load_encoder_and_tokenizer,
    tokenize_edus,
)
from iudex.rst.parsers.dmrst.configuration_dmrst import DMRSTConfig


class _GRUDecoder(nn.Module):
    """Unidirectional GRU decoder that maintains hidden state across decisions.

    Args:
        input_hidden_states: [1, length, input_size]
        last_hidden:         [num_layers, 1, hidden_size]

    Returns:
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

    Args:
        encoder_outputs: [length, hidden_size]  (EDU representations for the span)
        decoder_output:  [hidden_size]          (current decoder hidden state)

    Returns:
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
    """Deep biaffine classifier over joint nuclearity+relation labels.

    Args:
        input_left:  [1, hidden_size]
        input_right: [1, hidden_size]

    Returns:
        relation_weights:     [1, num_classes]  (softmax probabilities)
        log_relation_weights: [1, num_classes]  (log-softmax for NLL loss)
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
    """Per-token EDU-boundary classifier.

    Trains as a binary token-tagger that fires at EDU END positions; the class
    weight on the positive label is typically large (~10) since end-of-EDU
    tokens are rare. An optional second head can also be trained to fire at
    EDU START positions (paper §3.1.1 describes a 3-class B/I/E head; upstream
    code uses these two binary heads instead and that's what we mirror).
    """

    def __init__(self, hidden_size: int, pos_weight: float, dropout: float = 0.5, start_loss: bool = False):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, 2)
        self.start_linear = nn.Linear(hidden_size, 2) if start_loss else None
        self.register_buffer("class_weight", torch.tensor([1.0, pos_weight]))

    def loss(self, embeddings: torch.Tensor, edu_end_positions: list[int]) -> torch.Tensor:
        """Compute the segmentation loss against gold EDU ends.

        Args:
            embeddings:        [num_tokens, hidden_size]
            edu_end_positions: token indices of EDU ends (inclusive)

        Returns:
            scalar loss
        """
        num_tokens = embeddings.size(0)
        device = embeddings.device
        end_target = torch.zeros(num_tokens, dtype=torch.long, device=device)
        end_target[edu_end_positions] = 1
        logits = self.linear(self.dropout(embeddings))
        loss = F.cross_entropy(logits, end_target, weight=self.class_weight)

        if self.start_linear is not None:
            start_target = torch.zeros(num_tokens, dtype=torch.long, device=device)
            start_target[0] = 1
            # Every EDU end except the last document end is followed by an EDU start.
            for end in edu_end_positions[:-1]:
                start_target[end + 1] = 1
            start_logits = self.start_linear(self.dropout(embeddings))
            loss = loss + F.cross_entropy(start_logits, start_target, weight=self.class_weight)
        return loss

    @torch.no_grad()
    def predict_breaks(self, embeddings: torch.Tensor) -> list[int]:
        """Predict EDU end token indices from `embeddings`.

        Args:
            embeddings: [num_tokens, hidden_size]

        Returns:
            Sorted, deduped list of inclusive end indices. The last token is
            always forced to be a break so the final EDU is closed. We dedupe
            because argmax can fire on `last` independently of the force-append,
            and a duplicate would yield an empty (prev, end+1) interval downstream
            that NaNs out the per-EDU mean.
        """
        logits = self.linear(embeddings)
        preds = logits.argmax(-1).tolist()
        breaks = [i for i, p in enumerate(preds) if p == 1]
        last = embeddings.size(0) - 1
        if not breaks or breaks[-1] != last:
            breaks.append(last)
        return sorted(set(breaks))


class DMRSTParser(nn.Module):
    """DMRST parser (Liu, Shi & Chen, CODI 2021).

    Pipeline per document:
        tokens --(striding transformer)--> token embeddings
        per-EDU mean --(2-layer BiGRU)--> contextual EDU vectors
        edu_repr = reduce_dim([bigru_out, first_token, last_token])
        decode top-down with a unidirectional GRU decoder whose hidden state
        is carried across decisions (DFS, left-first); at each non-leaf span,
        a pointer attention picks the split position and a bilinear classifier
        picks the joint nuclearity+relation label.

    With `joint_segmentation=True`, a per-token EDU-boundary head trains
    alongside the parser and enables raw-text inference via `predict_from_text`.

    Training (`forward`) is teacher-forced and returns the split, label, and
    segmentation losses separately so the trainer can apply dynamic loss
    weighting; `loss` is their unweighted sum for non-DLW callers.
    """

    def __init__(self, config: DMRSTConfig):
        super().__init__()
        self.config = config
        self.label_index = determine_label_index(config.relation_types)
        self.stride = config.stride

        self.encoder, self.tokenizer, self.max_length = load_encoder_and_tokenizer(config.model_name)
        self.hidden_size = self.encoder.config.hidden_size

        # Freeze early encoder layers (BERT/RoBERTa/XLM-R submodule layout).
        if config.freeze_encoder_layers > 0:
            for p in self.encoder.embeddings.parameters():
                p.requires_grad = False
            for layer in self.encoder.encoder.layer[: config.freeze_encoder_layers]:
                for p in layer.parameters():
                    p.requires_grad = False

        hidden_size = self.hidden_size
        self.layer_norm = nn.LayerNorm(hidden_size, elementwise_affine=True)
        self.encoder_dropout = nn.Dropout(config.encoder_dropout)
        self.doc_gru = nn.GRU(
            hidden_size,
            hidden_size // 2,
            num_layers=2,
            batch_first=True,
            dropout=config.doc_gru_dropout,
            bidirectional=True,
        )
        self.reduce_dim = nn.Linear(hidden_size * 3, hidden_size, bias=False)

        self.decoder = _GRUDecoder(hidden_size, hidden_size, config.num_rnn_layers, config.decoder_dropout)
        self.pointer = _PointerAttention(config.attention_type, hidden_size)
        self.label_classifier = _LabelClassifier(
            hidden_size,
            hidden_size,
            len(self.label_index),
            bias=config.classifier_use_bias,
            dropout=config.labeler_dropout,
        )
        self.average_edu_level = config.average_edu_level

        self.segmenter = (
            _Segmenter(
                hidden_size,
                pos_weight=config.seg_pos_weight,
                dropout=config.encoder_dropout,
                start_loss=config.seg_start_loss,
            )
            if config.joint_segmentation
            else None
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def _encode(self, tree: RstTree) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full encoder pass using gold EDU segmentation.

        Computes the segmentation loss as a side-effect when joint segmentation
        is enabled and the model is in train mode; otherwise returns a zero scalar.

        Returns:
            edu_reprs:    [num_edus, hidden_size]
            decoder_init: [1, 1, hidden_size]
            seg_loss:     scalar tensor (zero when joint segmentation is disabled)
        """
        input_ids, edu_mapping = tokenize_edus(self.tokenizer, tree.edu_strings, self.device)
        embeddings = encode_tokens_strided(
            self.encoder, self.tokenizer, input_ids, self.max_length, self.stride
        )
        normed = self.layer_norm(embeddings.float())  # [num_tokens, hidden_size]

        if self.segmenter is not None and self.training:
            # End token (inclusive) of each gold EDU; segmenter operates on the
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
        edu_mapping: list[tuple[int, int]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build per-EDU representations from token embeddings.

        Shared back half of the encoder (after token encoding + LayerNorm
        + dropout): per-EDU token mean -> BiGRU -> concat(gru_out, first,
        last) -> reduce. Used by both the gold-EDU and predicted-EDU paths.

        Args:
            embeddings:  [num_tokens, hidden_size]
            edu_mapping: list of (start_token, end_token_exclusive) per EDU

        Returns:
            edu_reprs:    [num_edus, hidden_size]
            decoder_init: [1, 1, hidden_size]  (initial hidden state for the GRU decoder)
        """
        avg_edu_reprs = torch.stack([embeddings[b:e].mean(0) for b, e in edu_mapping])

        # Document-level BiGRU over averaged EDUs.
        # gru_out:    [1, num_edus, hidden_size]
        # gru_hidden: [num_layers * num_directions, 1, hidden_size // 2]
        gru_out, gru_hidden = self.doc_gru(avg_edu_reprs.unsqueeze(0))
        # Take last layer, concat both directions -> [1, 1, hidden_size].
        gru_hidden = gru_hidden.view(2, 2, 1, self.hidden_size // 2)[-1]
        decoder_init = gru_hidden.transpose(0, 1).reshape(1, 1, -1).contiguous()

        final_reprs = []
        for i, (b, e) in enumerate(edu_mapping):
            final_reprs.append(torch.cat([gru_out[0, i], embeddings[b], embeddings[e - 1]]))
        edu_reprs = self.reduce_dim(torch.stack(final_reprs))
        return edu_reprs, decoder_init

    def _label_inputs(
        self,
        edu_reprs: torch.Tensor,
        b: int,
        e: int,
        split_point: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Left/right child representations fed to the label classifier.

        For a 2-EDU span (or when `average_edu_level=False`), uses the edge EDU
        of each child; otherwise uses the mean of all EDUs in each child.

        Args:
            edu_reprs:   [num_edus, hidden_size]
            b, e:        EDU range of the span (exclusive at `e`)
            split_point: gold/predicted split position

        Returns:
            input_left:  [1, hidden_size]
            input_right: [1, hidden_size]
        """
        if e - b == 2 or not self.average_edu_level:
            return edu_reprs[split_point - 1].unsqueeze(0), edu_reprs[e - 1].unsqueeze(0)
        return edu_reprs[b:split_point].mean(0, keepdim=True), edu_reprs[split_point:e].mean(0, keepdim=True)

    def forward(self, tree: RstTree) -> dict[str, torch.Tensor]:
        """Teacher-forced loss for one gold tree.

        Walks the gold parse top-down (DFS, left-first) carrying decoder state
        across decisions. At each non-leaf span [b, e), the pointer scores the
        gold split position and the bilinear classifier scores the gold joint
        nuc+rel label; both losses are accumulated and averaged.

        Returns:
            {
                "loss":       split_loss + label_loss + seg_loss,
                "split_loss": scalar,
                "label_loss": scalar,
                "seg_loss":   scalar (zero when joint segmentation is disabled),
            }
        """
        num_edus = len(tree.edus)
        if num_edus < 2:
            zero = torch.zeros((), device=self.device, requires_grad=True)
            return {"loss": zero, "split_loss": zero, "label_loss": zero, "seg_loss": zero}

        edu_reprs, decoder_hidden, seg_loss = self._encode(tree)

        # Build a lookup: gold span (b, e) → (gold split point, gold label).
        gold_decisions: dict[tuple[int, int], tuple[int, str]] = {}
        for (left_range, right_range), nuc, rel in tree.spans_with_ranges():
            gold_decisions[(left_range[0], right_range[1])] = (right_range[0], f"{nuc}_{rel}")

        split_losses, label_losses = [], []
        stack = [(0, num_edus)]
        while stack:
            b, e = stack.pop()

            # Run decoder on the span's mean representation; this maintains
            # sequential decoder state across all decisions in the tree.
            decoder_input = edu_reprs[b:e].mean(0, keepdim=True).unsqueeze(0)
            decoder_output, decoder_hidden = self.decoder(decoder_input, last_hidden=decoder_hidden)

            gold_split, gold_label_str = gold_decisions[(b, e)]
            gold_label_idx = self.label_index.index(gold_label_str)

            if e - b == 2:
                # Single forced split — no pointer loss, label_inputs use the two EDUs directly.
                input_left = edu_reprs[b].unsqueeze(0)
                input_right = edu_reprs[b + 1].unsqueeze(0)
            else:
                # Pointer attends over candidate splits (EDU indices b..e-2 inclusive).
                split_logits = self.pointer(edu_reprs[b : e - 1], decoder_output.squeeze(0).squeeze(0))
                gold_pointer_idx = torch.tensor([gold_split - b - 1], device=self.device)
                split_losses.append(F.cross_entropy(split_logits, gold_pointer_idx))

                input_left, input_right = self._label_inputs(edu_reprs, b, e, gold_split)

                # Push right then left so left is on top — DFS left-first matches the
                # order in which a sequential decoder should see decisions.
                if e - gold_split > 1:
                    stack.append((gold_split, e))
                if gold_split - b > 1:
                    stack.append((b, gold_split))

            _, log_probs = self.label_classifier(input_left, input_right)
            label_target = torch.tensor([gold_label_idx], device=self.device)
            label_losses.append(F.nll_loss(log_probs, label_target))

        split_loss = (
            sum(split_losses) / len(split_losses) if split_losses else torch.zeros((), device=self.device)
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
    ) -> list[tuple[int, str, str]]:
        """Greedy top-down decode given precomputed EDU reprs and decoder init.

        Shared between the gold-EDU and predicted-EDU prediction paths.

        Args:
            edu_reprs:      [num_edus, hidden_size]
            decoder_hidden: [num_layers, 1, hidden_size]

        Returns:
            actions: list of (split_point, nuclearity, relation) tuples in DFS order.
        """
        num_edus = edu_reprs.size(0)
        actions: list[tuple[int, str, str]] = []
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
                split_logits = self.pointer(edu_reprs[b : e - 1], decoder_output.squeeze(0).squeeze(0))
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
    def predict(self, tree: RstTree) -> RstTree:
        """Greedy top-down decode using the gold EDU segmentation in `tree.edus`."""
        self.eval()
        num_edus = len(tree.edus)
        if num_edus < 2:
            return RstTree.from_parsing_actions([], tree.edus, relation_types=self.config.relation_types)

        edu_reprs, decoder_hidden, _ = self._encode(tree)
        actions = self._decode_actions(edu_reprs, decoder_hidden)
        return RstTree.from_parsing_actions(actions, tree.edus, relation_types=self.config.relation_types)

    @torch.no_grad()
    def predict_from_text(self, text: str) -> RstTree:
        """End-to-end inference from raw document text.

        Requires `joint_segmentation=True` in the config so the model has a
        trained segmenter to predict EDU boundaries.
        """
        if self.segmenter is None:
            raise RuntimeError("predict_from_text requires joint_segmentation=True")
        self.eval()

        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(ids) == 0:
            return RstTree.from_parsing_actions([], [], relation_types=self.config.relation_types)
        input_ids = torch.tensor(ids, dtype=torch.long, device=self.device)
        embeddings = encode_tokens_strided(
            self.encoder, self.tokenizer, input_ids, self.max_length, self.stride
        )
        normed = self.layer_norm(embeddings.float())

        breaks = self.segmenter.predict_breaks(normed)
        # `breaks` are inclusive end token indices. Convert to (start, end_exclusive).
        edu_mapping: list[tuple[int, int]] = []
        prev = 0
        for end_inclusive in breaks:
            edu_mapping.append((prev, end_inclusive + 1))
            prev = end_inclusive + 1
        edu_texts = [
            self.tokenizer.decode(
                ids[b:e], skip_special_tokens=True, clean_up_tokenization_spaces=True
            ).strip()
            for b, e in edu_mapping
        ]

        if len(edu_mapping) < 2:
            return RstTree.from_parsing_actions([], edu_texts, relation_types=self.config.relation_types)

        embeddings = self.encoder_dropout(normed)  # eval mode → dropout is identity
        edu_reprs, decoder_hidden = self._build_edu_reprs(embeddings, edu_mapping)
        actions = self._decode_actions(edu_reprs, decoder_hidden)
        return RstTree.from_parsing_actions(actions, edu_texts, relation_types=self.config.relation_types)

    @torch.no_grad()
    def predict_both(self, tree: RstTree) -> dict[str, Any]:
        """Single encoder pass yielding both the gold-EDU and end-to-end predictions.

        Used by dev evaluation when joint segmentation is enabled, to avoid two
        encoder passes per dev tree. When `self.segmenter is None`, only the
        gold-EDU keys are populated and the e2e keys are `None`.

        Returns:
            {
                "gold_pred":        RstTree (parse over gold EDUs),
                "gold_edu_mapping": list[(start_token, end_token_exclusive)],
                "gold_edu_ends":    list[int]  (inclusive end token indices),
                "e2e_pred":         RstTree or None,
                "pred_edu_mapping": list[(start, end)] or None,
                "pred_edu_ends":    list[int] or None,
            }
        """
        self.eval()
        input_ids, gold_edu_mapping = tokenize_edus(self.tokenizer, tree.edu_strings, self.device)
        token_embeddings = encode_tokens_strided(
            self.encoder, self.tokenizer, input_ids, self.max_length, self.stride
        )
        normed = self.layer_norm(token_embeddings.float())
        embeddings = self.encoder_dropout(normed)  # eval mode → identity

        # Gold-EDU path.
        gold_pred: RstTree
        if len(gold_edu_mapping) < 2:
            gold_pred = RstTree.from_parsing_actions(
                [], tree.edus, relation_types=self.config.relation_types
            )
        else:
            edu_reprs, decoder_hidden = self._build_edu_reprs(embeddings, gold_edu_mapping)
            actions = self._decode_actions(edu_reprs, decoder_hidden)
            gold_pred = RstTree.from_parsing_actions(
                actions, tree.edus, relation_types=self.config.relation_types
            )
        gold_edu_ends = [end - 1 for _, end in gold_edu_mapping]

        out: dict[str, Any] = {
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
        pred_edu_mapping: list[tuple[int, int]] = []
        prev = 0
        for end_inclusive in pred_ends:
            pred_edu_mapping.append((prev, end_inclusive + 1))
            prev = end_inclusive + 1
        ids_list = input_ids.tolist()
        pred_edu_texts = [
            self.tokenizer.decode(
                ids_list[b:e], skip_special_tokens=True, clean_up_tokenization_spaces=True
            ).strip()
            for b, e in pred_edu_mapping
        ]

        e2e_pred: RstTree
        if len(pred_edu_mapping) < 2:
            e2e_pred = RstTree.from_parsing_actions(
                [], pred_edu_texts, relation_types=self.config.relation_types
            )
        else:
            edu_reprs, decoder_hidden = self._build_edu_reprs(embeddings, pred_edu_mapping)
            actions = self._decode_actions(edu_reprs, decoder_hidden)
            e2e_pred = RstTree.from_parsing_actions(
                actions, pred_edu_texts, relation_types=self.config.relation_types
            )

        out["e2e_pred"] = e2e_pred
        out["pred_edu_mapping"] = pred_edu_mapping
        out["pred_edu_ends"] = pred_ends
        return out
