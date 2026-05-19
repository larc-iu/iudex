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


class _PointerAttention(nn.Module):
    """Pointer attention for selecting a split position within a span.

    A span of n EDUs can be split at n - 1 different points (at least one
    EDU on either side). Position k scores the split just after EDU b + k.

    With e_k = encoder_outputs[k] and d = decoder_output:
        biaffine: logit_k = (W1 e_k)·d + w2·e_k
        dot_product: logit_k = e_k · d

    Args:
        encoder_outputs: [n - 1, hidden_size]
        decoder_output: [hidden_size]

    Returns:
        logits: [1, n - 1]  (leading dim is F.cross_entropy's batch convention)
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
    """Deep biaffine classifier (Dozat & Manning) over joint nuclearity+relation
    labels, e.g. "SN_elaboration".

    Each side is projected (Linear → ELU → dropout) into a shared hidden space.
    The score for class c is then

        score_c = h_L^T B_c h_R  +  u_L^c · h_L  +  u_R^c · h_R  +  b_c

    where B is `bilinear` (the pairwise interaction term) and u_L, u_R are the
    per-side unary terms `linear_left`, `linear_right`.

    Args:
        input_left: [1, input_size]
        input_right: [1, input_size]

    Returns:
        logits: [1, num_classes]  (raw scores, caller applies softmax / cross_entropy)
    """

    def __init__(self, input_size, hidden_size, num_classes, bias=True, dropout=0.5):
        super().__init__()
        self.proj_left = nn.Linear(input_size, hidden_size, bias=False)
        self.proj_right = nn.Linear(input_size, hidden_size, bias=False)
        self.bilinear = nn.Bilinear(hidden_size, hidden_size, num_classes, bias=bias)
        self.linear_left = nn.Linear(hidden_size, num_classes, bias=False)
        self.linear_right = nn.Linear(hidden_size, num_classes, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_left, input_right):
        hidden_left = self.dropout(F.elu(self.proj_left(input_left)))
        hidden_right = self.dropout(F.elu(self.proj_right(input_right)))
        return (
            self.bilinear(hidden_left, hidden_right) + self.linear_left(hidden_left) + self.linear_right(hidden_right)
        )


class _Segmenter(nn.Module):
    """Per-token EDU-boundary classifier: binary tagger over EDU end
    positions, with an optional second head for EDU starts."""

    def __init__(self, hidden_size: int, pos_weight: float, dropout: float = 0.5, start_loss: bool = False):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, 2)
        self.start_linear = nn.Linear(hidden_size, 2) if start_loss else None
        self.register_buffer("class_weight", torch.tensor([1.0, pos_weight]))

    def loss(self, embeddings: torch.Tensor, edu_end_positions: list[int]) -> torch.Tensor:
        """Args:
            embeddings: [num_tokens, hidden_size]
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
        """Args:
            embeddings: [num_tokens, hidden_size]

        Returns:
            Sorted, deduped list of inclusive end indices. The last token is
            always forced to be a break so the final EDU is closed. Dedupe is
            needed because argmax can fire on `last` independently of the
            force-append, and a duplicate would yield an empty (prev, end+1)
            interval downstream that NaNs out the per-EDU mean.
        """
        logits = self.linear(embeddings)
        preds = logits.argmax(-1).tolist()
        breaks = [i for i, p in enumerate(preds) if p == 1]
        last = embeddings.size(0) - 1
        if not breaks or breaks[-1] != last:
            breaks.append(last)
        return sorted(set(breaks))


class DMRSTParser(nn.Module):
    """An end-to-end RST parser including its own segmenter.

    When `cfg.segmentation` is non-null, a per-token EDU-boundary head trains
    alongside the parser and `predict_from_text` becomes available.

    Training (`forward`) is teacher-forced and returns the split, label, and
    segmentation losses separately so the trainer can apply dynamic loss
    weighting. `loss` is their unweighted sum.
    """

    def __init__(self, config: DMRSTConfig):
        super().__init__()
        self.config = config
        self.label_index = determine_label_index(config.relation_types)
        self.stride = config.stride

        self.encoder, self.tokenizer, self.max_length = load_encoder_and_tokenizer(config.model_name)

        if config.freeze_embeddings:
            for p in self.encoder.embeddings.parameters():
                p.requires_grad = False
        if config.freeze_encoder_layers > 0:
            for layer in self.encoder.encoder.layer[: config.freeze_encoder_layers]:
                for p in layer.parameters():
                    p.requires_grad = False

        # Compile the encoder forward (not the module) so state_dict keys are
        # unchanged and existing checkpoints still load. dynamic=True avoids
        # per-shape recompiles on variable-length documents.
        if torch.cuda.is_available():
            self.encoder.forward = torch.compile(self.encoder.forward, dynamic=True)

        hidden_size = self.encoder.config.hidden_size
        self.layer_norm = nn.LayerNorm(hidden_size)
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

        self.decoder = nn.GRU(
            hidden_size,
            hidden_size,
            num_layers=config.num_rnn_layers,
            batch_first=True,
            dropout=(0 if config.num_rnn_layers == 1 else config.decoder_dropout),
        )
        self.pointer = _PointerAttention(config.attention_type, hidden_size)
        self.label_classifier = _LabelClassifier(
            hidden_size,
            hidden_size,
            len(self.label_index),
            bias=config.classifier_use_bias,
            dropout=config.labeler_dropout,
        )
        if config.label_input_pooling not in ("mean", "last_edu"):
            raise ValueError(f"Unknown label_input_pooling: {config.label_input_pooling!r}")
        self.label_input_pooling = config.label_input_pooling

        self.segmenter = (
            _Segmenter(
                hidden_size,
                pos_weight=config.segmentation.pos_weight,
                dropout=config.encoder_dropout,
                start_loss=config.segmentation.start_loss,
            )
            if config.segmentation is not None
            else None
        )

    @property
    def device(self):
        return next(self.parameters()).device

    @classmethod
    def from_pretrained(
        cls,
        repo_or_path: str,
        *,
        device: str | torch.device | None = None,
        revision: str | None = None,
        cache_dir: str | None = None,
        token: str | bool | None = None,
    ) -> "DMRSTParser":
        """Load from a HuggingFace Hub repo id, a local run dir, or a `.pt` file.

        See `iudex.rst.parsers.hfhub.load_parser_from_pretrained` for the
        full resolution rules (including how Hub vs. local paths are detected).
        """
        from iudex.rst.parsers.hfhub import load_parser_from_pretrained

        dev = (
            torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        return load_parser_from_pretrained(
            repo_or_path,
            parser_cls=cls,
            config_cls=DMRSTConfig,
            device=dev,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
        )

    def _encode(self, tree: RstTree) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full encoder pass using gold EDU segmentation.

        Also computes the segmentation loss when joint segmentation is enabled and
        the model is in train mode. Otherwise, returns a zero scalar.

        Returns:
            edu_reprs: [num_edus, hidden_size]
            decoder_init: [1, 1, hidden_size]
            seg_loss: scalar tensor (zero when joint segmentation is disabled)
        """
        input_ids, edu_mapping = tokenize_edus(self.tokenizer, tree.edu_strings, self.device)
        embeddings = encode_tokens_strided(self.encoder, self.tokenizer, input_ids, self.max_length, self.stride)
        normed = self.layer_norm(embeddings.float())  # [num_tokens, hidden_size]

        if self.segmenter is not None and self.training:
            # End token (inclusive) of each gold EDU. Segmenter operates on the
            # layer-normed embeddings BEFORE dropout.
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
        """Pool token embeddings into one vector per EDU, plus the decoder's initial hidden state.

        Each EDU's tokens are mean-pooled, run through a document-level BiGRU, then
        each EDU's final repr is `reduce_dim([bigru_out, first_token, last_token])`.
        The decoder's initial hidden state is the BiGRU's final (last-layer, both
        directions concatenated) hidden state.

        Args:
            embeddings: [num_tokens, hidden_size]
            edu_mapping: list of (start_token, end_token_exclusive) per EDU

        Returns:
            edu_reprs: [num_edus, hidden_size]
            decoder_init: [1, 1, hidden_size]
        """
        # [1, num_edus, hidden_size]. Unsqueeze gives the batch dim the GRU expects.
        avg_edu_reprs = torch.stack([embeddings[b:e].mean(0) for b, e in edu_mapping]).unsqueeze(0)

        # gru_out: [1, num_edus, hidden_size]
        # gru_hidden: [num_layers * num_directions, 1, hidden_size // 2]   (= [4, 1, H/2])
        gru_out, gru_hidden = self.doc_gru(avg_edu_reprs)

        # Decoder's initial state = BiGRU's top-layer final hidden state, with
        # the two directions concatenated:
        H = self.encoder.config.hidden_size
        # 1. Split the packed first dim into (layer, direction): [2, 2, 1, H/2].
        #    nn.GRU's convention is layers-outer, directions-inner.
        gru_hidden = gru_hidden.view(2, 2, 1, H // 2)
        # 2. Keep only the top layer -> [num_directions=2, batch=1, H/2].
        gru_hidden = gru_hidden[-1]
        # 3. Batch first, then flatten the two directions into one hidden dim
        #    of size 2 * (H/2) = H. Result: [num_decoder_layers=1, batch=1, H].
        decoder_init = gru_hidden.transpose(0, 1).reshape(1, 1, H).contiguous()

        final_reprs = []
        for i, (b, e) in enumerate(edu_mapping):
            final_reprs.append(torch.cat([gru_out[0, i], embeddings[b], embeddings[e - 1]]))
        edu_reprs = self.reduce_dim(torch.stack(final_reprs))
        return edu_reprs, decoder_init

    def _build_label_inputs(
        self,
        edu_reprs: torch.Tensor,
        b: int,
        e: int,
        split_point: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Left/right child representations fed to the label classifier.

        Pooled as specified by `self.label_input_pooling`. A 2-EDU span short-circuits
        to the edge form since both pooling modes collapse to picking the lone EDU.

        Args:
            edu_reprs: [num_edus, hidden_size]
            b, e: EDU range of the span (exclusive at `e`)
            split_point: gold/predicted split position

        Returns:
            input_left: [1, hidden_size]
            input_right: [1, hidden_size]
        """
        if e - b == 2 or self.label_input_pooling == "last_edu":
            last_edu_left = edu_reprs[split_point - 1].unsqueeze(0)
            last_edu_right = edu_reprs[e - 1].unsqueeze(0)
            return last_edu_left, last_edu_right
        else:
            mean_of_left = edu_reprs[b:split_point].mean(0, keepdim=True)
            mean_of_right = edu_reprs[split_point:e].mean(0, keepdim=True)
            return mean_of_left, mean_of_right

    def forward(self, tree: RstTree) -> dict[str, torch.Tensor]:
        """Teacher-forced loss for one gold tree.

        Top-down parser with a pointer-network decoder (Vinyals et al. 2015):
        the decoder GRU maintains hidden state across all parsing decisions,
        and at each non-leaf span, pointer attention picks the split position
        by attending over the span's EDU representations.

        Per gold span (b, e):
          1. Step the decoder on the span's mean EDU repr (output is the query).
          2. Pointer attention scores the gold split (cross-entropy loss).
          3. Bilinear classifier scores the gold (nuc, rel) label (CE loss).

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

            # One decoder step per span. The input (mean of the span's EDUs)
            # grounds the GRU in "what I'm deciding about now". The carried
            # hidden state grounds it in "what I've decided so far". The
            # output becomes the query for pointer attention below.
            decoder_input = edu_reprs[b:e].mean(0, keepdim=True).unsqueeze(0)
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)

            gold_split, gold_label_str = gold_decisions[(b, e)]
            gold_label_idx = self.label_index.index(gold_label_str)

            if e - b == 2:
                # Only one possible split. Skip pointer loss, use the two EDUs directly.
                input_left = edu_reprs[b].unsqueeze(0)
                input_right = edu_reprs[b + 1].unsqueeze(0)
            else:
                # Pointer attention: query is the decoder output, keys are the
                # n-1 candidate split anchors edu_reprs[b:e-1].
                split_logits = self.pointer(edu_reprs[b : e - 1], decoder_output.squeeze(0).squeeze(0))
                gold_pointer_idx = torch.tensor([gold_split - b - 1], device=self.device)
                split_losses.append(F.cross_entropy(split_logits, gold_pointer_idx))

                input_left, input_right = self._build_label_inputs(edu_reprs, b, e, gold_split)

                # Push right then left so left pops first. DFS left-first matches
                # the order in which the sequential decoder sees decisions.
                if e - gold_split > 1:
                    stack.append((gold_split, e))
                if gold_split - b > 1:
                    stack.append((b, gold_split))

            logits = self.label_classifier(input_left, input_right)
            label_target = torch.tensor([gold_label_idx], device=self.device)
            label_losses.append(F.cross_entropy(logits, label_target))

        split_loss = sum(split_losses) / len(split_losses) if split_losses else torch.zeros((), device=self.device)
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

        Args:
            edu_reprs: [num_edus, hidden_size]
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
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)

            if e - b == 2:
                split_point = b + 1
                input_left = edu_reprs[b].unsqueeze(0)
                input_right = edu_reprs[b + 1].unsqueeze(0)
            else:
                split_logits = self.pointer(edu_reprs[b : e - 1], decoder_output.squeeze(0).squeeze(0))
                split_point = b + split_logits.argmax(-1).item() + 1
                input_left, input_right = self._build_label_inputs(edu_reprs, b, e, split_point)

            logits = self.label_classifier(input_left, input_right)
            pred_label = self.label_index[logits.argmax(-1).item()]
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

        Requires `cfg.segmentation` to be non-null so the model has a trained
        segmenter to predict EDU boundaries.
        """
        if self.segmenter is None:
            raise RuntimeError("predict_from_text requires `cfg.segmentation` to be non-null")
        self.eval()

        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(ids) == 0:
            # A 0-EDU tree is unconstructible (`RstTree.__init__` requires
            # exactly one root). Single-EDU is handled below.
            raise ValueError("predict_from_text: input text tokenized to zero tokens")
        input_ids = torch.tensor(ids, dtype=torch.long, device=self.device)
        embeddings = encode_tokens_strided(self.encoder, self.tokenizer, input_ids, self.max_length, self.stride)
        normed = self.layer_norm(embeddings.float())

        breaks = self.segmenter.predict_breaks(normed)
        # `breaks` are inclusive end token indices. Convert to (start, end_exclusive).
        edu_mapping: list[tuple[int, int]] = []
        prev = 0
        for end_inclusive in breaks:
            edu_mapping.append((prev, end_inclusive + 1))
            prev = end_inclusive + 1
        edu_texts = [
            self.tokenizer.decode(ids[b:e], skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()
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

        When `self.segmenter is None`, only the gold-EDU keys are populated and
        the e2e keys are `None`.

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
        token_embeddings = encode_tokens_strided(self.encoder, self.tokenizer, input_ids, self.max_length, self.stride)
        normed = self.layer_norm(token_embeddings.float())
        embeddings = self.encoder_dropout(normed)  # eval mode → identity

        # Gold-EDU path.
        gold_pred: RstTree
        if len(gold_edu_mapping) < 2:
            gold_pred = RstTree.from_parsing_actions([], tree.edus, relation_types=self.config.relation_types)
        else:
            edu_reprs, decoder_hidden = self._build_edu_reprs(embeddings, gold_edu_mapping)
            actions = self._decode_actions(edu_reprs, decoder_hidden)
            gold_pred = RstTree.from_parsing_actions(actions, tree.edus, relation_types=self.config.relation_types)
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
            self.tokenizer.decode(ids_list[b:e], skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()
            for b, e in pred_edu_mapping
        ]

        e2e_pred: RstTree
        if len(pred_edu_mapping) < 2:
            e2e_pred = RstTree.from_parsing_actions([], pred_edu_texts, relation_types=self.config.relation_types)
        else:
            edu_reprs, decoder_hidden = self._build_edu_reprs(embeddings, pred_edu_mapping)
            actions = self._decode_actions(edu_reprs, decoder_hidden)
            e2e_pred = RstTree.from_parsing_actions(actions, pred_edu_texts, relation_types=self.config.relation_types)

        out["e2e_pred"] = e2e_pred
        out["pred_edu_mapping"] = pred_edu_mapping
        out["pred_edu_ends"] = pred_ends
        return out
