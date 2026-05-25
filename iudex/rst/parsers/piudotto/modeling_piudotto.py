"""Span-based end-to-end RST parser (piudotto)."""

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from iudex.rst.data.reader import determine_label_index
from iudex.rst.data.tree import RstTree
from iudex.rst.parsers.common.encoding import (
    encode_tokens_strided,
    load_encoder_and_tokenizer,
    tokenize_document,
)
from iudex.rst.parsers.common.segmentation import Segmenter
from iudex.rst.parsers.piudotto.configuration_piudotto import PiudottoConfig

_NEG_INF = float("-inf")


class _SpanPooler(nn.Module):
    """Pool a span of token embeddings into one per-EDU vector.

    "concat":    reduce(concat(first_token, last_token, mean(tokens))) → H. Endpoint
                 + mean pooling, the standard span representation for parsing over a
                 contextual encoder; a strong default.
    "attention": reduce(concat(first_token, last_token, attn_pool(tokens))) → H,
                 where attn_pool is a learned-query attention over the span's tokens
                 (a single learned scoring vector, softmaxed over the span). Rarely
                 beats "concat" with a strong encoder; can underfit small treebanks.
    """

    def __init__(self, hidden_size: int, pooling: str, dropout: float):
        super().__init__()
        if pooling not in ("concat", "attention"):
            raise ValueError(f"Unknown span_pooling: {pooling!r}")
        self.pooling = pooling
        self.dropout = nn.Dropout(dropout)
        self.reduce = nn.Linear(3 * hidden_size, hidden_size, bias=False)
        if pooling == "attention":
            self.attn_score = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, embeddings: torch.Tensor, edu_mapping: list[tuple[int, int]]) -> torch.Tensor:
        """
        Args:
            embeddings:  [num_tokens, H]
            edu_mapping: list of (start, end_exclusive) per EDU, contiguous and in
                         token order (as the readers produce them).

        Returns:
            edu_reprs: [num_edus, H]
        """
        for i, (b, e) in enumerate(edu_mapping):
            if b >= e:
                # Empty-token EDU (e.g. an `edu_strings[i]` that tokenizes to zero
                # pieces). Surface as a hard error rather than silently NaN-ing.
                # Cheap pure-Python check (no kernels) before the vectorized path.
                raise ValueError(
                    f"EDU {i} has an empty token range {(b, e)}; check the "
                    f"upstream tokenizer output for empty / strippable EDU text."
                )
        device = embeddings.device
        n_edus, H = len(edu_mapping), embeddings.size(1)
        starts = torch.tensor([b for b, _ in edu_mapping], device=device)
        ends = torch.tensor([e for _, e in edu_mapping], device=device)

        first = embeddings[starts]  # [num_edus, H]
        last = embeddings[ends - 1]  # [num_edus, H]
        if self.pooling == "concat":
            # Segment mean via prefix sums (same trick as `_score_chart`).
            prefix = torch.cat([embeddings.new_zeros(1, H), embeddings.cumsum(0)], dim=0)
            pooled = (prefix[ends] - prefix[starts]) / (ends - starts).unsqueeze(-1).to(embeddings.dtype)
        else:
            # Per-EDU softmax over the EDU's own tokens as one segmented softmax
            # over the flat token sequence (token_to_edu maps each token to its
            # EDU; relies on the EDUs being contiguous and in order).
            token_to_edu = torch.repeat_interleave(torch.arange(n_edus, device=device), ends - starts)
            scores = self.attn_score(embeddings).squeeze(-1)  # [num_tokens]
            seg_max = scores.new_full((n_edus,), float("-inf")).scatter_reduce(0, token_to_edu, scores, reduce="amax")
            weights = (scores - seg_max[token_to_edu]).exp()
            seg_sum = weights.new_zeros(n_edus).index_add_(0, token_to_edu, weights)
            weights = weights / seg_sum[token_to_edu]
            pooled = embeddings.new_zeros(n_edus, H).index_add_(0, token_to_edu, weights.unsqueeze(-1) * embeddings)
        return self.reduce(self.dropout(torch.cat([first, last, pooled], dim=-1)))


class _FeedForward(nn.Sequential):
    """Linear → GELU → Dropout → Linear."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float):
        super().__init__(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )


class _DeepBiAffine(nn.Module):
    """Deep biaffine scorer (Dozat & Manning) over (left, right) span reprs.

    Args:
        h_left:  [num_candidates, input_dim]
        h_right: [num_candidates, input_dim]

    Returns:
        scores: [num_candidates, output_dim]
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float, bias: bool = True):
        super().__init__()
        self.W_left = _FeedForward(input_dim, hidden_dim, hidden_dim, dropout)
        self.W_right = _FeedForward(input_dim, hidden_dim, hidden_dim, dropout)
        self.W_s = nn.Bilinear(hidden_dim, hidden_dim, output_dim, bias=bias)
        self.V_left = nn.Linear(hidden_dim, output_dim)
        self.V_right = nn.Linear(hidden_dim, output_dim)

    def forward(self, h_left: torch.Tensor, h_right: torch.Tensor) -> torch.Tensor:
        h_left = self.W_left(h_left)
        h_right = self.W_right(h_right)
        return self.W_s(h_left, h_right) + self.V_left(h_left) + self.V_right(h_right)


def _sinusoidal_pe(length: int, dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Sinusoidal positional encodings [length, dim] (Vaswani et al. 2017).

    The EDU encoder's attention is permutation-invariant, but EDU order is
    central to RST, so we add positions explicitly. `dim` is even for all
    supported encoders.
    """
    pos = torch.arange(length, device=device, dtype=torch.float).unsqueeze(1)
    div = torch.exp(torch.arange(0, dim, 2, device=device, dtype=torch.float) * (-math.log(10000.0) / dim))
    pe = torch.zeros(length, dim, device=device, dtype=torch.float)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe.to(dtype)


class _EduEncoder(nn.Module):
    """A small, randomly-initialized Transformer encoder over the per-EDU
    vectors, run before span scoring so EDUs contextualize against each other
    (the role dmrst's document-level BiGRU plays). One document at a time, so
    no padding mask is needed.

    `inner_size`, when smaller than `hidden_size`, runs the Transformer in a
    narrower bottleneck: down-project H→inner, contextualize, up-project
    inner→H, with an outer residual. This forces the contextual update through a
    low-capacity channel (a dmrst-BiGRU-style dimensionality squeeze, the main
    regularization knob) while the full-width pooled rep is preserved by the
    residual. With `inner_size=None` it runs at full width and replaces the
    reps directly (the Transformer's own residual stream preserves them)."""

    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        inner_size: int | None = None,
    ):
        super().__init__()
        d = inner_size or hidden_size
        if d % num_heads != 0:
            raise ValueError(f"edu_encoder_heads ({num_heads}) must divide the EDU encoder width ({d})")
        self.bottleneck = d != hidden_size
        self.in_proj = nn.Linear(hidden_size, d) if self.bottleneck else nn.Identity()
        self.out_proj = nn.Linear(d, hidden_size) if self.bottleneck else nn.Identity()
        if self.bottleneck:
            # Zero-init the up-projection so the residual branch starts at 0: the
            # EDU encoder begins as the identity (the no-EDU-encoder baseline) and
            # learns the contextual update from there, rather than injecting noise.
            nn.init.zeros_(self.out_proj.weight)
            nn.init.zeros_(self.out_proj.bias)
        layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=num_heads,
            dim_feedforward=2 * d,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        # enable_nested_tensor is incompatible with norm_first and would just warn.
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, edu_reprs: torch.Tensor) -> torch.Tensor:
        # edu_reprs: [num_edus, H] -> [num_edus, H]
        x = self.in_proj(edu_reprs)
        x = self.dropout(x + _sinusoidal_pe(x.size(0), x.size(1), x.device, x.dtype))
        x = self.encoder(x.unsqueeze(0)).squeeze(0)
        x = self.out_proj(x)
        # Outer residual only when bottlenecked: the down-projection breaks the
        # H-dim residual stream, so we re-add the pooled rep to preserve it.
        return edu_reprs + x if self.bottleneck else x


class PiudottoParser(nn.Module):
    """End-to-end RST parser with span-based biaffine scoring.

    Training (`forward`) is teacher-forced and returns split/label/seg losses
    separately so the trainer can apply EMA loss weighting. With
    `cfg.margin_training` set (non-null), training instead minimizes a Stern
    et al. 2017 max-margin objective against the cost-augmented CKY tree.
    Decoding is greedy by default; set `cfg.decoding = "cky"` for the globally
    optimal binary tree.
    """

    def __init__(self, config: PiudottoConfig, *, compile_encoder: bool = False):
        super().__init__()
        self.config = config
        self.label_index = determine_label_index(config.relation_types)
        self.stride = config.stride

        # Validate string-enum fields up front.
        if config.span_pooling not in ("concat", "attention"):
            raise ValueError(f"Unknown span_pooling: {config.span_pooling!r}")
        if config.label_input_pooling not in ("mean", "last_edu"):
            raise ValueError(f"Unknown label_input_pooling: {config.label_input_pooling!r}")
        if config.decoding not in ("cky", "greedy"):
            raise ValueError(f"Unknown decoding: {config.decoding!r}")

        self.encoder, self.tokenizer, self.max_length = load_encoder_and_tokenizer(
            config.model_name, peft_config=config.peft
        )

        # Compile the encoder forward (not the module) so state_dict keys are
        # unchanged and existing checkpoints still load. dynamic=True avoids
        # per-shape recompiles on variable-length documents. Off by default
        # (inference); training opts in, predict opts in via --compile-encoder.
        if compile_encoder and torch.cuda.is_available():
            self.encoder.forward = torch.compile(self.encoder.forward, dynamic=True)

        H = self.encoder.config.hidden_size
        self.encoder_dropout = nn.Dropout(config.encoder_dropout)
        self.span_pooler = _SpanPooler(H, config.span_pooling, config.encoder_dropout)
        self.split_scorer = _DeepBiAffine(
            H, config.classifier_hidden_size, 1, config.classifier_dropout, bias=config.classifier_use_bias
        )
        self.label_scorer = _DeepBiAffine(
            H,
            config.classifier_hidden_size,
            len(self.label_index),
            config.classifier_dropout,
            bias=config.classifier_use_bias,
        )

        # Optional EDU-level Transformer over the pooled per-EDU vectors. None
        # disables it (the original pure encoder-pooled design).
        self.edu_encoder = (
            _EduEncoder(
                H,
                config.edu_encoder_layers,
                config.edu_encoder_heads,
                config.edu_encoder_dropout,
                inner_size=config.edu_encoder_hidden_size,
            )
            if config.edu_encoder_layers > 0
            else None
        )

        self.label_input_pooling = config.label_input_pooling
        self.decoding = config.decoding
        # None → per-node CE; non-None → margin objective.
        self.margin = config.margin_training.margin if config.margin_training is not None else None

        self.segmenter = (
            Segmenter(
                H,
                scheme=config.segmentation.scheme,
                loss=config.segmentation.loss,
                pos_weight=config.segmentation.pos_weight,
                dropout=config.segmentation.dropout,
            )
            if config.segmentation is not None
            else None
        )

        # Faithful text reconstruction for end-to-end (segmenter) models, so
        # training matches the raw text `predict_from_text` receives. A
        # detokenized corpus's exact inter-EDU `prefix` markers are preferred
        # when present (see RstTree.edu_prefixes); the heuristic detokenizer is
        # the fallback. Gold-EDU-only models consume corpus-tokenized RS3/RS4
        # verbatim, so they get neither.
        self.detokenizer = config.detokenizer if self.segmenter is not None else None
        self.use_edu_prefixes = self.segmenter is not None

    @property
    def device(self) -> torch.device:
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
        compile_encoder: bool = False,
    ) -> "PiudottoParser":
        """Load from a HuggingFace Hub repo id, a local run dir, or a `.pt` file.

        See `iudex.rst.parsers.hfhub.load_parser_from_pretrained` for the
        full resolution rules.
        """
        from iudex.rst.parsers.hfhub import load_parser_from_pretrained

        dev = (
            torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        return load_parser_from_pretrained(
            repo_or_path,
            parser_cls=cls,
            config_cls=PiudottoConfig,
            device=dev,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
            compile_encoder=compile_encoder,
        )

    # ─── Encoding ──────────────────────────────────────────────────────────

    def _encode(self, tree: RstTree) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize + encode + (optionally) compute the segmentation loss.

        Encodes the EDUs as one continuous document (so train/inference
        tokenization match for the segmenter; see `tokenize_document`).

        Returns:
            edu_reprs: [num_edus, H]
            seg_loss:  scalar tensor (zero when joint segmentation is disabled
                       or the model is in eval mode)
        """
        input_ids, edu_mapping = tokenize_document(
            self.tokenizer,
            tree.edu_strings,
            self.device,
            detokenizer=self.detokenizer,
            prefixes=tree.edu_prefixes if self.use_edu_prefixes else None,
        )
        embeddings = encode_tokens_strided(
            self.encoder, self.tokenizer, input_ids, self.max_length, self.stride
        ).float()

        if self.segmenter is not None and self.training:
            # Segmenter operates on the encoder outputs BEFORE dropout.
            seg_loss = self.segmenter.loss(embeddings, edu_mapping)
        else:
            seg_loss = torch.zeros((), device=embeddings.device)

        edu_reprs = self._pool_edus(self.encoder_dropout(embeddings), edu_mapping)
        return edu_reprs, seg_loss

    def _pool_edus(self, embeddings: torch.Tensor, edu_mapping: list[tuple[int, int]]) -> torch.Tensor:
        """Pool tokens into per-EDU vectors, then (optionally) contextualize the
        EDUs against each other with the EDU-level Transformer."""
        edu_reprs = self.span_pooler(embeddings, edu_mapping)
        if self.edu_encoder is not None:
            edu_reprs = self.edu_encoder(edu_reprs)
        return edu_reprs

    # ─── Pooling helpers (used by per-node CE training) ────────────────────

    def _pool_span(self, edu_reprs: torch.Tensor, b: int, e: int) -> torch.Tensor:
        """Pool EDUs [b, e) into a single [1, H] vector per `label_input_pooling`."""
        if self.label_input_pooling == "mean":
            return edu_reprs[b:e].mean(0, keepdim=True)
        return edu_reprs[e - 1].unsqueeze(0)  # last_edu

    # ─── Full-chart scoring (used by CKY decoding and margin training) ─────

    def _score_chart(self, edu_reprs: torch.Tensor, return_label_logits: bool = False) -> dict[str, Any]:
        """Score every valid (b, k, e) triple, chunked by span width to bound memory.

        For all 0 ≤ b < k < e ≤ n, build pooled left = pool(edu_reprs[b:k]) and
        right = pool(edu_reprs[k:e]) per `label_input_pooling`, then score the
        split (1-D) and label (C-D) biaffines on the batched pools.

        Chunking: triples are enumerated in groups of constant span width
        e − b. Within a width w there are O(n · w) triples — total O(n²) at the
        widest tier — so peak `[num_triples_in_chunk, H]` memory is O(n² · H)
        instead of the O(n³ · H) you'd get from materializing all triples at
        once. The reduced charts `split_chart` / `best_label_*_chart` are
        O(n³) of scalars/longs, which is fine even at n ≈ 250.

        Args:
            edu_reprs:           [num_edus, H]
            return_label_logits: only `_forward_margin` needs the per-triple
                                 label_logits + triple_index lookup. Default
                                 False so CKY decoding and per-node CE training
                                 don't pay the O(n³ · C) storage cost.

        Returns a dict with:
            split_chart            [n+1, n+1, n+1]   (-inf at invalid positions)
            best_label_score_chart [n+1, n+1, n+1]   (max over C)
            best_label_idx_chart   [n+1, n+1, n+1]   (argmax over C)
            label_logits           [num_triples, C]  (only if return_label_logits)
            triple_index           {(b, k, e): position in label_logits}
        """
        # Chart fill (CKY / margin gold + augmented scores) sums many terms, so do
        # it in fp32 for stability under bf16 autocast (a no-op outside autocast).
        edu_reprs = edu_reprs.float()
        n = edu_reprs.size(0)
        device = edu_reprs.device
        H = edu_reprs.size(1)
        score_dtype = edu_reprs.dtype

        split_chart = torch.full((n + 1, n + 1, n + 1), _NEG_INF, device=device, dtype=score_dtype)
        best_label_score_chart = torch.full((n + 1, n + 1, n + 1), _NEG_INF, device=device, dtype=score_dtype)
        best_label_idx_chart = torch.zeros((n + 1, n + 1, n + 1), dtype=torch.long, device=device)

        if self.label_input_pooling == "mean":
            # Prefix-sum trick: P[i] = sum(edu_reprs[:i]); mean(edu_reprs[a:b]) = (P[b] - P[a]) / (b - a).
            prefix = torch.cat([torch.zeros(1, H, device=device, dtype=score_dtype), edu_reprs.cumsum(0)], dim=0)

        all_label_logits: list[torch.Tensor] = []
        triple_index: dict[tuple[int, int, int], int] = {}
        running_offset = 0

        for width in range(2, n + 1):
            # All (b, k, e) with e − b = width: b ∈ [0, n − width], k ∈ (b, e).
            num_b = n - width + 1
            ks_local = torch.arange(1, width, device=device)
            bs_t = torch.arange(num_b, device=device).repeat_interleave(width - 1)
            ks_t = bs_t + ks_local.repeat(num_b)
            es_t = bs_t + width

            if self.label_input_pooling == "mean":
                lefts = (prefix[ks_t] - prefix[bs_t]) / (ks_t - bs_t).unsqueeze(-1).to(score_dtype)
                rights = (prefix[es_t] - prefix[ks_t]) / (es_t - ks_t).unsqueeze(-1).to(score_dtype)
            else:  # last_edu
                lefts = edu_reprs[ks_t - 1]
                rights = edu_reprs[es_t - 1]

            split_logits = self.split_scorer(lefts, rights).squeeze(-1).float()
            label_logits = self.label_scorer(lefts, rights).float()

            split_chart[bs_t, ks_t, es_t] = split_logits
            best_label_score, best_label_idx = label_logits.max(-1)
            best_label_score_chart[bs_t, ks_t, es_t] = best_label_score
            best_label_idx_chart[bs_t, ks_t, es_t] = best_label_idx

            if return_label_logits:
                all_label_logits.append(label_logits)
                bs_list = bs_t.tolist()
                ks_list = ks_t.tolist()
                es_list = es_t.tolist()
                for off, (b, k, e) in enumerate(zip(bs_list, ks_list, es_list, strict=True)):
                    triple_index[(b, k, e)] = running_offset + off
                running_offset += len(bs_list)

        out: dict[str, Any] = {
            "split_chart": split_chart,
            "best_label_score_chart": best_label_score_chart,
            "best_label_idx_chart": best_label_idx_chart,
        }
        if return_label_logits:
            out["label_logits"] = (
                torch.cat(all_label_logits, dim=0)
                if all_label_logits
                else torch.empty(0, len(self.label_index), device=device, dtype=score_dtype)
            )
            out["triple_index"] = triple_index
        return out

    # ─── Training ──────────────────────────────────────────────────────────

    def forward(self, tree: RstTree) -> dict[str, torch.Tensor]:
        """Teacher-forced loss for one gold tree.

        With `cfg.margin_training` null (default), accumulates per-gold-span CE
        on splits and labels (cheap; no full chart). When set, computes the
        gold tree score and the cost-augmented CKY tree score, returning the
        margin hinge loss.

        Returns:
            {
                "loss":       (weighted) sum of split + label + seg losses,
                "split_loss": scalar (broken out for the trainer's EMA-weighting
                              machinery; "margin" mode reports the hinge under
                              "split_loss" with `label_loss=0` since the two
                              objectives can't be cleanly separated),
                "label_loss": scalar,
                "seg_loss":   scalar (zero when joint segmentation is disabled),
            }
        """
        num_edus = len(tree.edus)
        if num_edus < 2:
            zero = torch.zeros((), device=self.device, requires_grad=True)
            return {"loss": zero, "split_loss": zero, "label_loss": zero, "seg_loss": zero}

        edu_reprs, seg_loss = self._encode(tree)

        gold_decisions: dict[tuple[int, int], tuple[int, str]] = {}
        for (left_range, right_range), nuc, rel in tree.spans_with_ranges():
            gold_decisions[(left_range[0], right_range[1])] = (right_range[0], f"{nuc}_{rel}")

        if self.margin is None:
            split_loss, label_loss = self._forward_per_node_ce(edu_reprs, gold_decisions)
        else:
            split_loss, label_loss = self._forward_margin(edu_reprs, gold_decisions)

        return {
            "loss": split_loss + label_loss + seg_loss,
            "split_loss": split_loss,
            "label_loss": label_loss,
            "seg_loss": seg_loss,
        }

    def _forward_per_node_ce(
        self,
        edu_reprs: torch.Tensor,
        gold_decisions: dict[tuple[int, int], tuple[int, str]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-gold-span CE on splits and labels (no full chart materialization)."""
        split_losses: list[torch.Tensor] = []
        label_losses: list[torch.Tensor] = []
        for (b, e), (gold_k, gold_label_str) in gold_decisions.items():
            gold_label_idx = self.label_index.index(gold_label_str)
            # Build pooled (left, right) for every candidate k ∈ (b, e).
            lefts = torch.cat([self._pool_span(edu_reprs, b, k) for k in range(b + 1, e)], dim=0)
            rights = torch.cat([self._pool_span(edu_reprs, k, e) for k in range(b + 1, e)], dim=0)
            split_logits = self.split_scorer(lefts, rights).squeeze(-1)  # [e-b-1]
            label_logits = self.label_scorer(lefts, rights)  # [e-b-1, C]

            gold_k_idx = gold_k - b - 1
            if e - b > 2:
                split_target = torch.tensor([gold_k_idx], device=self.device)
                split_losses.append(F.cross_entropy(split_logits.unsqueeze(0), split_target))

            label_target = torch.tensor([gold_label_idx], device=self.device)
            label_losses.append(F.cross_entropy(label_logits[gold_k_idx].unsqueeze(0), label_target))

        split_loss = sum(split_losses) / len(split_losses) if split_losses else torch.zeros((), device=self.device)
        label_loss = sum(label_losses) / len(label_losses)
        return split_loss, label_loss

    def _forward_margin(
        self,
        edu_reprs: torch.Tensor,
        gold_decisions: dict[tuple[int, int], tuple[int, str]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Stern et al. 2017 max-margin loss against the cost-augmented CKY tree.

        Cost is per-decision Hamming over (b, k, e, label): every CKY-chosen
        decision that doesn't match the gold decision at its (b, e) span adds
        `self.margin` to the augmented score. The optimizer is driven to push
        the gold tree's score above the most-violated tree's score by a margin.

        Returns the hinge loss under `split_loss` and zero under `label_loss`
        (so the trainer's loss-weighting machinery sees a single scalar even
        though the margin couples the two decisions).
        """
        chart = self._score_chart(edu_reprs, return_label_logits=True)
        split_chart = chart["split_chart"]
        best_label_score_chart = chart["best_label_score_chart"]
        best_label_idx_chart = chart["best_label_idx_chart"]
        label_logits = chart["label_logits"]
        triple_index = chart["triple_index"]

        n = edu_reprs.size(0)

        # Score the gold tree from the same chart.
        gold_score = torch.zeros((), device=edu_reprs.device, dtype=label_logits.dtype)
        for (b, e), (gold_k, gold_label_str) in gold_decisions.items():
            gold_label_idx = self.label_index.index(gold_label_str)
            t_idx = triple_index[(b, gold_k, e)]
            gold_score = gold_score + split_chart[b, gold_k, e] + label_logits[t_idx, gold_label_idx]

        # Cost-augment label scores: every triple gets +margin (anything you pick
        # there is a non-gold decision), then for each gold (b, gold_k, e) re-score
        # so picking the gold label costs zero and picking any other label costs +margin.
        aug_label_score = best_label_score_chart + self.margin
        aug_label_idx = best_label_idx_chart.clone()
        for (b, e), (gold_k, gold_label_str) in gold_decisions.items():
            gold_label_idx = self.label_index.index(gold_label_str)
            t_idx = triple_index[(b, gold_k, e)]
            per_label = label_logits[t_idx].clone()
            per_label = per_label + self.margin
            per_label[gold_label_idx] = per_label[gold_label_idx] - self.margin
            best = per_label.max()
            best_arg = per_label.argmax()
            aug_label_score[b, gold_k, e] = best
            aug_label_idx[b, gold_k, e] = best_arg

        pred_score, _ = _cky_fill(split_chart, aug_label_score, aug_label_idx, n)
        hinge = torch.clamp(pred_score - gold_score, min=0.0)
        return hinge, torch.zeros((), device=edu_reprs.device)

    # ─── Inference ─────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict(self, tree: RstTree) -> RstTree:
        """Decode using the gold EDU segmentation in `tree.edus`."""
        self.eval()
        num_edus = len(tree.edus)
        if num_edus < 2:
            return RstTree.from_parsing_actions([], tree.edus, relation_types=self.config.relation_types)

        edu_reprs, _ = self._encode(tree)
        actions = self._decode_actions(edu_reprs)
        return RstTree.from_parsing_actions(actions, tree.edus, relation_types=self.config.relation_types)

    @torch.no_grad()
    def predict_from_text(self, text: str) -> RstTree:
        """End-to-end inference from raw document text. Requires `cfg.segmentation` to be non-null."""
        if self.segmenter is None:
            raise RuntimeError("predict_from_text requires `cfg.segmentation` to be non-null")
        self.eval()

        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(ids) == 0:
            return RstTree.from_parsing_actions([], [], relation_types=self.config.relation_types)
        input_ids = torch.tensor(ids, dtype=torch.long, device=self.device)
        embeddings = encode_tokens_strided(
            self.encoder, self.tokenizer, input_ids, self.max_length, self.stride
        ).float()

        breaks = self.segmenter.predict_breaks(embeddings)
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

        edu_reprs = self._pool_edus(self.encoder_dropout(embeddings), edu_mapping)  # eval mode → dropout identity
        actions = self._decode_actions(edu_reprs)
        return RstTree.from_parsing_actions(actions, edu_texts, relation_types=self.config.relation_types)

    @torch.no_grad()
    def predict_both(self, tree: RstTree) -> dict[str, Any]:
        """One encoder pass that yields both gold-EDU and e2e predictions.

        With `self.segmenter is None`, only the gold-EDU keys are populated
        and the e2e keys are `None`.

        Returns:
            {
                "gold_pred":        RstTree,
                "gold_edu_mapping": list[(start, end)],
                "gold_edu_ends":    list[int],
                "e2e_pred":         RstTree or None,
                "pred_edu_mapping": list[(start, end)] or None,
                "pred_edu_ends":    list[int] or None,
            }
        """
        self.eval()
        input_ids, gold_edu_mapping = tokenize_document(
            self.tokenizer,
            tree.edu_strings,
            self.device,
            detokenizer=self.detokenizer,
            prefixes=tree.edu_prefixes if self.use_edu_prefixes else None,
        )
        token_embeddings = encode_tokens_strided(
            self.encoder, self.tokenizer, input_ids, self.max_length, self.stride
        ).float()
        dropped = self.encoder_dropout(token_embeddings)  # eval mode → identity

        if len(gold_edu_mapping) < 2:
            gold_pred = RstTree.from_parsing_actions([], tree.edus, relation_types=self.config.relation_types)
        else:
            edu_reprs = self._pool_edus(dropped, gold_edu_mapping)
            actions = self._decode_actions(edu_reprs)
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

        pred_ends = self.segmenter.predict_breaks(token_embeddings)
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

        if len(pred_edu_mapping) < 2:
            e2e_pred = RstTree.from_parsing_actions([], pred_edu_texts, relation_types=self.config.relation_types)
        else:
            edu_reprs = self._pool_edus(dropped, pred_edu_mapping)
            actions = self._decode_actions(edu_reprs)
            e2e_pred = RstTree.from_parsing_actions(actions, pred_edu_texts, relation_types=self.config.relation_types)

        out["e2e_pred"] = e2e_pred
        out["pred_edu_mapping"] = pred_edu_mapping
        out["pred_edu_ends"] = pred_ends
        return out

    def _decode_actions(self, edu_reprs: torch.Tensor) -> list[tuple[int, str, str]]:
        """Dispatch decoding per `self.decoding`.

        Greedy scores only the spans it visits (`_greedy_actions`); CKY needs the
        full chart for the global optimum.
        """
        if self.decoding == "greedy":
            return self._greedy_actions(edu_reprs)
        n = edu_reprs.size(0)
        chart = self._score_chart(edu_reprs)
        return cky_decode(
            chart["split_chart"],
            chart["best_label_score_chart"],
            chart["best_label_idx_chart"],
            n,
            self.label_index,
        )

    def _greedy_actions(self, edu_reprs: torch.Tensor) -> list[tuple[int, str, str]]:
        """Greedy top-down decode that scores only the spans it visits.

        At each span (b, e) pick the argmax (split, label) with no lookahead, then
        recurse into the two children. This touches O(n) spans rather than the full
        O(n^3) chart `_score_chart` builds, so it's the cheap path for the default
        decoder. Scoring per span mirrors `_forward_per_node_ce`.
        """
        n = edu_reprs.size(0)
        if n < 2:
            return []
        actions: list[tuple[int, str, str]] = []
        stack: list[tuple[int, int]] = [(0, n)]
        while stack:
            b, e = stack.pop()
            if e - b < 2:
                continue
            lefts = torch.cat([self._pool_span(edu_reprs, b, k) for k in range(b + 1, e)], dim=0)
            rights = torch.cat([self._pool_span(edu_reprs, k, e) for k in range(b + 1, e)], dim=0)
            split_logits = self.split_scorer(lefts, rights).squeeze(-1)  # [e-b-1]
            label_logits = self.label_scorer(lefts, rights)  # [e-b-1, C]
            best_label_score, best_label_idx = label_logits.max(-1)
            best_k_idx = int((split_logits + best_label_score).argmax().item())
            k = b + 1 + best_k_idx
            nuc, rel = self.label_index[int(best_label_idx[best_k_idx].item())].split("_", 1)
            actions.append((k, nuc, rel))
            if e - k > 1:
                stack.append((k, e))
            if k - b > 1:
                stack.append((b, k))
        return actions


# ─── Decoders (top-level helpers, not methods) ─────────────────────────────


def _cky_fill(
    split_chart: torch.Tensor,
    label_score_chart: torch.Tensor,
    label_idx_chart: torch.Tensor,
    num_edus: int,
) -> tuple[torch.Tensor, dict[tuple[int, int], tuple[int, int]]]:
    """Inner CKY fill. Returns (root_score, back-pointers).

    back-pointers: {(b, e): (best_k, best_label_idx)} for every non-leaf span
    along the optimal tree (and along every other span that ever got filled).
    """
    n = num_edus
    device = split_chart.device
    chart = torch.zeros((n + 1, n + 1), device=device, dtype=split_chart.dtype)
    back: dict[tuple[int, int], tuple[int, int]] = {}
    for width in range(2, n + 1):
        for b in range(n - width + 1):
            e = b + width
            # Vectorize over k ∈ (b, e).
            split_vals = split_chart[b, b + 1 : e, e]  # [width-1]
            label_score_vals = label_score_chart[b, b + 1 : e, e]
            left_vals = chart[b, b + 1 : e]
            right_vals = chart[b + 1 : e, e]
            total = split_vals + label_score_vals + left_vals + right_vals
            best_k_idx = int(total.argmax().item())
            chart[b, e] = total[best_k_idx]
            best_k = b + 1 + best_k_idx
            back[(b, e)] = (best_k, int(label_idx_chart[b, best_k, e].item()))
    return chart[0, n], back


def cky_decode(
    split_chart: torch.Tensor,
    best_label_score_chart: torch.Tensor,
    best_label_idx_chart: torch.Tensor,
    num_edus: int,
    label_index: list[str],
) -> list[tuple[int, str, str]]:
    """Globally-optimal binary tree via CKY over EDU spans. O(n^3) fill + O(n) backtrace.

    Returns parsing actions in DFS order ready for `RstTree.from_parsing_actions`.
    """
    if num_edus < 2:
        return []
    _, back = _cky_fill(split_chart, best_label_score_chart, best_label_idx_chart, num_edus)
    return _backtrace(back, num_edus, label_index)


def _backtrace(
    back: dict[tuple[int, int], tuple[int, int]],
    num_edus: int,
    label_index: list[str],
) -> list[tuple[int, str, str]]:
    actions: list[tuple[int, str, str]] = []
    stack: list[tuple[int, int]] = [(0, num_edus)]
    while stack:
        b, e = stack.pop()
        if e - b < 2:
            continue
        k, label_idx = back[(b, e)]
        nuc, rel = label_index[label_idx].split("_", 1)
        actions.append((k, nuc, rel))
        if e - k > 1:
            stack.append((k, e))
        if k - b > 1:
            stack.append((b, k))
    return actions
