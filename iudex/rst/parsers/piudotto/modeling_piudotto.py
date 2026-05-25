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
from iudex.rst.parsers.common.pointer import PointerAttention
from iudex.rst.parsers.common.segmentation import Segmenter
from iudex.rst.parsers.piudotto.configuration_piudotto import PiudottoConfig


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
            # Segment mean via prefix sums.
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


class _TreeDecoder(nn.Module):
    """Autoregressive Transformer decoder over the top-down decision sequence.

    The non-RNN replacement for dmrst's recurrent pointer decoder. The
    "sequence" is the tree's internal-node spans in left-first preorder DFS, the
    order both the training pass and greedy decode visit them. Each step's input
    token is the pooled repr of the span being decided (plus a sinusoidal
    positional encoding over the decision-step index, since attention is
    order-agnostic and DFS order is the history signal). Causal self-attention
    over the prefix conditions each split decision on the decisions already
    committed (query t attends only to tokens 1..t); cross-attention reads the
    EDU representations. The per-step output is the pointer query.

    `inner_size`, when smaller than `hidden_size`, runs the decoder in a narrow
    bottleneck (down-project H->inner, decode, up-project inner->H) the same way
    `_EduEncoder` does, the main regularization knob on small treebanks. The
    cross-attention memory is projected to the same width.
    """

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
            raise ValueError(f"decoder_heads ({num_heads}) must divide the decoder width ({d})")
        self.bottleneck = d != hidden_size
        self.in_proj = nn.Linear(hidden_size, d) if self.bottleneck else nn.Identity()
        self.mem_proj = nn.Linear(hidden_size, d) if self.bottleneck else nn.Identity()
        self.out_proj = nn.Linear(d, hidden_size) if self.bottleneck else nn.Identity()
        layer = nn.TransformerDecoderLayer(
            d_model=d,
            nhead=num_heads,
            dim_feedforward=2 * d,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        # Normalize the output query so the pointer's dot products start at a sane
        # scale (the pooled EDU reprs are un-normalized and large-magnitude; see
        # `_split_logits`). norm_first layers leave the residual stream un-normed.
        self.query_norm = nn.LayerNorm(hidden_size)

    def forward(self, span_tokens: torch.Tensor, edu_reprs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            span_tokens: [num_decisions, H]  pooled span reprs in DFS order
            edu_reprs:   [num_edus, H]       cross-attention memory

        Returns:
            queries: [num_decisions, H]  one (layer-normed) pointer query per decision
        """
        m = span_tokens.size(0)
        x = self.in_proj(span_tokens)
        x = self.dropout(x + _sinusoidal_pe(m, x.size(1), x.device, x.dtype))
        memory = self.mem_proj(edu_reprs)
        # Boolean upper-triangular mask: True (= disallowed) above the diagonal,
        # so position t attends only to 1..t. dtype-agnostic, unlike a float mask
        # (avoids a query/mask dtype mismatch under bf16 autocast).
        causal = torch.triu(torch.ones(m, m, dtype=torch.bool, device=x.device), diagonal=1)
        h = self.decoder(x.unsqueeze(0), memory.unsqueeze(0), tgt_mask=causal).squeeze(0)
        return self.query_norm(self.out_proj(h))


class PiudottoParser(nn.Module):
    """End-to-end RST parser with span-based biaffine scoring.

    Training (`forward`) is teacher-forced and returns split/label/seg losses
    separately so the trainer can apply EMA loss weighting. Decoding is greedy
    top-down.

    Split scoring has two modes, gated by `cfg.decoder_layers`:
      0 (default): the history-free per-node deep biaffine over pooled left/right
        halves (`split_scorer`). Every span is scored independently.
      >0: an autoregressive Transformer decoder (`tree_decoder`) over the DFS
        decision sequence, with a pointer split head conditioned on the decode
        history (the non-RNN analog of dmrst's recurrent pointer decoder).
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
        self.label_scorer = _DeepBiAffine(
            H,
            config.classifier_hidden_size,
            len(self.label_index),
            config.classifier_dropout,
            bias=config.classifier_use_bias,
        )

        # Split scoring. With the decoder off, score every span independently with
        # a deep biaffine over pooled halves. With the decoder on, the pointer
        # head (query = decoder output) replaces it, so only one of the two is
        # ever built (keeps the checkpoint to the params the config actually uses).
        if config.decoder_layers > 0:
            if config.pointer_attention_type not in ("biaffine", "dot_product"):
                raise ValueError(f"Unknown pointer_attention_type: {config.pointer_attention_type!r}")
            self.split_scorer = None
            self.tree_decoder = _TreeDecoder(
                H,
                config.decoder_layers,
                config.decoder_heads,
                config.decoder_dropout,
                inner_size=config.decoder_hidden_size,
            )
            self.pointer = PointerAttention(config.pointer_attention_type, H)
            self.pointer_key_norm = nn.LayerNorm(H)
            self._pointer_scale = H**0.5
        else:
            self.split_scorer = _DeepBiAffine(
                H, config.classifier_hidden_size, 1, config.classifier_dropout, bias=config.classifier_use_bias
            )
            self.tree_decoder = None
            self.pointer = None

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

    def _split_logits(self, edu_reprs: torch.Tensor, b: int, e: int, query: torch.Tensor) -> torch.Tensor:
        """Scaled pointer split logits over candidate splits k ∈ (b, e).

        Scaled dot-product attention done right: LayerNorm the key EDU reprs (the
        query is already normed by the decoder) and divide by √H, so logits start
        at O(1) despite the un-normalized, large-magnitude pooled EDU reprs. dmrst
        gets the same effect for free from its `layer_norm` plus bounded GRU
        outputs; piudotto's pooled reprs have neither, so without this the pointer
        saturates its softmax at init (huge split loss and gradients).
        """
        keys = self.pointer_key_norm(edu_reprs[b : e - 1])  # [e-b-1, H]
        return self.pointer(keys, query) / self._pointer_scale  # [1, e-b-1]

    # ─── Training ──────────────────────────────────────────────────────────

    def forward(self, tree: RstTree) -> dict[str, torch.Tensor]:
        """Teacher-forced per-gold-span CE on splits and labels.

        With the decoder off, every span is scored independently
        (`_forward_per_node_ce`). With the decoder on, the gold spans are scored
        in one masked decoder pass over the DFS decision sequence
        (`_forward_with_decoder`); the loss decomposition is identical.

        Returns:
            {
                "loss":       sum of split + label + seg losses,
                "split_loss": scalar (broken out for the trainer's EMA weighting),
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

        if self.tree_decoder is not None:
            split_loss, label_loss = self._forward_with_decoder(edu_reprs, gold_decisions, num_edus)
        else:
            split_loss, label_loss = self._forward_per_node_ce(edu_reprs, gold_decisions)

        return {
            "loss": split_loss + label_loss + seg_loss,
            "split_loss": split_loss,
            "label_loss": label_loss,
            "seg_loss": seg_loss,
        }

    @staticmethod
    def _dfs_spans(
        gold_decisions: dict[tuple[int, int], tuple[int, str]],
        num_edus: int,
    ) -> list[tuple[int, int]]:
        """The tree's internal-node spans in left-first preorder DFS.

        This is the order the decoder lays out its decision tokens at training and
        the order greedy decode visits spans at inference. They must match, or the
        causal/positional structure won't correspond.
        """
        order: list[tuple[int, int]] = []
        stack = [(0, num_edus)]
        while stack:
            b, e = stack.pop()
            order.append((b, e))
            gold_k, _ = gold_decisions[(b, e)]
            if e - gold_k > 1:
                stack.append((gold_k, e))
            if gold_k - b > 1:
                stack.append((b, gold_k))
        return order

    def _forward_with_decoder(
        self,
        edu_reprs: torch.Tensor,
        gold_decisions: dict[tuple[int, int], tuple[int, str]],
        num_edus: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-gold-span CE with split scores from the AR decoder + pointer head.

        One masked decoder pass produces every per-decision query; the causal
        mask guarantees query t saw only tokens 1..t (the parse history up to that
        decision), so this matches the left-to-right flow of inference while being
        a single parallel forward.
        """
        spans = self._dfs_spans(gold_decisions, num_edus)
        span_tokens = torch.stack([edu_reprs[b:e].mean(0) for b, e in spans])  # [m, H]
        queries = self.tree_decoder(span_tokens, edu_reprs)  # [m, H]

        split_losses: list[torch.Tensor] = []
        label_losses: list[torch.Tensor] = []
        for t, (b, e) in enumerate(spans):
            gold_k, gold_label_str = gold_decisions[(b, e)]
            gold_label_idx = self.label_index.index(gold_label_str)

            if e - b > 2:
                split_logits = self._split_logits(edu_reprs, b, e, queries[t])  # [1, e-b-1]
                split_target = torch.tensor([gold_k - b - 1], device=self.device)
                split_losses.append(F.cross_entropy(split_logits, split_target))

            left = self._pool_span(edu_reprs, b, gold_k)
            right = self._pool_span(edu_reprs, gold_k, e)
            label_logits = self.label_scorer(left, right)  # [1, C]
            label_target = torch.tensor([gold_label_idx], device=self.device)
            label_losses.append(F.cross_entropy(label_logits, label_target))

        split_loss = sum(split_losses) / len(split_losses) if split_losses else torch.zeros((), device=self.device)
        label_loss = sum(label_losses) / len(label_losses)
        return split_loss, label_loss

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
        """Greedy top-down decode, dispatched on whether the AR decoder is on."""
        if self.tree_decoder is not None:
            return self._decode_with_decoder(edu_reprs)
        return self._greedy_actions(edu_reprs)

    def _decode_with_decoder(self, edu_reprs: torch.Tensor) -> list[tuple[int, str, str]]:
        """Autoregressive greedy decode with the Transformer decoder + pointer head.

        Visits spans in left-first preorder DFS (the order the decoder was trained
        on). Each step appends the current span's token, re-runs the decoder over
        the prefix, and uses the last query to pick the split + label. Re-running
        the whole prefix per step is the simple, obviously-correct route (it's the
        same function as training on a prefix); RST trees are small enough that the
        O(n) extra passes don't matter, and a KV-cache is a later optimization.
        """
        n = edu_reprs.size(0)
        if n < 2:
            return []
        actions: list[tuple[int, str, str]] = []
        tokens: list[torch.Tensor] = []
        stack: list[tuple[int, int]] = [(0, n)]
        while stack:
            b, e = stack.pop()
            tokens.append(edu_reprs[b:e].mean(0))
            query = self.tree_decoder(torch.stack(tokens), edu_reprs)[-1]  # [H]
            if e - b == 2:
                k = b + 1
            else:
                split_logits = self._split_logits(edu_reprs, b, e, query)  # [1, e-b-1]
                k = b + 1 + int(split_logits.argmax(-1).item())
            left = self._pool_span(edu_reprs, b, k)
            right = self._pool_span(edu_reprs, k, e)
            label_logits = self.label_scorer(left, right)
            nuc, rel = self.label_index[int(label_logits.argmax(-1).item())].split("_", 1)
            actions.append((k, nuc, rel))
            if e - k > 1:
                stack.append((k, e))
            if k - b > 1:
                stack.append((b, k))
        return actions

    def _greedy_actions(self, edu_reprs: torch.Tensor) -> list[tuple[int, str, str]]:
        """Greedy top-down decode that scores only the spans it visits (the
        history-free path, used when the AR decoder is off).

        At each span (b, e) pick the argmax (split, label) with no lookahead, then
        recurse into the two children. Scoring per span mirrors `_forward_per_node_ce`.
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
