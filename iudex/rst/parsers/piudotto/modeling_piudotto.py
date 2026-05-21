"""Span-based end-to-end RST parser (piudotto)."""

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
from iudex.rst.parsers.piudotto.configuration_piudotto import PiudottoConfig

_NEG_INF = float("-inf")


# Per-scheme tag layout. Tag indices are positional in `tag_names`. The
# transition mask and the first/last-position masks encode schema validity:
# they are added (as 0 / -inf biases) to the Viterbi scores at inference and,
# for the CRF objective, to the learned transition/start/end scores at training
# too — so the CRF only ever puts probability mass on schema-valid sequences.
#
# Conventions for building gold targets from EDU mappings:
#   BIE: multi-token EDU → B at first, I in middle, E at last.
#        1-token EDU → just E (treats the lone token as the "end" of a degenerate
#        EDU; keeps the rule "EDU ends are exactly the E positions" unambiguous).
#   BO:  first token of each EDU is B, every other token is O.
#   EO:  last token of each EDU is E, every other token is O.
_SCHEMES: dict[str, dict[str, Any]] = {
    "BIE": {
        "tag_names": ("B", "I", "E"),
        # [from, to] allowed-transition mask:
        #   B → I or E   (B cannot be immediately followed by another B)
        #   I → I or E   (I doesn't end an EDU; can't enter another EDU)
        #   E → B or E   (E ends an EDU; next must start a new one)
        "allowed_transitions": (
            (False, True, True),
            (False, True, True),
            (True, False, True),
        ),
        # First token must START an EDU: B (multi-token EDU) or E (1-token EDU).
        "first_token_allowed": (True, False, True),
        # Last token must CLOSE the final EDU: only E.
        "last_token_allowed": (False, False, True),
        # Class weights: B and E are rare boundary tags; I is common.
        "pos_weighted_indices": (0, 2),
    },
    "BO": {
        "tag_names": ("B", "O"),
        "allowed_transitions": ((True, True), (True, True)),
        "first_token_allowed": (True, False),  # first token must be B (an EDU starts)
        "last_token_allowed": (True, True),
        "pos_weighted_indices": (0,),
    },
    "EO": {
        "tag_names": ("E", "O"),
        "allowed_transitions": ((True, True), (True, True)),
        "first_token_allowed": (True, True),
        "last_token_allowed": (True, False),  # last token must be E (last EDU closes)
        "pos_weighted_indices": (0,),
    },
}


class _Segmenter(nn.Module):
    """Per-token EDU-boundary tagger over one of three schemes (BIE/BO/EO).

    Two trainable variants, selected by `loss`:

      "crf": linear-chain CRF with learned transition/start/end scores, trained
             by negative log-likelihood (forward algorithm partition - gold
             score). The scheme's structural masks are added to the learned
             scores at both training and decoding, so the CRF only ever scores
             schema-valid sequences. `pos_weight` is unused.
      "ce":  independent per-token class-weighted cross-entropy. Decoding is the
             same Viterbi but over the structural masks alone (no learned
             transitions). `pos_weight` upweights the rare boundary tag(s).

    Both decode through `_viterbi`, differing only in whether learned
    transitions are added to the structural masks. The CRF objective relies on
    the schemes being designed so every reachable tag has at least one
    structurally-valid predecessor; otherwise a partition reduction could be
    all -inf and NaN out.
    """

    def __init__(self, hidden_size: int, scheme: str, loss: str, pos_weight: float, dropout: float):
        super().__init__()
        if scheme not in _SCHEMES:
            raise ValueError(f"Unknown segmentation scheme: {scheme!r} (expected one of {sorted(_SCHEMES)})")
        if loss not in ("crf", "ce"):
            raise ValueError(f"Unknown segmentation loss: {loss!r} (expected 'crf' or 'ce')")
        info = _SCHEMES[scheme]
        n_tags = len(info["tag_names"])
        self.scheme = scheme
        self.loss_type = loss
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, n_tags)

        weights = torch.ones(n_tags)
        for idx in info["pos_weighted_indices"]:
            weights[idx] = pos_weight
        self.register_buffer("class_weight", weights)

        # Pre-baked structural masks as float tensors with 0 on allowed and
        # `_NEG_INF` on forbidden positions.
        def _to_bias(allowed_nested):
            allowed = torch.tensor(allowed_nested, dtype=torch.bool)
            return torch.where(
                allowed,
                torch.zeros_like(allowed, dtype=torch.float),
                torch.full_like(allowed, _NEG_INF, dtype=torch.float),
            )

        self.register_buffer("trans_bias", _to_bias(info["allowed_transitions"]))
        self.register_buffer("first_bias", _to_bias(info["first_token_allowed"]))
        self.register_buffer("last_bias", _to_bias(info["last_token_allowed"]))

        if loss == "crf":
            # transitions[i, j] = learned score for tag i → tag j.
            self.transitions = nn.Parameter(torch.zeros(n_tags, n_tags))
            self.start_transitions = nn.Parameter(torch.zeros(n_tags))
            self.end_transitions = nn.Parameter(torch.zeros(n_tags))

    def _scores(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Additive (transition, start, end) scores shared by the CRF objective
        and Viterbi. For "ce" these are the structural masks alone; for "crf"
        the learned scores are added on top of the masks."""
        if self.loss_type == "crf":
            return (
                self.transitions + self.trans_bias,
                self.start_transitions + self.first_bias,
                self.end_transitions + self.last_bias,
            )
        return self.trans_bias, self.first_bias, self.last_bias

    def _build_targets(self, num_tokens: int, edu_mapping: list[tuple[int, int]], device: torch.device) -> torch.Tensor:
        """Per-token gold tags for the current scheme.

        Args:
            num_tokens:  total token count in the document.
            edu_mapping: list of (start, end_exclusive) per EDU.

        Returns:
            targets: [num_tokens] long tensor of tag indices.
        """
        if self.scheme == "BIE":
            targets = torch.full((num_tokens,), 1, dtype=torch.long, device=device)  # I=1 fallback
            for start, end in edu_mapping:
                if end - start == 1:
                    targets[start] = 2  # E (1-token EDU)
                else:
                    targets[start] = 0  # B
                    targets[end - 1] = 2  # E
                    # interior already = 1 (I)
        elif self.scheme == "BO":
            targets = torch.full((num_tokens,), 1, dtype=torch.long, device=device)  # O=1
            for start, _ in edu_mapping:
                targets[start] = 0  # B
        else:  # EO
            targets = torch.full((num_tokens,), 1, dtype=torch.long, device=device)  # O=1
            for _, end in edu_mapping:
                targets[end - 1] = 0  # E
        return targets

    def loss(self, embeddings: torch.Tensor, edu_mapping: list[tuple[int, int]]) -> torch.Tensor:
        """Per-token CE ("ce") or CRF negative log-likelihood ("crf") against
        gold tags built from `edu_mapping`. The CRF NLL is normalized by token
        count to keep its magnitude comparable to the (mean) CE loss."""
        num_tokens = embeddings.size(0)
        if num_tokens == 0:
            return torch.zeros((), device=embeddings.device)
        targets = self._build_targets(num_tokens, edu_mapping, embeddings.device)
        logits = self.linear(self.dropout(embeddings))
        if self.loss_type == "ce":
            return F.cross_entropy(logits, targets, weight=self.class_weight)
        # The CRF partition is a logsumexp, which autocast does not promote to
        # fp32. Compute the objective in fp32 (a no-op outside autocast) so the
        # partition stays numerically stable under bf16 training.
        logits = logits.float()
        trans, start, end = self._scores()
        log_partition = _crf_log_partition(logits, trans, start, end)
        gold_score = _crf_gold_score(logits, targets, trans, start, end)
        return (log_partition - gold_score) / num_tokens

    @torch.no_grad()
    def predict_breaks(self, embeddings: torch.Tensor) -> list[int]:
        """Predict EDU end token indices (inclusive) via Viterbi.

        Returns:
            Sorted list of inclusive end positions. The final token is always
            an EDU end (the schema enforces it).
        """
        logits = self.linear(embeddings)
        trans, start, end = self._scores()
        tags = _viterbi(logits, trans, start, end)
        return _tags_to_breaks(tags, self.scheme)


def _crf_log_partition(
    logits: torch.Tensor,
    trans: torch.Tensor,
    start: torch.Tensor,
    end: torch.Tensor,
) -> torch.Tensor:
    """Forward-algorithm log-partition log Z over all schema-valid tag sequences.

    Args:
        logits: [num_tokens, n_tags]  per-token emission scores
        trans:  [n_tags, n_tags]      trans[i, j] = score of tag i → tag j
        start:  [n_tags]              score of starting in each tag
        end:    [n_tags]              score of ending in each tag

    Returns:
        scalar tensor log Z.
    """
    alpha = logits[0] + start  # [n_tags]
    for t in range(1, logits.size(0)):
        # alpha[i] + trans[i, j] over predecessors i, summed in log-space, + emission.
        alpha = torch.logsumexp(alpha.unsqueeze(1) + trans, dim=0) + logits[t]
    return torch.logsumexp(alpha + end, dim=0)


def _crf_gold_score(
    logits: torch.Tensor,
    tags: torch.Tensor,
    trans: torch.Tensor,
    start: torch.Tensor,
    end: torch.Tensor,
) -> torch.Tensor:
    """Unnormalized score of the single gold tag sequence `tags`.

    Gold tags always respect the scheme, so every transition term is finite.
    """
    score = start[tags[0]] + logits[0, tags[0]]
    for t in range(1, logits.size(0)):
        score = score + trans[tags[t - 1], tags[t]] + logits[t, tags[t]]
    return score + end[tags[-1]]


def _viterbi(
    logits: torch.Tensor,
    trans: torch.Tensor,
    start: torch.Tensor,
    end: torch.Tensor,
) -> list[int]:
    """Best (max-score) tag sequence under additive transition/start/end scores.

    Args:
        logits: [num_tokens, n_tags]
        trans:  [n_tags, n_tags]  trans[i, j] = score of tag i → tag j
        start:  [n_tags]          score of starting in each tag
        end:    [n_tags]          score of ending in each tag

    Returns:
        Best tag sequence as list[int] of length num_tokens.
    """
    num_tokens, n_tags = logits.shape
    if num_tokens == 0:
        return []
    alpha = logits[0] + start  # [n_tags]
    backpointers: list[torch.Tensor] = []
    for t in range(1, num_tokens):
        # score[i, j] = alpha[i] + trans[i, j] + logits[t, j]
        scores = alpha.unsqueeze(1) + trans + logits[t].unsqueeze(0)  # [n_tags, n_tags]
        alpha, argmax = scores.max(dim=0)
        backpointers.append(argmax)
    alpha = alpha + end
    last_state = int(alpha.argmax().item())
    tags = [last_state]
    for argmax in reversed(backpointers):
        last_state = int(argmax[last_state].item())
        tags.append(last_state)
    tags.reverse()
    return tags


def _tags_to_breaks(tags: list[int], scheme: str) -> list[int]:
    """Convert a tag sequence into inclusive EDU end positions per the scheme.

    Always force the final token to be a break so the last EDU is closed (the
    Viterbi mask already enforces this for BIE/EO; for BO we enforce it here
    since the BO scheme has no end tag).
    """
    n = len(tags)
    if n == 0:
        return []
    last = n - 1
    if scheme in ("BIE", "EO"):
        end_tag = 2 if scheme == "BIE" else 0  # BIE: E=2; EO: E=0
        breaks = [i for i, t in enumerate(tags) if t == end_tag]
    else:  # BO: starts at B; EDU ends are positions before each B (except the very first)
        starts = [i for i, t in enumerate(tags) if t == 0]
        if not starts:
            return [last]
        breaks = [s - 1 for s in starts[1:]]
        breaks.append(last)
    if not breaks or breaks[-1] != last:
        breaks.append(last)
    return sorted(set(breaks))


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
            edu_mapping: list of (start, end_exclusive) per EDU.

        Returns:
            edu_reprs: [num_edus, H]
        """
        rows = []
        for idx, (b, e) in enumerate(edu_mapping):
            if b >= e:
                # Empty-token EDU (e.g. an `edu_strings[i]` that tokenizes to
                # zero pieces). Surface as a hard error rather than silently
                # NaN-ing through `mean(0)` or `IndexError`-ing on `span[0]`.
                raise ValueError(
                    f"EDU {idx} has an empty token range {(b, e)}; check the "
                    f"upstream tokenizer output for empty / strippable EDU text."
                )
            span = embeddings[b:e]
            first = span[0]
            last = span[-1]
            if self.pooling == "concat":
                pooled = span.mean(0)
            else:
                weights = F.softmax(self.attn_score(span).squeeze(-1), dim=0)  # [span_len]
                pooled = (weights.unsqueeze(-1) * span).sum(0)
            rows.append(torch.cat([first, last, pooled]))
        return self.reduce(self.dropout(torch.stack(rows)))


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

        self.encoder, self.tokenizer, self.max_length = load_encoder_and_tokenizer(config.model_name)

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

        self.label_input_pooling = config.label_input_pooling
        self.decoding = config.decoding
        # None → per-node CE; non-None → margin objective.
        self.margin = config.margin_training.margin if config.margin_training is not None else None

        self.segmenter = (
            _Segmenter(
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

        edu_reprs = self.span_pooler(self.encoder_dropout(embeddings), edu_mapping)
        return edu_reprs, seg_loss

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

        edu_reprs = self.span_pooler(self.encoder_dropout(embeddings), edu_mapping)  # eval mode → dropout identity
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
            edu_reprs = self.span_pooler(dropped, gold_edu_mapping)
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
            edu_reprs = self.span_pooler(dropped, pred_edu_mapping)
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
