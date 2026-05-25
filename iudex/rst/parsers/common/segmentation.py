"""Shared scheme-based EDU-boundary segmenter (BIE / BO / EO, CRF or CE).

Used by any parser that does joint EDU segmentation: piudotto by default, and
dmrst when its `segmentation.scheme` is set. Factored here (rather than copied)
because it's a self-contained `nn.Module` reused verbatim across parsers, the
kind of thing we now share at the second use. The per-parser training loops, by
contrast, stay duplicated for self-contained reading.
"""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

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


class Segmenter(nn.Module):
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
    emit = logits[torch.arange(logits.size(0), device=logits.device), tags].sum()
    trans_score = trans[tags[:-1], tags[1:]].sum()
    return start[tags[0]] + end[tags[-1]] + emit + trans_score


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
