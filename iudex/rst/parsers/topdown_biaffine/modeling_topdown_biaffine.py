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
from iudex.rst.parsers.topdown_biaffine.configuration_topdown_biaffine import TopdownBiaffineConfig


class _FeedForward(nn.Sequential):
    """Two-layer GELU feed-forward block with dropout between the layers."""

    def __init__(self, input_dim, hidden_dim, output_dim, dropout_p):
        super().__init__(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, output_dim),
        )


class _DeepBiAffine(nn.Module):
    """Deep biaffine scorer used for both split and label decisions.

    Each side is projected with its own FFN, then combined as a bilinear term
    plus per-side linear terms (a.k.a. the deep biaffine of Dozat & Manning).

    Args:
        h_left:  [num_candidates, input_dim]
        h_right: [num_candidates, input_dim]

    Returns:
        scores: [num_candidates, output_dim]
    """

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
    """Top-down RST parser with biaffine split and label scoring."""

    def __init__(self, config: TopdownBiaffineConfig):
        super().__init__()
        self.config = config
        self.label_index = determine_label_index(config.relation_types)
        self.stride = config.stride

        self.encoder, self.tokenizer, self.max_length = load_encoder_and_tokenizer(config.model_name)
        self.hidden_size = self.encoder.config.hidden_size

        # Compile the encoder forward (not the module) so state_dict keys are
        # unchanged and existing checkpoints still load. dynamic=True avoids
        # per-shape recompiles on variable-length documents.
        if torch.cuda.is_available():
            self.encoder.forward = torch.compile(self.encoder.forward, dynamic=True)

        self.split_biaffine = _DeepBiAffine(self.hidden_size, config.ffn_hidden_size, 1, config.dropout)
        self.label_biaffine = _DeepBiAffine(
            self.hidden_size, config.ffn_hidden_size, len(self.label_index), config.dropout
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
    ) -> "TopdownBiaffineParser":
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
            config_cls=TopdownBiaffineConfig,
            device=dev,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
        )

    def _encode_tree(self, tree: RstTree) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize the tree's EDUs and return their token-level embeddings.

        Returns:
            embeddings:     shape [num_tokens, hidden_size]
            edu_boundaries: shape [num_edus, 2], each a pair of (start_token, end_token_exclusive)
        """
        input_ids, boundaries = tokenize_edus(self.tokenizer, tree.edu_strings, self.device)
        embeddings = encode_tokens_strided(
            self.encoder, self.tokenizer, input_ids, self.max_length, self.stride
        ).float()
        edu_boundaries = torch.tensor(boundaries, dtype=torch.long, device=self.device)
        return embeddings, edu_boundaries

    def _subspan_reprs_per_split(
        self,
        embeddings: torch.Tensor,
        edu_boundaries: torch.Tensor,
        b: int,
        e: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build left/right sub-span representations for every candidate split.

        For the span of EDUs [b, e) there are `num_splits = e - b - 1` candidate
        split points k ∈ [b+1, e), where k means "this EDU and all subsequent EDUs
        in the span will be split from all edus in the span less than k".

        For each split point k, we represent the left sub-span (EDUs [b, k)) and
        the right sub-span (EDUs [k, e)) as the average of their first and last
        token embeddings. We compute both representations for all candidate
        splits in one batched pass.

        Args:
            embeddings:     [num_tokens, hidden_size]
            edu_boundaries: [num_edus, 2]  rows of (start_token, end_token_exclusive)
            b, e:           EDU range of the span, exclusive at `e`

        Returns:
            packed_l: [num_splits, hidden_size]  left sub-span repr per split
            packed_r: [num_splits, hidden_size]  right sub-span repr per split
        """
        # The first token of the whole span is the first token of every LEFT sub-span;
        # the last token of the whole span is the last token of every RIGHT sub-span.
        span_first_h = embeddings[edu_boundaries[b, 0]]  # [hidden_size]
        span_last_h = embeddings[edu_boundaries[e - 1, 1] - 1]  # [hidden_size]

        # For each candidate split k ∈ [b+1, e):
        #   - last token of the LEFT sub-span  = last token of EDU (k - 1)
        #   - first token of the RIGHT sub-span = first token of EDU k
        ks = torch.arange(b + 1, e, device=embeddings.device)
        left_last_idx = edu_boundaries[ks - 1, 1] - 1  # remember -1 because end index is exclusive
        right_first_idx = edu_boundaries[ks, 0]

        # Average the two endpoints of each sub-span. Broadcasting: span_first_h
        # has shape [hidden_size] and is added elementwise to each row of
        # embeddings[left_last_idx], which has shape [num_splits, hidden_size].
        packed_l = (span_first_h + embeddings[left_last_idx]) / 2
        packed_r = (embeddings[right_first_idx] + span_last_h) / 2
        return packed_l, packed_r

    def forward(self, tree: RstTree) -> dict[str, torch.Tensor]:
        """Get teacher-forced loss for one gold tree.

        At each non-leaf span [b, e) we score every candidate split (split loss)
        and the (nuclearity, relation) label at the *gold* split (label loss),
        then recurse into the gold sub-spans, top-down.

        Returns:
            {"loss": scalar tensor} — mean of (split_loss + label_loss) / 2
        """
        num_edus = len(tree.edus)
        if num_edus < 2:
            return {"loss": torch.zeros((), device=self.device, requires_grad=True)}

        embeddings, edu_boundaries = self._encode_tree(tree)

        # Build a lookup: gold span (b, e) → (gold split point, gold label).
        gold_decisions: dict[tuple[int, int], tuple[int, str]] = {}
        for (left_range, right_range), nuc, rel in tree.spans_with_ranges():
            gold_decisions[(left_range[0], right_range[1])] = (right_range[0], f"{nuc}_{rel}")

        split_losses, label_losses = [], []
        stack = [(0, num_edus)]
        while stack:
            b, e = stack.pop()
            if e - b <= 1:  # single-EDU span: no decision to make
                continue

            packed_l, packed_r = self._subspan_reprs_per_split(embeddings, edu_boundaries, b, e)
            split_logits = self.split_biaffine(packed_l, packed_r).squeeze(-1)  # [num_splits]
            label_logits = self.label_biaffine(packed_l, packed_r)  # [num_splits, num_labels]

            gold_split, gold_label_str = gold_decisions[(b, e)]
            # Absolute EDU index → candidate-split index: gold_split ∈ [b+1, e)
            # maps to [0, num_splits - 1].
            gold_split_idx = gold_split - b - 1
            gold_label_idx = self.label_index.index(gold_label_str)

            # 2-EDU spans have a single forced split — no choice, no split loss.
            if e - b > 2:
                split_target = torch.tensor([gold_split_idx], device=self.device)
                split_losses.append(F.cross_entropy(split_logits.unsqueeze(0), split_target))

            label_target = torch.tensor([gold_label_idx], device=self.device)
            label_losses.append(F.cross_entropy(label_logits[gold_split_idx].unsqueeze(0), label_target))

            # Recurse into the gold sub-spans (teacher forcing).
            stack.append((gold_split, e))
            stack.append((b, gold_split))

        split_loss = sum(split_losses) / len(split_losses) if split_losses else torch.zeros((), device=self.device)
        label_loss = sum(label_losses) / len(label_losses)
        return {"loss": (split_loss + label_loss) / 2}

    @torch.no_grad()
    def predict(self, tree: RstTree) -> RstTree:
        """Greedy top-down decode using gold EDU segmentation from `tree.edus`.

        At each span [b, e), pick the argmax split and argmax label; recurse
        into both sub-spans. Returns a new tree built from the parsing actions.
        """
        self.eval()
        num_edus = len(tree.edus)
        if num_edus < 2:
            return RstTree.from_parsing_actions([], tree.edus, relation_types=self.config.relation_types)

        embeddings, edu_boundaries = self._encode_tree(tree)

        actions = []
        queue = [(0, num_edus)]
        while queue:
            b, e = queue.pop(0)
            if e - b <= 1:
                continue

            packed_l, packed_r = self._subspan_reprs_per_split(embeddings, edu_boundaries, b, e)
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

        return RstTree.from_parsing_actions(actions, tree.edus, relation_types=self.config.relation_types)
