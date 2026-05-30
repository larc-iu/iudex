import torch
import torch.nn as nn
import torch.nn.functional as F

from iudex.rst.data.reader import determine_label_index
from iudex.rst.data.tree import Reduce, RstTree, Shift
from iudex.rst.parsers.common.biaffine import DeepBiAffine, FeedForward
from iudex.rst.parsers.common.encoding import (
    encode_tokens_strided,
    load_encoder_and_tokenizer,
    tokenize_edus,
)
from iudex.rst.parsers.sr_biaffine.configuration_sr_biaffine import SRBiaffineConfig

# Action head output indices. The transition system has exactly two structural
# actions. The reduce's (nuclearity, relation) label is a separate head.
_SHIFT, _REDUCE = 0, 1


class SRBiaffineParser(nn.Module):
    """Transition-based (shift-reduce) RST parser, the bottom-up sibling of
    `topdown_biaffine`. Both come from Kobayashi et al. (2022) "A Simple and
    Strong Baseline for End-to-End Neural RST-style Discourse Parsing"
    (Findings of EMNLP 2022), which scores one shared span representation under
    three decoding strategies. This is the transition-based one. Assumes gold
    EDU segmentation.

    State: a stack of EDU spans `(b, e)` plus a queue of not-yet-shifted EDUs
    (a single cursor `q`, since EDUs shift in text order). At each step the
    parser reads the top two stack spans (s1 = top, s2 = second) and the
    queue-front EDU (q1), then:
      - the action head (an FFN over the concatenated s1/s2/q1 span reprs)
        chooses SHIFT vs REDUCE (a 2-way decision, like the reference's V1
        action head)
      - on REDUCE, the deep-biaffine label head over (s2, s1) picks the joint
        (nuclearity, relation) label, with s2 the left child and s1 the right

    Nuclearity thus lives in the label head, folded into the portmanteau
    `{nuc}_{rel}` label space (matching `topdown_biaffine`). The reference's
    recommended variant instead folds nuclearity into a joint shift/reduce
    `act_nuc` action head and keeps relation separate. See the README.

    A span `(b, e)` is represented as the mean of its first and last subtoken
    embeddings (matching `topdown_biaffine` and the reference implementation).
    Absent state slots (empty stack below the top, exhausted queue) contribute
    a zero vector, as in the reference.
    """

    def __init__(self, config: SRBiaffineConfig, *, compile_encoder: bool = False):
        super().__init__()
        self.config = config
        self.label_index = determine_label_index(config.relation_types)
        self.stride = config.stride

        self.encoder, self.tokenizer, self.max_length = load_encoder_and_tokenizer(
            config.model_name, peft_config=config.peft
        )
        self.hidden_size = self.encoder.config.hidden_size

        # Compile the encoder forward (not the module) so state_dict keys are
        # unchanged and existing checkpoints still load. dynamic=True avoids
        # per-shape recompiles on variable-length documents. Off by default
        # (inference); training opts in, predict opts in via --compile-encoder.
        if compile_encoder and torch.cuda.is_available():
            self.encoder.forward = torch.compile(self.encoder.forward, dynamic=True)

        # Action head sees the three state spans (s1, s2, q1) concatenated.
        self.action_head = FeedForward(3 * self.hidden_size, config.action_ffn_hidden_size, 2, config.dropout)
        self.label_biaffine = DeepBiAffine(
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
        compile_encoder: bool = False,
    ) -> "SRBiaffineParser":
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
            config_cls=SRBiaffineConfig,
            device=dev,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
            compile_encoder=compile_encoder,
        )

    def _encode_tree(self, tree: RstTree) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns:
        embeddings: [num_tokens, hidden_size]
        edu_boundaries: [num_edus, 2], (start_token, end_token_exclusive) per EDU
        """
        input_ids, boundaries = tokenize_edus(self.tokenizer, tree.edu_strings, self.device)
        embeddings = encode_tokens_strided(
            self.encoder, self.tokenizer, input_ids, self.max_length, self.stride
        ).float()
        edu_boundaries = torch.tensor(boundaries, dtype=torch.long, device=self.device)
        return embeddings, edu_boundaries

    def _span_reprs(
        self,
        embeddings: torch.Tensor,
        edu_boundaries: torch.Tensor,
        spans: list[tuple[int, int] | None],
    ) -> torch.Tensor:
        """Span representations for a batch of EDU spans.

        Each present span `(b, e)` is the mean of its first subtoken embedding
        (start of EDU b) and its last subtoken embedding (end of EDU e-1). An
        absent span (`None`, i.e. an empty stack slot or exhausted queue) is a
        zero vector.

        Returns: [len(spans), hidden_size]
        """
        out = embeddings.new_zeros(len(spans), self.hidden_size)
        present = [i for i, s in enumerate(spans) if s is not None]
        if present:
            first_idx = edu_boundaries[[spans[i][0] for i in present], 0]
            last_idx = edu_boundaries[[spans[i][1] - 1 for i in present], 1] - 1
            out[present] = (embeddings[first_idx] + embeddings[last_idx]) / 2
        return out

    def forward(self, tree: RstTree) -> dict[str, torch.Tensor]:
        """Teacher-forced loss for one gold tree.

        We replay the gold shift-reduce action sequence, recording the parser
        state (s1, s2, q1) seen before each action. The action head is trained
        on SHIFT/REDUCE over every step, the label head only on the reduce
        steps. Returns {"loss": (action_loss + label_loss) / 2}.
        """
        num_edus = len(tree.edus)
        if num_edus < 2:
            return {"loss": torch.zeros((), device=self.device, requires_grad=True)}

        embeddings, edu_boundaries = self._encode_tree(tree)

        # Replay the oracle, collecting per-step state spans and targets.
        stack: list[tuple[int, int]] = []
        q = 0
        s1s, s2s, q1s, action_targets = [], [], [], []
        reduce_left, reduce_right, label_targets = [], [], []
        for action in tree.to_shift_reduce(ltr=True):
            s1 = stack[-1] if len(stack) >= 1 else None
            s2 = stack[-2] if len(stack) >= 2 else None
            q1 = (q, q + 1) if q < num_edus else None
            s1s.append(s1)
            s2s.append(s2)
            q1s.append(q1)
            if isinstance(action, Shift):
                action_targets.append(_SHIFT)
                stack.append((q, q + 1))
                q += 1
            else:  # Reduce: s1 and s2 are guaranteed present by SR validity.
                action_targets.append(_REDUCE)
                reduce_left.append(s2)  # left child
                reduce_right.append(s1)  # right child
                label_targets.append(self.label_index.index(f"{action.nuc}_{action.rel}"))
                stack.pop()
                stack.pop()
                stack.append((s2[0], s1[1]))

        action_feats = torch.cat(
            [
                self._span_reprs(embeddings, edu_boundaries, s1s),
                self._span_reprs(embeddings, edu_boundaries, s2s),
                self._span_reprs(embeddings, edu_boundaries, q1s),
            ],
            dim=-1,
        )  # [num_steps, 3 * hidden_size]
        action_logits = self.action_head(action_feats)
        action_loss = F.cross_entropy(action_logits, torch.tensor(action_targets, device=self.device))

        left = self._span_reprs(embeddings, edu_boundaries, reduce_left)
        right = self._span_reprs(embeddings, edu_boundaries, reduce_right)
        label_logits = self.label_biaffine(left, right)  # [num_reduces, num_labels]
        label_loss = F.cross_entropy(label_logits, torch.tensor(label_targets, device=self.device))

        return {"loss": (action_loss + label_loss) / 2}

    @torch.no_grad()
    def predict(self, tree: RstTree) -> RstTree:
        """Greedy shift-reduce decode using gold EDU segmentation from `tree.edus`."""
        self.eval()
        num_edus = len(tree.edus)
        if num_edus < 2:
            return RstTree.from_parsing_actions([], tree.edus, relation_types=self.config.relation_types)

        embeddings, edu_boundaries = self._encode_tree(tree)

        stack: list[tuple[int, int]] = []
        q = 0
        actions: list[Shift | Reduce] = []
        # End state: the whole document reduced to one span, queue exhausted.
        while not (len(stack) == 1 and q >= num_edus):
            s1 = stack[-1] if len(stack) >= 1 else None
            s2 = stack[-2] if len(stack) >= 2 else None
            q1 = (q, q + 1) if q < num_edus else None

            s1r, s2r, q1r = self._span_reprs(embeddings, edu_boundaries, [s1, s2, q1])
            action_logits = self.action_head(torch.cat([s1r, s2r, q1r]))
            if q >= num_edus:  # queue exhausted: shift no longer legal
                action_logits[_SHIFT] = float("-inf")
            if len(stack) < 2:  # need two spans to reduce
                action_logits[_REDUCE] = float("-inf")

            if action_logits.argmax().item() == _SHIFT:
                stack.append((q, q + 1))
                q += 1
                actions.append(Shift())
            else:
                reps = self._span_reprs(embeddings, edu_boundaries, [s2, s1])  # left=s2, right=s1
                label_logits = self.label_biaffine(reps[0:1], reps[1:2])[0]
                nuc, rel = self.label_index[label_logits.argmax().item()].split("_", 1)
                actions.append(Reduce(nuc=nuc, rel=rel))
                stack.pop()
                stack.pop()
                stack.append((s2[0], s1[1]))

        return RstTree.from_shift_reduce(actions, edus=tree.edu_strings, relation_types=self.config.relation_types)
