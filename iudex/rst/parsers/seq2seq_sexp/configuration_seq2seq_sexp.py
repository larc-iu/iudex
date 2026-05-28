from dataclasses import dataclass, field

from tonga import FromParams

from iudex.rst.parsers.common.config import parse_config_dict


@dataclass
class _PeftConfig(FromParams):
    """LoRA fine-tuning of the seq2seq stack. Mirrors `Seq2SeqSRConfig._PeftConfig`.

    When `use_copy=True`, the lm_head is replaced at parser-init with a
    fresh, small Linear projecting to just the action vocab (~100 dims),
    so the old tied-weight issue between `embed_tokens` and
    `lm_head.out_proj` is gone. Only `embed_tokens` needs `modules_to_save`
    to keep the newly-added action-token rows trainable. By default we
    also mask gradients on the old pretrained rows of `embed_tokens` so
    only the ~100 new rows update (see
    `Seq2SeqSexpParser._mask_old_embedding_gradients`).

    When `use_copy=False`, the full pretrained lm_head stays in place
    (source subwords are predicted natively), and
    `train_only_new_embedding_rows` is auto-overridden to False (see
    `Seq2SeqSexpConfig.__post_init__`): the full head must learn to
    predict source ids, which requires all embedding rows to remain
    trainable.
    """

    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: str | list[str] = "all-linear"
    bias: str = "none"
    dora: bool = False
    # Module names whose full weights train alongside the LoRA adapters.
    # `embed_tokens` covers the input embedding (shared encoder/decoder via
    # `_retie_modules_to_save`). The lm_head is no longer in this list
    # because we replace it with a fresh small head and train it fully on
    # its own. Out of `modules_to_save` so PEFT doesn't wrap-and-copy it.
    modules_to_save: list[str] = field(default_factory=lambda: ["embed_tokens"])
    # When True (default), register a backward hook on the trainable
    # embed_tokens Parameter that zeros gradients for rows < original
    # vocab size. Only the newly-added action-token rows accumulate
    # gradient, the pretrained vocabulary embeddings stay frozen. A
    # regularization win on small datasets (old embeddings have been
    # trained on trillions of tokens and shouldn't drift under fine-tuning
    # on ~150 docs).
    train_only_new_embedding_rows: bool = True

    def __post_init__(self):
        if self.r < 1:
            raise ValueError(f"_PeftConfig.r must be >= 1 (got {self.r})")


@dataclass
class Seq2SeqSexpConfig(FromParams):
    train_dir: str
    dev_dir: str
    test_dir: str | None = None

    # Inferred at training time. Persisted so predict / from_pretrained know
    # the action vocabulary to register on the tokenizer.
    relation_types: list[tuple[str, str]] | None = None
    relation_map: dict[str, str] | None = None

    # Model
    model_name: str = "google/t5gemma-2-1b-1b"
    # Encoder input length (raw document subword tokens, no specials counted).
    max_input_length: int = 4096
    # Decoder target length. sexp adds ~2 tokens per internal node ('(' and
    # ')') plus 1 per leaf token vs the SR cousin's ~1 per reduce + 1 per
    # shift. The 8192 budget (vs SR's 5120) is matched to the expected
    # serialization length and is deliberately divergent.
    max_output_length: int = 8192
    # Required on a 4090 at T5Gemma-1B + 4K input + 8K output. Enters the
    # hash so flipping it triggers a fresh run.
    gradient_checkpointing: bool = True

    # LoRA. Null = full fine-tuning. When set, the base seq2seq stack is
    # frozen and only LoRA adapters + the modules in `peft.modules_to_save`
    # train. For T5Gemma2 / mT5 at 1B+ scale this is the practical default
    # since full FT blows up optimizer memory.
    peft: _PeftConfig | None = None

    # Training
    lr: float = 3e-5
    weight_decay: float = 0.01
    max_epochs: int = 10
    batch_size: int = 1
    grad_accum: int = 16
    # "adamw": standard, but two fp32-or-bf16 state tensors per param (~16-32 GB
    # for T5Gemma 2 1B-1B).
    # "adafactor": T5 paper's optimizer, factored 2nd moment, ~50 MB of state
    # total. Use this when the AdamW footprint OOMs.
    optimizer: str = "adafactor"
    num_warmup_steps: int | None = None
    max_grad_norm: float = 1.0
    amp: bool = True
    patience: int = 5
    log_every: int = 5
    validate_every: int | None = None
    checkpoint_every: int | None = None
    checkpoint_dir: str = "checkpoints"
    run_name: str | None = None
    seed: int = 42
    val_metric_name: str = "e2e_full_f1"

    # Decoding
    num_beams: int = 4
    use_validity_constraints: bool = True
    eval_decode_greedy: bool = True

    # Minimum number of content tokens required in a leaf before its close
    # paren becomes legal at decode time (inference-only, has no effect on
    # training). 1 is no constraint. Bump to 2 or 3 to suppress over-
    # segmentation (single-token EDUs out of one gold span). Exception: at
    # end-of-source the leaf may close even when below the threshold,
    # otherwise the final EDU can't commit.
    min_edu_length: int = 1

    # Cap per-epoch dev eval to the first N documents (in directory order) to
    # speed up training-time validation. Autoregressive generation at L>3K on
    # a 1B decoder is ~1 min/doc even with KV cache, full GUM dev (32 docs)
    # costs ~30 min/epoch. None = use the full dev set every epoch. The FINAL
    # dev/test eval always runs on the full split regardless of this setting.
    dev_max_docs: int | None = None

    # Batch size for dev/test predictions. Each batch shares one decoder pass
    # (weights stream from HBM once, KV cache strides across the batch) so
    # this is bandwidth-bound rather than compute-bound and roughly linear
    # in batch_size up to memory. 1 = original per-document loop.
    dev_batch_size: int = 1

    # Multiplier on the gradient contribution of structural-action positions
    # (open/close parens, labels) in the training loss. Default 1.0 (no
    # rebalance) since the replacement lm_head projects to ~100 action
    # classes. Copy CE is on the same scale as structural CE and the old
    # 262K-vocab justification for upweighting structurals doesn't apply.
    # Bump above 1.0 only if action positions are demonstrably starved.
    action_loss_weight: float = 1.0

    # Label smoothing on the CE loss. Standard seq2seq fine-tuning trick.
    # The action head is small (~100 classes) and GUM has ~150 train docs,
    # so hard targets overfit fast. 0.1 is the conventional default.
    label_smoothing: float = 0.1

    # Sexp-specific. The first is also the registry signature_field for
    # this parser (`traversal_order` is unique to sexp parsers).
    traversal_order: str = "postorder"
    # Controls how content positions are masked at decode time. Only
    # applies when `use_copy=False`. True (default) hard-masks content
    # positions to `source_ids[cursor]` (COPY-via-constraint, current
    # behavior). False admits any non-structural id at content positions
    # (free generation, closer to Hu and Wan 2023's apparent setup where
    # the model learns to copy via attention). Requires `use_copy=False`;
    # raises if both are True.
    constrain_content: bool = True
    # When True (default), action vocab includes a `<copy>` token, source
    # subwords are replaced by `<copy>` at training time, the lm_head is
    # replaced with a small fresh `Linear(hidden, head_vocab_size)` over
    # the action vocab (~100 classes), and the predict path substitutes
    # the current source id into the next decoder input when `<copy>` is
    # emitted. When False, source subwords appear verbatim in-stream and
    # the full pretrained lm_head is kept so source ids are scorable
    # natively. This mirrors Hu and Wan 2023 (TASLP, "RST Discourse
    # Parsing as Text-to-Text Generation"). Note the inherent asymmetry:
    # the head architecture differs across modes by design (no-copy
    # MUST keep the full head to score source ids), so a COPY ablation
    # conflates "having COPY" with "enables a small head". This is a
    # property of the COPY mechanism itself, not a confound we can
    # eliminate without breaking one mode or the other.
    use_copy: bool = True

    def __post_init__(self):
        if self.traversal_order not in ("preorder", "postorder"):
            raise ValueError(f"traversal_order must be 'preorder' or 'postorder' (got {self.traversal_order!r})")
        if self.use_copy and not self.constrain_content:
            raise ValueError(
                "constrain_content=False requires use_copy=False (free-content decoding "
                "is only meaningful when source subwords are scored natively by the lm_head)."
            )
        if self.use_copy is False and self.peft is not None and self.peft.train_only_new_embedding_rows:
            # Mutates self.peft in place. The override is durable because
            # `dataclasses.asdict(self)` serializes the post-init state into
            # `config.json` and into checkpoint hashes.
            from iudex.common.log import warn as _warn

            _warn(
                "[CONFIG OVERRIDE] use_copy=False is incompatible with train_only_new_embedding_rows=True. "
                "Auto-overriding to False so the full lm_head and embedding rows can train. "
                "Set explicitly in your config to silence."
            )
            self.peft.train_only_new_embedding_rows = False

    @classmethod
    def from_dict(cls, d: dict) -> "Seq2SeqSexpConfig":
        return parse_config_dict(cls, d)
