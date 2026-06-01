from dataclasses import dataclass, field

from tonga import FromParams

from iudex.rst.parsers.common.config import parse_config_dict


@dataclass
class _PeftConfig(FromParams):
    """LoRA fine-tuning of the causal LM. Mirrors `DecoderOnlySRConfig._PeftConfig`.

    Under `use_copy=True` the lm_head is replaced with a small fresh head and
    the input embedding stays fully trainable with pretrained-row gradients
    zeroed (`seqgen.mask_old_embedding_gradients`). `modules_to_save` /
    `train_only_new_embedding_rows` have no effect.

    Under `use_copy=False` the full pretrained tied lm_head is kept (source
    subwords scored natively), so `modules_to_save=['embed_tokens']` is honored
    to keep all embedding rows trainable, and `train_only_new_embedding_rows`
    is auto-overridden to False (see `DecoderOnlySexpConfig.__post_init__`).
    """

    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: str | list[str] = "all-linear"
    bias: str = "none"
    dora: bool = False
    # Honored only under use_copy=False (keeps all embedding rows trainable so
    # the full tied lm_head can score source ids). Ignored under use_copy=True.
    modules_to_save: list[str] = field(default_factory=lambda: ["embed_tokens"])
    # No-op under use_copy=True (the carve-out always trains only new rows).
    # Auto-forced to False under use_copy=False by the config __post_init__.
    train_only_new_embedding_rows: bool = True

    def __post_init__(self):
        if self.r < 1:
            raise ValueError(f"_PeftConfig.r must be >= 1 (got {self.r})")


@dataclass
class DecoderOnlySexpConfig(FromParams):
    train_dir: str
    dev_dir: str
    test_dir: str | None = None

    # Inferred at training time. Persisted so predict / from_pretrained know
    # the action vocabulary to register on the tokenizer.
    relation_types: list[tuple[str, str]] | None = None
    relation_map: dict[str, str] | None = None

    # Model. Default is the smallest publicly-released Gemma 3 instruction-
    # tuned checkpoint. The parser is architecture-agnostic and works with
    # any AutoModelForCausalLM (Llama, Qwen, etc.) as long as the tokenizer
    # exposes character-offset mapping for SentencePiece-style alignment.
    model_name: str = "google/gemma-3-1b-it"

    # Single-stream layout, so length budgets must accommodate
    # source + actions + 2 specials (BOS + SEP) in one sequence. Naming
    # mirrors decoder_only_sr so per-side caps stay readable. The trainer
    # enforces the combined length internally.
    max_input_length: int = 3072
    # sexp adds ~2 tokens per internal node ('(' and ')') plus 1 per leaf
    # token vs the SR cousin's ~1 per reduce + 1 per shift. The 5120 budget
    # matches decoder_only_sr (the per-stream growth happens to fit in the
    # same envelope at the GUM/RST-DT input ranges we run on).
    max_output_length: int = 5120
    gradient_checkpointing: bool = False

    # Distinguishes this parser's configs from the seq2seq_sexp side. The
    # signature_field for runs-list inference uses `use_copy`, but the
    # disambiguator falls back to default-fields match, where `causal_mode`
    # pulls this config into the decoder-only column.
    causal_mode: bool = True

    # LoRA. Null = full fine-tuning. When set, the base causal LM is frozen
    # and only LoRA adapters + the modules in `peft.modules_to_save` train.
    # For Gemma3 1B and up this is the practical default since full FT
    # blows up optimizer memory.
    peft: _PeftConfig | None = None

    # Training
    lr: float = 3e-5
    weight_decay: float = 0.01
    max_epochs: int = 10
    batch_size: int = 1
    grad_accum: int = 16
    # "adamw": standard, but two state tensors per param. "adafactor": T5
    # paper's optimizer, factored 2nd moment. Use this when the AdamW
    # footprint OOMs.
    optimizer: str = "adafactor"
    num_warmup_steps: int | None = None
    max_grad_norm: float = 1.0
    amp: bool = True
    patience: int = 5
    log_every: int = 5
    validate_every: int | None = None
    # Skip dev validation until this epoch (0 = validate from the start).
    # Generative parsers decode every dev doc to max_output_length while
    # undertrained, so early evals cost hours for a ~0 score. In HASH_EXCLUDE,
    # so changing it is resume-safe.
    begin_validation_epoch: int = 0
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
    # segmentation. Exception: at end-of-source the leaf may close even
    # when below the threshold, otherwise the final EDU can't commit.
    min_edu_length: int = 1

    # Cap per-epoch dev eval to the first N documents (in directory order)
    # to speed up training-time validation. None = use the full dev set
    # every epoch. The FINAL dev/test eval always runs on the full split
    # regardless of this setting.
    dev_max_docs: int | None = None

    # Batch size for dev/test predictions. KV-cache stride amortizes the
    # decoder weight stream across the batch, so this is bandwidth-bound
    # and roughly linear in batch_size up to memory.
    dev_batch_size: int = 1

    # Multiplier on the gradient contribution of structural-action
    # positions (open/close parens, labels) in the training loss. Default
    # 1.0 (no rebalance) since the replacement lm_head projects to ~100
    # action classes. Bump above 1.0 only if action positions are
    # demonstrably starved.
    action_loss_weight: float = 1.0

    # Label smoothing on the CE loss. Standard fine-tuning trick. The
    # action head is small (~100 classes) and GUM has ~150 train docs,
    # so hard targets overfit fast. Applied uniformly at the configured
    # value in both use_copy modes (no auto-scale). Under use_copy=False the
    # head is the full vocab rather than ~100 action classes, a known
    # head-size confound on the per-off-class smoothing mass.
    label_smoothing: float = 0.1

    # S-expression knobs. `use_copy` is the registry's signature_field.
    traversal_order: str = "postorder"
    # Controls how content positions are masked at decode time. Only
    # applies when `use_copy=False`. True (default) hard-masks content
    # positions to `source_ids[cursor]` (COPY-via-constraint, current
    # behavior). False admits any non-structural id at content positions
    # (free generation, closer to Hu and Wan 2023's apparent setup where
    # the model learns to copy via attention). Requires `use_copy=False`.
    # Raises if both are True.
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
    def from_dict(cls, d: dict) -> "DecoderOnlySexpConfig":
        return parse_config_dict(cls, d)
