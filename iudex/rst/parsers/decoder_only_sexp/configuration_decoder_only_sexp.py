from dataclasses import dataclass, field

from tonga import FromParams

from iudex.rst.parsers.common.config import parse_config_dict


@dataclass
class _PeftConfig(FromParams):
    """LoRA fine-tuning of the causal LM. Mirrors `DecoderOnlySRConfig._PeftConfig`.

    Action-token embedding rows are added via `resize_token_embeddings` and
    `modules_to_save=['embed_tokens']` keeps them trainable. The lm_head is
    handled at parser init: when `use_copy=True` it's replaced with a small
    fresh head over the action vocab. When `use_copy=False` it would stay
    as the pretrained full-vocab head (we'd have to predict source subwords),
    but that mode is not currently supported (see config docstring).
    """

    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: str | list[str] = "all-linear"
    bias: str = "none"
    dora: bool = False
    # Module names whose full weights train alongside the LoRA adapters.
    # `embed_tokens` keeps the newly-added action-token rows trainable.
    # Out of `modules_to_save` so PEFT doesn't wrap-and-copy it.
    modules_to_save: list[str] = field(default_factory=lambda: ["embed_tokens"])
    # When True (default), register a backward hook on the trainable
    # embed_tokens Parameter that zeros gradients for rows < original
    # vocab size. Only the newly-added action-token rows accumulate
    # gradient, the pretrained vocabulary embeddings stay frozen.
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

    # S-expression knobs. `use_copy` is the registry's signature_field.
    traversal_order: str = "postorder"
    # True only for now. use_copy=False is future work (see the paper's
    # limitations section): it confounds COPY-mode ablations with lm_head
    # architecture changes (asymmetric head between modes).
    use_copy: bool = True

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
    # so hard targets overfit fast. 0.1 is the conventional default.
    label_smoothing: float = 0.1

    def __post_init__(self):
        if self.traversal_order not in ("preorder", "postorder"):
            raise ValueError(f"traversal_order must be 'preorder' or 'postorder' (got {self.traversal_order!r})")
        if self.use_copy is False:
            raise NotImplementedError(
                "DecoderOnlySexpConfig: use_copy=False is not implemented. "
                "The two modes use asymmetric lm_heads (small vs full vocab), "
                "which confounds the ablation. Set use_copy=True (the canonical mode)."
            )

    @classmethod
    def from_dict(cls, d: dict) -> "DecoderOnlySexpConfig":
        return parse_config_dict(cls, d)
