from dataclasses import dataclass, field

from tonga import FromParams

from iudex.rst.parsers.common.config import parse_config_dict


@dataclass
class _PeftConfig(FromParams):
    """LoRA fine-tuning of the causal LM. Mirrors `Seq2SeqSRConfig._PeftConfig`.

    The lm_head is replaced at parser-init with a fresh, small Linear over
    the action vocab. The input embedding freezes its pretrained base and
    trains a small new-rows Parameter (see
    `DecoderOnlySRParser._carve_new_token_embeddings`).
    """

    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: str | list[str] = "all-linear"
    bias: str = "none"
    dora: bool = False
    # No-ops, kept so existing jsonnets load. The input embedding is no longer
    # routed through PEFT modules_to_save (frozen base + small new-rows
    # Parameter instead), so neither field has any effect.
    modules_to_save: list[str] = field(default_factory=lambda: ["embed_tokens"])
    train_only_new_embedding_rows: bool = True

    def __post_init__(self):
        if self.r < 1:
            raise ValueError(f"_PeftConfig.r must be >= 1 (got {self.r})")


@dataclass
class DecoderOnlySRConfig(FromParams):
    train_dir: str
    dev_dir: str
    test_dir: str | None = None

    relation_types: list[tuple[str, str]] | None = None
    relation_map: dict[str, str] | None = None

    # Model. Default is the smallest publicly-released Gemma 3 instruction-
    # tuned checkpoint; the parser is architecture-agnostic and works with
    # any AutoModelForCausalLM (Llama, Qwen, etc.) as long as the tokenizer
    # exposes character-offset mapping for SentencePiece-style alignment.
    model_name: str = "google/gemma-3-1b-it"

    # Single-stream layout, so length budgets must accommodate
    # source + actions + 2 specials (BOS + SEP) in one sequence. Naming
    # mirrors seq2seq_sr so per-side caps stay readable; the trainer
    # enforces the combined length internally.
    max_input_length: int = 3072
    max_output_length: int = 5120
    gradient_checkpointing: bool = False

    # Distinguishes this parser's configs from any other (the registry's
    # `signature_field` needs a unique field name per parser kind). Reading
    # it as True here is no-op information — it exists to tag a config.json
    # as belonging to decoder_only_sr.
    causal_mode: bool = True

    peft: _PeftConfig | None = None

    # Training
    lr: float = 3e-5
    weight_decay: float = 0.01
    max_epochs: int = 10
    batch_size: int = 1
    grad_accum: int = 16
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

    min_edu_length: int = 1

    dev_max_docs: int | None = None
    dev_batch_size: int = 1

    action_loss_weight: float = 1.0
    label_smoothing: float = 0.1

    @classmethod
    def from_dict(cls, d: dict) -> "DecoderOnlySRConfig":
        return parse_config_dict(cls, d)
