from dataclasses import dataclass, field

from tonga import FromParams

from iudex.rst.parsers.common.config import parse_config_dict


@dataclass
class _PeftConfig(FromParams):
    """LoRA fine-tuning of the seq2seq stack. Mirrors the seq2seq_sr peft
    config: the lm_head is replaced with a fresh small Linear, so only
    `embed_tokens` needs `modules_to_save`, and even those grads are masked
    to only the newly-added action-token rows.
    """

    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: str | list[str] = "all-linear"
    bias: str = "none"
    dora: bool = False
    modules_to_save: list[str] = field(default_factory=lambda: ["embed_tokens"])
    train_only_new_embedding_rows: bool = True

    def __post_init__(self):
        if self.r < 1:
            raise ValueError(f"_PeftConfig.r must be >= 1 (got {self.r})")


@dataclass
class Seq2SeqSexpConfig(FromParams):
    train_dir: str
    dev_dir: str
    test_dir: str | None = None

    relation_types: list[tuple[str, str]] | None = None
    relation_map: dict[str, str] | None = None

    # Model
    model_name: str = "google/t5gemma-2-1b-1b"
    max_input_length: int = 4096
    max_output_length: int = 8192
    gradient_checkpointing: bool = True

    # LoRA
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

    # Sexp-specific. The first is also the registry signature_field for
    # this parser (`traversal_order` is unique to sexp parsers).
    traversal_order: str = "postorder"
    use_copy: bool = True

    def __post_init__(self):
        if self.traversal_order not in ("preorder", "postorder"):
            raise ValueError(f"traversal_order must be 'preorder' or 'postorder' (got {self.traversal_order!r})")

    @classmethod
    def from_dict(cls, d: dict) -> "Seq2SeqSexpConfig":
        return parse_config_dict(cls, d)
