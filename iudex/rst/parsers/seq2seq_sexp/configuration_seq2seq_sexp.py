from dataclasses import dataclass, field

from tonga import FromParams

from iudex.rst.parsers.common.config import PeftConfig, parse_config_dict
from iudex.rst.parsers.common.curriculum import Curriculum, SimpleCurriculum


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
    # Decoder target length. sexp runs ~2x the SR action length (open/close
    # parens per node), hence the larger budget vs seq2seq_sr.
    max_output_length: int = 8192
    # Trade compute for activation memory. Enters the config hash.
    gradient_checkpointing: bool = True

    # LoRA. Null = full fine-tuning. When set, the base stack is frozen and only
    # LoRA adapters + `peft.modules_to_save` train. The practical default at 1B+.
    peft: PeftConfig | None = None

    # Curriculum strategy (Registrable). Default `SimpleCurriculum` reproduces
    # cold full-document training. `SubtreeSizeCurriculum` warms up on small
    # subtrees before full docs. The curriculum owns each phase's train trees,
    # dev set, and epoch budget (the run length).
    curriculum: Curriculum = field(default_factory=SimpleCurriculum)

    # Training
    lr: float = 3e-5
    weight_decay: float = 0.01
    batch_size: int = 1
    grad_accum: int = 16
    # "adamw" (two state tensors per param) or "adafactor" (factored 2nd moment,
    # far less memory). Use adafactor when the AdamW footprint OOMs.
    optimizer: str = "adafactor"
    num_warmup_steps: int | None = None
    max_grad_norm: float = 1.0
    amp: bool = True
    patience: int = 5
    log_every: int = 5
    # Skip dev validation until this epoch (0 = from the start). Useful here
    # because decoding undertrained dev docs is slow for a ~0 score. Resume-safe
    # (in HASH_EXCLUDE). Non-final curriculum phases skip validation regardless.
    begin_validation_epoch: int = 0
    # Run dev validation every N epochs (global epoch count shared with the
    # curriculum phase loop). 1 = every epoch. The final epoch always validates.
    validate_every: int = 1
    checkpoint_dir: str = "checkpoints"
    run_name: str | None = None
    seed: int = 42
    val_metric_name: str = "e2e_full_f1"

    # Decoding
    num_beams: int = 4
    use_validity_constraints: bool = True
    eval_decode_greedy: bool = True

    # Min content tokens in a leaf before its close paren is legal at decode
    # time (inference only). 1 = off, bump to 2-3 to suppress over-segmentation.
    # At end-of-source a leaf may close below the threshold (else the last EDU
    # can't commit).
    min_edu_length: int = 1

    # Cap per-epoch dev eval to the first N docs (directory order) to speed up
    # validation. None = full dev set each epoch. The final dev/test eval is
    # always on the full split regardless.
    dev_max_docs: int | None = None

    # Batch size for dev/test predictions (KV cache strides across the batch).
    # 1 = per-document loop. Bump up to memory.
    dev_batch_size: int = 1

    # Gradient multiplier on structural-action positions (parens, labels).
    # 1.0 = no rebalance. Bump only if action positions are demonstrably starved.
    action_loss_weight: float = 1.0

    # Per-document loss weight proportional to (#EDUs ** edu_loss_weight_exponent),
    # normalized to mean 1 over each phase's training set (Hu & Wan 2023 Eq. 2 uses
    # exponent 1). 0.0 disables it (all documents weighted equally). Recomputed per
    # curriculum phase over that phase's trees.
    edu_loss_weight_exponent: float = 0.0

    # Label smoothing on the CE loss. 0.1 is a reasonable default for the small
    # action head and few training docs.
    label_smoothing: float = 0.1

    # Order leaves/nodes are emitted in. "postorder" or "preorder".
    # The registry's signature_field for this parser.
    traversal_order: str = "postorder"
    # use_copy=False only. True (default) hard-masks content positions to the
    # source cursor (copy-via-constraint), False allows free content generation.
    # Raises if use_copy is True and constrain_content is False.
    constrain_content: bool = True
    # True (default): source subwords become a `<copy>` sentinel and the lm_head
    # is a small fresh head over the action vocab (~100 classes). The predict path
    # substitutes the source id when `<copy>` is emitted. False: source subwords
    # appear verbatim and the full pretrained lm_head is kept to score them (Hu and
    # Wan 2023, "RST Discourse Parsing as Text-to-Text Generation").
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
