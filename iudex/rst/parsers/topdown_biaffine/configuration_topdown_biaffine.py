from dataclasses import dataclass, field

from tonga import FromParams

from iudex.rst.parsers.common.config import PeftConfig, parse_config_dict
from iudex.rst.parsers.common.curriculum import Curriculum, SimpleCurriculum


@dataclass
class TopdownBiaffineConfig(FromParams):
    train_dir: str
    dev_dir: str
    test_dir: str | None = None

    # Inferred at training time from train_dir + dev_dir. Persisted so
    # predict / from_pretrained know the label space.
    relation_types: list[tuple[str, str]] | None = None

    # Optional fine→coarse relation remap applied by the reader. When set,
    # every non-"span" relname in the data must be a key (missing keys raise).
    # `relation_types` and the model's label space are in the mapped space.
    relation_map: dict[str, str] | None = None

    # Model
    model_name: str = "SpanBERT/spanbert-base-cased"
    ffn_hidden_size: int = 512
    dropout: float = 0.2
    stride: int = 100

    # LoRA encoder fine-tuning. See `PeftConfig`. Null (default) = full fine-tuning.
    peft: PeftConfig | None = None

    # Curriculum strategy (Registrable). Default `SimpleCurriculum` reproduces
    # cold full-document training. `SubtreeSizeCurriculum` warms up on small
    # subtrees before full docs. The curriculum owns each phase's train trees,
    # dev set, and epoch budget (the run length).
    curriculum: Curriculum = field(default_factory=SimpleCurriculum)

    # Training
    lr: float = 2e-4
    encoder_lr: float | None = None  # if set, encoder params use this LR instead of `lr`
    grad_accum: int = 1
    # bf16 autocast on the training forward (CUDA only; bf16 needs no GradScaler).
    # Set false for full-fp32 training. Inference is always fp32.
    amp: bool = True
    patience: int = 10
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    # Linear warmup before linear decay. None uses a 1-epoch warmup
    # (steps_per_epoch). 0 means no warmup. Any positive int is taken literally.
    num_warmup_steps: int | None = None
    log_every: int = 50
    # Skip dev validation until this epoch (0 = validate from the start). In
    # HASH_EXCLUDE, so changing it is resume-safe. Applies within a validating
    # phase. A curriculum's non-final phases skip validation regardless.
    begin_validation_epoch: int = 0
    # Per-document loss weight proportional to (#EDUs ** edu_loss_weight_exponent),
    # normalized to mean 1 over each phase's training set (Hu & Wan 2023 Eq. 2 uses
    # exponent 1). 0.0 disables it (all documents weighted equally). Recomputed per
    # curriculum phase over that phase's trees.
    edu_loss_weight_exponent: float = 0.0
    checkpoint_dir: str = "checkpoints"
    run_name: str | None = None
    seed: int = 42
    val_metric_name: str = "span_f1"

    @classmethod
    def from_dict(cls, d: dict) -> "TopdownBiaffineConfig":
        return parse_config_dict(cls, d)
