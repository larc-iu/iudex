from dataclasses import dataclass

from iudex.rst.parsers.common.config import parse_config_dict


@dataclass
class DMRSTConfig:
    train_dir: str
    dev_dir: str
    # Optional held-out test split. If set, final evaluation runs on both
    # dev and test after the dev table; if null, only dev is reported.
    test_dir: str | None = None

    # If null, the trainer infers the inventory from (relation, nuclearity)
    # pairs observed in train_dir + dev_dir.
    relation_types: list[tuple[str, str]] | None = None

    # Optional fine→coarse relation remap applied by the reader. When set,
    # every non-"span" relname in the data must be a key (missing keys raise).
    # `relation_types` (if inferred) and the model's label space are in the
    # mapped space.
    relation_map: dict[str, str] | None = None

    # Model
    model_name: str = "xlm-roberta-base"
    stride: int = 100
    attention_type: str = "dot_product"  # or "biaffine"
    classifier_use_bias: bool = True
    num_rnn_layers: int = 1
    encoder_dropout: float = 0.5
    decoder_dropout: float = 0.5
    labeler_dropout: float = 0.5
    doc_gru_dropout: float = 0.2
    # How to pool the EDUs of each child of a split into the single vector fed
    # to the label classifier:
    #   "mean":     average of all EDU representations in the child (paper default)
    #   "last_edu": the last EDU representation in the child (single-EDU edge stand-in)
    # The two collapse to the same thing for a 2-EDU span (split is forced, each
    # child has exactly one EDU).
    label_input_pooling: str = "mean"
    # Freezing embeddings and first n layers of the encoder.
    # Original implementation freezes embeddings and first 3 layers.
    freeze_embeddings: bool = True
    freeze_encoder_layers: int = 3

    # Joint EDU segmentation (cf. section 3.1.1 of paper). When True, training adds
    # a per-token binary classification loss over EDU end positions, and
    # `predict_from_text` is available for raw-text → tree inference.
    joint_segmentation: bool = True
    seg_pos_weight: float = 10.0  # class weight on the positive (EDU-end) label
    seg_start_loss: bool = False  # add a second binary head for EDU starts

    # Training
    lr: float = 1e-4
    encoder_lr: float | None = 2e-5
    max_epochs: int = 100
    grad_accum: int = 3
    patience: int = 10
    max_grad_norm: float = 5.0
    weight_decay: float = 0.01
    num_warmup_steps: int = 0
    log_every: int = 50
    validate_every: int | None = None
    checkpoint_every: int | None = None
    checkpoint_dir: str = "checkpoints"
    run_name: str | None = None
    seed: int = 42
    val_metric_name: str = "span_f1"

    # Dynamic loss weighting (paper §3.2)
    dlw_enabled: bool = True
    dlw_temperature: float = 2.0

    @classmethod
    def from_dict(cls, d: dict) -> "DMRSTConfig":
        return parse_config_dict(cls, d)
