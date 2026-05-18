from dataclasses import dataclass

from tonga import FromParams

from iudex.rst.parsers.common.config import parse_config_dict


@dataclass
class _SegmentationConfig(FromParams):
    """Joint per-token EDU-boundary head. Presence of this sub-config on
    `DMRSTConfig` is itself the "enabled" signal; set `segmentation: null`
    in jsonnet to disable joint segmentation (and lose `predict_from_text`).
    """

    pos_weight: float = 10.0  # class weight on the positive (EDU-end) label
    start_loss: bool = False  # second binary head for EDU starts


@dataclass
class _DLWConfig(FromParams):
    """Dynamic loss weighting (paper §3.2). Presence on `DMRSTConfig`
    enables DLW; `dlw: null` falls back to unweighted sum of component losses.

    The weight update compares the mean of the most recent `window // 2`
    optimizer steps' component losses against the mean of the preceding
    `window // 2` (or the rest, for odd `window`). With `window=2` (default)
    this collapses to `L_k(t-1) / L_k(t-2)`, exactly reproducing the paper's
    formulation. Larger windows yield much less noisy ratios — useful for
    whole-tree training where per-step loss is dominated by single-document
    variance. `window=2` is the minimum.
    """

    temperature: float = 2.0
    window: int = 2

    def __post_init__(self):
        if self.window < 2:
            raise ValueError(f"_DLWConfig.window must be >= 2 (got {self.window})")


@dataclass
class DMRSTConfig(FromParams):
    train_dir: str
    dev_dir: str
    # Optional held-out test split. If set, final evaluation runs on both
    # dev and test after the dev table; if null, only dev is reported.
    test_dir: str | None = None

    # Populated at training time by inferring (relation, nuclearity) pairs
    # from train_dir + dev_dir; not user-configurable. Persists in the
    # checkpointed config so predict/from_pretrained know the label space.
    relation_types: list[tuple[str, str]] | None = None

    # Optional fine→coarse relation remap applied by the reader. When set,
    # every non-"span" relname in the data must be a key (missing keys raise).
    # `relation_types` and the model's label space are in the mapped space.
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
    #   "mean":     average of all EDU representations in the child
    #   "last_edu": the last EDU representation in the child
    # The two collapse to the same thing for a 2-EDU span (split is forced, each
    # child has exactly one EDU).
    label_input_pooling: str = "mean"
    freeze_embeddings: bool = True
    freeze_encoder_layers: int = 3

    # Joint EDU segmentation (paper §3.1.1). When non-null, training adds a
    # per-token binary classification loss over EDU end positions and
    # `predict_from_text` is available for raw-text inference.
    segmentation: _SegmentationConfig | None = None

    # Dynamic loss weighting (paper §3.2). Set to `null` for unweighted sum.
    dlw: _DLWConfig | None = None

    # Training
    lr: float = 1e-4
    encoder_lr: float | None = 2e-5
    max_epochs: int = 50
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

    @classmethod
    def from_dict(cls, d: dict) -> "DMRSTConfig":
        return parse_config_dict(cls, d)
