from dataclasses import dataclass, field

from tonga import FromParams

from iudex.rst.parsers.common.config import PeftConfig, parse_config_dict
from iudex.rst.parsers.common.curriculum import Curriculum, SimpleCurriculum
from iudex.rst.parsers.common.detokenization import Detokenizer


@dataclass
class _SegmentationConfig(FromParams):
    """Joint per-token EDU-boundary head. Set `segmentation: null` in
    jsonnet to disable (and lose `predict_from_text`)."""

    pos_weight: float = 10.0  # upweighted because EDU ends are rare
    start_loss: bool = False
    # When `scheme` is set (BIE/BO/EO), use the shared scheme-based segmenter
    # (iudex.rst.parsers.common.segmentation) with `loss` (crf/ce) and `dropout`,
    # instead of the paper's binary end-tagger; `start_loss` is then ignored.
    scheme: str | None = None
    loss: str = "crf"
    dropout: float = 0.5


@dataclass
class _DLWConfig(FromParams):
    """Dynamic loss weighting (paper §3.2). Set `dlw: null` for unweighted sum.

    The weight update compares the mean of the most recent `window // 2`
    optimizer steps' component losses against the mean of the preceding
    `window // 2` (or the rest, for odd `window`). With `window=2` (default)
    this collapses to `L_k(t-1) / L_k(t-2)`, reproducing the paper's
    formulation.
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
    test_dir: str | None = None

    # Inferred at training time from train_dir + dev_dir. Persisted so
    # predict / from_pretrained know the label space.
    relation_types: list[tuple[str, str]] | None = None

    # Optional fine→coarse relation remap applied by the reader. When set,
    # every non-"span" relname in the data must be a key (missing keys raise).
    # `relation_types` and the model's label space are in the mapped space.
    relation_map: dict[str, str] | None = None

    # Model
    model_name: str = "xlm-roberta-base"
    stride: int = 100
    # When set, encode with DMRST's original fixed sliding-window scheme
    # (reference module.py EncoderRNN): `encoder_window_size` content tokens per
    # window, `stride` context tokens discarded per interior side, no [CLS]/[SEP].
    # Requires encoder_window_size + 2*stride <= the encoder's positional budget.
    # None (default) uses the shared max-length striding (CLS/SEP per chunk).
    encoder_window_size: int | None = None
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

    # LoRA encoder fine-tuning. See `PeftConfig`. Mutually exclusive with the
    # freeze fields above (set both to off when enabling peft).
    peft: PeftConfig | None = None

    # Curriculum strategy (Registrable). Default `SimpleCurriculum` reproduces
    # cold full-document training. `SubtreeSizeCurriculum` warms up on small
    # subtrees before full docs. The curriculum owns each phase's train trees,
    # dev set, and epoch budget (the run length).
    curriculum: Curriculum = field(default_factory=SimpleCurriculum)

    # Joint EDU segmentation (paper §3.1.1). See `_SegmentationConfig`.
    segmentation: _SegmentationConfig | None = None

    # Detokenizer for EDU text. Applied only when `segmentation` is non-null, so
    # end-to-end-from-text models train on natural text matching the raw input
    # `predict_from_text` receives. Registrable; see common.detokenization.
    detokenizer: Detokenizer | None = None

    # Dynamic loss weighting (paper §3.2). See `_DLWConfig`.
    dlw: _DLWConfig | None = None

    # Training
    lr: float = 1e-4
    encoder_lr: float | None = 2e-5
    grad_accum: int = 3
    # bf16 autocast on the training forward (CUDA only; bf16 needs no GradScaler).
    # Set false for full-fp32 training. Inference is always fp32.
    amp: bool = True
    patience: int = 10
    max_grad_norm: float = 5.0
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
    def from_dict(cls, d: dict) -> "DMRSTConfig":
        return parse_config_dict(cls, d)
