from dataclasses import dataclass

from tonga import FromParams

from iudex.rst.parsers.common.config import parse_config_dict
from iudex.rst.parsers.common.detokenization import Detokenizer


@dataclass
class _SegmentationConfig(FromParams):
    """Joint per-token EDU-boundary head. Presence on `PiudottoConfig` is
    itself the "enabled" signal; set `segmentation: null` in jsonnet to
    disable joint segmentation (and lose `predict_from_text`).
    """

    # Per-token tagging scheme used by the segmentation head:
    #   "BIE": 3-class B(egin) / I(nside) / E(nd) (default)
    #   "BO":  2-class B(egin EDU) / O(ther)
    #   "EO":  2-class E(nd EDU)   / O(ther) (DMRST's approach)
    scheme: str = "BIE"

    # Training objective for the head:
    #   "crf": linear-chain CRF with learned transition/start/end scores, trained
    #          by negative log-likelihood. The scheme's structural constraints are
    #          added as masks on top of the learned scores, so the CRF only ever
    #          puts mass on schema-valid sequences. `pos_weight` is ignored.
    #   "ce":  independent per-token class-weighted cross-entropy; decoding is a
    #          constrained Viterbi over the structural masks alone (no learned
    #          transitions). `pos_weight` upweights the rare boundary tag(s).
    loss: str = "crf"
    pos_weight: float = 10.0  # class weight on the boundary tag(s); "ce" loss only
    dropout: float = 0.2


@dataclass
class _EMAConfig(FromParams):
    """EMA-based loss weighting for piudotto. Each optimizer step,

        w_k ∝ exp((curr_loss_k / max(ema_loss_k, 1e-3)) / temperature)
        ema_loss_k = momentum * ema_loss_k + (1 - momentum) * curr_loss_k

    Component losses are `split`, `label`, and (when joint segmentation is
    on) `seg`. Set `ema: null` in jsonnet for an unweighted sum.

    Why EMA instead of DMRST's window-based DLW: per-step loss under
    whole-tree training is dominated by per-document variance, and a hard
    2-step ratio amplifies that noise (and degenerates on small RST trees
    where `split_loss` can be 0). EMA gives a smoothly-decaying baseline
    (effective window ≈ `1 / (1 - momentum)` steps; default 0.95 ≈ 20
    steps) — much more stable, and a single zero step only nudges the
    baseline by `(1 - momentum)`.
    """

    momentum: float = 0.95
    temperature: float = 2.0

    def __post_init__(self):
        if not (0.0 < self.momentum < 1.0):
            raise ValueError(f"_EMAConfig.momentum must be in (0, 1), got {self.momentum}")


@dataclass
class _MarginObjectiveConfig(FromParams):
    """Stern et al. 2017 max-margin objective. Runs CKY each step (~2x
    slower than per-node CE) but optimizes a global tree-level signal.
    Set `margin_training: null` for per-node CE.
    """

    margin: float = 1.0


@dataclass
class PiudottoConfig(FromParams):
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

    # Encoder. Fully fine-tuned (no layer freezing) with a smaller `encoder_lr`.
    model_name: str = "jhu-clsp/ettin-encoder-150m"
    # Docs longer than the model's context length require strided encoding, but a
    # modern long-context encoder will outrun pretty much any RST doc.
    stride: int = 100
    # Light dropout: the encoder is fully fine-tuned (and carries its own internal
    # dropout), so the task-side heads only need a touch of regularization.
    encoder_dropout: float = 0.2

    # Per-EDU span representation built from token embeddings.
    #   "concat":    reduce(concat(first_token, last_token, mean(tokens)))
    #   "attention": learned additive attention over the span's tokens, concat with
    #                first+last endpoints, then reduce. Tends to underfit on small
    #                treebanks.
    span_pooling: str = "concat"

    # Deep biaffine scorer (Dozat & Manning).
    classifier_hidden_size: int = 256
    classifier_dropout: float = 0.2
    classifier_use_bias: bool = True

    # How to pool the EDUs of each child of a split into the single vector fed
    # to the label / split classifier:
    #   "mean":     average of all EDU representations in the child
    #   "last_edu": the last EDU representation in the child
    label_input_pooling: str = "mean"

    # Joint EDU segmentation. When non-null, training adds a per-token
    # segmenter over EDU boundaries and `predict_from_text` is available
    # for raw-text → tree inference.
    segmentation: _SegmentationConfig | None = None

    # Detokenizer for EDU text. Applied only when `segmentation` is non-null, so
    # end-to-end-from-text models train on natural text matching the raw input
    # `predict_from_text` receives. Registrable; see common.detokenization.
    detokenizer: Detokenizer | None = None

    # Tree decoding strategy at inference time:
    #   "greedy": top-down argmax at each level (default)
    #   "cky":    O(n^3) chart fill, globally optimal binary tree
    decoding: str = "greedy"

    # Training objective: per-node teacher-forced CE by default. Set
    # `margin_training: {margin: 1.0}` to switch to Stern-style max-margin
    # against the cost-augmented best non-gold tree (CKY at train time).
    margin_training: _MarginObjectiveConfig | None = None

    # Training
    lr: float = 1e-4
    encoder_lr: float | None = 2e-5
    max_epochs: int = 50
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
    validate_every: int | None = None
    checkpoint_every: int | None = None
    checkpoint_dir: str = "checkpoints"
    run_name: str | None = None
    seed: int = 42
    val_metric_name: str = "span_f1"

    # EMA-based loss weighting. Set to `null` for unweighted sum.
    ema: _EMAConfig | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "PiudottoConfig":
        return parse_config_dict(cls, d)
