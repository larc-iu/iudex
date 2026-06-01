from dataclasses import dataclass, field

from tonga import FromParams

from iudex.rst.parsers.common.config import parse_config_dict
from iudex.rst.parsers.common.curriculum import Curriculum, SimpleCurriculum
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
class _PeftConfig(FromParams):
    """LoRA fine-tuning of the encoder. Null (default) = full fine-tuning.
    When set, the base encoder is frozen and only the low-rank adapters train,
    so a higher `encoder_lr` (~1e-4) is appropriate.
    """

    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    # Which encoder submodules get adapters. "all-linear" (attention + FFN) suits
    # tasks far from the MLM pretraining objective, like discourse parsing; pass an
    # explicit list (e.g. ["query", "value"]) for the classic attention-only LoRA.
    target_modules: str | list[str] = "all-linear"
    bias: str = "none"
    # DoRA (Liu et al. 2024): decompose each adapted weight into magnitude +
    # direction; only direction passes through the low-rank decomposition while
    # magnitude is a separate per-output-dim trainable vector.
    dora: bool = False

    def __post_init__(self):
        if self.r < 1:
            raise ValueError(f"_PeftConfig.r must be >= 1 (got {self.r})")


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

    # Encoder. Fully fine-tuned (no layer freezing) with a smaller `encoder_lr`,
    # unless `peft` is set (LoRA adapters only). See `_PeftConfig`.
    model_name: str = "jhu-clsp/ettin-encoder-150m"
    peft: _PeftConfig | None = None

    # Curriculum strategy (Registrable). Default `SimpleCurriculum` reproduces
    # cold full-document training. `SubtreeSizeCurriculum` warms up on small
    # subtrees before full docs. The curriculum owns each phase's train trees,
    # dev set, and epoch budget (the run length).
    curriculum: Curriculum = field(default_factory=SimpleCurriculum)
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

    # Optional EDU-level Transformer encoder over the pooled per-EDU vectors,
    # run before span scoring so EDUs contextualize against each other (the role
    # dmrst's document-level BiGRU plays). Randomly initialized. 0 layers
    # disables it.
    edu_encoder_layers: int = 0
    # Bottleneck width for the EDU encoder. None runs it at the full encoder
    # hidden size; set it smaller (e.g. 256/128) to down-project H→width,
    # contextualize, up-project back with a residual, squeezing the contextual
    # update through a low-capacity channel (regularization). Must be divisible
    # by `edu_encoder_heads`.
    edu_encoder_hidden_size: int | None = None
    edu_encoder_heads: int = 8
    edu_encoder_dropout: float = 0.2

    # Autoregressive Transformer decoder over the top-down decision sequence,
    # the non-RNN replacement for dmrst's recurrent pointer decoder. Causal
    # self-attention over the DFS-ordered split decisions conditions each split
    # on the decisions already committed (the parse history); cross-attention
    # reads the EDU reprs. 0 layers disables it, recovering the history-free
    # per-node deep-biaffine split scorer. When on, a pointer head replaces that
    # biaffine for the split decision.
    decoder_layers: int = 0
    # Bottleneck width (down-project H->width, decode, up-project), like the EDU
    # encoder; None runs at full width. Must be divisible by decoder_heads.
    decoder_hidden_size: int | None = None
    decoder_heads: int = 8
    decoder_dropout: float = 0.2
    # Pointer split head used by the decoder: "biaffine" or "dot_product".
    pointer_attention_type: str = "biaffine"
    # Order the decoder visits internal nodes (the decision sequence its causal
    # self-attention runs over). Only meaningful with decoder_layers > 0.
    #   "dfs": left-first preorder (matches dmrst's GRU thread).
    #   "bfs": level order (coarse-to-fine).
    decoder_order: str = "dfs"

    # Joint EDU segmentation. When non-null, training adds a per-token
    # segmenter over EDU boundaries and `predict_from_text` is available
    # for raw-text → tree inference.
    segmentation: _SegmentationConfig | None = None

    # Detokenizer for EDU text. Applied only when `segmentation` is non-null, so
    # end-to-end-from-text models train on natural text matching the raw input
    # `predict_from_text` receives. Registrable; see common.detokenization.
    detokenizer: Detokenizer | None = None

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

    # EMA-based loss weighting. Set to `null` for unweighted sum.
    ema: _EMAConfig | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "PiudottoConfig":
        return parse_config_dict(cls, d)
