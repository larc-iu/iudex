from dataclasses import dataclass, field

from tonga import FromParams

from iudex.rst.parsers.common.config import PeftConfig, parse_config_dict
from iudex.rst.parsers.common.curriculum import Curriculum, SimpleCurriculum


@dataclass
class DecoderOnlySRConfig(FromParams):
    train_dir: str
    dev_dir: str
    test_dir: str | None = None

    # Inferred at training time. Persisted so predict / from_pretrained know
    # the action vocabulary to register on the tokenizer.
    relation_types: list[tuple[str, str]] | None = None
    relation_map: dict[str, str] | None = None

    # Any AutoModelForCausalLM works (Llama, Qwen, ...) as long as the tokenizer
    # exposes character-offset mapping for source alignment.
    model_name: str = "google/gemma-3-1b-it"

    # Single-stream layout: source + actions + specials share one sequence. The
    # trainer enforces the combined length and drops trees that overflow it.
    max_input_length: int = 3072
    max_output_length: int = 5120
    gradient_checkpointing: bool = False

    # Parser-kind tag (a unique field name per parser, used to identify a saved
    # config). Reading the value is a no-op.
    causal_mode: bool = True

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
    checkpoint_dir: str = "checkpoints"
    run_name: str | None = None
    seed: int = 42
    val_metric_name: str = "e2e_full_f1"

    # Decoding
    num_beams: int = 4
    # Honored by greedy decoding only. Beam search always applies the validity
    # mask regardless of this flag (a feasible tree per beam is required to
    # reconstruct), so setting it False has no effect under num_beams > 1.
    use_validity_constraints: bool = True
    eval_decode_greedy: bool = True

    # Min `<copy>` actions before `<shift>` is legal at decode time (inference
    # only). 1 = no constraint, bump to 2-3 to suppress over-segmentation.
    # At end-of-source `<shift>` is always legal (else the last EDU can't commit).
    min_edu_length: int = 1

    # Cap per-epoch dev eval to the first N docs (directory order) to speed up
    # validation. None = full dev set each epoch. The final dev/test eval is
    # always on the full split regardless.
    dev_max_docs: int | None = None

    # Batch size for dev/test predictions (KV cache strides across the batch).
    # 1 = per-document loop. Bump up to memory.
    dev_batch_size: int = 1

    # Gradient multiplier on structural-action positions (`<shift>`, `<reduce_*>`).
    # 1.0 = no rebalance (copy and structural CE share a scale under the small
    # action head). Bump only if action positions are demonstrably starved.
    action_loss_weight: float = 1.0

    # Per-document loss weight proportional to (#EDUs ** edu_loss_weight_exponent),
    # normalized to mean 1 over each phase's training set (Hu & Wan 2023 Eq. 2 uses
    # exponent 1). 0.0 disables it (all documents weighted equally). Recomputed per
    # curriculum phase over that phase's trees.
    edu_loss_weight_exponent: float = 0.0

    # Label smoothing on the CE loss. 0.1 is a reasonable default for the small
    # action head and few training docs.
    label_smoothing: float = 0.1

    @classmethod
    def from_dict(cls, d: dict) -> "DecoderOnlySRConfig":
        return parse_config_dict(cls, d)
