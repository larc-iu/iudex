from dataclasses import dataclass, field

from tonga import FromParams

from iudex.rst.parsers.common.config import parse_config_dict


@dataclass
class _PeftConfig(FromParams):
    """LoRA fine-tuning of the seq2seq stack. Mirrors `DMRSTConfig._PeftConfig`.

    The lm_head is replaced at parser-init with a fresh, small Linear over
    the action vocab. The input embedding is handled by carving the new
    action-token rows into a small trainable Parameter and freezing the
    pretrained base matrix (see `Seq2SeqSRParser._carve_new_token_embeddings`).
    """

    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: str | list[str] = "all-linear"
    bias: str = "none"
    dora: bool = False
    # No-ops, kept so existing jsonnets load. The input embedding is no longer
    # routed through PEFT modules_to_save (it carries a frozen base + small
    # trainable new-rows Parameter instead), so neither field has any effect.
    modules_to_save: list[str] = field(default_factory=lambda: ["embed_tokens"])
    train_only_new_embedding_rows: bool = True

    def __post_init__(self):
        if self.r < 1:
            raise ValueError(f"_PeftConfig.r must be >= 1 (got {self.r})")


@dataclass
class Seq2SeqSRConfig(FromParams):
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
    # Decoder target length. Output is source subwords + n SHIFTs + n-1 REDUCEs,
    # so headroom matters: a 1.5K-source / 50-EDU doc lands around 1.6K tokens.
    max_output_length: int = 6144
    # Required on a 4090 at T5Gemma-1B + 4K input + 6K output. Enters the
    # hash so flipping it triggers a fresh run.
    gradient_checkpointing: bool = True

    # LoRA. Null = full fine-tuning. When set, the base seq2seq stack is
    # frozen and only LoRA adapters + the modules in `peft.modules_to_save`
    # train. For T5Gemma2 / mT5 at 1B+ scale this is the practical default
    # since full FT blows up optimizer memory.
    peft: _PeftConfig | None = None

    # Training
    lr: float = 3e-5
    weight_decay: float = 0.01
    max_epochs: int = 10
    batch_size: int = 1
    grad_accum: int = 16
    # "adamw": standard, but two fp32-or-bf16 state tensors per param (~16-32 GB
    # for T5Gemma 2 1B-1B).
    # "adafactor": T5 paper's optimizer, factored 2nd moment, ~50 MB of state
    # total. Use this when the AdamW footprint OOMs.
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

    # Minimum number of `<copy>` actions required before `<shift>` becomes
    # legal at decode time (inference-only — has no effect on training). 1
    # is no constraint. Bump to 2 or 3 to suppress over-segmentation
    # (splits like "Education" + "and early loves" out of one gold EDU);
    # the model trained on real EDUs of length >=1 may still want to shift
    # after a single copy, but the constraint forces it to continue copying.
    # Exception: at the end of the source, shift is always legal regardless
    # of this setting (otherwise the final EDU can't be committed).
    min_edu_length: int = 1

    # Cap per-epoch dev eval to the first N documents (in directory order) to
    # speed up training-time validation. Autoregressive generation at L≈3K on
    # a 1B decoder is ~1 min/doc even with KV cache; full GUM dev (32 docs)
    # costs ~30 min/epoch. None = use the full dev set every epoch. The FINAL
    # dev/test eval always runs on the full split regardless of this setting.
    dev_max_docs: int | None = None

    # Batch size for dev/test predictions. Each batch shares one decoder pass
    # (weights stream from HBM once, KV cache strides across the batch) so
    # this is bandwidth-bound rather than compute-bound and roughly linear
    # in batch_size up to memory. 1 = original per-document loop.
    dev_batch_size: int = 1

    # Multiplier on the gradient contribution of structural-action positions
    # (`<shift>`, `<reduce_*>`) in the training loss. Default 1.0 (no
    # rebalance) since the replacement lm_head projects to ~100 action
    # classes — copy CE is on the same scale as structural CE and the old
    # 262K-vocab justification for upweighting structurals doesn't apply.
    # Bump above 1.0 only if action positions are demonstrably starved
    # (e.g. copy CE near zero but structural CE high); higher values bias
    # the model toward over-emitting `<shift>` at inference.
    action_loss_weight: float = 1.0

    # Label smoothing on the CE loss. Standard seq2seq fine-tuning trick.
    # The action head is small (~100 classes) and GUM has ~150 train docs,
    # so hard targets overfit fast; 0.1 is the conventional default.
    label_smoothing: float = 0.1

    @classmethod
    def from_dict(cls, d: dict) -> "Seq2SeqSRConfig":
        return parse_config_dict(cls, d)
