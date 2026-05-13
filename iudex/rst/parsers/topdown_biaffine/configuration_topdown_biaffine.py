import dataclasses
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class TopdownBiaffineConfig:
    relation_types: List[Tuple[str, str]]
    train_dir: str
    dev_dir: str

    # Model
    model_name: str = "SpanBERT/spanbert-base-cased"
    ffn_hidden_size: int = 512
    dropout: float = 0.2
    stride: int = 100
    attn_implementation: Optional[str] = None

    # Training
    lr: float = 2e-4
    encoder_lr: Optional[float] = None  # if set, encoder params use this LR instead of `lr`
    max_epochs: int = 100
    grad_accum: int = 1
    patience: int = 10
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    num_warmup_steps: int = 0
    log_every: int = 50
    validate_every: Optional[int] = None
    checkpoint_every: Optional[int] = None
    checkpoint_dir: str = "checkpoints"
    run_name: Optional[str] = None
    seed: int = 42
    val_metric_name: str = "span_f1"

    @classmethod
    def from_dict(cls, d: dict) -> "TopdownBiaffineConfig":
        """Validate `d` against this dataclass's fields and instantiate.

        Raises ValueError on unknown keys. Missing required fields raise TypeError
        via dataclass __init__. Missing optional fields use dataclass defaults.
        """
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(
                f"Unknown config field(s): {sorted(unknown)}. "
                f"Valid fields: {sorted(known)}"
            )
        d = dict(d)
        if "relation_types" in d:
            d["relation_types"] = [tuple(r) for r in d["relation_types"]]
        return cls(**d)
