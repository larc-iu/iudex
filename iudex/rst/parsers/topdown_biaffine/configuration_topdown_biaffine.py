import dataclasses
from dataclasses import dataclass


@dataclass
class TopdownBiaffineConfig:
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
    model_name: str = "SpanBERT/spanbert-base-cased"
    ffn_hidden_size: int = 512
    dropout: float = 0.2
    stride: int = 100

    # Training
    lr: float = 2e-4
    encoder_lr: float | None = None  # if set, encoder params use this LR instead of `lr`
    max_epochs: int = 100
    grad_accum: int = 1
    patience: int = 10
    max_grad_norm: float = 1.0
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
    def from_dict(cls, d: dict) -> "TopdownBiaffineConfig":
        """Validate `d` against this dataclass's fields and instantiate.

        Args:
            d: usually a dict produced by `tonga.Params.from_file(...).as_dict()`.

        Raises:
            ValueError: if `d` contains keys that are not fields of this dataclass.
            TypeError: from `__init__` if a required field is missing.
        """
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"Unknown config field(s): {sorted(unknown)}. Valid fields: {sorted(known)}")
        d = dict(d)
        if d.get("relation_types") is not None:
            d["relation_types"] = [tuple(r) for r in d["relation_types"]]
        return cls(**d)
