"""Shared config-parsing helper and the shared PEFT/LoRA sub-configs."""

from dataclasses import dataclass, field
from typing import TypeVar

from tonga import FromParams

T = TypeVar("T", bound=FromParams)


def parse_config_dict(cls: type[T], d: dict) -> T:
    """Instantiate `cls` from a plain dict (e.g. `tonga.Params.as_dict()` output)."""
    return cls.from_params(d)


@dataclass
class PeftConfig(FromParams):
    """LoRA fine-tuning hyperparameters, shared by every parser (a `null` `peft`
    field means full fine-tuning). This is the maximal superset of knobs: not
    all are read by every parser, but the field set is deliberately not tied to
    model details so one definition serves all eight.

    `r`, `alpha`, `dropout`, `target_modules`, `bias`, `dora` are the core LoRA
    knobs, read everywhere (duck-typed by `common.encoding._wrap_lora` for the
    encoder parsers, by each generative parser's `_install_peft`).

    `modules_to_save` / `train_only_new_embedding_rows` matter only to the
    generative parsers, and only in the full-vocab `*_sexp` `use_copy=false`
    mode, where the lm_head must learn to emit source subwords. In every other
    mode the input embedding is kept fully trainable with a backward hook zeroing
    pretrained-row gradients (see `seqgen.mask_old_embedding_gradients`), so the
    two fields are inert. Encoder parsers ignore them entirely. They are kept on
    the shared class so existing jsonnets load unchanged."""

    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: str | list[str] = "all-linear"
    bias: str = "none"
    dora: bool = False
    modules_to_save: list[str] = field(default_factory=lambda: ["embed_tokens"])
    train_only_new_embedding_rows: bool = True

    def __post_init__(self):
        if self.r < 1:
            raise ValueError(f"PeftConfig.r must be >= 1 (got {self.r})")
