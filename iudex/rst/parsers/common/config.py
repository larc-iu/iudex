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
    """LoRA fine-tuning hyperparameters, shared by every parser (`peft: null`
    means full fine-tuning).

    `r`, `alpha`, `dropout`, `target_modules`, `bias`, `dora` are the core LoRA
    knobs, read by every parser. `modules_to_save` / `train_only_new_embedding_rows`
    are read only by the generative parsers in the full-vocab `*_sexp`
    `use_copy=false` mode (where the lm_head must learn to emit source subwords);
    in every other mode they are inert, and encoder parsers ignore them."""

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
