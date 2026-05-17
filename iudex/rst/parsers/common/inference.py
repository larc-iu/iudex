"""Shared checkpoint-resolution and model-loading for RST parser CLIs."""

import dataclasses
import os
import sys
from typing import TypeVar

import torch
from tonga import Params

from iudex.common.log import console
from iudex.rst.training import derive_run_id

ConfigT = TypeVar("ConfigT")
ParserT = TypeVar("ParserT", bound=torch.nn.Module)


def resolve_checkpoint(
    config_path: str | None,
    checkpoint_path: str | None,
    config_cls: type,
    train_module: str,
) -> str:
    """Return the .pt path to load.

    Exactly one of the two paths is non-None. With `checkpoint_path`, use the
    path as-is; with `config_path`, derive the run dir from the resolved
    config and look up `best_model.pt`. Both branches exit non-zero with a
    helpful message if the file is missing.

    Args:
        config_cls:    the parser's config dataclass (used to re-derive run_id)
        train_module:  dotted module path for the "train first with" hint
    """
    if checkpoint_path:
        if not os.path.exists(checkpoint_path):
            console.print(f"[bold red]Checkpoint not found:[/bold red] [path]{checkpoint_path}[/path]")
            sys.exit(1)
        return checkpoint_path

    cfg = config_cls.from_dict(Params.from_file(config_path).as_dict(quiet=True))
    run_id, _ = derive_run_id(dataclasses.asdict(cfg), cfg.run_name)
    run_dir = os.path.join(cfg.checkpoint_dir, run_id)
    derived_path = os.path.join(run_dir, "best_model.pt")
    if not os.path.exists(derived_path):
        console.print(
            f"[bold red]No trained model found for this config.[/bold red]\n"
            f"  Expected: [path]{derived_path}[/path]\n"
            f"  Train first with:\n"
            f"    python -m {train_module} {config_path}"
        )
        sys.exit(1)
    return derived_path


def load_parser_from_checkpoint(
    checkpoint_path: str,
    device: torch.device,
    config_cls: type[ConfigT],
    parser_cls: type[ParserT],
) -> ParserT:
    """Rehydrate a parser from a checkpoint: rebuild the config, init the
    model, load weights, move to `device`, and put it in eval mode."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = config_cls.from_dict(checkpoint["config"])
    model = parser_cls(cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval()
