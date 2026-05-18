"""Training utilities for iudex RST parsers.

A bag of helpers consumed by parser-specific training scripts (e.g.
`iudex/rst/parsers/<name>/train_<name>.py`). There is no `Trainer` class —
each `train_<name>.py` owns its own loop. Each utility takes its inputs
explicitly and assumes only standard `nn.Module` interfaces.
"""

import hashlib
import json
import logging
import os
import random
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn
from rich.panel import Panel
from rich.pretty import Pretty
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from iudex.common.log import console, dim, warn

logger = logging.getLogger(__name__)


# Fields stripped before hashing the config to compute `run_id`. Anything in
# this list can change between runs without invalidating an existing
# checkpoint.
#
# Categories:
#   - display:        run_name
#   - inferred:       relation_types (populated post-hash from train/dev data;
#                     would silently mismatch between train and predict)
#   - storage:        checkpoint_dir (moving runs shouldn't reissue run_ids)
#   - training-loop:  max_epochs, patience, validate_every, checkpoint_every,
#                     log_every, val_metric_name (don't change the weights)
#   - eval-only:      test_dir (read only at final eval)
DEFAULT_HASH_EXCLUDE: tuple[str, ...] = (
    "run_name",
    "relation_types",
    "checkpoint_dir",
    "max_epochs",
    "patience",
    "log_every",
    "validate_every",
    "checkpoint_every",
    "val_metric_name",
    "test_dir",
)


def config_hash(obj: Any) -> str:
    """First 12 hex chars (48-bit prefix) of SHA-256 over a JSON-serializable
    object (typically `dataclasses.asdict(cfg)`). Collision-resistance is
    birthday-bound ≈ √2⁴⁸ ≈ 16M.

    Raises TypeError if `obj` contains a value json doesn't know how to encode
    — silently `str()`-ifying would make hashes platform- and
    Python-version-dependent. Add an explicit serializer for any new type.
    """
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:12]


def set_seeds(seed: int) -> None:
    """Seed torch (and cuda, if present) plus the stdlib `random` module."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)


def gpu_mem_gb(device: torch.device) -> tuple[float, float] | None:
    """Return (allocated_gb, reserved_gb) for CUDA devices, else None."""
    if device.type != "cuda":
        return None
    return (
        torch.cuda.memory_allocated(device) / 1024**3,
        torch.cuda.memory_reserved(device) / 1024**3,
    )


def derive_run_id(
    config_dict: dict,
    run_name: str | None = None,
    *,
    hash_exclude: tuple[str, ...] = DEFAULT_HASH_EXCLUDE,
) -> tuple[str, str]:
    """Compute the run id ("{run_name}-{hash}" or just "{hash}") and hash.

    Fields named in `hash_exclude` are stripped before hashing so they don't
    affect run identity (see `DEFAULT_HASH_EXCLUDE`). No I/O — safe to call
    from inference paths. Returns (run_id, cfg_hash).
    """
    hashable = {k: v for k, v in config_dict.items() if k not in hash_exclude}
    cfg_hash = config_hash(hashable)
    run_id = f"{run_name}-{cfg_hash}" if run_name else cfg_hash
    return run_id, cfg_hash


def prepare_run_dir(
    config_dict: dict,
    checkpoint_dir: str,
    run_name: str | None = None,
    *,
    hash_exclude: tuple[str, ...] = DEFAULT_HASH_EXCLUDE,
) -> tuple[str, str]:
    """Derive `{checkpoint_dir}/{run_name}-{hash}` (or just `/{hash}`) and
    `mkdir -p` it. Returns (run_dir, cfg_hash).

    On first creation (no `last.pt` present), prints a hint about the closest
    sibling run if one exists, so an accidental field change that branched
    into a new run is visible.

    Does NOT write `config.json`; callers should `write_run_config` separately
    once they've resolved any inferred-at-training fields (e.g. relation_types).
    """
    run_id, cfg_hash = derive_run_id(config_dict, run_name, hash_exclude=hash_exclude)
    run_dir = os.path.join(checkpoint_dir, run_id)
    is_fresh = not os.path.exists(os.path.join(run_dir, "last.pt"))
    os.makedirs(run_dir, exist_ok=True)
    if is_fresh:
        _hint_closest_sibling(run_dir, checkpoint_dir, config_dict, hash_exclude=hash_exclude)
    return run_dir, cfg_hash


def write_run_config(run_dir: str, config_dict: dict) -> None:
    """Write `{run_dir}/config.json` for audit. Overwrites if present.

    Train scripts call this *after* resolving inferred fields (relation_types,
    etc.) so the on-disk audit, the embedded checkpoint config, and any
    Hub-published config.json all match.
    """
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)


def _hint_closest_sibling(
    new_run_dir: str,
    checkpoint_dir: str,
    new_config: dict,
    *,
    hash_exclude: tuple[str, ...],
) -> None:
    """Print a hint if there's a sibling run whose config differs by only a
    handful of fields. Silent if there are no siblings, or the closest one
    differs in more than 5 hash-affecting fields. Excluded fields are ignored.
    """
    if not os.path.isdir(checkpoint_dir):
        return
    best: tuple[int, str, list[str]] | None = None
    for entry in os.listdir(checkpoint_dir):
        sibling = os.path.join(checkpoint_dir, entry)
        if sibling == new_run_dir or not os.path.isdir(sibling):
            continue
        sibling_cfg_path = os.path.join(sibling, "config.json")
        if not os.path.exists(sibling_cfg_path):
            continue
        try:
            with open(sibling_cfg_path, encoding="utf-8") as f:
                sibling_cfg = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        keys = (set(new_config) | set(sibling_cfg)) - set(hash_exclude)
        diffs = sorted(k for k in keys if new_config.get(k) != sibling_cfg.get(k))
        if not diffs:
            continue
        if best is None or len(diffs) < best[0]:
            best = (len(diffs), entry, diffs)
    if best is None or best[0] > 5:
        return
    n, run_id, diffs = best
    field_list = ", ".join(diffs[:5]) + (f", +{len(diffs) - 5} more" if len(diffs) > 5 else "")
    dim(
        f"  New run. Closest existing run: [path]{run_id}[/path] "
        f"(differs in {n} field{'s' if n != 1 else ''}: {field_list}).\n"
        f"  If you meant to resume that, revert those fields. Otherwise this is fine."
    )


def resume_or_init(
    run_dir: str,
    *,
    model: nn.Module,
    optimizer,
    scheduler,
    expected_hash: str,
) -> dict[str, Any]:
    """Try to resume `{run_dir}/last.pt`. If it exists with matching hash, restore
    model/optimizer/scheduler state and return the saved training state. Otherwise
    return fresh state.

    Returned dict keys: global_step, epoch, best_val, stale_validations.
    """
    ckpt = try_resume(os.path.join(run_dir, "last.pt"), expected_hash=expected_hash)
    if ckpt is None:
        return {"global_step": 0, "epoch": 0, "best_val": -1.0, "stale_validations": 0}
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return {
        "global_step": ckpt["global_step"],
        "epoch": ckpt["epoch"],
        "best_val": ckpt.get("best_val", -1.0),
        "stale_validations": ckpt.get("stale_validations", 0),
    }


def build_optimizer(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    *,
    submodule_lrs: Sequence[tuple[nn.Module, float]] = (),
) -> AdamW:
    """Build AdamW with no-decay on biases and norm weights, and per-submodule LRs.

    Params belonging to each `(submodule, sub_lr)` entry use `sub_lr`; everything
    else uses `lr`. If a param belongs to multiple listed submodules, the first
    listed wins. With an empty `submodule_lrs`, every param uses `lr` (two groups,
    decay vs no-decay).
    """

    def _is_no_decay(name: str) -> bool:
        if name.endswith(".bias"):
            return True
        parts = name.split(".")
        return len(parts) >= 2 and "norm" in parts[-2].lower() and parts[-1] == "weight"

    id_to_lr: dict[int, float] = {}
    for submod, sub_lr in submodule_lrs:
        for p in submod.parameters():
            id_to_lr.setdefault(id(p), sub_lr)

    buckets: dict[tuple[float, bool], list[nn.Parameter]] = {}
    for name, p in model.named_parameters():
        bucket_lr = id_to_lr.get(id(p), lr)
        nd = _is_no_decay(name)
        buckets.setdefault((bucket_lr, nd), []).append(p)

    return AdamW(
        [
            {"params": params, "lr": bucket_lr, "weight_decay": 0.0 if nd else weight_decay}
            for (bucket_lr, nd), params in buckets.items()
        ]
    )


def make_scheduler(optimizer, warmup_steps: int, total_steps: int):
    """Linear warmup, then linear decay to zero."""

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return max(0.0, float(total_steps - step) / float(max(1, total_steps - warmup_steps)))

    return LambdaLR(optimizer, lr_lambda)


def save_checkpoint(path: str, model: nn.Module, optimizer, scheduler, **extra) -> None:
    """Save model + training state. Caller passes any other state (config, step, etc.) via **extra.

    Also writes a `<path>.json` sidecar with the scalar `extra` fields, so
    list-style tools (e.g. `python -m iudex runs list`) can read run metadata
    without `torch.load`-ing the full checkpoint.
    """
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            **extra,
        },
        path,
    )
    sidecar = {k: v for k, v in extra.items() if k != "config" and isinstance(v, (int, float, str, bool))}
    sidecar_path = (path[:-3] if path.endswith(".pt") else path) + ".json"
    with open(sidecar_path, "w", encoding="utf-8") as f:
        json.dump(sidecar, f, indent=2)


def try_resume(checkpoint_path: str, *, expected_hash: str) -> dict[str, Any] | None:
    """Return the checkpoint dict iff it exists and its config_hash matches; otherwise None.

    A mismatch indicates a hand-copied `last.pt` or a checkpoint left behind
    by a previous hash scheme; we warn loudly so the user can stop us before
    we overwrite it.
    """
    if not os.path.exists(checkpoint_path):
        return None
    ckpt = torch.load(checkpoint_path, weights_only=False)
    found_hash = ckpt.get("config_hash")
    if found_hash != expected_hash:
        warn(
            f"Config hash mismatch on [path]{checkpoint_path}[/path] "
            f"(checkpoint={found_hash!r}, expected={expected_hash!r}). "
            f"Starting fresh — this run will overwrite the existing last.pt."
        )
        return None
    console.print(
        f"[bold cyan]Resuming[/bold cyan] from step {ckpt.get('global_step', '?')}, epoch {ckpt.get('epoch', '?')}"
    )
    return ckpt


def make_progress_bar() -> Progress:
    """Configured rich Progress for whole-tree training. `with make_progress_bar() as p:`"""
    return Progress(
        SpinnerColumn("dots"),
        TextColumn("[epoch]Epoch {task.fields[epoch]}[/epoch]"),
        BarColumn(bar_width=30, style="magenta", complete_style="bold magenta", finished_style="green"),
        MofNCompleteColumn(),
        TextColumn("[dim]|[/dim]"),
        TextColumn("{task.fields[loss_str]}"),
        TextColumn("{task.fields[lr_str]}"),
        TextColumn("{task.fields[mem_str]}"),
        TextColumn("[dim]|[/dim]"),
        TimeElapsedColumn(),
        TextColumn("[dim]/[/dim]"),
        TimeRemainingColumn(),
        TextColumn("[dim](total {task.fields[total_elapsed]})[/dim]"),
        console=console,
        transient=True,
    )


def config_panel(cfg_dict: dict) -> Panel:
    """Render a config dict (typically `dataclasses.asdict(cfg)`) as a pretty panel."""
    return Panel(Pretty(cfg_dict), title="[bold cyan]Config[/bold cyan]", border_style="cyan")


def device_panel(device: torch.device, *, seed: int, checkpoint_dir: str) -> Panel:
    info = Table(show_header=False, padding=(0, 2), box=None)
    info.add_column(style="bold cyan")
    info.add_column()
    info.add_row("Device", f"[bold]{device}[/bold]")
    if device.type == "cuda":
        info.add_row(
            "GPU",
            f"{torch.cuda.get_device_name(device)} "
            f"([green]{torch.cuda.get_device_properties(device).total_memory / 1024**3:.1f} GB[/green])",
        )
    info.add_row("Seed", str(seed))
    info.add_row("Checkpoint dir", f"[path]{checkpoint_dir}[/path]")
    return Panel(info, title="[bold magenta]iudex[/bold magenta] trainer", border_style="magenta")


def model_panel(model: nn.Module, *, num_train_trees: int, grad_accum: int) -> Panel:
    n_params = sum(p.numel() for p in model.parameters())
    n_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    dt = Table(show_header=False, padding=(0, 2), box=None)
    dt.add_column(style="bold cyan")
    dt.add_column()
    dt.add_row("Parameters", f"[bold]{n_params:,}[/bold] total, [bold]{n_train_params:,}[/bold] trainable")
    dt.add_row("Training trees", f"[bold]{num_train_trees:,}[/bold]")
    dt.add_row("Grad accum", str(grad_accum))
    return Panel(dt, title="[bold cyan]Data & Model[/bold cyan]", border_style="cyan")


def schedule_panel(
    *,
    steps_per_epoch: int,
    total_steps: int,
    warmup_steps: int,
    lr: float,
    encoder_lr: float | None = None,
) -> Panel:
    sched = Table(show_header=False, padding=(0, 2), box=None)
    sched.add_column(style="bold cyan")
    sched.add_column()
    sched.add_row("Steps/epoch", f"{steps_per_epoch:,}")
    sched.add_row("Total steps", f"{total_steps:,}")
    sched.add_row("Warmup steps", f"{warmup_steps:,}")
    sched.add_row("LR", f"{lr:.2e}")
    if encoder_lr is not None:
        sched.add_row("Encoder LR", f"{encoder_lr:.2e}")
    return Panel(sched, title="[bold yellow]Schedule[/bold yellow]", border_style="yellow")
