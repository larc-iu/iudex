"""Training utilities shared across parser-specific `train_<name>.py` scripts.

The on-disk layout written here (`last.pt`/`last.json`,
`best_model.pt`/`best_model.json`, `config.json`) is the contract
`iudex.runs` reads. Frameworks that bypass these helpers don't show up
in `iudex runs list`.
"""

import hashlib
import json
import logging
import os
import random
import signal
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
from torch.utils.tensorboard import SummaryWriter

from iudex.common.log import console, dim, warn

logger = logging.getLogger(__name__)


# Framework-agnostic fields stripped before hashing for `run_id`. Changing
# any of these leaves an existing run resumable. Frameworks extend this
# (e.g. `iudex.rst.HASH_EXCLUDE` adds `relation_types`) and pass the
# combined tuple via `hash_exclude=`.
DEFAULT_HASH_EXCLUDE: tuple[str, ...] = (
    "run_name",
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
    """First 12 hex chars of SHA-256 over a JSON-serializable obj. Raises
    TypeError on non-JSON values (silent `str()`-ifying would make hashes
    platform- and Python-version-dependent)."""
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:12]


def set_seeds(seed: int) -> None:
    """Seed torch/cuda/random and opt into TF32 matmul (no-op on non-Ampere)."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.set_float32_matmul_precision("high")


def install_abort_handler():
    """SIGINT soft-abort. First Ctrl-C sets `flag.value = True`. The
    second restores the default handler so a hung cleanup path is still
    hard-killable. Returns the flag object."""

    class _Flag:
        value = False

    flag = _Flag()

    def handler(signum, frame):
        if flag.value:
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            return
        flag.value = True
        console.print("\n[yellow]Abort received; finishing the current step and writing the best model.[/yellow]")

    signal.signal(signal.SIGINT, handler)
    return flag


def gpu_mem_gb(device: torch.device) -> tuple[float, float] | None:
    """(allocated_gb, reserved_gb) for CUDA devices, else None."""
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
    affect run identity (see `DEFAULT_HASH_EXCLUDE`). No I/O, so safe to call
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
    """`mkdir -p` the derived run dir and return (run_dir, cfg_hash). On
    first creation, hints at the closest sibling run (catches accidental
    field bumps that branched into a new hash). Does NOT write
    `config.json`. Callers do that after resolving inferred fields, so
    the on-disk audit, embedded ckpt config, and Hub config.json all match.
    """
    run_id, cfg_hash = derive_run_id(config_dict, run_name, hash_exclude=hash_exclude)
    run_dir = os.path.join(checkpoint_dir, run_id)
    is_fresh = not os.path.exists(os.path.join(run_dir, "last.pt"))
    os.makedirs(run_dir, exist_ok=True)
    if is_fresh:
        _hint_closest_sibling(run_dir, checkpoint_dir, config_dict, hash_exclude=hash_exclude)
    return run_dir, cfg_hash


def write_run_config(run_dir: str, config_dict: dict) -> None:
    """Write `{run_dir}/config.json` (overwrites). Call after resolving
    inferred fields so audit / embedded ckpt config / Hub config all match.
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
    """Print a hint if a sibling run differs by ≤5 hash-affecting fields."""
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
    """Restore from `{run_dir}/last.pt` if hash matches, else fresh state.
    Returns dict with keys: global_step, epoch, best_val, stale_validations.
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
    """AdamW with no-decay on biases / norm weights, plus per-submodule LRs.
    Params in each `(submodule, sub_lr)` entry use `sub_lr` (first listed
    wins on overlap). Everything else uses `lr`.
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
    """Save model + training state to `path`. Also writes a `<path>.json`
    sidecar with the scalar `extra` fields so `iudex runs list` can read
    metadata without `torch.load`-ing the full checkpoint.
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
    """Checkpoint dict iff it exists and config_hash matches, else None.
    Mismatch warns loudly (hand-copied last.pt, or a previous hash scheme)
    so the user can stop us before overwriting.
    """
    if not os.path.exists(checkpoint_path):
        return None
    ckpt = torch.load(checkpoint_path, weights_only=False)
    found_hash = ckpt.get("config_hash")
    if found_hash != expected_hash:
        warn(
            f"Config hash mismatch on [path]{checkpoint_path}[/path] "
            f"(checkpoint={found_hash!r}, expected={expected_hash!r}). "
            f"Starting fresh. This run will overwrite the existing last.pt."
        )
        return None
    console.print(
        f"[bold cyan]Resuming[/bold cyan] from step {ckpt.get('global_step', '?')}, epoch {ckpt.get('epoch', '?')}"
    )
    return ckpt


class TBLogger:
    """Thin SummaryWriter wrapper writing to `{run_dir}/tb`. `log_scalars`
    namespaces each value under `prefix/` so TensorBoard groups train/ vs dev/.
    On resume a new event file is appended into the same dir; TensorBoard merges
    by tag and step."""

    def __init__(self, run_dir: str):
        self.writer = SummaryWriter(log_dir=os.path.join(run_dir, "tb"))

    def log_scalars(self, prefix: str, scalars: dict[str, float], step: int) -> None:
        for name, value in scalars.items():
            self.writer.add_scalar(f"{prefix}/{name}", value, step)

    def close(self) -> None:
        self.writer.close()


def make_progress_bar() -> Progress:
    """Rich Progress configured for whole-tree training."""
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
