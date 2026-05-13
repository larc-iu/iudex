"""Training utilities for iudex RST parsers.

This module is a bag of utility functions; parser-specific training scripts
(e.g. `iudex/rst/parsers/<name>/train_<name>.py`) import what they need and
own their own training loop. There is no `Trainer` class here intentionally —
the goal is that one file per parser tells the full training story top-to-bottom.

Each utility takes its inputs explicitly. We avoid assuming anything about
model or cfg structure beyond standard `nn.Module` interfaces.
"""
import hashlib
import json
import logging
import os
import random
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

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

from iudex.common.log import console, rule, success
from iudex.rst.data.metrics import compute_parseval_metrics
from iudex.rst.data.tree import RstPpTree

logger = logging.getLogger(__name__)


def config_hash(obj: Any) -> str:
    """First 12 hex chars (48-bit prefix) of SHA-256 over any JSON-serializable
    object (typically `dataclasses.asdict(cfg)`). Short enough to read in run-dir
    names, long enough that collisions across hundreds of runs are vanishingly
    rare (birthday-bound ≈ √2⁴⁸ ≈ 16M).
    """
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:12]


def set_seeds(seed: int) -> None:
    """Seed torch (and cuda, if present) plus the stdlib `random` module."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)


def gpu_mem_gb(device: torch.device) -> Optional[Tuple[float, float]]:
    """Return (allocated_gb, reserved_gb) for CUDA devices, else None."""
    if device.type != "cuda":
        return None
    return (
        torch.cuda.memory_allocated(device) / 1024 ** 3,
        torch.cuda.memory_reserved(device) / 1024 ** 3,
    )


def derive_run_id(
    config_dict: dict,
    run_name: Optional[str] = None,
    *,
    hash_exclude: Tuple[str, ...] = ("run_name",),
) -> Tuple[str, str]:
    """Pure: compute the run id ("{run_name}-{hash}" or just "{hash}") and hash.

    Display-only fields named in `hash_exclude` are stripped before hashing so
    they don't affect run identity. No I/O — safe to call from inference paths
    that just need to locate a previously-prepared run directory.
    Returns (run_id, cfg_hash).
    """
    hashable = {k: v for k, v in config_dict.items() if k not in hash_exclude}
    cfg_hash = config_hash(hashable)
    run_id = f"{run_name}-{cfg_hash}" if run_name else cfg_hash
    return run_id, cfg_hash


def prepare_run_dir(
    config_dict: dict,
    checkpoint_dir: str,
    run_name: Optional[str] = None,
    *,
    hash_exclude: Tuple[str, ...] = ("run_name",),
) -> Tuple[str, str]:
    """Derive `{checkpoint_dir}/{run_name}-{hash}` (or just `/{hash}`), `mkdir -p`
    it, and write `config.json` for audit. Returns (run_dir, cfg_hash).
    """
    run_id, cfg_hash = derive_run_id(config_dict, run_name, hash_exclude=hash_exclude)
    run_dir = os.path.join(checkpoint_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2, default=str)
    return run_dir, cfg_hash


def resume_or_init(
    run_dir: str,
    *,
    model: nn.Module,
    optimizer,
    scheduler,
    expected_hash: str,
) -> Dict[str, Any]:
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


def final_evaluation(
    *,
    model: nn.Module,
    run_dir: str,
    predict_fn: Callable[["RstPpTree"], "RstPpTree"],
    dev_pairs: List[Tuple[str, "RstPpTree"]],
    val_metric_name: str,
    best_val: float,
    test_pairs: Optional[List[Tuple[str, "RstPpTree"]]] = None,
) -> None:
    """Reload `{run_dir}/best_model.pt` if present, re-evaluate, and print results.
    Always reports dev; reports test too when `test_pairs` is given. Falls back to
    printing the recorded best metric if no best checkpoint exists."""
    rule("Final Evaluation")
    best_path = os.path.join(run_dir, "best_model.pt")
    if not os.path.exists(best_path):
        success(f"Training complete. Best {val_metric_name}: {best_val:.4f}")
        return
    ckpt = torch.load(best_path, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    dev_metrics = evaluate(
        predict_fn, dev_pairs,
        output_dir=os.path.join(run_dir, "dev_predictions", "final"),
    )
    console.print(metrics_table(dev_metrics, title="Final Dev Results"))
    if test_pairs is not None:
        test_metrics = evaluate(
            predict_fn, test_pairs,
            output_dir=os.path.join(run_dir, "test_predictions", "final"),
        )
        console.print(metrics_table(test_metrics, title="Final Test Results"))


def build_optimizer(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    *,
    submodule_lrs: Sequence[Tuple[nn.Module, float]] = (),
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

    id_to_lr: Dict[int, float] = {}
    for submod, sub_lr in submodule_lrs:
        for p in submod.parameters():
            id_to_lr.setdefault(id(p), sub_lr)

    buckets: Dict[Tuple[float, bool], List[nn.Parameter]] = {}
    for name, p in model.named_parameters():
        bucket_lr = id_to_lr.get(id(p), lr)
        nd = _is_no_decay(name)
        buckets.setdefault((bucket_lr, nd), []).append(p)

    return AdamW([
        {"params": params, "lr": bucket_lr, "weight_decay": 0.0 if nd else weight_decay}
        for (bucket_lr, nd), params in buckets.items()
    ])


def make_scheduler(optimizer, warmup_steps: int, total_steps: int):
    """Linear warmup, then linear decay to zero."""
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return max(0.0, float(total_steps - step) / float(max(1, total_steps - warmup_steps)))
    return LambdaLR(optimizer, lr_lambda)


def save_checkpoint(path: str, model: nn.Module, optimizer, scheduler, **extra) -> None:
    """Save model + training state. Caller passes any other state (config, step, etc.) via **extra."""
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        **extra,
    }, path)


def try_resume(checkpoint_path: str, *, expected_hash: str) -> Optional[Dict[str, Any]]:
    """Return the checkpoint dict iff it exists and its config_hash matches; otherwise None."""
    if not os.path.exists(checkpoint_path):
        return None
    ckpt = torch.load(checkpoint_path, weights_only=False)
    if ckpt.get("config_hash") != expected_hash:
        logger.info("Config hash mismatch; starting fresh")
        return None
    console.print(
        f"[bold cyan]Resuming[/bold cyan] from step {ckpt.get('global_step', '?')}, "
        f"epoch {ckpt.get('epoch', '?')}"
    )
    return ckpt


def metrics_table(metrics: Dict[str, float], title: str) -> Table:
    table = Table(title=title, show_header=True, header_style="bold cyan", padding=(0, 1))
    table.add_column("Metric", style="dim")
    table.add_column("F1", justify="right", style="bold green")
    for name in ["span", "nuc", "rel", "full"]:
        table.add_row(name.upper(), f"{metrics[f'{name}_f1']:.4f}")
    return table


@torch.no_grad()
def evaluate(
    predict_fn: Callable[[RstPpTree], RstPpTree],
    dev_pairs: List[Tuple[str, RstPpTree]],
    output_dir: Optional[str] = None,
) -> Dict[str, float]:
    """Run `predict_fn` over each (path, gold_tree) pair and report Parseval F1s.

    Caller is responsible for setting the underlying model to eval mode if applicable.
    If `output_dir` is given, write predicted .rs4 files keyed by the input filename stem.
    """
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    totals = {f"{m}_{x}_count": 0 for m in ["span", "nuc", "rel", "full"] for x in ["p", "r"]}
    totals["num_spans"] = 0

    for filepath, gold in dev_pairs:
        pred = predict_fn(gold)
        m = compute_parseval_metrics(gold, pred)
        for k in totals:
            totals[k] += m[k]
        if output_dir is not None:
            basename = os.path.splitext(os.path.basename(filepath))[0] + ".rs4"
            with open(os.path.join(output_dir, basename), "w", encoding="utf-8") as f:
                f.write(pred.to_rs4_string())

    n = totals["num_spans"]
    if n == 0:
        return {f"{m}_f1": 0.0 for m in ["span", "nuc", "rel", "full"]}

    def f1(p, r):
        return (2 * p * r) / (p + r) if (p + r) > 0 else 0.0

    return {
        f"{m}_f1": f1(totals[f"{m}_p_count"] / n, totals[f"{m}_r_count"] / n)
        for m in ["span", "nuc", "rel", "full"]
    }


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
    info.add_column(style="bold cyan"); info.add_column()
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
    dt.add_column(style="bold cyan"); dt.add_column()
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
    encoder_lr: Optional[float] = None,
) -> Panel:
    sched = Table(show_header=False, padding=(0, 2), box=None)
    sched.add_column(style="bold cyan"); sched.add_column()
    sched.add_row("Steps/epoch", f"{steps_per_epoch:,}")
    sched.add_row("Total steps", f"{total_steps:,}")
    sched.add_row("Warmup steps", f"{warmup_steps:,}")
    sched.add_row("LR", f"{lr:.2e}")
    if encoder_lr is not None:
        sched.add_row("Encoder LR", f"{encoder_lr:.2e}")
    return Panel(sched, title="[bold yellow]Schedule[/bold yellow]", border_style="yellow")
