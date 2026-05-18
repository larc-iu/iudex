"""Inspect existing iudex training runs.

Usage:
    python -m iudex runs list [--checkpoint-dir checkpoints/]
"""

import argparse
import json
import os
import sys
from datetime import datetime

from rich.table import Table

from iudex.common.log import console

# Each parser dataclass has at least one field no other parser has, so we can
# tag run rows by parser kind from the on-disk config.json alone.
_PARSER_KIND_BY_SIGNATURE_FIELD = [
    ("attention_type", "dmrst"),
    ("ffn_hidden_size", "topdown_biaffine"),
]


def _infer_parser_kind(config: dict) -> str:
    for key, kind in _PARSER_KIND_BY_SIGNATURE_FIELD:
        if key in config:
            return kind
    return "?"


def _read_best_meta(run_dir: str) -> tuple[str, str]:
    """(best_val_str, step_str) read from the best_model.json sidecar.

    Falls back to "-" / "-" if no sidecar (e.g. a run that hasn't validated yet).
    """
    sidecar = os.path.join(run_dir, "best_model.json")
    if not os.path.exists(sidecar):
        return ("(no best)" if not os.path.exists(os.path.join(run_dir, "best_model.pt")) else "-"), "-"
    try:
        with open(sidecar, encoding="utf-8") as f:
            meta = json.load(f)
    except (OSError, json.JSONDecodeError):
        return "-", "-"
    val = meta.get("best_val")
    val_str = f"{val:.4f}" if isinstance(val, (int, float)) and val >= 0 else "-"
    step = meta.get("global_step")
    step_str = str(step) if isinstance(step, int) else "-"
    return val_str, step_str


def list_runs(checkpoint_dir: str) -> None:
    if not os.path.isdir(checkpoint_dir):
        console.print(f"[bold red]No such directory:[/bold red] [path]{checkpoint_dir}[/path]")
        sys.exit(1)

    rows: list[tuple[str, ...]] = []
    for entry in sorted(os.listdir(checkpoint_dir)):
        run_dir = os.path.join(checkpoint_dir, entry)
        config_path = os.path.join(run_dir, "config.json")
        if not os.path.isdir(run_dir) or not os.path.exists(config_path):
            continue
        try:
            with open(config_path, encoding="utf-8") as f:
                cfg = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        kind = _infer_parser_kind(cfg)
        run_name = cfg.get("run_name") or "-"
        model_name = cfg.get("model_name", "?")
        train_dir = cfg.get("train_dir") or "?"
        best_val_str, step_str = _read_best_meta(run_dir)

        # mtime from the freshest signal so the column reflects when the run
        # last did real work (not just when config.json was written).
        mtime_src = next(
            (
                p
                for p in (
                    os.path.join(run_dir, "best_model.pt"),
                    os.path.join(run_dir, "last.pt"),
                    config_path,
                )
                if os.path.exists(p)
            ),
            config_path,
        )
        modified = datetime.fromtimestamp(os.path.getmtime(mtime_src)).strftime("%Y-%m-%d %H:%M")
        rows.append((entry, run_name, kind, model_name, train_dir, best_val_str, step_str, modified))

    if not rows:
        console.print(f"[dim]No runs found in[/dim] [path]{checkpoint_dir}[/path]")
        return

    table = Table(
        title=f"Runs in {checkpoint_dir}",
        show_header=True,
        header_style="bold cyan",
        padding=(0, 1),
    )
    table.add_column("run_id", style="bold")
    table.add_column("run_name", style="dim")
    table.add_column("parser")
    table.add_column("model_name")
    table.add_column("train_dir")
    table.add_column("best_val", justify="right", style="bold green")
    table.add_column("step", justify="right", style="dim")
    table.add_column("modified", style="dim")
    for row in rows:
        table.add_row(*row)
    console.print(table)


def main():
    parser = argparse.ArgumentParser(description="Inspect iudex training runs")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)
    p_list = subparsers.add_parser("list", help="List runs under a checkpoint directory")
    p_list.add_argument("--checkpoint-dir", default="checkpoints", help="Root checkpoint dir to walk")
    args = parser.parse_args()
    if args.subcommand == "list":
        list_runs(args.checkpoint_dir)


if __name__ == "__main__":
    main()
