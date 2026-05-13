"""Inference for the dmrst parser.

Two ways to identify the model:

    # From a config: looks up the run dir's best_model.pt
    python -m iudex.rst.parsers.dmrst.predict_dmrst \\
        --config configs/dmrst.jsonnet --input X --output-dir Y

    # From an explicit checkpoint: useful for shared .pt files, or to pick a
    # specific intermediate checkpoint (e.g. last.pt instead of best_model.pt)
    python -m iudex.rst.parsers.dmrst.predict_dmrst \\
        --checkpoint checkpoints/<run_id>/best_model.pt --input X --output-dir Y
"""
import argparse
import dataclasses
import logging
import os
import sys
from glob import glob
from pathlib import Path

import torch
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn
from tonga import Params

from iudex.common.log import console, setup_logging
from iudex.rst.data.reader import read_rst_file
from iudex.rst.parsers.dmrst.configuration_dmrst import DMRSTConfig
from iudex.rst.parsers.dmrst.modeling_dmrst import DMRSTParser
from iudex.rst.training import derive_run_id

setup_logging()
logger = logging.getLogger(__name__)


def _resolve_checkpoint(config_path: str, checkpoint_path: str) -> str:
    """Return the .pt path to load. Exactly one of the two args is non-None."""
    if checkpoint_path:
        if not os.path.exists(checkpoint_path):
            console.print(f"[bold red]Checkpoint not found:[/bold red] [path]{checkpoint_path}[/path]")
            sys.exit(1)
        return checkpoint_path

    cfg = DMRSTConfig.from_dict(Params.from_file(config_path).as_dict(quiet=True))
    run_id, _ = derive_run_id(dataclasses.asdict(cfg), cfg.run_name)
    run_dir = os.path.join(cfg.checkpoint_dir, run_id)
    ckpt = os.path.join(run_dir, "best_model.pt")
    if not os.path.exists(ckpt):
        console.print(
            f"[bold red]No trained model found for this config.[/bold red]\n"
            f"  Expected: [path]{ckpt}[/path]\n"
            f"  Train first with:\n"
            f"    python -m iudex.rst.parsers.dmrst.train_dmrst {config_path}"
        )
        sys.exit(1)
    return ckpt


def load_model(checkpoint_path: str, device: torch.device) -> DMRSTParser:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = DMRSTConfig.from_dict(ckpt["config"])
    model = DMRSTParser(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device).eval()


def main():
    parser = argparse.ArgumentParser(description="Predict RST trees")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--config", help="Jsonnet config; load best_model.pt from the derived run dir")
    src.add_argument("--checkpoint", help="Direct path to a .pt checkpoint")
    inp = parser.add_mutually_exclusive_group(required=True)
    inp.add_argument("--input", help="RS3/RS4 file or directory (uses gold EDU segmentation)")
    inp.add_argument(
        "--input-text",
        help="Raw .txt file or directory of .txt (requires joint_segmentation=True)",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    ckpt_path = _resolve_checkpoint(args.config, args.checkpoint)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_model(ckpt_path, device)
    console.print(f"[dim]Loaded model from[/dim] [path]{ckpt_path}[/path]")

    os.makedirs(args.output_dir, exist_ok=True)
    if args.input is not None:
        if os.path.isdir(args.input):
            paths = sorted(glob(str(Path(args.input) / "*.rs3"))) + sorted(glob(str(Path(args.input) / "*.rs4")))
        else:
            paths = [args.input]
    else:
        if model.segmenter is None:
            console.print(
                "[bold red]This model has no segmenter[/bold red] — train with "
                "`joint_segmentation: true` to use --input-text."
            )
            sys.exit(1)
        if os.path.isdir(args.input_text):
            paths = sorted(glob(str(Path(args.input_text) / "*.txt")))
        else:
            paths = [args.input_text]

    with Progress(
        SpinnerColumn("dots"),
        TextColumn("[bold cyan]Predicting[/bold cyan]"),
        BarColumn(bar_width=30, style="magenta", complete_style="bold magenta", finished_style="green"),
        MofNCompleteColumn(),
        TextColumn("[dim]|[/dim]"),
        TextColumn("{task.fields[current_file]}"),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("predict", total=len(paths), current_file="")
        for filepath in paths:
            progress.update(task, current_file=f"[dim]{Path(filepath).name}[/dim]")
            if args.input is not None:
                tree = read_rst_file(
                    filepath,
                    relation_types=model.relation_types,
                    relation_map=getattr(model.config, "relation_map", None),
                )
                pred = model.predict(tree)
            else:
                with open(filepath, "r", encoding="utf-8") as f:
                    text = f.read()
                pred = model.predict_from_text(text)
            out = os.path.join(args.output_dir, Path(filepath).stem + ".rs4")
            with open(out, "w", encoding="utf-8") as f:
                f.write(pred.to_rs4_string())
            progress.advance(task)

    console.print(f"[bold green]Done![/bold green] Wrote {len(paths)} predictions to [path]{args.output_dir}[/path]")


if __name__ == "__main__":
    main()
