import argparse
import logging
import os
import sys
from glob import glob
from pathlib import Path

import torch
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)

from iudex.common.log import console, setup_logging
from iudex.rst.data.reader import read_rst_file
from iudex.rst.parsers.common.inference import load_parser_from_checkpoint, resolve_checkpoint
from iudex.rst.parsers.dmrst.configuration_dmrst import DMRSTConfig
from iudex.rst.parsers.dmrst.modeling_dmrst import DMRSTParser

setup_logging()
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Predict RST trees")
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--config", help="Jsonnet config; load best_model.pt from the derived run dir")
    source_group.add_argument("--checkpoint", help="Direct path to a .pt checkpoint")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input", help="RS3/RS4 file or directory (uses gold EDU segmentation)")
    input_group.add_argument(
        "--input-text",
        help="Raw .txt file or directory of .txt (requires joint_segmentation=True)",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    checkpoint_path = resolve_checkpoint(
        args.config,
        args.checkpoint,
        DMRSTConfig,
        "iudex.rst.parsers.dmrst.train_dmrst",
    )
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_parser_from_checkpoint(checkpoint_path, device, DMRSTConfig, DMRSTParser)
    console.print(f"[dim]Loaded model from[/dim] [path]{checkpoint_path}[/path]")

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
                    relation_types=model.config.relation_types,
                    relation_map=model.config.relation_map,
                )
                pred = model.predict(tree)
            else:
                with open(filepath, encoding="utf-8") as f:
                    text = f.read()
                pred = model.predict_from_text(text)
            out = os.path.join(args.output_dir, Path(filepath).stem + ".rs4")
            with open(out, "w", encoding="utf-8") as f:
                f.write(pred.to_rs4_string())
            progress.advance(task)

    console.print(f"[bold green]Done![/bold green] Wrote {len(paths)} predictions to [path]{args.output_dir}[/path]")


if __name__ == "__main__":
    main()
