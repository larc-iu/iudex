import argparse
import logging
import os
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
from iudex.rst.parsers.hfhub import load_parser_from_pretrained
from iudex.rst.parsers.common.inference import load_parser_from_checkpoint, resolve_source
from iudex.rst.parsers.topdown_biaffine.configuration_topdown_biaffine import TopdownBiaffineConfig
from iudex.rst.parsers.topdown_biaffine.modeling_topdown_biaffine import TopdownBiaffineParser

setup_logging()
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Predict RST trees")
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--config", help="Jsonnet config; load best_model.pt from the derived run dir")
    source_group.add_argument("--checkpoint", help="Direct path to a .pt checkpoint")
    source_group.add_argument(
        "--hub-id", dest="hub_id", help="HuggingFace Hub repo id (e.g. larc-iu/topdown_biaffine-rstdt-coarse)"
    )
    parser.add_argument("--input", required=True, help="RS3/RS4 file or directory")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    kind, source = resolve_source(
        args.config,
        args.checkpoint,
        args.hub_id,
        TopdownBiaffineConfig,
        "iudex.rst.parsers.topdown_biaffine.train_topdown_biaffine",
    )
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    if kind == "hub":
        model = load_parser_from_pretrained(
            source, parser_cls=TopdownBiaffineParser, config_cls=TopdownBiaffineConfig, device=device
        )
    else:
        model = load_parser_from_checkpoint(source, device, TopdownBiaffineConfig, TopdownBiaffineParser)
    console.print(f"[dim]Loaded model from[/dim] [path]{source}[/path]")

    os.makedirs(args.output_dir, exist_ok=True)
    if os.path.isdir(args.input):
        paths = sorted(glob(str(Path(args.input) / "*.rs3"))) + sorted(glob(str(Path(args.input) / "*.rs4")))
    else:
        paths = [args.input]

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
            tree = read_rst_file(
                filepath,
                relation_types=model.config.relation_types,
                relation_map=model.config.relation_map,
            )
            pred = model.predict(tree)
            out = os.path.join(args.output_dir, Path(filepath).stem + ".rs4")
            with open(out, "w", encoding="utf-8") as f:
                f.write(pred.to_rs4_string())
            progress.advance(task)

    console.print(f"[bold green]Done![/bold green] Wrote {len(paths)} predictions to [path]{args.output_dir}[/path]")


if __name__ == "__main__":
    main()
