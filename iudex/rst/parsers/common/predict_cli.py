"""Shared `iudex <parser> predict` CLI. Each parser's `predict_<name>.py`
is a thin shim that calls `run_predict(name)`."""

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
from iudex.rst.parsers import PARSERS
from iudex.rst.parsers.common.inference import load_parser_from_checkpoint, resolve_source
from iudex.rst.parsers.hfhub import load_parser_from_pretrained

setup_logging()
logger = logging.getLogger(__name__)


def run_predict(parser_name: str) -> None:
    spec = PARSERS[parser_name]
    config_cls = spec.load_config_cls()
    parser_cls = spec.load_parser_cls()

    argp = argparse.ArgumentParser(prog=f"iudex {parser_name} predict", description="Predict RST trees")
    source_group = argp.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--config", help="Jsonnet config; load best_model.pt from the derived run dir")
    source_group.add_argument("--checkpoint", help="Direct path to a .pt checkpoint")
    source_group.add_argument(
        "--hub-id",
        dest="hub_id",
        help=f"HuggingFace Hub repo id (e.g. larc-iu/{parser_name}-rstdt-coarse)",
    )

    if spec.supports_text:
        input_group = argp.add_mutually_exclusive_group(required=True)
        input_group.add_argument("--input", help="RS3/RS4 file or directory (uses gold EDU segmentation)")
        input_group.add_argument(
            "--text-file",
            dest="text_file",
            help="Path to a raw .txt file or a directory of .txt files (requires a model with segmentation)",
        )
        input_group.add_argument(
            "--text",
            help="Inline raw text string; parsed tree is written to stdout as RS4",
        )
    else:
        argp.add_argument("--input", required=True, help="RS3/RS4 file or directory")

    argp.add_argument("--output-dir", help="Required unless --text is used")
    argp.add_argument("--device", default=None)
    argp.add_argument(
        "--compile-encoder",
        dest="compile_encoder",
        action="store_true",
        help="torch.compile the encoder forward (CUDA only). Off by default for inference; "
        "pays an initial compile cost to speed up bulk prediction.",
    )
    args = argp.parse_args()

    # `text` / `text_file` only exist on parsers with supports_text=True.
    text_arg = getattr(args, "text", None)
    text_file_arg = getattr(args, "text_file", None)

    if text_arg is None and not args.output_dir:
        argp.error("--output-dir is required when --input or --text-file is used")

    kind, source = resolve_source(args.config, args.checkpoint, args.hub_id, config_cls, parser_name)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    if kind == "hub":
        model = load_parser_from_pretrained(
            source, parser_cls=parser_cls, config_cls=config_cls, device=device, compile_encoder=args.compile_encoder
        )
    else:
        model = load_parser_from_checkpoint(
            source, device, config_cls, parser_cls, compile_encoder=args.compile_encoder
        )
    console.print(f"[dim]Loaded model from[/dim] [path]{source}[/path]")

    if text_arg is not None:
        _require_segmenter(model, "--text")
        print(model.predict_from_text(text_arg).to_rs4_string())
        return

    os.makedirs(args.output_dir, exist_ok=True)
    if text_file_arg is not None:
        _require_segmenter(model, "--text-file")
        paths = _glob_or_single(text_file_arg, ("*.txt",))
    else:
        paths = _glob_or_single(args.input, ("*.rs3", "*.rs4"))

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
            if text_file_arg is not None:
                with open(filepath, encoding="utf-8") as f:
                    pred = model.predict_from_text(f.read())
            else:
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

    console.print(
        f"[bold green]Done![/bold green] Wrote {len(paths)} predictions to "
        f"[path]{os.path.abspath(args.output_dir)}[/path]"
    )


def _glob_or_single(path: str, patterns: tuple[str, ...]) -> list[str]:
    if os.path.isdir(path):
        out: list[str] = []
        for pat in patterns:
            out.extend(sorted(glob(str(Path(path) / pat))))
        return out
    return [path]


def _require_segmenter(model, flag: str) -> None:
    if model.segmenter is None:
        console.print(
            f"[bold red]This model has no segmenter[/bold red]. Train with "
            f"a non-null `segmentation:` block in your jsonnet to use {flag}."
        )
        sys.exit(1)
