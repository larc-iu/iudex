"""Rich-based logging and console helpers."""

import logging

from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme

theme = Theme(
    {
        "info": "cyan",
        "warning": "yellow",
        "error": "bold red",
        "metric": "bold green",
        "metric.name": "dim",
        "epoch": "bold magenta",
        "step": "dim cyan",
        "lr": "dim yellow",
        "loss": "bold orange1",
        "gpu": "bold green",
        "path": "underline blue",
    }
)

console = Console(theme=theme)


def success(msg: str) -> None:
    console.print(f"[bold green]{msg}[/bold green]")


def warn(msg: str) -> None:
    console.print(f"[bold yellow]{msg}[/bold yellow]")


def dim(msg: str) -> None:
    console.print(f"[dim]{msg}[/dim]")


def rule(title: str) -> None:
    console.rule(f"[bold magenta]{title}[/bold magenta]")


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(
                console=console,
                rich_tracebacks=True,
                tracebacks_show_locals=True,
                show_path=False,
                markup=True,
            )
        ],
        force=True,
    )
