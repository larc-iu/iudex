"""Sexy logging for iudex, powered by Rich."""

import logging

from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme

# Project-wide console with a custom theme
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
    """Bold-green status line."""
    console.print(f"[bold green]{msg}[/bold green]")


def warn(msg: str) -> None:
    """Bold-yellow status line."""
    console.print(f"[bold yellow]{msg}[/bold yellow]")


def dim(msg: str) -> None:
    """Dimmed status line."""
    console.print(f"[dim]{msg}[/dim]")


def rule(title: str) -> None:
    """Horizontal rule with a bold-magenta title."""
    console.rule(f"[bold magenta]{title}[/bold magenta]")


def setup_logging(level: int = logging.INFO) -> None:
    """Configure logging with Rich for beautiful terminal output."""
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
