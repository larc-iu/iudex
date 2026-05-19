"""Inspect and manage iudex training runs.

Usage:
    python -m iudex runs list   [--checkpoint-dir DIR]
    python -m iudex runs show   <run_id>            [--checkpoint-dir DIR]
    python -m iudex runs diff   <run_id_a> <run_id_b>  [--checkpoint-dir DIR]
    python -m iudex runs rename <run_id> <new_run_name> [--checkpoint-dir DIR]
    python -m iudex runs delete <run_id> [--yes]    [--checkpoint-dir DIR]
    python -m iudex runs delete-all                 [--checkpoint-dir DIR]

`<run_id>` may be any unique prefix of an actual run dir name; ambiguous
prefixes error out listing the candidates.

Framework-agnostic: walks every framework declared in `iudex.FRAMEWORKS`,
unioning each framework's `PARSERS` registry to tag run rows by parser
kind via the `signature_field` (a config field unique to each parser
dataclass).

Assumes the sidecar conventions written by `iudex.common.training`
(`config.json`, `best_model.json`, `best_model.pt`, `last.pt`). Frameworks
that bypass those helpers won't show up here.
"""

import argparse
import importlib
import json
import os
import re
import shutil
import sys
from datetime import datetime

from rich.panel import Panel
from rich.pretty import Pretty
from rich.table import Table

import iudex
from iudex.common.log import console, dim, success

# Last 12 hex chars of the run dir name are the config hash; everything
# before is the optional run_name.
_HASH_SUFFIX_RE = re.compile(r"(?:^|-)([0-9a-f]{12})$")


def _all_parsers() -> dict:
    """Merge `PARSERS` across every framework in `iudex.FRAMEWORKS`. Aborts
    on a `signature_field` collision — each parser's signature_field must
    be globally unique so `_infer_parser_kind` is well-defined.
    """
    merged: dict = {}
    by_sig: dict[str, str] = {}
    for fw_path in iudex.FRAMEWORKS:
        fw = importlib.import_module(fw_path)
        for name, spec in fw.PARSERS.items():
            merged[name] = spec
            owner = by_sig.get(spec.signature_field)
            if owner is not None and owner != name:
                sys.stderr.write(
                    f"iudex.runs: parsers {owner!r} and {name!r} both claim "
                    f"signature_field {spec.signature_field!r}. Pick a "
                    f"field unique to one of them.\n"
                )
                sys.exit(2)
            by_sig[spec.signature_field] = name
    return merged


def _infer_parser_kind(config: dict, parsers: dict) -> str:
    for name, spec in parsers.items():
        if spec.signature_field in config:
            return name
    return "?"


def _list_run_dirs(checkpoint_dir: str) -> list[str]:
    """Return sorted run-dir basenames under `checkpoint_dir` that look like
    actual runs (have a `config.json`)."""
    if not os.path.isdir(checkpoint_dir):
        return []
    out = []
    for entry in sorted(os.listdir(checkpoint_dir)):
        if os.path.exists(os.path.join(checkpoint_dir, entry, "config.json")):
            out.append(entry)
    return out


def _resolve_run_id(checkpoint_dir: str, partial: str) -> str:
    """Find the unique run dir whose name starts with `partial`. Exits
    non-zero if no match or multiple matches (listing the candidates)."""
    matches = [e for e in _list_run_dirs(checkpoint_dir) if e.startswith(partial)]
    if not matches:
        console.print(f"[bold red]No run matching[/bold red] [path]{partial}[/path] in [path]{checkpoint_dir}[/path]")
        sys.exit(1)
    if len(matches) > 1:
        console.print(f"[bold red]Ambiguous run id[/bold red] [path]{partial}[/path]:")
        for m in matches:
            console.print(f"  [path]{m}[/path]")
        sys.exit(1)
    return matches[0]


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


def _dir_size(path: str) -> int:
    """Total bytes under `path`, following dirents only (no symlinks)."""
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    return total


def _format_size(n: int) -> str:
    unit = "B"
    val: float = float(n)
    for u in ("KB", "MB", "GB", "TB"):
        if val < 1024:
            break
        val /= 1024
        unit = u
    return f"{val:.1f} {unit}" if unit != "B" else f"{n} B"


def _latest_mtime(run_dir: str) -> float:
    """Use the freshest signal (best_model.pt → last.pt → config.json) so
    callers see when the run last did real work."""
    for name in ("best_model.pt", "last.pt", "config.json"):
        p = os.path.join(run_dir, name)
        if os.path.exists(p):
            return os.path.getmtime(p)
    return os.path.getmtime(run_dir)


# ---------------------------------------------------------------------------
# `runs list`


def list_runs(checkpoint_dir: str) -> None:
    if not os.path.isdir(checkpoint_dir):
        console.print(f"[bold red]No such directory:[/bold red] [path]{checkpoint_dir}[/path]")
        sys.exit(1)

    parsers = _all_parsers()
    rows: list[tuple[str, ...]] = []
    for entry in _list_run_dirs(checkpoint_dir):
        run_dir = os.path.join(checkpoint_dir, entry)
        try:
            with open(os.path.join(run_dir, "config.json"), encoding="utf-8") as f:
                cfg = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        kind = _infer_parser_kind(cfg, parsers)
        run_name = cfg.get("run_name") or "-"
        model_name = cfg.get("model_name", "?")
        train_dir = cfg.get("train_dir") or "?"
        best_val_str, step_str = _read_best_meta(run_dir)
        modified = datetime.fromtimestamp(_latest_mtime(run_dir)).strftime("%Y-%m-%d %H:%M")
        rows.append((entry, run_name, kind, model_name, train_dir, best_val_str, step_str, modified))

    if not rows:
        console.print(f"[dim]No runs found in[/dim] [path]{checkpoint_dir}[/path]")
        return

    table = Table(title=f"Runs in {checkpoint_dir}", show_header=True, header_style="bold cyan", padding=(0, 1))
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


# ---------------------------------------------------------------------------
# `runs show`


def show_run(checkpoint_dir: str, partial: str) -> None:
    run_id = _resolve_run_id(checkpoint_dir, partial)
    run_dir = os.path.join(checkpoint_dir, run_id)
    parsers = _all_parsers()

    with open(os.path.join(run_dir, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)

    kind = _infer_parser_kind(cfg, parsers)
    best_val_str, step_str = _read_best_meta(run_dir)
    size = _format_size(_dir_size(run_dir))
    modified = datetime.fromtimestamp(_latest_mtime(run_dir)).strftime("%Y-%m-%d %H:%M")

    header = Table(show_header=False, padding=(0, 2), box=None)
    header.add_column(style="bold cyan")
    header.add_column()
    header.add_row("run_id", run_id)
    header.add_row("run_name", cfg.get("run_name") or "-")
    header.add_row("parser", kind)
    header.add_row("run_dir", run_dir)
    header.add_row("modified", modified)
    header.add_row("size", size)
    header.add_row("best_val", best_val_str)
    header.add_row("step", step_str)
    console.print(Panel(header, title=f"[bold magenta]Run[/bold magenta] {run_id}", border_style="magenta"))

    console.print(Panel(Pretty(cfg), title="[bold cyan]Config[/bold cyan]", border_style="cyan"))

    fm_path = os.path.join(run_dir, "final_metrics.json")
    if os.path.exists(fm_path):
        try:
            with open(fm_path, encoding="utf-8") as f:
                final = json.load(f)
            metrics = Table(show_header=True, header_style="bold cyan", padding=(0, 1))
            metrics.add_column("split", style="bold")
            all_keys: list[str] = []
            for d in final.values():
                if isinstance(d, dict):
                    for k in d:
                        if k not in all_keys:
                            all_keys.append(k)
            for k in all_keys:
                metrics.add_column(k, justify="right")
            for split, d in final.items():
                if not isinstance(d, dict):
                    continue
                metrics.add_row(
                    split, *[f"{d[k]:.4f}" if k in d and isinstance(d[k], (int, float)) else "-" for k in all_keys]
                )
            console.print(Panel(metrics, title="[bold green]Final metrics[/bold green]", border_style="green"))
        except (OSError, json.JSONDecodeError):
            pass

    files = Table(show_header=False, padding=(0, 2), box=None)
    files.add_column(style="bold")
    files.add_column(justify="right", style="dim")
    for name in sorted(os.listdir(run_dir)):
        full = os.path.join(run_dir, name)
        if os.path.isdir(full):
            count = sum(1 for _ in os.scandir(full))
            files.add_row(f"{name}/", f"{count} entries")
        else:
            files.add_row(name, _format_size(os.path.getsize(full)))
    console.print(Panel(files, title="[bold yellow]Files[/bold yellow]", border_style="yellow"))


# ---------------------------------------------------------------------------
# `runs diff`


def diff_runs(checkpoint_dir: str, partial_a: str, partial_b: str) -> None:
    a = _resolve_run_id(checkpoint_dir, partial_a)
    b = _resolve_run_id(checkpoint_dir, partial_b)
    if a == b:
        console.print(f"[dim]{a} and {b} resolve to the same run.[/dim]")
        return

    with open(os.path.join(checkpoint_dir, a, "config.json"), encoding="utf-8") as f:
        cfg_a = json.load(f)
    with open(os.path.join(checkpoint_dir, b, "config.json"), encoding="utf-8") as f:
        cfg_b = json.load(f)

    keys = sorted(set(cfg_a) | set(cfg_b))
    diffs = [k for k in keys if cfg_a.get(k) != cfg_b.get(k)]

    if not diffs:
        console.print(f"[green]No differences[/green] between [path]{a}[/path] and [path]{b}[/path].")
        return

    sentinel = object()
    table = Table(
        title=f"Diff: {a} vs {b}",
        show_header=True,
        header_style="bold cyan",
        padding=(0, 1),
    )
    table.add_column("field", style="bold")
    table.add_column(a[:16])
    table.add_column(b[:16])
    for k in diffs:
        va = cfg_a.get(k, sentinel)
        vb = cfg_b.get(k, sentinel)
        sa = "[dim]<missing>[/dim]" if va is sentinel else _short_repr(va)
        sb = "[dim]<missing>[/dim]" if vb is sentinel else _short_repr(vb)
        table.add_row(k, sa, sb)
    console.print(table)


def _short_repr(v) -> str:
    """One-line repr suitable for a table cell; truncates long collections."""
    if isinstance(v, (dict, list)) and len(str(v)) > 60:
        return f"{type(v).__name__}({len(v)} items)"
    return repr(v)


# ---------------------------------------------------------------------------
# `runs rename`


def rename_run(checkpoint_dir: str, partial: str, new_run_name: str) -> None:
    if not new_run_name or "/" in new_run_name:
        console.print(f"[bold red]Invalid run_name:[/bold red] {new_run_name!r}")
        sys.exit(1)
    run_id = _resolve_run_id(checkpoint_dir, partial)
    run_dir = os.path.join(checkpoint_dir, run_id)

    m = _HASH_SUFFIX_RE.search(run_id)
    if m is None:
        console.print(f"[bold red]Can't extract hash from run id[/bold red] {run_id!r}")
        sys.exit(1)
    cfg_hash = m.group(1)
    new_run_id = f"{new_run_name}-{cfg_hash}"
    new_run_dir = os.path.join(checkpoint_dir, new_run_id)
    if os.path.exists(new_run_dir):
        console.print(f"[bold red]Target already exists:[/bold red] [path]{new_run_dir}[/path]")
        sys.exit(1)

    cfg_path = os.path.join(run_dir, "config.json")
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    old_run_name = cfg.get("run_name")
    cfg["run_name"] = new_run_name
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    os.rename(run_dir, new_run_dir)

    success(f"Renamed [path]{run_id}[/path] → [path]{new_run_id}[/path]")
    if old_run_name != new_run_name:
        dim(
            f"  Note: embedded run_name inside last.pt / best_model.pt is unchanged "
            f"({old_run_name!r}). It isn't used for run-dir resolution. If you use "
            f"`predict --config <jsonnet>`, update the jsonnet's run_name to {new_run_name!r}."
        )


# ---------------------------------------------------------------------------
# `runs delete` / `runs delete-all`


def _summarize_run(checkpoint_dir: str, run_id: str, parsers: dict) -> str:
    """One-line summary used in delete prompts: parser, best_val, size."""
    run_dir = os.path.join(checkpoint_dir, run_id)
    try:
        with open(os.path.join(run_dir, "config.json"), encoding="utf-8") as f:
            cfg = json.load(f)
    except (OSError, json.JSONDecodeError):
        cfg = {}
    kind = _infer_parser_kind(cfg, parsers)
    run_name = cfg.get("run_name") or "-"
    best_val_str, _ = _read_best_meta(run_dir)
    size = _format_size(_dir_size(run_dir))
    return f"[path]{run_id}[/path]  name={run_name}  parser={kind}  best_val={best_val_str}  size={size}"


def delete_run(checkpoint_dir: str, partial: str, assume_yes: bool) -> None:
    run_id = _resolve_run_id(checkpoint_dir, partial)
    run_dir = os.path.join(checkpoint_dir, run_id)
    parsers = _all_parsers()

    console.print("[bold yellow]About to delete:[/bold yellow]")
    console.print(f"  {_summarize_run(checkpoint_dir, run_id, parsers)}")

    if not assume_yes:
        try:
            answer = console.input("[bold]Delete this run? (y/N):[/bold] ").strip().lower()
        except EOFError:
            answer = ""
        if answer not in ("y", "yes"):
            console.print("[dim]Aborted.[/dim]")
            return

    shutil.rmtree(run_dir)
    success(f"Deleted [path]{run_dir}[/path]")


def delete_all_runs(checkpoint_dir: str) -> None:
    parsers = _all_parsers()
    run_ids = _list_run_dirs(checkpoint_dir)
    if not run_ids:
        console.print(f"[dim]No runs to delete in[/dim] [path]{checkpoint_dir}[/path]")
        return

    total_size = sum(_dir_size(os.path.join(checkpoint_dir, r)) for r in run_ids)
    console.print(
        f"[bold yellow]About to delete {len(run_ids)} run(s) from[/bold yellow] [path]{checkpoint_dir}[/path]:"
    )
    for r in run_ids:
        console.print(f"  {_summarize_run(checkpoint_dir, r, parsers)}")
    console.print(f"\n[bold]Total: {len(run_ids)} runs, {_format_size(total_size)}[/bold]")
    console.print("[bold red]This cannot be undone.[/bold red]")

    try:
        answer = console.input("Type [bold]delete all[/bold] to confirm: ").strip()
    except EOFError:
        answer = ""
    if answer != "delete all":
        console.print("[dim]Aborted (did not type 'delete all').[/dim]")
        return

    for r in run_ids:
        shutil.rmtree(os.path.join(checkpoint_dir, r))
    success(f"Deleted {len(run_ids)} run(s).")


# ---------------------------------------------------------------------------
# CLI wiring


def main():
    parser = argparse.ArgumentParser(prog="iudex runs", description="Inspect and manage iudex training runs")

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--checkpoint-dir", default="checkpoints", help="Root checkpoint dir to walk")

    sub = parser.add_subparsers(dest="subcommand", required=True)

    sub.add_parser("list", parents=[common], help="List runs under a checkpoint directory")

    p_show = sub.add_parser("show", parents=[common], help="Deep-inspect a single run")
    p_show.add_argument("run_id", help="Run id (or unique prefix)")

    p_diff = sub.add_parser("diff", parents=[common], help="Show config-field differences between two runs")
    p_diff.add_argument("run_id_a", help="First run id (or unique prefix)")
    p_diff.add_argument("run_id_b", help="Second run id (or unique prefix)")

    p_rename = sub.add_parser("rename", parents=[common], help="Set a run_name (renames the dir to <name>-<hash>)")
    p_rename.add_argument("run_id", help="Run id (or unique prefix)")
    p_rename.add_argument("new_run_name", help="New run_name")

    p_delete = sub.add_parser("delete", parents=[common], help="Delete a single run (prompts unless --yes)")
    p_delete.add_argument("run_id", help="Run id (or unique prefix)")
    p_delete.add_argument("--yes", action="store_true", help="Skip the y/N prompt")

    sub.add_parser(
        "delete-all", parents=[common], help="Delete every run in the checkpoint dir (requires typing 'delete all')"
    )

    args = parser.parse_args()

    if args.subcommand == "list":
        list_runs(args.checkpoint_dir)
    elif args.subcommand == "show":
        show_run(args.checkpoint_dir, args.run_id)
    elif args.subcommand == "diff":
        diff_runs(args.checkpoint_dir, args.run_id_a, args.run_id_b)
    elif args.subcommand == "rename":
        rename_run(args.checkpoint_dir, args.run_id, args.new_run_name)
    elif args.subcommand == "delete":
        delete_run(args.checkpoint_dir, args.run_id, args.yes)
    elif args.subcommand == "delete-all":
        delete_all_runs(args.checkpoint_dir)


if __name__ == "__main__":
    main()
