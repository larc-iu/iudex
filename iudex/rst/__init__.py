"""Rhetorical Structure Theory

What the top-level dispatcher (`iudex/__main__.py`) needs from a framework
module is three attributes:

  - `PARSERS`: the parser registry (defined in `iudex.rst.parsers`).
  - `PARSER_SCOPED_COMMANDS`: `{cmd: module_path}` for commands invoked as
    `iudex <parser> <cmd>`. The module's `main()` is called with
    `parser_kind=<parser>`.
  - `GLOBAL_COMMANDS`: `{cmd: module_path}` for commands invoked as
    `iudex <cmd>` (no parser noun) that are framework-specific. The
    module's `main()` is called with no arguments. Project-level globals
    (e.g. `runs`) live in `iudex.__init__.GLOBAL_COMMANDS` instead.

To add a sibling framework (e.g. `iudex.pdtb`), give it the same three
attributes on its `__init__.py` and add its dotted name to `FRAMEWORKS` in
`iudex/__init__.py`. The dispatcher merges the three dicts across all
frameworks and refuses to start on a name collision.
"""

from iudex.common.training import DEFAULT_HASH_EXCLUDE
from iudex.rst.parsers import PARSERS

PARSER_SCOPED_COMMANDS: dict[str, str] = {
    "push": "iudex.rst.parsers.hfhub.push",
}

GLOBAL_COMMANDS: dict[str, str] = {}

# Combined hash_exclude for RST parsers: the framework-agnostic defaults
# plus RST-specific inferred-at-training fields. RST trainers and inference
# helpers pass this via `hash_exclude=` when calling `derive_run_id` /
# `prepare_run_dir`.
#
# - `relation_types`: inferred post-hash from train/dev data; would otherwise
#   silently mismatch between train and predict.
HASH_EXCLUDE: tuple[str, ...] = DEFAULT_HASH_EXCLUDE + ("relation_types",)

__all__ = ["PARSERS", "PARSER_SCOPED_COMMANDS", "GLOBAL_COMMANDS", "HASH_EXCLUDE"]
