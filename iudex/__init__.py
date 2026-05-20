"""This module declares the project's CLI surface that lives above any one
framework:

  - `FRAMEWORKS`: dotted paths of framework modules (e.g. `iudex.rst`).
     Each framework module exposes `PARSERS`, `PARSER_SCOPED_COMMANDS`,
     and `GLOBAL_COMMANDS`. The dispatcher (`iudex/__main__.py`) imports
     each and merges the three.
  - `GLOBAL_COMMANDS`: `{cmd: module_path}` for project-level commands
    that aren't owned by any single framework (e.g. `runs`, which walks
    every framework's parser registry to tag rows by parser kind).

To add a sibling framework, append its dotted path to `FRAMEWORKS` and
give its `__init__.py` the three required attributes.
"""

__version__ = "0.1.0a4"

FRAMEWORKS: list[str] = ["iudex.rst"]

GLOBAL_COMMANDS: dict[str, str] = {
    "runs": "iudex.runs",
}
