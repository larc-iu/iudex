"""iudex top-level dispatcher.

Usage:
    python -m iudex <parser_name> <command> [args...]
    python -m iudex runs <subcommand> [args...]

Examples:
    python -m iudex topdown_biaffine train configs/topdown_biaffine_rstdt.jsonnet
    python -m iudex topdown_biaffine predict --config configs/topdown_biaffine_rstdt.jsonnet \\
        --input data/test --output-dir out/
    python -m iudex dmrst predict --hub-id larc-iu/dmrst-rstdt-coarse \\
        --input-text doc.txt --output-dir out/
    python -m iudex dmrst push --config configs/dmrst_rstdt.jsonnet --repo-id larc-iu/dmrst-rstdt-coarse
    python -m iudex runs list

To add a new parser, register its package path in `PARSERS`. Each parser folder
is expected to provide `train_<name>.py` and `predict_<name>.py` (or any other
`<command>_<name>.py`) — the dispatcher imports `<package>.<command>_<name>`
and calls its `main()`.

Commands whose implementation is parser-agnostic (e.g. `push`) live in
`SHARED_COMMANDS`. The dispatcher routes `iudex <parser> <cmd>` to the shared
module and passes the parser name as `main(parser_kind=...)`, so per-parser
shim files are unnecessary.

Top-level (parser-less) commands live in `TOP_LEVEL_COMMANDS`. They dispatch to
a module's `main()` directly without the `<command>_<parser>` naming dance.
"""

import importlib
import sys

# {parser_name: importable package containing this parser's command scripts}
PARSERS = {
    "topdown_biaffine": "iudex.rst.parsers.topdown_biaffine",
    "dmrst": "iudex.rst.parsers.dmrst",
}

# {command_name: module path}. Implementation is shared across parsers; the
# dispatcher calls `main(parser_kind=...)` with the parser name from argv.
SHARED_COMMANDS = {
    "push": "iudex.rst.parsers.hfhub.push",
}

# {command_name: module path} for commands with no associated parser.
TOP_LEVEL_COMMANDS = {
    "runs": "iudex.rst.runs",
}


def main():
    if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help"):
        print(__doc__.strip())
        print(f"\nKnown parsers: {', '.join(sorted(PARSERS))}")
        print(f"Top-level commands: {', '.join(sorted(TOP_LEVEL_COMMANDS))}")
        sys.exit(0)

    if len(sys.argv) < 2:
        sys.stderr.write(__doc__.strip() + "\n\n")
        sys.stderr.write(f"Known parsers: {', '.join(sorted(PARSERS))}\n")
        sys.stderr.write(f"Top-level commands: {', '.join(sorted(TOP_LEVEL_COMMANDS))}\n")
        sys.exit(2)

    head = sys.argv[1]

    if head in TOP_LEVEL_COMMANDS:
        module = importlib.import_module(TOP_LEVEL_COMMANDS[head])
        sys.argv = [TOP_LEVEL_COMMANDS[head]] + sys.argv[2:]
        module.main()
        return

    if len(sys.argv) < 3:
        sys.stderr.write(__doc__.strip() + "\n\n")
        sys.stderr.write(f"Known parsers: {', '.join(sorted(PARSERS))}\n")
        sys.exit(2)

    parser_name = head
    command = sys.argv[2]

    if parser_name not in PARSERS:
        sys.stderr.write(f"Unknown parser: {parser_name!r}\n")
        sys.stderr.write(f"Known parsers: {', '.join(sorted(PARSERS))}\n")
        sys.stderr.write(f"Top-level commands: {', '.join(sorted(TOP_LEVEL_COMMANDS))}\n")
        sys.exit(2)

    if command in SHARED_COMMANDS:
        shared_path = SHARED_COMMANDS[command]
        module = importlib.import_module(shared_path)
        sys.argv = [f"iudex {parser_name} {command}"] + sys.argv[3:]
        module.main(parser_kind=parser_name)
        return

    module_path = f"{PARSERS[parser_name]}.{command}_{parser_name}"
    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        sys.stderr.write(f"No such command {command!r} for parser {parser_name!r}\n")
        sys.stderr.write(f"  (tried to import {module_path}: {e})\n")
        sys.exit(2)

    sys.argv = [module_path] + sys.argv[3:]
    module.main()


if __name__ == "__main__":
    main()
