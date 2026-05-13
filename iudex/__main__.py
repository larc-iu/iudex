"""iudex top-level dispatcher.

Usage:
    python -m iudex <parser_name> <command> [args...]

Examples:
    python -m iudex topdown_biaffine train configs/topdown_biaffine.jsonnet
    python -m iudex topdown_biaffine predict --config configs/topdown_biaffine.jsonnet \\
        --input data/test --output-dir out/

To add a new parser, register its package path in `PARSERS` below. Each parser
folder is expected to provide `train_<name>.py` and `predict_<name>.py` (or any
other `<command>_<name>.py`) — the dispatcher imports
`<package>.<command>_<name>` and calls its `main()`.
"""

import importlib
import sys

# {parser_name: importable package containing this parser's command scripts}
PARSERS = {
    "topdown_biaffine": "iudex.rst.parsers.topdown_biaffine",
    "dmrst": "iudex.rst.parsers.dmrst",
}


def main():
    if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help"):
        print(__doc__.strip())
        print(f"\nKnown parsers: {', '.join(sorted(PARSERS))}")
        sys.exit(0)

    if len(sys.argv) < 3:
        sys.stderr.write(__doc__.strip() + "\n\n")
        sys.stderr.write(f"Known parsers: {', '.join(sorted(PARSERS))}\n")
        sys.exit(2)

    parser_name = sys.argv[1]
    command = sys.argv[2]

    if parser_name not in PARSERS:
        sys.stderr.write(f"Unknown parser: {parser_name!r}\n")
        sys.stderr.write(f"Known parsers: {', '.join(sorted(PARSERS))}\n")
        sys.exit(2)

    module_path = f"{PARSERS[parser_name]}.{command}_{parser_name}"
    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        sys.stderr.write(f"No such command {command!r} for parser {parser_name!r}\n")
        sys.stderr.write(f"  (tried to import {module_path}: {e})\n")
        sys.exit(2)

    # Hand off remaining args to the module's own main() — argparse there will
    # see the right sys.argv (with module_path as argv[0]).
    sys.argv = [module_path] + sys.argv[3:]
    module.main()


if __name__ == "__main__":
    main()
