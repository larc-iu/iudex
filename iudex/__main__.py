"""iudex top-level dispatcher.

Frameworks (RST, eventually PDTB, ...) declare their CLI surface as
three module-level attributes on the framework's `__init__.py`:

  - `PARSERS`                  — the parser registry
  - `PARSER_SCOPED_COMMANDS`   — `{cmd: module_path}` for `iudex <parser> <cmd>`
  - `GLOBAL_COMMANDS`          — `{cmd: module_path}` for `iudex <cmd>`

The dispatcher imports each framework named in `iudex.FRAMEWORKS`, merges
the three dicts (after seeding the global-commands map with project-level
entries from `iudex.GLOBAL_COMMANDS`), and refuses to start on a name
collision. To add a framework, add its dotted path to `iudex.FRAMEWORKS`
and give it the three attributes.
"""

import importlib
import sys

import iudex


def _merge_frameworks() -> tuple[dict, dict, dict]:
    """Import every framework module and merge its three dicts. Seeds the
    global-commands map with project-level entries from `iudex.GLOBAL_COMMANDS`
    so things like `runs` are always available without being owned by a
    framework. Aborts on a name collision (two frameworks claiming the same
    parser name, or a framework redeclaring a project-level global command).
    """
    parsers: dict = {}
    parser_scoped: dict = {}
    global_cmds: dict = dict(iudex.GLOBAL_COMMANDS)
    for fw_path in iudex.FRAMEWORKS:
        fw = importlib.import_module(fw_path)
        _merge_no_collide(parsers, fw.PARSERS, fw_path, "parser name")
        _merge_no_collide(parser_scoped, fw.PARSER_SCOPED_COMMANDS, fw_path, "parser-scoped command")
        _merge_no_collide(global_cmds, fw.GLOBAL_COMMANDS, fw_path, "global command")
    return parsers, parser_scoped, global_cmds


def _merge_no_collide(dst: dict, src: dict, fw_path: str, kind: str) -> None:
    for k, v in src.items():
        if k in dst and dst[k] is not v:
            sys.stderr.write(
                f"iudex: {kind} {k!r} declared by both an earlier framework and {fw_path!r}. Rename one of them.\n"
            )
            sys.exit(2)
        dst[k] = v


PARSERS, PARSER_SCOPED_COMMANDS, GLOBAL_COMMANDS = _merge_frameworks()


def main():
    if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help"):
        print(__doc__.strip())
        print(f"\nKnown parsers: {', '.join(sorted(PARSERS))}")
        print(f"Global commands: {', '.join(sorted(GLOBAL_COMMANDS))}")
        sys.exit(0)

    if len(sys.argv) < 2:
        sys.stderr.write(__doc__.strip() + "\n\n")
        sys.stderr.write(f"Known parsers: {', '.join(sorted(PARSERS))}\n")
        sys.stderr.write(f"Global commands: {', '.join(sorted(GLOBAL_COMMANDS))}\n")
        sys.exit(2)

    head = sys.argv[1]

    if head in GLOBAL_COMMANDS:
        module = importlib.import_module(GLOBAL_COMMANDS[head])
        sys.argv = [GLOBAL_COMMANDS[head]] + sys.argv[2:]
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
        sys.stderr.write(f"Global commands: {', '.join(sorted(GLOBAL_COMMANDS))}\n")
        sys.exit(2)

    if command in PARSER_SCOPED_COMMANDS:
        shared_path = PARSER_SCOPED_COMMANDS[command]
        module = importlib.import_module(shared_path)
        sys.argv = [f"iudex {parser_name} {command}"] + sys.argv[3:]
        module.main(parser_kind=parser_name)
        return

    module_path = f"{PARSERS[parser_name].package}.{command}_{parser_name}"
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
