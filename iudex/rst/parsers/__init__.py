"""Registry of RST parsers.

One ParserSpec per parser. This is the single source of truth used by:

  - the top-level dispatcher (`iudex/__main__.py`) to route commands
  - the shared `push` CLI (`hfhub/push.py`) and the shared `predict` CLI
    (`common/predict_cli.py`) to look up the Config / Parser classes
  - `runs list` (`iudex/runs.py`) to tag rows by parser kind.

To add a parser: add one entry below. Module paths follow the convention
`<package>/{configuration,modeling,train,predict}_<name>.py` and class
names follow `<Name>Config` / `<Name>Parser`.
"""

import importlib
from dataclasses import dataclass


@dataclass(frozen=True)
class ParserSpec:
    name: str
    package: str
    config_cls: str  # class name inside <package>/configuration_<name>.py
    parser_cls: str  # class name inside <package>/modeling_<name>.py
    # Exposes `--text` / `--text-file` in the predict CLI. Set on parsers
    # that implement `predict_from_text` (i.e. have a segmentation head).
    # Runtime still checks `model.segmenter is not None` since the segmenter
    # may be disabled per-config even on a `supports_text=True` parser.
    supports_text: bool
    # A config field present only on this parser. Used by `runs list` to tag
    # a config.json with its parser kind without having to import any parser
    # modules. Must be unique across registered parsers.
    signature_field: str

    def load_config_cls(self) -> type:
        mod = importlib.import_module(f"{self.package}.configuration_{self.name}")
        return getattr(mod, self.config_cls)

    def load_parser_cls(self) -> type:
        mod = importlib.import_module(f"{self.package}.modeling_{self.name}")
        return getattr(mod, self.parser_cls)


PARSERS: dict[str, ParserSpec] = {
    "topdown_biaffine": ParserSpec(
        name="topdown_biaffine",
        package="iudex.rst.parsers.topdown_biaffine",
        config_cls="TopdownBiaffineConfig",
        parser_cls="TopdownBiaffineParser",
        supports_text=False,
        signature_field="ffn_hidden_size",
    ),
    "dmrst": ParserSpec(
        name="dmrst",
        package="iudex.rst.parsers.dmrst",
        config_cls="DMRSTConfig",
        parser_cls="DMRSTParser",
        supports_text=True,
        signature_field="attention_type",
    ),
}
