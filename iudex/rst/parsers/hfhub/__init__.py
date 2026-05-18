"""HuggingFace Hub distribution for iudex RST parsers.

Public API re-exported here so callers can write
`from iudex.rst.parsers.hfhub import load_parser_from_pretrained` instead of
reaching into `hub.py`. `push.py` is the shared CLI entry point dispatched
from `iudex <parser> push ...` (see `SHARED_COMMANDS` in `iudex/__main__.py`).
"""

from iudex.rst.parsers.hfhub.datasets import DATASETS, lookup
from iudex.rst.parsers.hfhub.hub import (
    load_parser_from_pretrained,
    push_parser_to_hub,
    render_model_card,
)

__all__ = [
    "DATASETS",
    "load_parser_from_pretrained",
    "lookup",
    "push_parser_to_hub",
    "render_model_card",
]
