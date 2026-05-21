"""Rhetorical Structure Theory framework. See `iudex/__init__.py` for the
dispatcher contract these three module-level attributes satisfy."""

from iudex.common.training import DEFAULT_HASH_EXCLUDE
from iudex.rst.parsers import PARSERS

PARSER_SCOPED_COMMANDS: dict[str, str] = {
    "push": "iudex.rst.parsers.hfhub.push",
}

GLOBAL_COMMANDS: dict[str, str] = {}

# RST adds `relation_types` (inferred post-hash from train/dev data, which
# would otherwise silently mismatch between train and predict) and `amp`
# (a bf16-autocast training-precision knob, not an architecture choice, so
# excluding it keeps fp32-trained runs resumable after enabling it).
HASH_EXCLUDE: tuple[str, ...] = DEFAULT_HASH_EXCLUDE + ("relation_types", "amp")

__all__ = ["PARSERS", "PARSER_SCOPED_COMMANDS", "GLOBAL_COMMANDS", "HASH_EXCLUDE"]
