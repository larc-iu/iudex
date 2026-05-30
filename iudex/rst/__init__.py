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
# The generative decode/eval-only knobs change no trained weights (they only
# steer decoding/eval), so excluding them keeps runs resumable across changes.
# They exist only on the generative configs, harmless elsewhere (exclusion is
# by key name against the asdict, so absent keys are simply ignored).
HASH_EXCLUDE: tuple[str, ...] = DEFAULT_HASH_EXCLUDE + (
    "relation_types",
    "amp",
    "num_beams",
    "use_validity_constraints",
    "eval_decode_greedy",
    "min_edu_length",
    "dev_max_docs",
    "dev_batch_size",
    "constrain_content",
)

__all__ = ["PARSERS", "PARSER_SCOPED_COMMANDS", "GLOBAL_COMMANDS", "HASH_EXCLUDE"]
