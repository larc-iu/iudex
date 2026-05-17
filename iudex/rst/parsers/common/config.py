"""Shared config-parsing helpers for RST parser configs."""

import dataclasses
from typing import TypeVar

T = TypeVar("T")


def parse_config_dict(cls: type[T], d: dict) -> T:
    """Validate `d` against `cls`'s dataclass fields and instantiate.

    `d` is usually a dict produced by `tonga.Params.from_file(...).as_dict()`.
    Tuples are not JSON-representable, so `relation_types` (if present) is
    promoted from list-of-lists to list-of-tuples here.

    Raises:
        ValueError: if `d` contains keys that are not fields of `cls`.
        TypeError:  from `cls.__init__` if a required field is missing.
    """
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(d) - known
    if unknown:
        raise ValueError(f"Unknown config field(s): {sorted(unknown)}. Valid fields: {sorted(known)}")
    d = dict(d)
    if d.get("relation_types") is not None:
        d["relation_types"] = [tuple(r) for r in d["relation_types"]]
    return cls(**d)
