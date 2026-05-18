"""Shared config-parsing helpers for RST parser configs.

Each parser's config is a `@dataclass` inheriting `tonga.FromParams`, which
provides:

  - Recursive construction of nested sub-config dataclasses.
  - Tuple promotion for fields typed as `list[tuple[str, str]]` (jsonnet has
    no tuples, so they arrive as lists).
"""

from typing import TypeVar

from tonga import FromParams

T = TypeVar("T", bound=FromParams)


def parse_config_dict(cls: type[T], d: dict) -> T:
    """Instantiate `cls` from a plain dict (usually from `tonga.Params.as_dict()`).

    `cls` must inherit `FromParams`. Unknown keys raise `ConfigurationError`;
    nested dataclass fields are constructed recursively.
    """
    return cls.from_params(d)
