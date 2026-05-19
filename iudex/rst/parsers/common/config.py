"""Shared config-parsing helper."""

from typing import TypeVar

from tonga import FromParams

T = TypeVar("T", bound=FromParams)


def parse_config_dict(cls: type[T], d: dict) -> T:
    """Instantiate `cls` from a plain dict (e.g. `tonga.Params.as_dict()` output)."""
    return cls.from_params(d)
