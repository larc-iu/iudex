"""Shared `iudex <parser> push` CLI. Dispatched via `iudex/__main__.py`."""

import argparse
import logging

from iudex.common.log import setup_logging
from iudex.rst.parsers import PARSERS
from iudex.rst.parsers.common.inference import resolve_checkpoint
from iudex.rst.parsers.hfhub import push_parser_to_hub

setup_logging()
logger = logging.getLogger(__name__)


def main(parser_kind: str) -> None:
    if parser_kind not in PARSERS:
        raise ValueError(f"Unknown parser_kind: {parser_kind!r} (known: {sorted(PARSERS)})")
    spec = PARSERS[parser_kind]
    config_cls = spec.load_config_cls()

    parser = argparse.ArgumentParser(
        prog=f"iudex {parser_kind} push",
        description=f"Push a {parser_kind} checkpoint to the HuggingFace Hub",
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--config", help="Jsonnet config; push best_model.pt from the derived run dir")
    source_group.add_argument("--checkpoint", help="Direct path to a .pt checkpoint")
    parser.add_argument(
        "--repo-id",
        dest="repo_id",
        required=True,
        help=f"Target HF repo id, e.g. larc-iu/{parser_kind}-rstdt-coarse",
    )
    parser.add_argument("--private", action="store_true", help="Create the repo as private")
    parser.add_argument("--message", default=f"Upload {parser_kind} parser", help="Commit message")
    parser.add_argument("--token", default=None, help="HF token (falls back to cached login)")
    args = parser.parse_args()

    checkpoint_path = resolve_checkpoint(args.config, args.checkpoint, config_cls, parser_kind)
    push_parser_to_hub(
        checkpoint_path,
        args.repo_id,
        parser_kind=parser_kind,
        private=args.private,
        commit_message=args.message,
        token=args.token,
    )
