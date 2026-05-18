"""Upload a trained parser checkpoint to the HuggingFace Hub.

Dispatched from `iudex <parser_kind> push ...` via the shared-command path in
`iudex/__main__.py`. The dispatcher passes `parser_kind` as a kwarg; everything
else (Config class, train-module path used for the "train first with" hint)
falls out of `_resolve` below.

Not meant to be invoked directly with `python -m`; use the dispatcher.
"""

import argparse
import logging

from iudex.common.log import setup_logging
from iudex.rst.parsers.common.inference import resolve_checkpoint
from iudex.rst.parsers.hfhub import push_parser_to_hub

setup_logging()
logger = logging.getLogger(__name__)


def _resolve(parser_kind: str):
    """Return (Config class, train-module path, human-readable kind name)."""
    if parser_kind == "dmrst":
        from iudex.rst.parsers.dmrst.configuration_dmrst import DMRSTConfig

        return DMRSTConfig, "iudex.rst.parsers.dmrst.train_dmrst", "DMRST"
    if parser_kind == "topdown_biaffine":
        from iudex.rst.parsers.topdown_biaffine.configuration_topdown_biaffine import TopdownBiaffineConfig

        return TopdownBiaffineConfig, "iudex.rst.parsers.topdown_biaffine.train_topdown_biaffine", "top-down biaffine"
    raise ValueError(f"Unknown parser_kind: {parser_kind!r}")


def main(parser_kind: str) -> None:
    config_cls, train_module, human = _resolve(parser_kind)
    parser = argparse.ArgumentParser(
        prog=f"iudex {parser_kind} push",
        description=f"Push a {human} checkpoint to the HuggingFace Hub",
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
    parser.add_argument("--message", default=f"Upload {human} parser", help="Commit message")
    parser.add_argument("--token", default=None, help="HF token (falls back to cached login)")
    args = parser.parse_args()

    checkpoint_path = resolve_checkpoint(args.config, args.checkpoint, config_cls, train_module)
    push_parser_to_hub(
        checkpoint_path,
        args.repo_id,
        parser_kind=parser_kind,
        private=args.private,
        commit_message=args.message,
        token=args.token,
    )
