"""HuggingFace Hub distribution for iudex RST parsers.

On load, both the parser repo and the underlying encoder (e.g.
`xlm-roberta-base`) are fetched and cached. The encoder weights are then
immediately overwritten by `load_state_dict` (strict mode catches
architecture drift).
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, TypeVar

import torch
import torch.nn as nn
from huggingface_hub import CommitOperationAdd, HfApi, snapshot_download

from iudex.common.log import console, success
from iudex.rst.parsers.common.inference import load_parser_from_checkpoint
from iudex.rst.parsers.hfhub.datasets import lookup as lookup_dataset

ConfigT = TypeVar("ConfigT")
ParserT = TypeVar("ParserT", bound=nn.Module)

HUB_WEIGHTS_NAME = "best_model.pt"
HUB_CONFIG_NAME = "config.json"
HUB_CARD_NAME = "README.md"

_HUB_ID_PATTERN = re.compile(r"^[\w.\-]+/[\w.\-]+$")

# Per-parser model-card metadata. `paper_*` keys present iff this parser
# re-implements an external paper (toggles intro wording + bibtex block).
_PARSER_META: dict[str, dict[str, str]] = {
    "dmrst": {
        "human_name": "DMRST parser",
        "module_path": "iudex.rst.parsers.dmrst.modeling_dmrst",
        "class_name": "DMRSTParser",
        "description": "an end-to-end RST parser with joint EDU segmentation, a GRU decoder, and pointer attention",
        "paper_title": "DMRST: A Joint Framework for Document-Level Multilingual RST Discourse Segmentation and Parsing",
        "paper_authors": "Zhengyuan Liu, Ke Shi, Nancy F. Chen",
        "paper_venue": "CODI 2021",
        "paper_url": "https://aclanthology.org/2021.codi-main.15/",
    },
    "topdown_biaffine": {
        "human_name": "Top-down Biaffine RST parser",
        "module_path": "iudex.rst.parsers.topdown_biaffine.modeling_topdown_biaffine",
        "class_name": "TopdownBiaffineParser",
        "description": "a greedy top-down RST parser with biaffine split and label scoring (assumes gold EDU segmentation)",
        "paper_title": "A Simple and Strong Baseline for End-to-End Neural RST-style Discourse Parsing",
        "paper_authors": "Naoki Kobayashi, Tsutomu Hirao, Hidetaka Kamigaito, Manabu Okumura, Masaaki Nagata",
        "paper_venue": "Findings of EMNLP 2022",
        "paper_url": "https://aclanthology.org/2022.findings-emnlp.501/",
    },
    # iudex original (no external paper): the `else` branch of render_model_card
    # supplies the intro/citation, so no `paper_*` keys.
    "piudotto": {
        "human_name": "piudotto RST parser",
        "module_path": "iudex.rst.parsers.piudotto.modeling_piudotto",
        "class_name": "PiudottoParser",
        "description": "an end-to-end span-based RST parser with joint EDU segmentation, biaffine span scoring, "
        "and greedy or CKY decoding",
    },
}


def _is_hub_id(s: str) -> bool:
    """True if `s` looks like `org/name` and isn't an existing path or `.pt` file."""
    if os.path.exists(s) or s.endswith(".pt"):
        return False
    return bool(_HUB_ID_PATTERN.match(s))


def load_parser_from_pretrained(
    repo_or_path: str,
    *,
    parser_cls: type[ParserT],
    config_cls: type[ConfigT],
    device: torch.device,
    revision: str | None = None,
    cache_dir: str | None = None,
    token: str | bool | None = None,
    compile_encoder: bool = False,
) -> ParserT:
    """Load a parser from a Hub repo id, a local run directory, or a `.pt` file.

    Hub ids pull `best_model.pt` / `config.json` / `README.md` via
    `snapshot_download`. Directories look for `best_model.pt`. `.pt` paths
    load as-is.
    """
    if _is_hub_id(repo_or_path):
        snapshot_dir = snapshot_download(
            repo_id=repo_or_path,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
            allow_patterns=[HUB_WEIGHTS_NAME, HUB_CONFIG_NAME, HUB_CARD_NAME],
        )
        checkpoint_path = os.path.join(snapshot_dir, HUB_WEIGHTS_NAME)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Hub repo {repo_or_path!r} has no {HUB_WEIGHTS_NAME}")
    elif os.path.isdir(repo_or_path):
        checkpoint_path = os.path.join(repo_or_path, HUB_WEIGHTS_NAME)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"No {HUB_WEIGHTS_NAME} in {repo_or_path}")
    else:
        checkpoint_path = repo_or_path
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(checkpoint_path)
    return load_parser_from_checkpoint(checkpoint_path, device, config_cls, parser_cls, compile_encoder=compile_encoder)


def push_parser_to_hub(
    checkpoint_path: str,
    repo_id: str,
    *,
    parser_kind: str,
    private: bool = False,
    commit_message: str = "Upload parser",
    token: str | bool | None = None,
    extra_card_fields: dict[str, Any] | None = None,
) -> str:
    """Upload `best_model.pt`, `config.json`, and a generated `README.md`
    to `repo_id` in a single commit. Returns the repo URL.
    """
    if parser_kind not in _PARSER_META:
        raise ValueError(f"Unknown parser_kind: {parser_kind!r}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    checkpoint_meta = {
        k: checkpoint.get(k) for k in ("best_val", "global_step", "epoch", "config_hash") if k in checkpoint
    }

    # Pick up adjacent final_metrics.json if the train script wrote one. It
    # gives us dev + test corpus metrics keyed by split. Absence is fine.
    run_dir = os.path.dirname(checkpoint_path)
    final_metrics_path = os.path.join(run_dir, "final_metrics.json")
    final_metrics: dict | None = None
    if os.path.exists(final_metrics_path):
        try:
            with open(final_metrics_path, encoding="utf-8") as f:
                final_metrics = json.load(f)
        except (OSError, json.JSONDecodeError):
            final_metrics = None

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True, token=token)

    # Prefer the adjacent run-dir config.json (byte-for-byte audit), fall back
    # to serializing the dict embedded in the checkpoint.
    adjacent_config = os.path.join(run_dir, HUB_CONFIG_NAME)
    if os.path.exists(adjacent_config):
        config_blob: str | bytes = adjacent_config
    else:
        config_blob = json.dumps(config, indent=2, default=str).encode()

    card = render_model_card(
        parser_kind=parser_kind,
        config=config,
        checkpoint_meta=checkpoint_meta,
        final_metrics=final_metrics,
        repo_id=repo_id,
        extra=extra_card_fields,
    )

    # Single commit so an interrupted push can't leave the repo with new
    # weights against a stale README / config.
    console.print(f"Uploading 3 files to [cyan]{repo_id}[/cyan] in a single commit...")
    api.create_commit(
        repo_id=repo_id,
        repo_type="model",
        operations=[
            CommitOperationAdd(path_in_repo=HUB_WEIGHTS_NAME, path_or_fileobj=checkpoint_path),
            CommitOperationAdd(path_in_repo=HUB_CONFIG_NAME, path_or_fileobj=config_blob),
            CommitOperationAdd(path_in_repo=HUB_CARD_NAME, path_or_fileobj=card.encode()),
        ],
        commit_message=commit_message,
        token=token,
    )

    url = f"https://huggingface.co/{repo_id}"
    success(f"Pushed to {url}")
    return url


def _format_relation_labels(config: dict[str, Any]) -> str:
    """Render the 'Relation labels' line. Sorted alphabetically for stable output."""
    relation_types = config.get("relation_types") or []
    relation_map = config.get("relation_map")

    distinct: list[str] = []
    seen: set[str] = set()
    for entry in relation_types:
        name = entry[0] if isinstance(entry, (list, tuple)) else str(entry)
        if name not in seen:
            seen.add(name)
            distinct.append(name)
    distinct.sort()

    if relation_map is None:
        descriptor = f"{len(distinct)} labels"
    else:
        descriptor = (
            f"Mapped from an original label inventory with {len(relation_map)} items "
            f"to {len(distinct)} labels. Mapped labels"
        )

    if not distinct:
        return f"**Relation labels:** {descriptor}.\n"
    label_list = ", ".join(f"`{n}`" for n in distinct)
    return f"**Relation labels:** {descriptor}:\n\n{label_list}\n"


def _render_data_section(
    *,
    config: dict[str, Any],
    train_dir: str,
    checkpoint_meta: dict[str, Any],
    final_metrics: dict[str, dict[str, float]] | None,
) -> str:
    """Render the "Data" section: dataset info, relation labels, corpus metrics."""
    parts: list[str] = ["## Data\n"]

    dataset = lookup_dataset(train_dir)
    if dataset is not None:
        name = dataset.get("name", "?")
        url = dataset.get("url")
        lang = dataset.get("language")
        header = f"**[{name}]({url})**" if url else f"**{name}**"
        if lang:
            header += f" ({lang})"
        parts.append(header + ".\n")
        if dataset.get("description"):
            parts.append(dataset["description"] + "\n")
    elif train_dir:
        parts.append(f"Trained on `{train_dir}` (no entry in the iudex dataset registry).\n")

    parts.append("\n" + _format_relation_labels(config))

    metric_name = config.get("val_metric_name", "?")
    if final_metrics:
        # Stable split ordering: dev first, then test, then anything else alpha.
        split_order = ["dev", "test"] + sorted(set(final_metrics) - {"dev", "test"})
        present = [s for s in split_order if s in final_metrics]
        all_keys: list[str] = []
        for s in present:
            for k in final_metrics[s]:
                if k not in all_keys:
                    all_keys.append(k)
        parts.append("\n### Metrics\n\n")
        parts.append("| Split | " + " | ".join(all_keys) + " |\n")
        parts.append("| --- | " + " | ".join("---" for _ in all_keys) + " |\n")
        for s in present:
            row = " | ".join(
                f"{final_metrics[s][k]:.4f}" if isinstance(final_metrics[s].get(k), (int, float)) else "-"
                for k in all_keys
            )
            parts.append(f"| {s} | {row} |\n")
    else:
        best_val = checkpoint_meta.get("best_val")
        parts.append("\n### Metrics\n\n")
        if isinstance(best_val, (int, float)) and best_val >= 0:
            parts.append(f"- **Dev {metric_name}**: {best_val:.4f}\n")
            parts.append("- _(no `final_metrics.json` sidecar, test metric not recorded)_\n")
        else:
            parts.append("- _(no metrics recorded on this checkpoint)_\n")

    return "".join(parts)


def render_model_card(
    *,
    parser_kind: str,
    config: dict[str, Any],
    checkpoint_meta: dict[str, Any],
    final_metrics: dict[str, dict[str, float]] | None = None,
    repo_id: str,
    extra: dict[str, Any] | None = None,
) -> str:
    """Generate the README.md model card. Plain string templates, no jinja.

    `final_metrics`, when present, maps split name → metric-dict (e.g.
    `{"dev": {...}, "test": {...}}`). It supersedes `checkpoint_meta['best_val']`
    for the displayed metrics table.
    """
    meta = _PARSER_META[parser_kind]
    encoder = config.get("model_name", "")
    train_dir = config.get("train_dir") or ""

    # Any parser with a non-null `segmentation` sub-config exposes
    # `predict_from_text`. Otherwise predictions require pre-segmented
    # RS3/RS4 trees (the parser's `predict` takes an `RstTree`).
    if config.get("segmentation") is not None:
        predict_snippet = (
            "tree = parser.predict_from_text(\n"
            '    "Although the experiment was carefully designed, "\n'
            '    "the results were inconclusive. "\n'
            '    "We plan to repeat it tonight."\n'
            ")"
        )
        cli_snippet = (
            f"iudex {parser_kind} predict \\\n"
            f"    --hub-id {repo_id} \\\n"
            '    --text "Although the experiment was carefully designed, '
            'the results were inconclusive. We plan to repeat it tonight."'
        )
        cli_batch_note = (
            "To parse a raw `.txt` file or a directory of them instead, use `--text-file <path> --output-dir out/`.\n"
        )
    else:
        predict_snippet = (
            "# This parser requires gold EDU segmentation, so the input must be an RS3/RS4 file.\n"
            "from iudex.rst.data.reader import read_rst_file\n"
            "gold = read_rst_file(\n"
            '    "doc.rs3",\n'
            "    relation_types=parser.config.relation_types,\n"
            "    relation_map=parser.config.relation_map,\n"
            ")\n"
            "tree = parser.predict(gold)"
        )
        cli_snippet = (
            f"iudex {parser_kind} predict \\\n"
            f"    --hub-id {repo_id} \\\n"
            "    --input <doc.rs3> \\\n"
            "    --output-dir out/"
        )
        cli_batch_note = "`--input` also accepts a directory of `.rs3` / `.rs4` files.\n"

    extras_section = ""
    if extra:
        extras_section = "## Notes\n\n" + "".join(f"- **{k}**: {v}\n" for k, v in extra.items()) + "\n"

    config_block = json.dumps(config, indent=2, default=str)

    front_matter = (
        "---\n"
        "library_name: iudex\n"
        f"base_model: {encoder}\n"
        "tags:\n"
        "  - discourse-parsing\n"
        "  - rst\n"
        f"  - {parser_kind}\n"
        "---\n\n"
    )

    data_section = _render_data_section(
        config=config,
        train_dir=train_dir,
        checkpoint_meta=checkpoint_meta,
        final_metrics=final_metrics,
    )

    if meta.get("paper_url"):
        intro = (
            f"A pretrained [{meta['human_name']}]({meta['paper_url']}) trained with "
            "[IUDEX](https://github.com/larc-iu/iudex)."
        )
        citation_block = f"""If you use this model, please cite both the underlying paper:

```bibtex
@inproceedings{{{parser_kind}_paper,
  title = {{{meta["paper_title"]}}},
  author = {{{meta["paper_authors"]}}},
  booktitle = {{{meta["paper_venue"]}}},
  url = {{{meta["paper_url"]}}},
}}
```

And the IUDEX library:

```bibtex
@misc{{gessler-iudex-2026,
  author       = {{Gessler, Luke}},
  title        = {{{{IUDEX: The Indiana University Discourse Exhibition}}}},
  year         = {{2026}},
  howpublished = {{\\url{{https://github.com/larc-iu/iudex}}}},
}}
```"""
    else:
        intro = (
            f"A pretrained {meta['human_name']} ({meta['description']}) "
            "developed and trained in [IUDEX](https://github.com/larc-iu/iudex)."
        )
        citation_block = """If you use this model, please cite the IUDEX library:

```bibtex
@misc{gessler-iudex-2026,
  author       = {Gessler, Luke},
  title        = {{IUDEX: The Indiana University Discourse Exhibition}},
  year         = {2026},
  howpublished = {\\url{https://github.com/larc-iu/iudex}},
}
```"""

    body = f"""# {repo_id}

{intro}

This model uses [`{encoder}`](https://huggingface.co/{encoder}) as its underlying encoder.

{data_section}
## Usage

### CLI

```
{cli_snippet}
```

{cli_batch_note}
### Python

```python
from {meta["module_path"]} import {meta["class_name"]}

parser = {meta["class_name"]}.from_pretrained("{repo_id}")
{predict_snippet}
print(tree.to_rs4_string())
```

## Citation

{citation_block}
{extras_section}## Full training configuration

See below for the full training configuration this model was trained with.

```json
{config_block}
```
"""
    return front_matter + body
