"""Shared utilities for the two ICL pilot scripts (sr_words and sexp).

These scripts are disposable. Comments here are deliberately liberal compared
to the iudex package itself, so the pilot is easy to debug.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# Ensure repo root is importable when running this script directly.
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from iudex.common.log import console, dim, rule, success, warn, wrote  # noqa: E402
from iudex.rst.data.metrics import evaluate_parseval, metrics_table  # noqa: E402
from iudex.rst.data.reader import infer_relation_types, read_rst_dir  # noqa: E402
from iudex.rst.data.tree import RstTree  # noqa: E402


TRAIN_DIR = "/home/luke/local/iudex/data/gum_12.1.0_notok/train"
DEV_DIR = "/home/luke/local/iudex/data/gum_12.1.0_notok/dev"
OUT_ROOT = Path("/home/luke/local/iudex/icl_pilot")


@dataclass
class TaskBundle:
    relation_types: List[Tuple[str, str]]
    train_trees: List[RstTree]
    dev_pairs: List[Tuple[str, RstTree]]
    icl_sample: List[RstTree]


def load_corpora(seed: int = 42, k: int = 5) -> TaskBundle:
    """Load train + dev trees and sample k random train docs for in-context use."""
    relation_types = infer_relation_types([TRAIN_DIR, DEV_DIR])
    train_trees = [t for _, t in read_rst_dir(TRAIN_DIR, relation_types=relation_types)]
    dev_pairs = read_rst_dir(DEV_DIR, relation_types=relation_types)
    rng = random.Random(seed)
    icl_sample = rng.sample(train_trees, k)
    return TaskBundle(
        relation_types=relation_types,
        train_trees=train_trees,
        dev_pairs=dev_pairs,
        icl_sample=icl_sample,
    )


def doc_id(rs4_path: str) -> str:
    return Path(rs4_path).stem


def doc_text(tree: RstTree) -> str:
    """Raw input text = single-space-joined EDUs."""
    return " ".join(tree.edu_strings)


def make_run_dir(fmt: str, model_slug: str) -> Path:
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    d = OUT_ROOT / f"{fmt}_{model_slug}_{ts}"
    (d / "raw").mkdir(parents=True, exist_ok=True)
    (d / "preds").mkdir(parents=True, exist_ok=True)
    return d


def safe_model_slug(model: str) -> str:
    return model.replace("/", "_").replace(":", "_")


def call_llm(model: str, prompt: str, max_tokens: int = 65536, temperature: float = 0.0) -> str:
    """Single call via litellm. Lets exceptions propagate so callers can catch
    per-doc."""
    import litellm  # imported lazily so the module imports cheaply

    resp = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return resp.choices[0].message.content


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, indent=2))
    wrote(str(path))


def write_text(path: Path, text: str) -> None:
    path.write_text(text)
    wrote(str(path))


def run_eval(
    fmt: str,
    model: str,
    *,
    build_prefix: Callable[[TaskBundle], str],
    parse_response: Callable[[str, RstTree, TaskBundle], RstTree],
    output_suffix: str = "",
    limit: Optional[int] = None,
    seed: int = 42,
    k: int = 5,
) -> Dict[str, Any]:
    """The shared end-to-end loop. `build_prefix` returns the static prompt
    prefix string (description + relation inventory + k formatted examples,
    ending with whatever lead-in the format wants). `parse_response` converts
    a model response (plus the dev tree, for EDU strings) into an `RstTree`.
    Any exception raised by `parse_response` is treated as a parse failure for
    that document."""
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )

    bundle = load_corpora(seed=seed, k=k)
    if limit is not None:
        bundle.dev_pairs = bundle.dev_pairs[:limit]

    run_dir = make_run_dir(fmt + output_suffix, safe_model_slug(model))
    rule(f"ICL pilot: {fmt} via {model}")
    dim(f"run dir: {run_dir}")
    dim(f"k={k}, dev docs={len(bundle.dev_pairs)}, relation types={len(bundle.relation_types)}")

    prefix = build_prefix(bundle)
    write_text(run_dir / "prompt_prefix.txt", prefix)
    dim(f"prefix chars: {len(prefix)} (~{len(prefix) // 4} tokens)")

    gold_trees: List[RstTree] = []
    pred_trees: List[RstTree] = []
    per_doc: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    total_in_chars = 0
    total_out_chars = 0

    progress = Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )
    with progress:
        task = progress.add_task("ICL dev", total=len(bundle.dev_pairs))
        for path, gold in bundle.dev_pairs:
            did = doc_id(path)
            dtext = doc_text(gold)
            full_prompt = prefix + f"\n--- Now parse this document ---\nINPUT: {dtext}\nOUTPUT: "
            total_in_chars += len(full_prompt)

            t0 = time.time()
            try:
                response = call_llm(model, full_prompt)
            except Exception as e:  # API failure
                warn(f"[{did}] API call failed: {e!r}")
                failures.append({"doc_id": did, "stage": "api", "error": repr(e)})
                progress.advance(task)
                continue
            elapsed = time.time() - t0
            total_out_chars += len(response)

            # Persist the raw response before we try to parse, so we can inspect
            # whatever format mess the model returned.
            (run_dir / "raw" / f"{did}.txt").write_text(response)

            try:
                pred = parse_response(response, gold, bundle)
            except Exception as e:
                warn(f"[{did}] parse failed: {e!r}")
                failures.append({"doc_id": did, "stage": "parse", "error": repr(e)})
                progress.advance(task)
                continue

            # Check that pred has the same number of EDUs as gold; Parseval
            # assumes a shared EDU set, indexed lockstep.
            if len(pred.edu_strings) != len(gold.edu_strings):
                warn(
                    f"[{did}] EDU count mismatch: pred={len(pred.edu_strings)} "
                    f"vs gold={len(gold.edu_strings)}; treating as parse failure"
                )
                failures.append(
                    {
                        "doc_id": did,
                        "stage": "edu_count",
                        "gold_edus": len(gold.edu_strings),
                        "pred_edus": len(pred.edu_strings),
                    }
                )
                progress.advance(task)
                continue

            # Force same EDU strings as gold so Parseval scoring sees lockstep
            # spans (the model may have lightly perturbed surface tokens; we
            # care about structure/labels). We rebuild the tree with gold EDU
            # text spliced in. This is safe because we matched EDU counts.
            try:
                pred = _replace_edu_text(pred, gold.edu_strings, bundle.relation_types)
            except Exception as e:
                warn(f"[{did}] EDU text splice failed: {e!r}")
                failures.append({"doc_id": did, "stage": "splice", "error": repr(e)})
                progress.advance(task)
                continue

            try:
                rs4_str = pred.to_rs4_string()
                (run_dir / "preds" / f"{did}.rs4").write_text(rs4_str)
            except Exception as e:
                warn(f"[{did}] rs4 serialization failed: {e!r} (still counted)")

            gold_trees.append(gold)
            pred_trees.append(pred)
            per_doc.append(
                {
                    "doc_id": did,
                    "gold_edus": len(gold.edu_strings),
                    "pred_edus": len(pred.edu_strings),
                    "response_chars": len(response),
                    "elapsed_s": elapsed,
                }
            )
            progress.advance(task)

    # Aggregate. Per the instructions: skip failures, report failure count.
    if gold_trees:
        parseval = evaluate_parseval(gold_trees, pred_trees)
    else:
        parseval = {f"{m}_f1": 0.0 for m in ("span", "nuc", "rel", "full")}

    results = {
        "model": model,
        "format": fmt + output_suffix,
        "k": k,
        "seed": seed,
        "num_docs": len(bundle.dev_pairs),
        "num_parsed": len(gold_trees),
        "num_parse_failures": len(failures),
        "failures": failures,
        "per_doc": per_doc,
        "parseval": parseval,
        "prefix_chars": len(prefix),
        "total_input_chars": total_in_chars,
        "total_output_chars": total_out_chars,
        "raw_responses_dir": str(run_dir / "raw"),
        "preds_dir": str(run_dir / "preds"),
    }
    write_json(run_dir / "results.json", results)

    table = metrics_table(parseval, title=f"{fmt}{output_suffix} via {model}")
    console.print(table)
    if failures:
        warn(f"{len(failures)}/{len(bundle.dev_pairs)} docs had parse failures")
    success(f"Done. Results: {run_dir / 'results.json'}")
    return results


def _replace_edu_text(pred: RstTree, gold_edu_strings: List[str], relation_types) -> RstTree:
    """Splice gold EDU strings into the predicted tree, preserving structure.

    The model is free to garble whitespace inside EDUs; we only care about its
    structural decisions. We rebuild the tree from the predicted parsing
    actions plus the gold EDU strings.
    """
    actions = pred.parsing_actions()
    return RstTree.from_parsing_actions(actions, gold_edu_strings, relation_types=relation_types)


def fmt_relation_inventory(relation_types: List[Tuple[str, str]]) -> str:
    """Compact bullet list of reduce tokens, grouped by relation kind, for the
    prompt. We list both the multinuc and mononuc reduce-token forms so the
    model sees the legal label vocabulary."""
    from iudex.rst.data.tree import Reduce

    lines = []
    for rel, kind in relation_types:
        if kind == "multinuc":
            tok = Reduce("NN", rel).to_token()
            lines.append(f"- {tok}  (multinuclear; both children are nuclei of relation '{rel}')")
        else:
            ns = Reduce("NS", rel).to_token()
            sn = Reduce("SN", rel).to_token()
            lines.append(f"- {ns}  (left is nucleus, right is satellite; relation '{rel}')")
            lines.append(f"- {sn}  (left is satellite, right is nucleus; relation '{rel}')")
    return "\n".join(lines)


def fmt_sexp_relation_inventory(relation_types: List[Tuple[str, str]]) -> str:
    """Same idea, but list the head tags as they appear in s-exp internal
    nodes: `(NUC:relation child1 child2)`."""
    lines = []
    for rel, kind in relation_types:
        if kind == "multinuc":
            lines.append(f"- NN:{rel}  (multinuclear; both children are nuclei)")
        else:
            lines.append(f"- NS:{rel}  (left is nucleus, right is satellite)")
            lines.append(f"- SN:{rel}  (left is satellite, right is nucleus)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI plumbing
# ---------------------------------------------------------------------------


def parse_args(prog: str) -> Any:
    import argparse

    p = argparse.ArgumentParser(prog=prog)
    p.add_argument("--model", default="claude-opus-4-7", help="litellm model string")
    p.add_argument("--limit", type=int, default=None, help="only process the first N dev docs")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--k", type=int, default=5)
    return p.parse_args()
