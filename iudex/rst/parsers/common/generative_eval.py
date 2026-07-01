"""Shared dev/test evaluation for the four generative RST parsers.

All four (`seq2seq_sr`, `decoder_only_sr`, `seq2seq_sexp`, `decoder_only_sexp`)
share a byte-identical evaluation path: batched end-to-end decode then Parseval +
segmentation F1 (token-range keyed, since predicted EDU counts can differ from
gold so an EDU-index-keyed Parseval would crash), plus an optional
gold-EDU-forced Parseval. Every per-parser difference (batched-greedy vs per-doc,
SR vs sexp serialization, encoder-decoder vs causal) lives inside
`model.predict_batch` / `model.predict_with_gold_edus`, so this orchestration is
identical across the four and lifted here.

This is a plain function call, NOT inversion of control: each `train_<name>.py`
still owns its training loop top-to-bottom and CALLS `evaluate_on_dev(model,
dev_pairs, ...)`, getting metrics back. `model` is duck-typed (see
`GenerativeParser`): any parser exposing `eval()`, `tokenizer`,
`predict_batch(...)`, and `predict_with_gold_edus(...)`.
"""

import os
import time
from typing import Protocol

import torch

from iudex.common.log import console, dim
from iudex.rst.data.metrics import compute_parseval_metrics, f1
from iudex.rst.data.seg_metrics import evaluate_seg_and_e2e
from iudex.rst.data.tree import RstTree
from iudex.rst.parsers.common.seqgen import align_edus_to_tokens, reconstruct_text


class GenerativeParser(Protocol):
    """The slice of the generative-parser interface this eval needs."""

    tokenizer: object

    def eval(self) -> object: ...
    def predict_batch(self, trees: list[RstTree], *, num_beams: int | None = None) -> list[RstTree]: ...
    def predict_with_gold_edus(self, tree: RstTree) -> RstTree: ...


def _gold_edu_token_mapping(model: GenerativeParser, tree: RstTree) -> tuple[list[int], list[tuple[int, int]]]:
    """EDU end-positions and per-EDU `(start, end_exclusive)` token-position
    ranges in the ENCODER'S whole-document tokenization space, the same space as
    the pred mappings the inference loop produces by cursor tracking. Delegates
    to `align_edus_to_tokens` (see it for why whole-doc tiling, not per-EDU
    tokenization, is required)."""
    text = reconstruct_text(tree)
    _, mapping = align_edus_to_tokens(model.tokenizer, text, tree.edus)
    edu_ends = [end - 1 for _, end in mapping]
    return edu_ends, mapping


def _pred_edu_token_mapping(pred_tree: RstTree) -> tuple[list[int], list[tuple[int, int]]]:
    """Pull the per-EDU source-position ranges that the inference loop
    stashed on the tree. Already in the encoder's source-id token space,
    so no re-tokenization needed."""
    ranges = getattr(pred_tree, "_pred_edu_source_ranges", None)
    if ranges is None:
        # Fallback: degenerate single-EDU empty tree.
        return [], []
    edu_ends = [end - 1 for _, end in ranges]
    return edu_ends, list(ranges)


def _write_rs4(tree: RstTree, output_dir: str, basename: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, basename), "w", encoding="utf-8") as f:
        f.write(tree.to_rs4_string())


@torch.no_grad()
def _evaluate_gold_edu(model: GenerativeParser, dev_pairs: list[tuple[str, RstTree]]) -> dict[str, float]:
    """Run the gold-EDU-forced predict path over every dev pair and
    aggregate Parseval. Per-tree shifts equal gold EDU counts by
    construction (forced segmentation), so `compute_parseval_metrics`
    sees aligned span counts. Trees where the alignment still drifts
    (e.g. severe input truncation drops gold EDUs) are skipped with a
    warning rather than aborting the whole pass.

    Output keys: `gold_edu_{span,nuc,rel,full}_f1`.
    """
    totals = {f"{m}_{x}_count": 0 for m in ("span", "nuc", "rel", "full") for x in ("p", "r")}
    totals["num_spans"] = 0
    skipped = 0
    eval_t0 = time.monotonic()
    for filepath, gold in dev_pairs:
        pred = model.predict_with_gold_edus(gold)
        if len(pred.edus) != len(gold.edus):
            skipped += 1
            continue
        try:
            per_tree = compute_parseval_metrics(gold, pred)
        except ValueError:
            skipped += 1
            continue
        for k in totals:
            totals[k] += per_tree[k]
    dim(
        f"  gold-EDU eval: {time.monotonic() - eval_t0:.1f}s over {len(dev_pairs)} docs"
        + (f" ({skipped} skipped for EDU-count drift)" if skipped else "")
    )
    num_spans = totals["num_spans"]
    if num_spans == 0:
        return {f"gold_edu_{m}_f1": 0.0 for m in ("span", "nuc", "rel", "full")}
    return {
        f"gold_edu_{m}_f1": f1(totals[f"{m}_p_count"] / num_spans, totals[f"{m}_r_count"] / num_spans)
        for m in ("span", "nuc", "rel", "full")
    }


@torch.no_grad()
def evaluate_on_dev(
    model: GenerativeParser,
    dev_pairs: list[tuple[str, RstTree]],
    *,
    num_beams: int | None = None,
    batch_size: int = 1,
    output_dir: str | None = None,
    eval_gold_edu: bool = False,
) -> dict[str, float]:
    """End-to-end Parseval + segmentation F1. No gold-EDU Parseval here:
    these parsers always use their own segmentation, so EDU counts can differ
    from gold and `evaluate_parseval` (which keys spans by EDU index) would
    crash. Token-range keyed `evaluate_seg_and_e2e` handles the alignment.

    `num_beams` overrides the parser's configured beam width. Per-epoch dev
    eval passes 1 (greedy) when `cfg.eval_decode_greedy`; final test eval
    passes the full `cfg.num_beams`. `batch_size > 1` groups dev documents
    into one `predict_batch` call per chunk; whether that actually batches
    the decode is up to the parser (the SR parsers batch their greedy path,
    the sexp parsers decode per-document, so for them it only chunks the
    progress logging). `output_dir`, when set, writes each pred tree as
    `{basename}.rs4` for later inspection. `eval_gold_edu` adds the
    gold-EDU-forced Parseval (`gold_edu_*` keys).

    Per-batch wall-time + EDU counts are printed via `dim()` so a slow or
    pathological prediction is visible in real time.
    """
    model.eval()
    gold_trees: list[RstTree] = []
    seg_data: list[dict] = []
    eval_t0 = time.monotonic()
    for chunk_start in range(0, len(dev_pairs), batch_size):
        chunk = dev_pairs[chunk_start : chunk_start + batch_size]
        chunk_t0 = time.monotonic()
        preds = model.predict_batch([gold for _, gold in chunk], num_beams=num_beams)
        chunk_dt = time.monotonic() - chunk_t0
        # Per-batch summary: one line covering all docs in the chunk.
        names = ",".join(os.path.basename(fp) for fp, _ in chunk)
        gold_counts = [len(g.edus) for _, g in chunk]
        pred_counts = [len(p.edus) for p in preds]
        dim(
            f"  dev {chunk_start + 1}-{chunk_start + len(chunk)}/{len(dev_pairs)} "
            f"[{names}]: gold_edus={gold_counts} pred_edus={pred_counts} {chunk_dt:.1f}s"
        )
        for (filepath, gold), pred in zip(chunk, preds):
            gold_trees.append(gold)
            gold_ends, gold_map = _gold_edu_token_mapping(model, gold)
            pred_ends, pred_map = _pred_edu_token_mapping(pred)
            seg_data.append(
                {
                    "gold_edu_ends": gold_ends,
                    "pred_edu_ends": pred_ends,
                    "e2e_pred": pred,
                    "gold_edu_mapping": gold_map,
                    "pred_edu_mapping": pred_map,
                }
            )
            if output_dir is not None:
                basename = os.path.splitext(os.path.basename(filepath))[0] + ".rs4"
                _write_rs4(pred, output_dir, basename)
    dim(f"  dev eval total: {time.monotonic() - eval_t0:.1f}s over {len(dev_pairs)} documents")
    if output_dir is not None:
        console.print(f"[dim]Wrote {len(dev_pairs)} predictions under[/dim] [path]{os.path.abspath(output_dir)}[/path]")
    metrics = evaluate_seg_and_e2e(gold_trees, seg_data)
    if eval_gold_edu:
        metrics.update(_evaluate_gold_edu(model, dev_pairs))
    return metrics
