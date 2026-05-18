"""RST Parseval — per-tree counts, corpus aggregation, and Rich table renderer.

Segmentation and end-to-end Parseval (which require different EDU sets between
gold and predicted trees) live in `iudex.rst.data.seg_metrics`.
"""

from typing import Any, Dict, List

from rich.table import Table

from iudex.rst.data.tree import RstTree


def f1(x, y):
    if x is None or y is None:
        return None
    return (2 * x * y) / (x + y) if (x + y) > 0 else 0.0


def _spans_to_ranges(tree: RstTree, coarse: bool = False):
    """Parent-keyed enumeration: one entry per binary action, keyed on the
    parent's (first_edu, last_edu). Yields `n-1` entries for a binary tree
    with `n` EDUs — only the parser's actual split decisions are scored, and
    leaf-EDU spans are not. This is the Morey/Muller/Asher (2017) "standard"
    Parseval that excludes trivially-correct leaf matches.
    """
    result = set()
    for (left, right), nuc, rel in tree.spans():
        range_key = (left[0], right[-1])
        rel = rel.split("-")[0] if coarse else rel
        result.add((range_key, nuc, rel))
    return result


def _spans_to_sibling_ranges(tree: RstTree, coarse: bool = False):
    """Sibling-keyed enumeration: two entries per binary action — one per
    child — keyed on the child's (first_edu, last_edu). Yields `2*(n-1)`
    entries for a binary tree with `n` EDUs, including every leaf-EDU span.
    Matches DMRST's `getEvalData` (their default, `use_org_Parseval=False`)
    and the original Marcu (2000) Parseval. When gold and predicted trees
    share the same EDU set, the `n` leaf spans match trivially, inflating
    span F1 by roughly 10-15 points relative to the parent-keyed form.

    Per-child labels follow the DMRST convention: the nucleus child carries
    ("N", "span") as a placeholder; the satellite child carries ("S", rel);
    multinuclear children both carry ("N", rel).
    """
    result = set()
    for (left, right), nuc, rel in tree.spans():
        if coarse:
            rel = rel.split("-")[0]
        left_range = (left[0], left[-1])
        right_range = (right[0], right[-1])
        if nuc == "NS":
            result.add((left_range, "N", "span"))
            result.add((right_range, "S", rel))
        elif nuc == "SN":
            result.add((left_range, "S", rel))
            result.add((right_range, "N", "span"))
        elif nuc == "NN":
            result.add((left_range, "N", rel))
            result.add((right_range, "N", rel))
        else:
            raise ValueError(f"Unknown nuclearity: {nuc}")
    return result


def compute_parseval_metrics(
    gold: RstTree,
    pred: RstTree,
    coarse: bool = False,
    original_parseval: bool = False,
) -> Dict[str, Any]:
    """Compute RST Parseval metrics for two trees over the same EDU set.

    Defaults to the Morey/Muller/Asher "standard" form (parent-keyed: one
    entry per action). Set `original_parseval=True` to switch to the
    sibling-keyed form used by DMRST and the original Marcu (2000) Parseval,
    which counts each non-root constituent (including every leaf-EDU span)
    separately — kept for direct comparison to numbers reported in papers
    using that convention. The two scales are not interchangeable: original
    Parseval typically reports ~10-15 points higher than standard.
    """
    enumerate_spans = _spans_to_sibling_ranges if original_parseval else _spans_to_ranges
    gold_spans = enumerate_spans(gold, coarse)
    pred_spans = enumerate_spans(pred, coarse)

    if len(gold_spans) != len(pred_spans):
        raise ValueError("Mismatched span lengths! Are these trees really equivalent?")
    num_spans = len(gold_spans)

    if num_spans == 0:
        # Both trees are single-EDU (or empty). No constituents to score.
        zeros = {f"{m}_{x}_count": 0 for m in ("span", "nuc", "rel", "full") for x in ("p", "r")}
        return {
            **zeros,
            **{f"{m}_{x}": 0.0 for m in ("span", "nuc", "rel", "full") for x in ("p", "r")},
            "num_spans": 0,
            **{f"{m}_f1": 0.0 for m in ("span", "nuc", "rel", "full")},
        }

    gold_span_index = {s[0]: s[1:] for s in gold_spans}
    pred_span_index = {s[0]: s[1:] for s in pred_spans}

    span_p = 0
    span_r = 0
    nuc_p = 0
    nuc_r = 0
    rel_p = 0
    rel_r = 0
    full_p = 0
    full_r = 0
    for pred_span, pred_nuc, pred_rel in pred_spans:
        if pred_span in gold_span_index:
            gold_nuc, gold_rel = gold_span_index[pred_span]
            span_p += 1
            nuc_p += 1 if pred_nuc == gold_nuc else 0
            rel_p += 1 if pred_rel == gold_rel else 0
            full_p += 1 if pred_nuc == gold_nuc and pred_rel == gold_rel else 0
    for gold_span, gold_nuc, gold_rel in gold_spans:
        if gold_span in pred_span_index:
            pred_nuc, pred_rel = pred_span_index[gold_span]
            span_r += 1
            nuc_r += 1 if gold_nuc == pred_nuc else 0
            rel_r += 1 if gold_rel == pred_rel else 0
            full_r += 1 if pred_nuc == gold_nuc and pred_rel == gold_rel else 0

    output = {
        "span_p_count": span_p,
        "span_r_count": span_r,
        "nuc_p_count": nuc_p,
        "nuc_r_count": nuc_r,
        "rel_p_count": rel_p,
        "rel_r_count": rel_r,
        "full_p_count": full_p,
        "full_r_count": full_r,
        "span_p": span_p / num_spans,
        "span_r": span_r / num_spans,
        "nuc_p": nuc_p / num_spans,
        "nuc_r": nuc_r / num_spans,
        "rel_p": rel_p / num_spans,
        "rel_r": rel_r / num_spans,
        "full_p": full_p / num_spans,
        "full_r": full_r / num_spans,
        "num_spans": num_spans,
    }
    output["span_f1"] = f1(output["span_p"], output["span_r"])
    output["nuc_f1"] = f1(output["nuc_p"], output["nuc_r"])
    output["rel_f1"] = f1(output["rel_p"], output["rel_r"])
    output["full_f1"] = f1(output["full_p"], output["full_r"])
    return output


def evaluate_parseval(gold_trees: List[RstTree], gold_preds: List[RstTree]) -> Dict[str, float]:
    """Aggregate per-tree Parseval counts over a corpus into span/nuc/rel/full F1.

    Both lists are indexed in lockstep — `gold_preds[i]` is the parser's
    same-EDU-set output for `gold_trees[i]`. Returns four `*_f1` keys.
    """
    if len(gold_trees) != len(gold_preds):
        raise ValueError(f"gold_trees/gold_preds length mismatch: {len(gold_trees)} vs {len(gold_preds)}")
    totals = {f"{m}_{x}_count": 0 for m in ("span", "nuc", "rel", "full") for x in ("p", "r")}
    totals["num_spans"] = 0
    for gold, pred in zip(gold_trees, gold_preds):
        per_tree = compute_parseval_metrics(gold, pred)
        for k in totals:
            totals[k] += per_tree[k]
    num_spans = totals["num_spans"]
    if num_spans == 0:
        return {f"{m}_f1": 0.0 for m in ("span", "nuc", "rel", "full")}
    return {
        f"{m}_f1": f1(totals[f"{m}_p_count"] / num_spans, totals[f"{m}_r_count"] / num_spans)
        for m in ("span", "nuc", "rel", "full")
    }


def metrics_table(metrics: Dict[str, float], title: str) -> Table:
    """Rich table with up to three sections (gold Parseval, segmentation, e2e Parseval).

    Segmentation/e2e sections render only when their keys (`seg_f1`,
    `e2e_span_f1`) are present in `metrics`.
    """
    table = Table(title=title, show_header=True, header_style="bold cyan", padding=(0, 1))
    table.add_column("Metric", style="dim")
    table.add_column("F1", justify="right", style="bold green")

    table.add_row("[bold]Gold-EDU Parseval[/bold]", "")
    for name in ("span", "nuc", "rel", "full"):
        table.add_row(f"  {name.upper()}", f"{metrics[f'{name}_f1']:.4f}")

    if "seg_f1" in metrics:
        table.add_section()
        table.add_row("[bold]Segmentation[/bold]", "")
        table.add_row("  P", f"{metrics['seg_p']:.4f}")
        table.add_row("  R", f"{metrics['seg_r']:.4f}")
        table.add_row("  F1", f"{metrics['seg_f1']:.4f}")

    if "e2e_span_f1" in metrics:
        table.add_section()
        table.add_row("[bold]End-to-End Parseval[/bold]", "")
        for name in ("span", "nuc", "rel", "full"):
            table.add_row(f"  {name.upper()}", f"{metrics[f'e2e_{name}_f1']:.4f}")
    return table
