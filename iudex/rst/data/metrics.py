from typing import Any, Dict, List, Set, Tuple

from iudex.rst.data.tree import RstPpTree


def f1(x, y):
    if x is None or y is None:
        return None
    return (2 * x * y) / (x + y) if (x + y) > 0 else 0.0


def _spans_to_ranges(tree: RstPpTree, coarse: bool = False):
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


def _spans_to_sibling_ranges(tree: RstPpTree, coarse: bool = False):
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
    gold: RstPpTree,
    pred: RstPpTree,
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


def spans_to_subtoken_ranges(
    tree: RstPpTree,
    edu_mapping: List[Tuple[int, int]],
    coarse: bool = False,
) -> Set[Tuple[Tuple[int, int], str, str]]:
    """Mirror `_spans_to_ranges` but map each constituent's `(first_edu, last_edu)`
    range through `edu_mapping` to inclusive-subtoken coordinates. Used for
    end-to-end Parseval where gold and predicted trees can have different EDU
    sets — matching by subtoken range sidesteps the EDU-index alignment problem.

    `edu_mapping` is `[(subtoken_start, subtoken_end_exclusive), ...]` per EDU.
    Returned key is `(edu_mapping[left[0]][0], edu_mapping[right[-1]][1] - 1)`.
    """
    result: Set[Tuple[Tuple[int, int], str, str]] = set()
    for (left, right), nuc, rel in tree.spans():
        first_subtok = edu_mapping[left[0]][0]
        last_subtok = edu_mapping[right[-1]][1] - 1
        rel = rel.split("-")[0] if coarse else rel
        result.add(((first_subtok, last_subtok), nuc, rel))
    return result


def compute_seg_metrics(gold_ends: List[int], pred_ends: List[int]) -> Dict[str, int]:
    """Position-level segmentation counts (Carlson/Marcu convention).

    `gold_ends` and `pred_ends` are inclusive subtoken end indices, including
    the forced terminal break — matches upstream `Metric.py:getSegMeasure`.
    The terminal subtoken is always a gold end and always a predicted end,
    so it contributes a guaranteed +1 to `seg_correct` per document.
    """
    gold_set = set(gold_ends)
    pred_set = set(pred_ends)
    return {
        "seg_correct": len(gold_set & pred_set),
        "seg_pred_count": len(pred_set),
        "seg_gold_count": len(gold_set),
    }


def compute_e2e_parseval(
    gold_subtok_spans: Set[Tuple[Tuple[int, int], str, str]],
    pred_subtok_spans: Set[Tuple[Tuple[int, int], str, str]],
) -> Dict[str, int]:
    """End-to-end Parseval counts using subtoken-range keys (gold and pred can
    have different EDU sets). Mirrors `compute_parseval_metrics`'s matching
    logic exactly; only the key type changed.

    Returns counts only — caller micro-aggregates. Pred and gold span counts
    can differ now, so per-tree P/R use different denominators; computing F1
    here would invite inconsistent aggregation.
    """
    gold_index = {s[0]: s[1:] for s in gold_subtok_spans}
    pred_index = {s[0]: s[1:] for s in pred_subtok_spans}

    span_p = nuc_p = rel_p = full_p = 0
    span_r = nuc_r = rel_r = full_r = 0
    for pred_span, pred_nuc, pred_rel in pred_subtok_spans:
        if pred_span in gold_index:
            gold_nuc, gold_rel = gold_index[pred_span]
            span_p += 1
            nuc_p += 1 if pred_nuc == gold_nuc else 0
            rel_p += 1 if pred_rel == gold_rel else 0
            full_p += 1 if pred_nuc == gold_nuc and pred_rel == gold_rel else 0
    for gold_span, gold_nuc, gold_rel in gold_subtok_spans:
        if gold_span in pred_index:
            pred_nuc, pred_rel = pred_index[gold_span]
            span_r += 1
            nuc_r += 1 if gold_nuc == pred_nuc else 0
            rel_r += 1 if gold_rel == pred_rel else 0
            full_r += 1 if pred_nuc == gold_nuc and pred_rel == gold_rel else 0

    return {
        "e2e_span_p_count": span_p,
        "e2e_span_r_count": span_r,
        "e2e_nuc_p_count": nuc_p,
        "e2e_nuc_r_count": nuc_r,
        "e2e_rel_p_count": rel_p,
        "e2e_rel_r_count": rel_r,
        "e2e_full_p_count": full_p,
        "e2e_full_r_count": full_r,
        "e2e_num_pred_spans": len(pred_subtok_spans),
        "e2e_num_gold_spans": len(gold_subtok_spans),
    }
