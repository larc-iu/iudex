from typing import Any, Dict

from iudex.rst.data.tree import RstPpTree


def f1(x, y):
    if x is None or y is None:
        return None
    return (2 * x * y) / (x + y) if (x + y) > 0 else 0.0


def _spans_to_ranges(tree: RstPpTree, coarse: bool = False):
    """Convert sibling-pair spans to individual constituent ranges for standard RST Parseval."""
    result = set()
    for (left, right), nuc, rel in tree.spans():
        range_key = (left[0], right[-1])
        rel = rel.split("-")[0] if coarse else rel
        result.add((range_key, nuc, rel))
    return result


def compute_parseval_metrics(gold: RstPpTree, pred: RstPpTree, coarse: bool = False) -> Dict[str, Any]:
    """
    Compute standard RST Parseval metrics for two trees.
    Each internal node is identified by its constituent range (first_edu, last_edu),
    matching the standard formulation where individual constituents are compared.
    """
    gold_spans = _spans_to_ranges(gold, coarse)
    pred_spans = _spans_to_ranges(pred, coarse)

    if len(gold_spans) != len(pred_spans):
        raise ValueError("Mismatched span lengths! Are these trees really equivalent?")
    num_spans = len(gold_spans)

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
