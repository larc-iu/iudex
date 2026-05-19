"""Segmentation and end-to-end RST metrics.

Joint EDU segmentation produces gold-vs-predicted EDU mismatches, so the
end-to-end Parseval here keys constituents by token-range rather than EDU
index (sidesteps the alignment problem). Segmentation F1 itself is a
position-level set comparison over EDU end indices.

For gold-EDU-only Parseval, see `iudex.rst.data.metrics`.
"""

from typing import Dict, List, Set, Tuple

from iudex.rst.data.metrics import f1
from iudex.rst.data.tree import RstTree


def spans_to_token_ranges(
    tree: RstTree,
    edu_mapping: List[Tuple[int, int]],
    coarse: bool = False,
) -> Set[Tuple[Tuple[int, int], str, str]]:
    """Mirror Parseval's parent-keyed span enumeration but map each
    constituent's `(first_edu, last_edu)` through `edu_mapping` to inclusive-
    token coordinates. Used for end-to-end Parseval where gold and predicted
    trees can have different EDU sets, so token-range matching sidesteps the
    EDU-index alignment problem.

    `edu_mapping` is `[(token_start, token_end_exclusive), ...]` per EDU.
    Returned key is `(edu_mapping[left[0]][0], edu_mapping[right[-1]][1] - 1)`.
    """
    result: Set[Tuple[Tuple[int, int], str, str]] = set()
    for (left, right), nuc, rel in tree.spans():
        first_tok = edu_mapping[left[0]][0]
        last_tok = edu_mapping[right[-1]][1] - 1
        rel = rel.split("-")[0] if coarse else rel
        result.add(((first_tok, last_tok), nuc, rel))
    return result


def compute_seg_metrics(gold_ends: List[int], pred_ends: List[int]) -> Dict[str, int]:
    """Position-level segmentation counts (Carlson/Marcu convention).

    `gold_ends` and `pred_ends` are inclusive token end indices, including
    the forced terminal break, matching the original `Metric.py:getSegMeasure`.
    The terminal token is always a gold end and always a predicted end,
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
    gold_tok_spans: Set[Tuple[Tuple[int, int], str, str]],
    pred_tok_spans: Set[Tuple[Tuple[int, int], str, str]],
) -> Dict[str, int]:
    """End-to-end Parseval counts using token-range keys (gold and pred can
    have different EDU sets). Mirrors `compute_parseval_metrics`'s matching
    logic exactly. Only the key type changed.

    Returns counts only, caller micro-aggregates. Pred and gold span counts
    can differ now, so per-tree P/R use different denominators. Computing F1
    here would invite inconsistent aggregation.
    """
    gold_index = {s[0]: s[1:] for s in gold_tok_spans}
    pred_index = {s[0]: s[1:] for s in pred_tok_spans}

    span_p = nuc_p = rel_p = full_p = 0
    span_r = nuc_r = rel_r = full_r = 0
    for pred_span, pred_nuc, pred_rel in pred_tok_spans:
        if pred_span in gold_index:
            gold_nuc, gold_rel = gold_index[pred_span]
            span_p += 1
            nuc_p += 1 if pred_nuc == gold_nuc else 0
            rel_p += 1 if pred_rel == gold_rel else 0
            full_p += 1 if pred_nuc == gold_nuc and pred_rel == gold_rel else 0
    for gold_span, gold_nuc, gold_rel in gold_tok_spans:
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
        "e2e_num_pred_spans": len(pred_tok_spans),
        "e2e_num_gold_spans": len(gold_tok_spans),
    }


def evaluate_seg_and_e2e(
    gold_trees: List[RstTree],
    seg_data: List[dict],
) -> Dict[str, float]:
    """Aggregate per-tree segmentation and end-to-end Parseval counts.

    Each `seg_data[i]` must be a dict with:
      - `gold_edu_ends`: List[int]
      - `pred_edu_ends`: List[int]
      - `e2e_pred`:       `RstTree | None`  (None if the segmenter produced <2 EDUs)
      - `gold_edu_mapping`: List[Tuple[int, int]]  (per-EDU token spans, gold)
      - `pred_edu_mapping`: List[Tuple[int, int]]  (per-EDU token spans, predicted)

    Returns keys `seg_p`, `seg_r`, `seg_f1`, `e2e_{span,nuc,rel,full}_{p,r,f1}`.
    """
    if len(seg_data) != len(gold_trees):
        raise ValueError(f"seg_data length mismatch: {len(seg_data)} vs {len(gold_trees)}")

    totals_seg = {"seg_correct": 0, "seg_pred_count": 0, "seg_gold_count": 0}
    totals_e2e = {f"e2e_{m}_{x}_count": 0 for m in ("span", "nuc", "rel", "full") for x in ("p", "r")}
    totals_e2e["e2e_num_pred_spans"] = 0
    totals_e2e["e2e_num_gold_spans"] = 0

    for gold, sd in zip(gold_trees, seg_data):
        m_seg = compute_seg_metrics(sd["gold_edu_ends"], sd["pred_edu_ends"])
        for k in totals_seg:
            totals_seg[k] += m_seg[k]

        gold_tok_spans = spans_to_token_ranges(gold, sd["gold_edu_mapping"])
        e2e_pred = sd["e2e_pred"]
        if e2e_pred is not None and len(e2e_pred.edus) >= 2:
            pred_tok_spans = spans_to_token_ranges(e2e_pred, sd["pred_edu_mapping"])
        else:
            pred_tok_spans = set()
        m_e2e = compute_e2e_parseval(gold_tok_spans, pred_tok_spans)
        for k in totals_e2e:
            totals_e2e[k] += m_e2e[k]

    out: Dict[str, float] = {}
    seg_correct = totals_seg["seg_correct"]
    seg_pred = totals_seg["seg_pred_count"]
    seg_gold = totals_seg["seg_gold_count"]
    out["seg_p"] = seg_correct / seg_pred if seg_pred > 0 else 0.0
    out["seg_r"] = seg_correct / seg_gold if seg_gold > 0 else 0.0
    out["seg_f1"] = (2 * seg_correct) / (seg_pred + seg_gold) if (seg_pred + seg_gold) > 0 else 0.0

    num_pred = totals_e2e["e2e_num_pred_spans"]
    num_gold = totals_e2e["e2e_num_gold_spans"]
    for m in ("span", "nuc", "rel", "full"):
        p = totals_e2e[f"e2e_{m}_p_count"] / num_pred if num_pred > 0 else 0.0
        r = totals_e2e[f"e2e_{m}_r_count"] / num_gold if num_gold > 0 else 0.0
        out[f"e2e_{m}_p"] = p
        out[f"e2e_{m}_r"] = r
        out[f"e2e_{m}_f1"] = f1(p, r) or 0.0
    return out
