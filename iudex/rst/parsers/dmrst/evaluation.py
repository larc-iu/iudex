"""Dev evaluation for the dmrst parser.

Always computes gold-EDU Parseval (the four span/nuc/rel/full F1s). When the
model has a trained `_Segmenter`, additionally computes:
  - Segmentation F1: position-level intersection over predicted vs gold EDU
    end subtoken indices.
  - End-to-end Parseval: span F1 with predicted EDUs, where spans are matched
    by inclusive subtoken-range tuples (so gold and pred can have different
    EDU sets).

Avoids two encoder passes per dev tree by routing through `DMRSTParser.predict_both`
when the segmenter is present.
"""
import os
from typing import Callable, Dict, List, Optional, Tuple

import torch
from rich.table import Table

from iudex.rst.data.metrics import (
    compute_e2e_parseval,
    compute_parseval_metrics,
    compute_seg_metrics,
    f1,
    spans_to_subtoken_ranges,
)
from iudex.rst.data.tree import RstPpTree
from iudex.rst.parsers.dmrst.modeling_dmrst import DMRSTParser


def _write_rs4(tree: RstPpTree, output_dir: str, basename: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, basename), "w", encoding="utf-8") as f:
        f.write(tree.to_rs4_string())


@torch.no_grad()
def evaluate_dmrst(
    model: DMRSTParser,
    dev_pairs: List[Tuple[str, RstPpTree]],
    output_dir: Optional[str] = None,
) -> Dict[str, float]:
    """Single dev pass. When the model has a segmenter, returns gold-EDU
    Parseval + segmentation F1 + end-to-end Parseval (subtoken-range matching).
    Otherwise returns only the gold-EDU Parseval F1s (matches the generic
    `iudex.rst.training.evaluate`).

    When `output_dir` is given:
      - segmenter present: writes gold-EDU predictions to `{output_dir}/gold/`
        and e2e predictions to `{output_dir}/e2e/`.
      - segmenter absent: writes predictions directly to `{output_dir}` (no
        subdir split — matches the generic evaluator's existing layout).
    """
    use_seg = model.segmenter is not None

    totals_parseval = {f"{m}_{x}_count": 0 for m in ("span", "nuc", "rel", "full") for x in ("p", "r")}
    totals_parseval["num_spans"] = 0

    totals_seg = {"seg_correct": 0, "seg_pred_count": 0, "seg_gold_count": 0}

    totals_e2e = {
        f"e2e_{m}_{x}_count": 0
        for m in ("span", "nuc", "rel", "full")
        for x in ("p", "r")
    }
    totals_e2e["e2e_num_pred_spans"] = 0
    totals_e2e["e2e_num_gold_spans"] = 0

    for filepath, gold in dev_pairs:
        basename = os.path.splitext(os.path.basename(filepath))[0] + ".rs4"

        if not use_seg:
            pred = model.predict(gold)
            m = compute_parseval_metrics(gold, pred)
            for k in totals_parseval:
                totals_parseval[k] += m[k]
            if output_dir is not None:
                _write_rs4(pred, output_dir, basename)
            continue

        both = model.predict_both(gold)
        gold_pred = both["gold_pred"]
        e2e_pred = both["e2e_pred"]
        gold_edu_mapping = both["gold_edu_mapping"]
        pred_edu_mapping = both["pred_edu_mapping"]

        # Gold-EDU Parseval.
        m_par = compute_parseval_metrics(gold, gold_pred)
        for k in totals_parseval:
            totals_parseval[k] += m_par[k]

        # Segmentation F1.
        m_seg = compute_seg_metrics(both["gold_edu_ends"], both["pred_edu_ends"])
        for k in totals_seg:
            totals_seg[k] += m_seg[k]

        # End-to-end Parseval (subtoken-range matching).
        gold_subtok_spans = spans_to_subtoken_ranges(gold, gold_edu_mapping)
        if e2e_pred is not None and len(e2e_pred.edus) >= 2:
            pred_subtok_spans = spans_to_subtoken_ranges(e2e_pred, pred_edu_mapping)
        else:
            pred_subtok_spans = set()
        m_e2e = compute_e2e_parseval(gold_subtok_spans, pred_subtok_spans)
        for k in totals_e2e:
            totals_e2e[k] += m_e2e[k]

        if output_dir is not None:
            _write_rs4(gold_pred, os.path.join(output_dir, "gold"), basename)
            if e2e_pred is not None:
                _write_rs4(e2e_pred, os.path.join(output_dir, "e2e"), basename)

    # Aggregate gold-EDU Parseval (micro-average).
    n = totals_parseval["num_spans"]
    if n == 0:
        out: Dict[str, float] = {f"{m}_f1": 0.0 for m in ("span", "nuc", "rel", "full")}
    else:
        out = {
            f"{m}_f1": f1(totals_parseval[f"{m}_p_count"] / n, totals_parseval[f"{m}_r_count"] / n)
            for m in ("span", "nuc", "rel", "full")
        }

    if not use_seg:
        return out

    # Segmentation aggregates.
    seg_correct = totals_seg["seg_correct"]
    seg_pred = totals_seg["seg_pred_count"]
    seg_gold = totals_seg["seg_gold_count"]
    out["seg_p"] = seg_correct / max(seg_pred, 1) if seg_pred > 0 else 0.0
    out["seg_r"] = seg_correct / max(seg_gold, 1) if seg_gold > 0 else 0.0
    out["seg_f1"] = (2 * seg_correct) / max(seg_pred + seg_gold, 1) if (seg_pred + seg_gold) > 0 else 0.0

    # End-to-end Parseval aggregates: P denominator = total pred spans, R denominator = total gold spans.
    np_ = totals_e2e["e2e_num_pred_spans"]
    ng = totals_e2e["e2e_num_gold_spans"]
    for m in ("span", "nuc", "rel", "full"):
        p = totals_e2e[f"e2e_{m}_p_count"] / np_ if np_ > 0 else 0.0
        r = totals_e2e[f"e2e_{m}_r_count"] / ng if ng > 0 else 0.0
        out[f"e2e_{m}_p"] = p
        out[f"e2e_{m}_r"] = r
        out[f"e2e_{m}_f1"] = f1(p, r) or 0.0
    return out


def dmrst_metrics_table(metrics: Dict[str, float], title: str) -> Table:
    """Three sections — Gold-EDU Parseval / Segmentation / End-to-End Parseval.
    Latter two are omitted when their keys aren't in `metrics` (segmenter off).
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


def legal_val_metric_names(joint_segmentation: bool) -> List[str]:
    """The set of keys `evaluate_dmrst` will produce for a given config —
    used by the trainer to validate `cfg.val_metric_name` at startup."""
    names = ["span_f1", "nuc_f1", "rel_f1", "full_f1"]
    if joint_segmentation:
        names += ["seg_p", "seg_r", "seg_f1"]
        for m in ("span", "nuc", "rel", "full"):
            names += [f"e2e_{m}_p", f"e2e_{m}_r", f"e2e_{m}_f1"]
    return names
