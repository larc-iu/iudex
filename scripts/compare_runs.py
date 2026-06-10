"""Paired significance test between two prediction dirs on the same documents.

compare_runs.py --gold DIR --config CONFIG_JSON --pred-a DIR --pred-b DIR
                [--mode gold_edu|e2e_char] [--label A] [--label-b B]

Reads gold trees (relations mapped through the run config's relation_map, the
same mapping the parsers trained against) and two dirs of predicted .rs4
files, then runs `iudex.rst.data.metrics.paired_permutation_test` per metric
(span/nuc/rel/full). Use this instead of eyeballing corpus F1 deltas: on a
37-doc split one document moves FULL F1 by ~0.03 of its own swing, so an
unpaired comparison can't distinguish ~0.02 deltas from noise. The paired
test can, because both systems are scored on the same documents.

Modes:
  gold_edu  EDU-index Parseval. Requires every prediction to share the gold
            EDU set (encoder parsers, or gold-EDU-forced generative output).
  e2e_char  Constituent keys are (start, end) offsets in whitespace-stripped
            characters of the concatenated EDU text. Exact for copy-based
            parsers (predictions preserve source characters). STRICTER than
            the runtime token-range e2e Parseval, so absolute numbers differ
            slightly from training logs. Paired deltas are the point here.
"""

import argparse
import json
import os

from iudex.rst.data.metrics import compute_parseval_metrics, paired_permutation_test
from iudex.rst.data.seg_metrics import compute_e2e_parseval
from iudex.rst.data.reader import read_rst_dir


def char_ranges(tree):
    """Parent-keyed constituent set in whitespace-stripped char coordinates."""
    lengths = [len("".join(e.text.split())) for e in tree.edus]
    starts = []
    pos = 0
    for n in lengths:
        starts.append(pos)
        pos += n
    out = set()
    for (left, right), nuc, rel in tree.spans():
        key = (starts[left[0]], starts[right[-1]] + lengths[right[-1]] - 1)
        out.add((key, nuc, rel))
    return out


def per_doc_counts_gold_edu(gold, pred):
    m = compute_parseval_metrics(gold, pred)
    return {
        name: (m[f"{name}_p_count"], m[f"{name}_r_count"], m["num_spans"], m["num_spans"])
        for name in ("span", "nuc", "rel", "full")
    }


def per_doc_counts_e2e_char(gold, pred):
    m = compute_e2e_parseval(char_ranges(gold), char_ranges(pred))
    return {
        name: (
            m[f"e2e_{name}_p_count"],
            m[f"e2e_{name}_r_count"],
            m["e2e_num_pred_spans"],
            m["e2e_num_gold_spans"],
        )
        for name in ("span", "nuc", "rel", "full")
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True, help="gold .rs3/.rs4 dir")
    ap.add_argument("--config", required=True, help="a run dir config.json supplying relation_types/relation_map")
    ap.add_argument("--pred-a", required=True)
    ap.add_argument("--pred-b", required=True)
    ap.add_argument("--mode", choices=("gold_edu", "e2e_char"), default="gold_edu")
    ap.add_argument("--label-a", default=None)
    ap.add_argument("--label-b", default=None)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    relation_types = tuple(tuple(rt) for rt in cfg["relation_types"])
    relation_map = cfg.get("relation_map")

    golds = {
        os.path.splitext(os.path.basename(p))[0]: t
        for p, t in read_rst_dir(args.gold, relation_types=relation_types, relation_map=relation_map)
    }
    preds = {}
    for side, d in (("a", args.pred_a), ("b", args.pred_b)):
        preds[side] = {
            os.path.splitext(os.path.basename(p))[0]: t
            for p, t in read_rst_dir(d, relation_types=relation_types)
        }

    docs = sorted(set(golds) & set(preds["a"]) & set(preds["b"]))
    missing = sorted(set(golds) - set(docs))
    if missing:
        print(f"NB: {len(missing)} gold docs missing from a pred dir, excluded: {missing}")
    if not docs:
        raise SystemExit("no shared documents between gold and both pred dirs")

    count_fn = per_doc_counts_gold_edu if args.mode == "gold_edu" else per_doc_counts_e2e_char
    counts = {"a": {m: [] for m in ("span", "nuc", "rel", "full")}, "b": {m: [] for m in ("span", "nuc", "rel", "full")}}
    skipped = []
    for doc in docs:
        try:
            doc_counts = {side: count_fn(golds[doc], preds[side][doc]) for side in ("a", "b")}
        except ValueError:
            # gold_edu mode with a prediction that does not share the gold EDU set
            skipped.append(doc)
            continue
        for side in ("a", "b"):
            for m in ("span", "nuc", "rel", "full"):
                counts[side][m].append(doc_counts[side][m])
    if skipped:
        print(f"NB: {len(skipped)} docs skipped for EDU-set mismatch (use --mode e2e_char?): {skipped}")

    label_a = args.label_a or args.pred_a
    label_b = args.label_b or args.pred_b
    n = len(counts["a"]["span"])
    print(f"\nPaired permutation test over {n} docs ({args.mode})")
    print(f"  A = {label_a}\n  B = {label_b}")
    print(f"{'metric':>7} {'F1(A)':>8} {'F1(B)':>8} {'delta':>8} {'95% CI':>18} {'p':>8}")
    for m in ("span", "nuc", "rel", "full"):
        r = paired_permutation_test(counts["a"][m], counts["b"][m], seed=args.seed)
        ci = f"[{r['ci_low']:+.4f},{r['ci_high']:+.4f}]"
        print(f"{m:>7} {r['f1_a']:>8.4f} {r['f1_b']:>8.4f} {r['delta']:>+8.4f} {ci:>18} {r['p_value']:>8.4f}")


if __name__ == "__main__":
    main()
