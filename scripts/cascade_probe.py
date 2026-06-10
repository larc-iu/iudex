"""Cascade-vs-intrinsic decomposition of wide-span recall (gold-EDU decode).

cascade_probe.py <parser> <ckpt.pt> [split] [--save-preds DIR] [--json PATH]

Companion to checkpoints_overnight/_scripts/decompose.py, which measured
width-stratified span recall under free structure decode and deferred the
cascade question. This probe answers it without forced decoding: for every
gold constituent, check whether BOTH of its gold children appear in the
predicted tree. A parent miss with both children present is a genuine
attachment error at that junction (the model had the right pieces and merged
one of them elsewhere). A parent miss with a missing child is (at least
partly) upward cascade from a lower error.

Per width bucket this reports:
  recall          plain span recall (should reproduce decompose.py)
  ctx             P(both gold children present in prediction)
  att|ctx         P(parent present | both children present), the intrinsic
                  attachment success rate with correct local context
  miss%att        fraction of misses that are attachment errors (ctx held)
  nuc|hit rel|hit conditional label accuracy on matched spans

Reading the result: if att|ctx stays high at wide widths and most misses are
cascade, the wide-span collapse is error propagation and the lever is
mid-level accuracy. If att|ctx falls with width, high-level attachment
decisions themselves fail and a structural lever (stack scratchpad) has a
target.
"""

import argparse
import json
import math
import os
import sys

import torch

from iudex.common.log import wrote
from iudex.rst.data.reader import read_rst_dir
from iudex.rst.parsers.common.inference import load_parser_from_checkpoint
from iudex.rst.parsers.seq2seq_sr.configuration_seq2seq_sr import Seq2SeqSRConfig
from iudex.rst.parsers.seq2seq_sr.modeling_seq2seq_sr import Seq2SeqSRParser
from iudex.rst.parsers.sr_biaffine.configuration_sr_biaffine import SRBiaffineConfig
from iudex.rst.parsers.sr_biaffine.modeling_sr_biaffine import SRBiaffineParser
from iudex.rst.parsers.topdown_biaffine.configuration_topdown_biaffine import TopdownBiaffineConfig
from iudex.rst.parsers.topdown_biaffine.modeling_topdown_biaffine import TopdownBiaffineParser

P = {
    "sr_biaffine": (SRBiaffineConfig, SRBiaffineParser),
    "topdown_biaffine": (TopdownBiaffineConfig, TopdownBiaffineParser),
    "seq2seq_sr": (Seq2SeqSRConfig, Seq2SeqSRParser),
}

BUCKETS = [(2, 2), (3, 4), (5, 8), (9, 16), (17, math.inf)]


def bucket_of(w):
    for lo, hi in BUCKETS:
        if lo <= w <= hi:
            return (lo, hi)
    return None


def bucket_label(b):
    lo, hi = b
    if hi == math.inf:
        return f"{lo}+"
    return f"{lo}" if lo == hi else f"{lo}-{hi}"


def constituents(tree):
    """{(start_edu, end_edu): (nuc, rel, left_child_key, right_child_key)}.

    Child keys are (start, end) EDU ranges. A width-1 child is a leaf EDU,
    trivially present under gold-EDU decode.
    """
    out = {}
    for (left, right), nuc, rel in tree.spans():
        key = (left[0], right[-1])
        out[key] = (nuc, rel, (left[0], left[-1]), (right[0], right[-1]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parser", choices=sorted(P))
    ap.add_argument("checkpoint")
    ap.add_argument("split", nargs="?", default="test")
    ap.add_argument("--save-preds", default=None, help="dir to write gold-EDU .rs4 predictions")
    ap.add_argument("--json", default=None, help="path to dump per-bucket counts as JSON")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config_cls, parser_cls = P[args.parser]
    print(f"loading {args.parser} {args.checkpoint} split={args.split}", flush=True)
    m = load_parser_from_checkpoint(args.checkpoint, dev, config_cls, parser_cls)
    cfg = m.config
    data_dir = {"dev": cfg.dev_dir, "test": cfg.test_dir, "train": cfg.train_dir}[args.split]
    trees = read_rst_dir(data_dir, relation_types=cfg.relation_types, relation_map=cfg.relation_map)

    pred_fn = m.predict_with_gold_edus if hasattr(m, "predict_with_gold_edus") else m.predict

    if args.save_preds:
        os.makedirs(args.save_preds, exist_ok=True)

    # counts[bucket] keys: n, hit, ctx, hit_ctx, nuc_hit, rel_hit
    zero = dict(n=0, hit=0, ctx=0, hit_ctx=0, nuc_hit=0, rel_hit=0)
    counts = {b: dict(zero) for b in BUCKETS}
    for i, (path, tree) in enumerate(trees):
        if len(tree.edus) < 2:
            continue
        pred = pred_fn(tree)
        if args.save_preds:
            out_path = os.path.join(args.save_preds, os.path.splitext(os.path.basename(path))[0] + ".rs4")
            with open(out_path, "w") as f:
                f.write(pred.to_rs4_string())
        gold = constituents(tree)
        pk = constituents(pred)

        def present(key):
            return key[0] == key[1] or key in pk

        for key, (gn, gr, lchild, rchild) in gold.items():
            b = bucket_of(key[1] - key[0] + 1)
            if b is None:
                continue
            c = counts[b]
            c["n"] += 1
            ctx = present(lchild) and present(rchild)
            hit = key in pk
            c["ctx"] += ctx
            c["hit"] += hit
            c["hit_ctx"] += hit and ctx
            if hit:
                pn, pr = pk[key][0], pk[key][1]
                c["nuc_hit"] += pn == gn
                c["rel_hit"] += pr == gr
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(trees)} docs", flush=True)

    if args.save_preds:
        wrote(os.path.abspath(args.save_preds))

    print(f"\n### {args.parser} {args.split} cascade-vs-intrinsic (gold-EDU decode)")
    hdr = f"{'bucket':>8} {'n':>5} {'recall':>7} {'ctx':>7} {'att|ctx':>8} {'miss%att':>9} {'nuc|hit':>8} {'rel|hit':>8}"
    print(hdr)
    tot = dict(zero)
    for b in BUCKETS + ["ALL"]:
        if b == "ALL":
            c, lbl = tot, "ALL"
        else:
            c, lbl = counts[b], bucket_label(b)
            for k in tot:
                tot[k] += c[k]
        n, hit, ctx, hit_ctx = c["n"], c["hit"], c["ctx"], c["hit_ctx"]
        if n == 0:
            print(f"{lbl:>8} {n:>5}" + " --" * 6)
            continue
        misses = n - hit
        att_misses = ctx - hit_ctx
        cells = [
            f"{hit / n:>7.3f}",
            f"{ctx / n:>7.3f}",
            f"{hit_ctx / ctx:>8.3f}" if ctx else f"{'--':>8}",
            f"{att_misses / misses:>9.3f}" if misses else f"{'--':>9}",
            f"{c['nuc_hit'] / hit:>8.3f}" if hit else f"{'--':>8}",
            f"{c['rel_hit'] / hit:>8.3f}" if hit else f"{'--':>8}",
        ]
        print(f"{lbl:>8} {n:>5} " + " ".join(cells), flush=True)

    if args.json:
        payload = {
            "parser": args.parser,
            "checkpoint": args.checkpoint,
            "split": args.split,
            "buckets": {bucket_label(b): counts[b] for b in BUCKETS},
        }
        with open(args.json, "w") as f:
            json.dump(payload, f, indent=2)
        wrote(os.path.abspath(args.json))


if __name__ == "__main__":
    main()
