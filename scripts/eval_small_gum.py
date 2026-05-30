"""Standing "real-lite" regression regime for the five new parsers.

Full GUM (211/32/32), tiny architecture-appropriate models, greedy decode.
NOT for SOTA numbers: each run exercises the whole pipeline at corpus scale so
that scale/behavioral regressions surface that toy smoke tests cannot, e.g. a
relation-label collapse (rel-F1 near zero while span-F1 is healthy), a
segmentation degenerating to 1 EDU/doc, an OOM, or a plain crash. Disposable:
everything lands under checkpoints_eval_small/.

Models:
  seq2seq_*        google-t5/t5-small        (encoder-decoder, full FT)
  decoder_only_*   google/gemma-3-270m       (causal, full FT, grad-ckpt)
  sr_biaffine      jhu-clsp/ettin-encoder-400m (gold-EDU Parseval, fast)

Matrix is 11 runs: the 3 single-config parsers + 4 sexp variants each
(preorder/postorder x copy/no-copy). The no-copy sexp runs use free-content
generation (constrain_content=false), the most failure-prone decode path.

Scheduler polls `nvidia-smi` free memory and launches a queued run whenever
free >= --reserve-gb and fewer than --max-concurrent are running (so ~2 fit on
a 24 GB card). Each finished run's last.pt + dev_predictions are purged to save
disk; best_model.pt + final_metrics.json + config.json are kept.

Usage:
  python scripts/eval_small_gum.py                 # generate configs, run matrix, print table
  python scripts/eval_small_gum.py --report        # re-print the table from existing runs (no training)
  python scripts/eval_small_gum.py --reserve-gb 11 --max-concurrent 2 --poll 30
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.expanduser("~/.mambaforge/envs/gud/bin/python")
CFG_DIR = os.path.join(REPO, "configs", "_eval_small")
LOG_DIR = os.path.join(REPO, "logs_eval_small")
CKPT_DIR = "checkpoints_eval_small"  # relative; matches checkpoint_dir in configs

GUM_TOK = "data/gum_12.1.0"  # biaffine: gold EDUs, tokenized text
GUM_NOTOK = "data/gum_12.1.0_notok"  # generative: raw text
T5 = "google-t5/t5-small"
GEMMA = "google/gemma-3-270m"
ETTIN = "jhu-clsp/ettin-encoder-400m"


def _gen_base(model: str) -> dict:
    """Common knobs for the four generative parsers (full FT, greedy decode)."""
    return {
        "train_dir": f"{GUM_NOTOK}/train",
        "dev_dir": f"{GUM_NOTOK}/dev",
        "test_dir": f"{GUM_NOTOK}/test",
        "relation_types": None,
        "relation_map": None,
        "model_name": model,
        "peft": None,  # full FT: these backbones are tiny
        "gradient_checkpointing": True,
        "amp": True,
        "weight_decay": 0.05,
        "max_epochs": 15,  # right past the knee: tiny models go "alive"
        "batch_size": 1,  # (non-trivial seg + structured trees) ~epoch 11,
        "grad_accum": 8,  # then plateau, so 15 exposes bugs without waste
        "optimizer": "adafactor",
        "num_warmup_steps": None,
        "max_grad_norm": 1.0,
        "patience": 4,
        "log_every": 10,
        "validate_every": None,  # epoch-end
        "checkpoint_every": None,
        "checkpoint_dir": CKPT_DIR,
        "seed": 42,
        "val_metric_name": "e2e_full_f1",
        "action_loss_weight": 1.0,
        "label_smoothing": 0.1,
        "dev_max_docs": 16,  # bounds per-epoch eval; FINAL eval is full
        "num_beams": 1,  # greedy throughout (fast, deterministic)
        "use_validity_constraints": True,
        "eval_decode_greedy": True,
        "min_edu_length": 1,
    }


def _seq2seq(model: str, max_output: int, lr: float) -> dict:
    return {
        **_gen_base(model),
        "max_input_length": 4096,
        "max_output_length": max_output,
        "lr": lr,
        "dev_batch_size": 8,
    }


def _decoder_only(model: str, lr: float) -> dict:
    return {
        **_gen_base(model),
        "causal_mode": True,
        "max_input_length": 3072,
        "max_output_length": 5120,
        # Eval once at the end. Decoder-only per-epoch eval (autoregressive
        # greedy over long single streams) dominates wall-clock; suppress it
        # with a huge validate_every. It is hash-excluded, so a partially-done
        # run resumes and picks this up without forking its run dir. The final
        # full dev/test eval still runs (on the last-epoch model).
        "validate_every": 1_000_000_000,
        "lr": lr,
        "dev_batch_size": 4,
    }


def _sr_biaffine() -> dict:
    return {
        "relation_map": None,
        "model_name": ETTIN,
        "ffn_hidden_size": 512,
        "action_ffn_hidden_size": 512,
        "dropout": 0.2,
        "stride": 100,
        "peft": None,
        "train_dir": f"{GUM_TOK}/train",
        "dev_dir": f"{GUM_TOK}/dev",
        "test_dir": f"{GUM_TOK}/test",
        "lr": 2e-4,
        "encoder_lr": 1e-5,
        "max_epochs": 18,  # discriminative on gold EDUs, converges faster
        "grad_accum": 1,
        "patience": 5,
        "max_grad_norm": 1.0,
        "weight_decay": 0.01,
        "num_warmup_steps": 500,
        "log_every": 5,
        "validate_every": None,
        "checkpoint_every": None,
        "checkpoint_dir": CKPT_DIR,
        "seed": 42,
        "val_metric_name": "span_f1",
    }


def _sexp_variant(base: dict, traversal: str, use_copy: bool) -> dict:
    # use_copy=True requires constrain_content=True (config __post_init__);
    # use_copy=False uses free-content generation, the riskiest decode path.
    return {
        **base,
        "traversal_order": traversal,
        "use_copy": use_copy,
        "constrain_content": True if use_copy else False,
    }


def build_matrix() -> list[tuple[str, str, dict]]:
    s2s_sexp = _seq2seq(T5, max_output=6144, lr=1e-3)
    do_sexp = _decoder_only(GEMMA, lr=5e-4)
    matrix: list[tuple[str, str, dict]] = [
        ("sr_biaffine", "sr_biaffine", _sr_biaffine()),
        ("seq2seq_sr", "seq2seq_sr", _seq2seq(T5, max_output=5120, lr=1e-3)),
        ("decoder_only_sr", "decoder_only_sr", _decoder_only(GEMMA, lr=5e-4)),
    ]
    for trav in ("postorder", "preorder"):
        for uc in (True, False):
            tag = f"{trav[:3]}_{'copy' if uc else 'nocopy'}"
            matrix.append((f"seq2seq_sexp_{tag}", "seq2seq_sexp", _sexp_variant(s2s_sexp, trav, uc)))
            matrix.append((f"decoder_only_sexp_{tag}", "decoder_only_sexp", _sexp_variant(do_sexp, trav, uc)))
    return matrix


# ---------- scheduler ----------


def free_mb() -> int:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    return int(out.split("\n")[0]) if out else 0


def write_cfg(name: str, cfg: dict) -> str:
    os.makedirs(CFG_DIR, exist_ok=True)
    cfg = {**cfg, "run_name": name}
    path = os.path.join(CFG_DIR, f"{name}.jsonnet")
    json.dump(cfg, open(path, "w"), indent=2)
    return path


def launch(name: str, parser: str, cfg_path: str) -> subprocess.Popen:
    os.makedirs(LOG_DIR, exist_ok=True)
    log = open(os.path.join(LOG_DIR, f"{name}.log"), "w")
    return subprocess.Popen(
        [PY, "-m", "iudex", parser, "train", cfg_path], cwd=REPO, stdout=log, stderr=subprocess.STDOUT
    )


def purge_run(name: str) -> None:
    """Drop the resumable last.pt and dev_predictions; keep best + metrics."""
    for rundir in glob.glob(os.path.join(REPO, CKPT_DIR, f"{name}-*")):
        for victim in ("last.pt",):
            p = os.path.join(rundir, victim)
            if os.path.exists(p):
                os.remove(p)
        dp = os.path.join(rundir, "dev_predictions")
        if os.path.isdir(dp):
            shutil.rmtree(dp, ignore_errors=True)


def _has_final_metrics(name: str) -> bool:
    return bool(glob.glob(os.path.join(REPO, CKPT_DIR, f"{name}-*", "final_metrics.json")))


def run_matrix(matrix, reserve_mb: int, max_concurrent: int, poll: int, settle: int) -> dict:
    # Skip runs already complete (final_metrics.json present) so a relaunch
    # doesn't retrain finished runs. Partially-done runs (no final_metrics yet)
    # stay queued and resume from their last.pt.
    done_already = [name for name, _p, _c in matrix if _has_final_metrics(name)]
    queue = [(name, p, c) for name, p, c in matrix if not _has_final_metrics(name)]
    running: list[dict] = []
    results: dict[str, dict] = {}
    if done_already:
        print(f"[regime] skipping {len(done_already)} already-complete: {', '.join(done_already)}", flush=True)
    print(f"[regime] {len(queue)} runs queued | reserve={reserve_mb}MB max_concurrent={max_concurrent}", flush=True)
    while queue or running:
        for r in list(running):
            rc = r["proc"].poll()
            if rc is not None:
                dur = time.time() - r["start"]
                results[r["name"]] = {"rc": rc, "sec": round(dur, 1)}
                purge_run(r["name"])
                running.remove(r)
                print(
                    f"[regime] DONE {r['name']} rc={rc} ({dur / 60:.1f} min) | {len(queue)} queued, {len(running)} running",
                    flush=True,
                )
        launched = False
        if queue and len(running) < max_concurrent:
            fm = free_mb()
            if fm >= reserve_mb:
                name, parser, cfg = queue.pop(0)
                cfg_path = write_cfg(name, cfg)
                proc = launch(name, parser, cfg_path)
                running.append({"name": name, "proc": proc, "start": time.time()})
                print(
                    f"[regime] LAUNCH {name} ({parser}) | free was {fm}MB | {len(queue)} queued, {len(running)} running",
                    flush=True,
                )
                launched = True
        # After a launch, let the new process allocate before re-polling free VRAM.
        time.sleep(settle if launched else poll)
    return results


# ---------- report ----------


def _flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out


def _pick(flat: dict, metric: str):
    """Best F1 for a metric, preferring test split then dev."""
    cands = [
        (k, v) for k, v in flat.items() if metric in k.lower() and "f1" in k.lower() and isinstance(v, (int, float))
    ]
    if not cands:
        return None
    for pref in ("test", "dev", ""):
        for k, v in cands:
            if pref in k.lower():
                return float(v)
    return float(cands[0][1])


def load_metrics(name: str):
    files = glob.glob(os.path.join(REPO, CKPT_DIR, f"{name}-*", "final_metrics.json"))
    if not files:
        return None
    flat = _flatten(json.load(open(sorted(files)[0])))
    return {m: _pick(flat, m) for m in ("seg", "span", "nuc", "rel", "full")}


def fmt(x):
    return f"{x:.3f}" if isinstance(x, float) else "  -  "


def report(matrix, results: dict | None):
    print("\n" + "=" * 92)
    print("REAL-LITE GUM REGIME  (tiny models, full GUM, greedy)  — sanity floors, NOT quality targets")
    print("=" * 92)
    header = f"{'run':<28} {'seg':>6} {'span':>6} {'nuc':>6} {'rel':>6} {'full':>6}  {'min':>5}  flags"
    print(header)
    print("-" * 92)
    for name, _parser, _cfg in matrix:
        m = load_metrics(name)
        rc = (results or {}).get(name, {})
        flags = []
        if rc.get("rc") not in (0, None):
            flags.append(f"CRASH(rc={rc['rc']})")
        if m is None:
            flags.append("NO-METRICS")
            row = f"{name:<28} " + " ".join(f"{'  -  ':>6}" for _ in range(5))
        else:
            row = (
                f"{name:<28} {fmt(m['seg']):>6} {fmt(m['span']):>6} {fmt(m['nuc']):>6} "
                f"{fmt(m['rel']):>6} {fmt(m['full']):>6}"
            )
            # tripwire: relation labeling collapsed while structure is fine
            if isinstance(m["span"], float) and m["span"] > 0.10 and isinstance(m["rel"], float) and m["rel"] < 0.02:
                flags.append("REL-COLLAPSE")
            # tripwire: e2e parser barely segments (sr_biaffine has no seg, skip)
            if isinstance(m["seg"], float) and m["seg"] < 0.05:
                flags.append("UNDER-SEG")
        mins = rc.get("sec", 0) / 60 if rc else 0
        print(f"{row}  {mins:>5.1f}  {' '.join(flags) if flags else 'ok'}")
    print("-" * 92)
    print(
        "tripwires: REL-COLLAPSE = span>0.10 but rel<0.02 (relation head broken, cf. bug #1); UNDER-SEG = seg-F1<0.05."
    )
    print("=" * 92, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true", help="only re-print the table from existing runs")
    ap.add_argument("--reserve-gb", type=float, default=11.0, help="free VRAM (GB) required before launching a run")
    ap.add_argument("--max-concurrent", type=int, default=2)
    ap.add_argument("--poll", type=int, default=30, help="seconds between scheduler ticks")
    ap.add_argument("--settle", type=int, default=75, help="seconds to wait after a launch before re-polling VRAM")
    args = ap.parse_args()

    matrix = build_matrix()
    if args.report:
        report(matrix, None)
        return
    results = run_matrix(matrix, int(args.reserve_gb * 1024), args.max_concurrent, args.poll, args.settle)
    report(matrix, results)


if __name__ == "__main__":
    main()
