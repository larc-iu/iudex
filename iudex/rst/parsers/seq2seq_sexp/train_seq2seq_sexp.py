import argparse
import dataclasses
import json
import logging
import math
import os
import time
from collections import deque

import torch
from tonga import Params
from torch.utils.data import DataLoader, Dataset

from iudex.common.log import console, dim, rule, setup_logging, success, warn, wrote
from iudex.common.training import (
    TBLogger,
    build_optimizer,
    config_panel,
    device_panel,
    gpu_mem_gb,
    install_abort_handler,
    make_progress_bar,
    make_scheduler,
    model_panel,
    prepare_run_dir,
    resume_or_init,
    save_checkpoint,
    schedule_panel,
    set_seeds,
    weight_decay_panel,
    write_run_config,
)
from iudex.rst import HASH_EXCLUDE
from iudex.rst.data.metrics import compute_parseval_metrics, f1, metrics_table
from iudex.rst.data.reader import infer_relation_types, read_rst_dir
from iudex.rst.data.seg_metrics import evaluate_seg_and_e2e
from iudex.rst.data.tree import RstTree
from iudex.rst.parsers.common.encoding import align_edus_to_tokens
from iudex.rst.parsers.seq2seq_sexp.configuration_seq2seq_sexp import Seq2SeqSexpConfig
from iudex.rst.parsers.seq2seq_sexp.modeling_seq2seq_sexp import (
    Seq2SeqSexpParser,
    _reconstruct_text,
)

setup_logging()
logger = logging.getLogger(__name__)


class _Seq2SeqSexpDataset(Dataset):
    def __init__(self, items: list[dict]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        return self.items[idx]


def _build_dataset(model: Seq2SeqSexpParser, pairs: list[tuple[str, RstTree]]) -> tuple[_Seq2SeqSexpDataset, int]:
    items: list[dict] = []
    dropped = 0
    for _, tree in pairs:
        encoded = model.encode_target(tree)
        if encoded is None:
            dropped += 1
            continue
        labels, decoder_input_ids = encoded
        text = _reconstruct_text(tree)
        enc = model.encode_input(text)
        items.append(
            {
                "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
                "labels": labels,
                "decoder_input_ids": decoder_input_ids,
            }
        )
    return _Seq2SeqSexpDataset(items), dropped


def _build_optimizer(model: Seq2SeqSexpParser, cfg: Seq2SeqSexpConfig):
    if cfg.optimizer == "adamw":
        return build_optimizer(model, cfg.lr, cfg.weight_decay)
    if cfg.optimizer == "adafactor":
        from transformers import Adafactor

        return Adafactor(
            [p for p in model.parameters() if p.requires_grad],
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            scale_parameter=False,
            relative_step=False,
            warmup_init=False,
        )
    raise ValueError(f"Unknown optimizer {cfg.optimizer!r}; expected 'adamw' or 'adafactor'.")


def _make_collator(pad_id: int):
    def collate(batch: list[dict]) -> dict[str, torch.Tensor]:
        max_in = max(len(item["input_ids"]) for item in batch)
        max_lbl = max(len(item["labels"]) for item in batch)
        input_ids = torch.full((len(batch), max_in), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), max_in), dtype=torch.long)
        labels = torch.full((len(batch), max_lbl), -100, dtype=torch.long)
        decoder_input_ids = torch.full((len(batch), max_lbl), pad_id, dtype=torch.long)
        for i, item in enumerate(batch):
            n = len(item["input_ids"])
            input_ids[i, :n] = torch.tensor(item["input_ids"], dtype=torch.long)
            attention_mask[i, :n] = torch.tensor(item["attention_mask"], dtype=torch.long)
            m = len(item["labels"])
            labels[i, :m] = torch.tensor(item["labels"], dtype=torch.long)
            d = len(item["decoder_input_ids"])
            decoder_input_ids[i, :d] = torch.tensor(item["decoder_input_ids"], dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "decoder_input_ids": decoder_input_ids,
        }

    return collate


def _gold_edu_token_mapping(model: Seq2SeqSexpParser, tree: RstTree) -> tuple[list[int], list[tuple[int, int]]]:
    """Per-EDU `(start, end_exclusive)` ranges in encoder whole-doc token
    space. Same logic as the seq2seq_sr helper of the same name."""
    text = _reconstruct_text(tree)
    _full_input_ids, spans = align_edus_to_tokens(model.tokenizer, text, tree.edus)
    mapping = list(spans)
    edu_ends = [end - 1 for _, end in mapping]
    return edu_ends, mapping


def _pred_edu_token_mapping(pred_tree: RstTree) -> tuple[list[int], list[tuple[int, int]]]:
    ranges = getattr(pred_tree, "_pred_edu_source_ranges", None)
    if ranges is None:
        return [], []
    edu_ends = [end - 1 for _, end in ranges]
    return edu_ends, list(ranges)


def _write_rs4(tree: RstTree, output_dir: str, basename: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, basename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(tree.to_rs4_string())


@torch.no_grad()
def _evaluate_gold_edu(
    model: Seq2SeqSexpParser,
    dev_pairs: list[tuple[str, RstTree]],
) -> dict[str, float]:
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
def _evaluate_on_dev(
    model: Seq2SeqSexpParser,
    dev_pairs: list[tuple[str, RstTree]],
    *,
    num_beams: int | None = None,
    batch_size: int = 1,
    output_dir: str | None = None,
    eval_gold_edu: bool = False,
) -> dict[str, float]:
    model.eval()
    gold_trees: list[RstTree] = []
    seg_data: list[dict] = []
    eval_t0 = time.monotonic()
    for chunk_start in range(0, len(dev_pairs), batch_size):
        chunk = dev_pairs[chunk_start : chunk_start + batch_size]
        chunk_t0 = time.monotonic()
        preds = model.predict_batch([gold for _, gold in chunk], num_beams=num_beams)
        chunk_dt = time.monotonic() - chunk_t0
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
        wrote(os.path.abspath(output_dir))
    metrics = evaluate_seg_and_e2e(gold_trees, seg_data)
    if eval_gold_edu:
        metrics.update(_evaluate_gold_edu(model, dev_pairs))
    return metrics


def train(cfg: Seq2SeqSexpConfig) -> None:
    set_seeds(cfg.seed)

    run_dir, cfg_hash = prepare_run_dir(
        dataclasses.asdict(cfg), cfg.checkpoint_dir, cfg.run_name, hash_exclude=HASH_EXCLUDE
    )

    if cfg.relation_map is not None:
        dim(f"Applying `relation_map` ({len(cfg.relation_map)} entries) to all read trees.")
    cfg.relation_types = infer_relation_types([cfg.train_dir, cfg.dev_dir], relation_map=cfg.relation_map)
    dim(
        f"Inferred {len(cfg.relation_types)} (relation, kind) pairs from "
        f"{cfg.train_dir} + {cfg.dev_dir}" + (" (after relation_map)." if cfg.relation_map is not None else ".")
    )

    cfg_dict = dataclasses.asdict(cfg)
    write_run_config(run_dir, cfg_dict)
    tb = TBLogger(run_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = cfg.amp and device.type == "cuda"
    model = Seq2SeqSexpParser(cfg).to(device)

    train_pairs = read_rst_dir(cfg.train_dir, relation_types=cfg.relation_types, relation_map=cfg.relation_map)
    dev_pairs = read_rst_dir(cfg.dev_dir, relation_types=cfg.relation_types, relation_map=cfg.relation_map)
    test_pairs = (
        read_rst_dir(cfg.test_dir, relation_types=cfg.relation_types, relation_map=cfg.relation_map)
        if cfg.test_dir is not None
        else None
    )

    train_ds, dropped = _build_dataset(model, train_pairs)
    if dropped > 0:
        warn(
            f"Dropped {dropped}/{len(train_pairs)} training trees whose target exceeded "
            f"max_output_length={cfg.max_output_length}. Bump it if this is a large fraction."
        )

    collate = _make_collator(model.tokenizer.pad_token_id)
    rng_seed = torch.Generator()
    rng_seed.manual_seed(cfg.seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate,
        generator=rng_seed,
    )

    # ceil, not floor: the trailing partial accumulation window is stepped too
    # (see the optimizer-step guard `(batch_idx + 1) == len(train_loader)`).
    steps_per_epoch = max(1, math.ceil(len(train_loader) / cfg.grad_accum))
    total_steps = steps_per_epoch * cfg.max_epochs
    warmup = cfg.num_warmup_steps if cfg.num_warmup_steps is not None else max(1, int(0.1 * total_steps))

    console.print(config_panel(cfg_dict))
    console.print(device_panel(device, seed=cfg.seed, checkpoint_dir=run_dir))
    console.print(model_panel(model, num_train_trees=len(train_ds), grad_accum=cfg.grad_accum))
    console.print(
        schedule_panel(
            steps_per_epoch=steps_per_epoch,
            total_steps=total_steps,
            warmup_steps=warmup,
            lr=cfg.lr,
            encoder_lr=None,
        )
    )

    optimizer = _build_optimizer(model, cfg)
    if cfg.optimizer == "adamw":
        console.print(weight_decay_panel(model, optimizer))
    scheduler = make_scheduler(optimizer, warmup, total_steps)

    resumed = resume_or_init(
        run_dir,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        expected_hash=cfg_hash,
    )
    global_step = resumed["global_step"]
    start_epoch = resumed["epoch"]
    best_val = resumed["best_val"]
    stale = resumed["stale_validations"]

    def _save(path: str, epoch: int) -> None:
        save_checkpoint(
            path,
            model,
            optimizer,
            scheduler,
            config=cfg_dict,
            config_hash=cfg_hash,
            global_step=global_step,
            epoch=epoch,
            best_val=best_val,
            stale_validations=stale,
        )

    dev_beams = 1 if cfg.eval_decode_greedy else cfg.num_beams
    per_epoch_dev = dev_pairs if cfg.dev_max_docs is None else dev_pairs[: cfg.dev_max_docs]
    if cfg.dev_max_docs is not None and cfg.dev_max_docs < len(dev_pairs):
        dim(
            f"Per-epoch dev eval is capped to the first {len(per_epoch_dev)}/"
            f"{len(dev_pairs)} documents (cfg.dev_max_docs). Final eval still uses all."
        )

    def _validate(epoch: int) -> None:
        nonlocal best_val, stale
        pred_dir = os.path.join(run_dir, "dev_predictions", f"epoch{epoch}_step{global_step}")
        # Per-epoch validation deliberately skips the gold-EDU pass to save
        # time. Final dev/test eval below always runs it.
        metrics = _evaluate_on_dev(
            model,
            per_epoch_dev,
            num_beams=dev_beams,
            batch_size=cfg.dev_batch_size,
            output_dir=pred_dir,
        )
        tb.log_scalars("dev", metrics, global_step)
        console.print(metrics_table(metrics, title=f"Dev @ step {global_step}"))
        score = metrics.get(cfg.val_metric_name, 0.0)
        if score > best_val:
            best_val = score
            stale = 0
            _save(os.path.join(run_dir, "best_model.pt"), epoch)
            success(f"  New best! {cfg.val_metric_name}={best_val:.4f}")
        else:
            stale += 1
            dim(f"  No improvement ({stale}/{cfg.patience})")
        model.train()

    aborted = install_abort_handler()
    training_complete = start_epoch >= cfg.max_epochs or stale >= cfg.patience
    if training_complete:
        reason = "max_epochs reached" if start_epoch >= cfg.max_epochs else "patience exhausted"
        dim(f"Skipping training: {reason} on prior run; jumping to final evaluation.")

    recent_losses: deque = deque(maxlen=200)
    recent_action_losses: deque = deque(maxlen=200)
    recent_copy_losses: deque = deque(maxlen=200)
    if not training_complete:
        rule("Training")
    training_start = time.monotonic()

    for epoch in range(start_epoch, cfg.max_epochs):
        if stale >= cfg.patience or aborted.value:
            break
        epoch_start = time.monotonic()
        model.train()
        total_loss = 0.0
        num_batches = 0
        epoch_step = 0

        with make_progress_bar() as progress:
            task = progress.add_task(
                "training",
                total=steps_per_epoch,
                epoch=f"{epoch + 1}/{cfg.max_epochs}",
                loss_str="loss=-.----",
                lr_str="",
                mem_str="",
                total_elapsed="0:00:00",
            )

            for batch_idx, batch in enumerate(train_loader):
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
                    out = model(batch)
                loss = out["loss"]
                if cfg.grad_accum > 1:
                    loss = loss / cfg.grad_accum
                loss.backward()
                raw_loss = loss.item() * (cfg.grad_accum if cfg.grad_accum > 1 else 1)
                recent_losses.append(raw_loss)
                if "action_loss" in out:
                    recent_action_losses.append(float(out["action_loss"].item()))
                    recent_copy_losses.append(float(out["copy_loss"].item()))
                total_loss += raw_loss
                num_batches += 1

                is_step = (batch_idx + 1) % cfg.grad_accum == 0 or (batch_idx + 1) == len(train_loader)
                if not is_step:
                    continue

                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                epoch_step += 1

                avg_loss = sum(recent_losses) / len(recent_losses)
                lr_display = "/".join(f"{lr:.1e}" for lr in sorted(set(scheduler.get_last_lr())))
                mem = gpu_mem_gb(device)
                mem_str = f"[gpu]max_mem={mem[1]:.1f}GB[/gpu]" if mem else ""
                secs = int(time.monotonic() - training_start)
                progress.update(
                    task,
                    advance=1,
                    loss_str=f"loss=[bold orange1]{avg_loss:.4f}[/bold orange1]",
                    lr_str=f"lr=[dim]{lr_display}[/dim]",
                    mem_str=mem_str,
                    total_elapsed=f"{secs // 3600}:{(secs % 3600) // 60:02d}:{secs % 60:02d}",
                )

                if epoch_step % cfg.log_every == 0:
                    tb_train = {"loss": avg_loss, "lr": max(scheduler.get_last_lr()), "grad_norm": float(grad_norm)}
                    if recent_action_losses:
                        tb_train["action_loss"] = sum(recent_action_losses) / len(recent_action_losses)
                        tb_train["copy_loss"] = sum(recent_copy_losses) / len(recent_copy_losses)
                    if mem:
                        tb_train["gpu_mem_gb"] = mem[1]
                    tb.log_scalars("train", tb_train, global_step)
                    mem_log = f"  mem=[dim]{mem[1]:.1f}GB[/dim]" if mem else ""
                    split_log = ""
                    if recent_action_losses:
                        a = sum(recent_action_losses) / len(recent_action_losses)
                        c = sum(recent_copy_losses) / len(recent_copy_losses)
                        split_log = f"  act=[dim]{a:.3f}[/dim]  cpy=[dim]{c:.3f}[/dim]"
                    progress.console.print(
                        f"  [step]step {epoch_step}/{steps_per_epoch}[/step]  "
                        f"loss=[loss]{avg_loss:.4f}[/loss]{split_log}  "
                        f"grad=[dim]{grad_norm:.4f}[/dim]  "
                        f"lr=[dim]{lr_display}[/dim]{mem_log}"
                    )

                if cfg.validate_every and global_step % cfg.validate_every == 0:
                    _validate(epoch + 1)
                    if stale >= cfg.patience or aborted.value:
                        break

                if cfg.checkpoint_every and global_step % cfg.checkpoint_every == 0:
                    _save(os.path.join(run_dir, "last.pt"), epoch + 1)

        if num_batches > 0 and stale < cfg.patience:
            console.print(
                f"  [epoch]Epoch {epoch + 1}/{cfg.max_epochs}[/epoch] "
                f"[dim]({time.monotonic() - epoch_start:.1f}s)[/dim]  "
                f"loss=[loss]{total_loss / num_batches:.4f}[/loss]"
            )
            if not cfg.validate_every:
                _validate(epoch + 1)
            _save(os.path.join(run_dir, "last.pt"), epoch + 1)
            if stale >= cfg.patience or aborted.value:
                warn(f"\nEarly stopping after {cfg.patience} validations without improvement")
                break
        elif stale >= cfg.patience or aborted.value:
            warn(f"\nEarly stopping at step {global_step}")
            break

    rule("Final Evaluation")
    best_path = os.path.join(run_dir, "best_model.pt")
    if os.path.exists(best_path):
        checkpoint = torch.load(best_path, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        dev_m = _evaluate_on_dev(
            model,
            dev_pairs,
            num_beams=cfg.num_beams,
            batch_size=cfg.dev_batch_size,
            output_dir=os.path.join(run_dir, "dev_predictions", "final"),
            eval_gold_edu=True,
        )
        console.print(metrics_table(dev_m, title="Final Dev Results"))
        final_metrics: dict[str, dict[str, float]] = {"dev": dev_m}
        if test_pairs is not None:
            test_m = _evaluate_on_dev(
                model,
                test_pairs,
                num_beams=cfg.num_beams,
                batch_size=cfg.dev_batch_size,
                output_dir=os.path.join(run_dir, "test_predictions", "final"),
                eval_gold_edu=True,
            )
            console.print(metrics_table(test_m, title="Final Test Results"))
            final_metrics["test"] = test_m
            tb.log_scalars("test", test_m, global_step)
        metrics_path = os.path.join(run_dir, "final_metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(final_metrics, f, indent=2)
        wrote(metrics_path)
    else:
        success(f"Training complete. Best {cfg.val_metric_name}: {best_val:.4f}")
    tb.close()


def main():
    parser = argparse.ArgumentParser(description="Train the seq2seq_sexp parser")
    parser.add_argument("config", help="Path to a jsonnet config file")
    args = parser.parse_args()
    cfg = Seq2SeqSexpConfig.from_dict(Params.from_file(args.config).as_dict(quiet=True))
    train(cfg)


if __name__ == "__main__":
    main()
