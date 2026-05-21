import argparse
import dataclasses
import json
import logging
import math
import os
import random
import time
from collections import deque

import torch
from tonga import Params

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
    save_checkpoint,
    schedule_panel,
    set_seeds,
    try_resume,
    write_run_config,
)
from iudex.rst import HASH_EXCLUDE
from iudex.rst.data.metrics import evaluate_parseval, metrics_table
from iudex.rst.data.reader import infer_relation_types, read_rst_dir
from iudex.rst.data.seg_metrics import evaluate_seg_and_e2e
from iudex.rst.data.tree import RstTree
from iudex.rst.parsers.dmrst.configuration_dmrst import DMRSTConfig
from iudex.rst.parsers.dmrst.modeling_dmrst import DMRSTParser

setup_logging()
logger = logging.getLogger(__name__)


def _write_rs4(tree: RstTree, output_dir: str, basename: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, basename), "w", encoding="utf-8") as f:
        f.write(tree.to_rs4_string())


@torch.no_grad()
def _evaluate_on_dev(
    model: DMRSTParser,
    dev_pairs: list[tuple[str, RstTree]],
    output_dir: str | None = None,
) -> dict[str, float]:
    """Run the model over `dev_pairs` and aggregate Parseval (+ seg + e2e when
    joint segmentation is on). One encoder pass per tree via `predict_both`."""
    use_seg = model.segmenter is not None
    gold_trees: list[RstTree] = []
    gold_preds: list[RstTree] = []
    seg_data: list[dict] | None = [] if use_seg else None
    for filepath, gold in dev_pairs:
        basename = os.path.splitext(os.path.basename(filepath))[0] + ".rs4"
        gold_trees.append(gold)
        if not use_seg:
            pred = model.predict(gold)
            gold_preds.append(pred)
            if output_dir is not None:
                _write_rs4(pred, output_dir, basename)
            continue
        both = model.predict_both(gold)
        gold_preds.append(both["gold_pred"])
        seg_data.append(
            {
                k: both[k]
                for k in (
                    "gold_edu_ends",
                    "pred_edu_ends",
                    "e2e_pred",
                    "gold_edu_mapping",
                    "pred_edu_mapping",
                )
            }
        )
        if output_dir is not None:
            _write_rs4(both["gold_pred"], os.path.join(output_dir, "gold"), basename)
            if both["e2e_pred"] is not None:
                _write_rs4(both["e2e_pred"], os.path.join(output_dir, "e2e"), basename)
    metrics = evaluate_parseval(gold_trees, gold_preds)
    if seg_data is not None:
        metrics.update(evaluate_seg_and_e2e(gold_trees, seg_data))
    if output_dir is not None:
        console.print(f"[dim]Wrote {len(dev_pairs)} predictions under[/dim] [path]{os.path.abspath(output_dir)}[/path]")
    return metrics


def train(cfg: DMRSTConfig) -> None:
    """Full training loop with dynamic loss weighting (paper §3.2): the
    `split_loss`, `label_loss`, and (when joint segmentation is on)
    `seg_loss` are combined with weights that adapt to the recent
    per-component rate of decrease at every optimizer step.
    """
    set_seeds(cfg.seed)

    run_dir, cfg_hash = prepare_run_dir(
        dataclasses.asdict(cfg), cfg.checkpoint_dir, cfg.run_name, hash_exclude=HASH_EXCLUDE
    )

    if cfg.relation_map is not None:
        dim(f"Applying `relation_map` ({len(cfg.relation_map)} entries) to all read trees.")
    cfg.relation_types = infer_relation_types([cfg.train_dir, cfg.dev_dir], relation_map=cfg.relation_map)
    dim(
        f"Inferred {len(cfg.relation_types)} (relation, kind) pairs from "
        f"{cfg.train_dir} + {cfg.dev_dir}"
        + (" (after relation_map)." if cfg.relation_map is not None else ".")
        + " See Config panel below for the full list."
    )

    # Resolved cfg_dict (post-inference) is written to the audit config.json,
    # embedded in the .pt, and uploaded to the Hub.
    cfg_dict = dataclasses.asdict(cfg)
    write_run_config(run_dir, cfg_dict)
    tb = TBLogger(run_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = cfg.amp and device.type == "cuda"
    model = DMRSTParser(cfg, compile_encoder=True).to(device)
    train_trees = [
        t for _, t in read_rst_dir(cfg.train_dir, relation_types=cfg.relation_types, relation_map=cfg.relation_map)
    ]
    dev_pairs = read_rst_dir(cfg.dev_dir, relation_types=cfg.relation_types, relation_map=cfg.relation_map)
    test_pairs = (
        read_rst_dir(cfg.test_dir, relation_types=cfg.relation_types, relation_map=cfg.relation_map)
        if cfg.test_dir is not None
        else None
    )

    steps_per_epoch = max(1, len(train_trees) // cfg.grad_accum)
    total_steps = steps_per_epoch * cfg.max_epochs
    warmup = cfg.num_warmup_steps if cfg.num_warmup_steps > 0 else steps_per_epoch

    console.print(config_panel(cfg_dict))
    console.print(device_panel(device, seed=cfg.seed, checkpoint_dir=run_dir))
    console.print(model_panel(model, num_train_trees=len(train_trees), grad_accum=cfg.grad_accum))
    console.print(
        schedule_panel(
            steps_per_epoch=steps_per_epoch,
            total_steps=total_steps,
            warmup_steps=warmup,
            lr=cfg.lr,
            encoder_lr=cfg.encoder_lr,
        )
    )

    optimizer = build_optimizer(
        model,
        cfg.lr,
        cfg.weight_decay,
        submodule_lrs=[(model.encoder, cfg.encoder_lr)] if cfg.encoder_lr is not None else [],
    )
    scheduler = make_scheduler(optimizer, warmup, total_steps)

    # DLW (paper §3.2): weights at step t are `r_k = L_k(t-1) / L_k(t-2)`,
    # so we keep the last 3 per-component step losses.
    components = ["split", "label"] + (["seg"] if cfg.segmentation is not None else [])
    loss_history = {k: [] for k in components}
    curr_sums = {k: 0.0 for k in components}
    weights = {k: 1.0 for k in components}

    checkpoint = try_resume(os.path.join(run_dir, "last.pt"), expected_hash=cfg_hash)
    if checkpoint is None:
        global_step, start_epoch, best_val, stale = 0, 0, -1.0, 0
    else:
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        global_step = checkpoint["global_step"]
        start_epoch = checkpoint["epoch"]
        best_val = checkpoint.get("best_val", -1.0)
        stale = checkpoint.get("stale_validations", 0)
        loaded_history = checkpoint.get("dlw_loss_history")
        loaded_weights = checkpoint.get("dlw_weights")
        if loaded_history is not None:
            # Drop components that aren't active in this config. Init missing ones empty.
            loss_history = {k: list(loaded_history.get(k, [])) for k in components}
        if loaded_weights is not None and isinstance(loaded_weights, dict):
            weights = {k: float(loaded_weights.get(k, 1.0)) for k in components}
        if loaded_history is None and loaded_weights is None:
            dim("  DLW state missing in checkpoint; restarting from empty history")

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
            dlw_loss_history=loss_history,
            dlw_weights=dict(weights),
        )

    def _validate(epoch: int) -> None:
        nonlocal best_val, stale
        pred_dir = os.path.join(run_dir, "dev_predictions", f"epoch{epoch}_step{global_step}")
        model.eval()
        metrics = _evaluate_on_dev(model, dev_pairs, output_dir=pred_dir)
        tb.log_scalars("dev", metrics, global_step)
        console.print(metrics_table(metrics, title=f"Dev @ step {global_step}"))
        score = metrics[cfg.val_metric_name]
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

    recent_losses = deque(maxlen=200)
    rng = random.Random(cfg.seed)
    if not training_complete:
        rule("Training")
    training_start = time.monotonic()

    for epoch in range(start_epoch, cfg.max_epochs):
        if stale >= cfg.patience or aborted.value:
            break
        trees = list(train_trees)
        rng.shuffle(trees)
        epoch_start = time.monotonic()
        model.train()
        total_loss = 0.0
        num_trees = 0
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

            for tree_idx, tree in enumerate(trees):
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
                    out = model(tree)
                loss = sum(weights[k] * out[f"{k}_loss"] for k in components)
                if cfg.grad_accum > 1:
                    loss = loss / cfg.grad_accum
                loss.backward()
                for k in components:
                    curr_sums[k] += out[f"{k}_loss"].item()
                raw_loss = loss.item() * (cfg.grad_accum if cfg.grad_accum > 1 else 1)
                recent_losses.append(raw_loss)
                total_loss += raw_loss
                num_trees += 1

                is_step = (tree_idx + 1) % cfg.grad_accum == 0 or (tree_idx + 1) == len(trees)
                if not is_step:
                    continue

                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                epoch_step += 1

                # Dynamic loss weighting (paper §3.2, lagged ratio). Store this
                # step's component losses, then (once we have at least 3
                # stored) compute next step's weights from `r = list[-1] / list[-2]`.
                # Adaptation effectively begins at step 4.
                if cfg.dlw is not None:
                    window = cfg.dlw.window
                    for k in components:
                        loss_history[k].append(curr_sums[k])
                        if len(loss_history[k]) > window + 1:
                            loss_history[k] = loss_history[k][-(window + 1) :]
                    # Need `window + 1` stored before triggering, one warm-up
                    # step beyond the window itself, matching the original code's
                    # `> 2` guard when `window == 2`.
                    if len(loss_history[components[0]]) > window:
                        temperature = cfg.dlw.temperature
                        num_components = len(components)
                        # Compare mean of recent half vs mean of earlier half.
                        # For window=2: half=1, recent=history[-1], earlier=history[-2]
                        # which is exactly the original paper's L(t-1)/L(t-2) ratio.
                        # For window=20: half=10, ten-step means each side.
                        half = max(window // 2, 1)
                        recent = {k: sum(loss_history[k][-half:]) / half for k in components}
                        earlier = {k: sum(loss_history[k][-window:-half]) / max(window - half, 1) for k in components}
                        # Subtract max before exp for numerical stability: when one
                        # component's loss collapses to near-zero, its ratio can
                        # explode past `math.exp`'s domain (≈709). The constant
                        # cancels in the softmax normalization below.
                        exp_args = {k: (recent[k] / max(earlier[k], 1e-8)) / temperature for k in components}
                        max_arg = max(exp_args.values())
                        expw = {k: math.exp(exp_args[k] - max_arg) for k in components}
                        norm = sum(expw.values())
                        weights = {k: num_components * expw[k] / norm for k in components}
                curr_sums = {k: 0.0 for k in components}

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
                    if mem:
                        tb_train["gpu_mem_gb"] = mem[1]
                    if cfg.dlw is not None:
                        for k in components:
                            tb_train[f"loss_{k}"] = loss_history[k][-1]
                            tb_train[f"weight_{k}"] = weights[k]
                    tb.log_scalars("train", tb_train, global_step)
                    mem_log = f"  mem=[dim]{mem[1]:.1f}GB[/dim]" if mem else ""
                    w_log = (
                        "  w=[dim]" + "/".join(f"{weights[k]:.2f}" for k in components) + "[/dim]"
                        if cfg.dlw is not None
                        else ""
                    )
                    progress.console.print(
                        f"  [step]step {epoch_step}/{steps_per_epoch}[/step]  "
                        f"loss=[loss]{avg_loss:.4f}[/loss]  "
                        f"grad=[dim]{grad_norm:.4f}[/dim]  "
                        f"lr=[dim]{lr_display}[/dim]{w_log}{mem_log}"
                    )

                if cfg.validate_every and global_step % cfg.validate_every == 0:
                    _validate(epoch + 1)
                    if stale >= cfg.patience or aborted.value:
                        break

                if cfg.checkpoint_every and global_step % cfg.checkpoint_every == 0:
                    _save(os.path.join(run_dir, "last.pt"), epoch + 1)

        if num_trees > 0 and stale < cfg.patience:
            console.print(
                f"  [epoch]Epoch {epoch + 1}/{cfg.max_epochs}[/epoch] "
                f"[dim]({time.monotonic() - epoch_start:.1f}s)[/dim]  "
                f"loss=[loss]{total_loss / num_trees:.4f}[/loss]"
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
            output_dir=os.path.join(run_dir, "dev_predictions", "final"),
        )
        console.print(metrics_table(dev_m, title="Final Dev Results"))
        final_metrics: dict[str, dict[str, float]] = {"dev": dev_m}
        if test_pairs is not None:
            test_m = _evaluate_on_dev(
                model,
                test_pairs,
                output_dir=os.path.join(run_dir, "test_predictions", "final"),
            )
            console.print(metrics_table(test_m, title="Final Test Results"))
            final_metrics["test"] = test_m
            tb.log_scalars("test", test_m, global_step)
        # Sidecar for downstream tools (e.g. hub.py model card) so they don't
        # need to torch.load the checkpoint just to read corpus-level numbers.
        metrics_path = os.path.join(run_dir, "final_metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(final_metrics, f, indent=2)
        wrote(metrics_path)
    else:
        success(f"Training complete. Best {cfg.val_metric_name}: {best_val:.4f}")
    tb.close()


def main():
    parser = argparse.ArgumentParser(description="Train the dmrst parser")
    parser.add_argument("config", help="Path to a jsonnet config file")
    args = parser.parse_args()
    cfg = DMRSTConfig.from_dict(Params.from_file(args.config).as_dict(quiet=True))
    train(cfg)


if __name__ == "__main__":
    main()
