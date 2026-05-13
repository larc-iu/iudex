"""Training entrypoint for the dmrst parser.

Owns its own training loop top-to-bottom. Shared utilities are imported from
`iudex.rst.training`. Adds dynamic loss weighting (paper §3.2): the trainer
combines the model's split_loss and label_loss with weights that adapt to the
recent rate of decrease in each component, recomputed at every optimizer step.

Usage:
    python -m iudex.rst.parsers.dmrst.train_dmrst configs/dmrst.jsonnet
"""
import argparse
import dataclasses
import logging
import math
import os
import random
import time
from collections import deque

import torch
from tonga import Params

from iudex.common.log import console, dim, rule, setup_logging, success, warn
from iudex.rst.data.reader import infer_relation_types, read_rst_dir
from iudex.rst.parsers.dmrst.configuration_dmrst import DMRSTConfig
from iudex.rst.parsers.dmrst.evaluation import (
    dmrst_metrics_table,
    evaluate_dmrst,
    legal_val_metric_names,
)
from iudex.rst.parsers.dmrst.modeling_dmrst import DMRSTParser
from iudex.rst.training import (
    build_optimizer,
    config_panel,
    device_panel,
    gpu_mem_gb,
    make_progress_bar,
    make_scheduler,
    model_panel,
    prepare_run_dir,
    save_checkpoint,
    schedule_panel,
    set_seeds,
    try_resume,
)

setup_logging()
logger = logging.getLogger(__name__)


def train(cfg: DMRSTConfig) -> None:
    """Train. `cfg` is the single source of truth; we serialize it via
    `dataclasses.asdict` for hashing, checkpoint storage, and `config.json` audit.
    Multiple runs with different configs coexist under `cfg.checkpoint_dir/`.
    """
    set_seeds(cfg.seed)
    if cfg.relation_map is not None:
        dim(f"Applying `relation_map` ({len(cfg.relation_map)} entries) to all read trees.")
    if cfg.relation_types is None:
        cfg.relation_types = infer_relation_types(
            [cfg.train_dir, cfg.dev_dir], relation_map=cfg.relation_map
        )
        dim(
            f"Inferred {len(cfg.relation_types)} (relation, kind) pairs from "
            f"{cfg.train_dir} + {cfg.dev_dir}"
            + (" (after relation_map)." if cfg.relation_map is not None else ".")
            + " See Config panel below for the full list."
        )
    else:
        dim(f"Using explicit `relation_types` from config ({len(cfg.relation_types)} pairs).")

    legal_metrics = legal_val_metric_names(cfg.joint_segmentation)
    if cfg.val_metric_name not in legal_metrics:
        raise ValueError(
            f"val_metric_name={cfg.val_metric_name!r} is not produced by evaluate_dmrst "
            f"with joint_segmentation={cfg.joint_segmentation}. "
            f"Legal keys: {legal_metrics}"
        )
    if cfg.joint_segmentation and cfg.val_metric_name in {"span_f1", "nuc_f1", "rel_f1", "full_f1"}:
        warn(
            f"val_metric_name={cfg.val_metric_name!r} is a gold-EDU metric. "
            f"With joint_segmentation=True, consider `e2e_full_f1` (end-to-end parse) "
            f"or `seg_f1` (segmentation-only) for early stopping."
        )

    cfg_dict = dataclasses.asdict(cfg)
    run_dir, cfg_hash = prepare_run_dir(cfg_dict, cfg.checkpoint_dir, cfg.run_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DMRSTParser(cfg).to(device)
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
    console.print(schedule_panel(
        steps_per_epoch=steps_per_epoch, total_steps=total_steps, warmup_steps=warmup,
        lr=cfg.lr, encoder_lr=cfg.encoder_lr,
    ))

    optimizer = build_optimizer(
        model, cfg.lr, cfg.weight_decay,
        submodule_lrs=[(model.encoder, cfg.encoder_lr)] if cfg.encoder_lr is not None else [],
    )
    scheduler = make_scheduler(optimizer, warmup, total_steps)

    # DLW state — `loss_history` keeps the per-component losses from recent
    # optimizer steps (only the last 3 are needed; older entries are dropped).
    # `weights` are the coefficients applied to the current step's forward;
    # they are recomputed at the end of each step from the ratio of the two
    # most-recent stored step-losses (paper §3.2 / upstream `Training.py`):
    # `r_k = L_k(t-1) / L_k(t-2)` so weights at step t use steps t-1 and t-2.
    # Components are `split` + `label`, plus `seg` when joint segmentation is on.
    components = ["split", "label"] + (["seg"] if cfg.joint_segmentation else [])
    loss_history = {k: [] for k in components}
    curr_sums = {k: 0.0 for k in components}
    weights = {k: 1.0 for k in components}

    ckpt = try_resume(os.path.join(run_dir, "last.pt"), expected_hash=cfg_hash)
    if ckpt is None:
        global_step, start_epoch, best_val, stale = 0, 0, -1.0, 0
    else:
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        global_step = ckpt["global_step"]
        start_epoch = ckpt["epoch"]
        best_val = ckpt.get("best_val", -1.0)
        stale = ckpt.get("stale_validations", 0)
        loaded_history = ckpt.get("dlw_loss_history")
        loaded_weights = ckpt.get("dlw_weights")
        if loaded_history is not None:
            # Drop components that aren't active in this config; init missing ones empty.
            loss_history = {k: list(loaded_history.get(k, [])) for k in components}
        if loaded_weights is not None and isinstance(loaded_weights, dict):
            weights = {k: float(loaded_weights.get(k, 1.0)) for k in components}
        if loaded_history is None and loaded_weights is None:
            dim("  DLW state missing in checkpoint; restarting from empty history")

    def _save(path: str, epoch: int) -> None:
        save_checkpoint(
            path, model, optimizer, scheduler,
            config=cfg_dict, config_hash=cfg_hash,
            global_step=global_step, epoch=epoch, best_val=best_val, stale_validations=stale,
            dlw_loss_history=loss_history, dlw_weights=dict(weights),
        )

    def _validate(epoch: int) -> None:
        nonlocal best_val, stale
        pred_dir = os.path.join(run_dir, "dev_predictions", f"epoch{epoch}_step{global_step}")
        model.eval()
        m = evaluate_dmrst(model, dev_pairs, output_dir=pred_dir)
        console.print(dmrst_metrics_table(m, title=f"Dev @ step {global_step}"))
        score = m[cfg.val_metric_name]
        if score > best_val:
            best_val = score
            stale = 0
            _save(os.path.join(run_dir, "best_model.pt"), epoch)
            success(f"  New best! {cfg.val_metric_name}={best_val:.4f}")
        else:
            stale += 1
            dim(f"  No improvement ({stale}/{cfg.patience})")
        model.train()

    recent_losses = deque(maxlen=200)
    rng = random.Random(cfg.seed)
    rule("Training")
    training_start = time.monotonic()

    for epoch in range(start_epoch, cfg.max_epochs):
        trees = list(train_trees)
        rng.shuffle(trees)
        epoch_start = time.monotonic()
        model.train()
        total_loss = 0.0
        num_trees = 0
        epoch_step = 0

        with make_progress_bar() as progress:
            task = progress.add_task(
                "training", total=steps_per_epoch,
                epoch=f"{epoch+1}/{cfg.max_epochs}",
                loss_str="loss=-.----", lr_str="", mem_str="", total_elapsed="0:00:00",
            )

            for tree_idx, tree in enumerate(trees):
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

                # Dynamic loss weighting (lagged ratio per paper §3.2 / upstream).
                # Store this step's component losses, then — once we have at least 3
                # stored — compute next step's weights from `r = list[-1] / list[-2]`,
                # i.e. the ratio of the two most-recent step losses. Effectively no
                # adaptation occurs before step 4, matching the upstream `> 2` guard.
                if cfg.dlw_enabled:
                    for k in components:
                        loss_history[k].append(curr_sums[k])
                        if len(loss_history[k]) > 3:
                            loss_history[k] = loss_history[k][-3:]
                    if len(loss_history[components[0]]) > 2:
                        T = cfg.dlw_temperature
                        K = len(components)
                        expw = {
                            k: math.exp(
                                (loss_history[k][-1] / max(loss_history[k][-2], 1e-8)) / T
                            )
                            for k in components
                        }
                        Z = sum(expw.values())
                        weights = {k: K * expw[k] / Z for k in components}
                curr_sums = {k: 0.0 for k in components}

                avg = sum(recent_losses) / len(recent_losses)
                lr_str_inner = "/".join(f"{lr:.1e}" for lr in sorted(set(scheduler.get_last_lr())))
                mem = gpu_mem_gb(device)
                mem_str = f"[gpu]max_mem={mem[1]:.1f}GB[/gpu]" if mem else ""
                secs = int(time.monotonic() - training_start)
                progress.update(
                    task, advance=1,
                    loss_str=f"loss=[bold orange1]{avg:.4f}[/bold orange1]",
                    lr_str=f"lr=[dim]{lr_str_inner}[/dim]",
                    mem_str=mem_str,
                    total_elapsed=f"{secs//3600}:{(secs%3600)//60:02d}:{secs%60:02d}",
                )

                if epoch_step % cfg.log_every == 0:
                    mem_log = f"  mem=[dim]{mem[0]:.1f}/{mem[1]:.1f}GB[/dim]" if mem else ""
                    w_log = (
                        "  w=[dim]" + "/".join(f"{weights[k]:.2f}" for k in components) + "[/dim]"
                        if cfg.dlw_enabled else ""
                    )
                    progress.console.print(
                        f"  [step]step {epoch_step}/{steps_per_epoch}[/step]  "
                        f"loss=[loss]{avg:.4f}[/loss]  "
                        f"grad=[dim]{grad_norm:.4f}[/dim]  "
                        f"lr=[dim]{lr_str_inner}[/dim]{w_log}{mem_log}"
                    )

                if cfg.validate_every and global_step % cfg.validate_every == 0:
                    _validate(epoch + 1)
                    if stale >= cfg.patience:
                        break

                if cfg.checkpoint_every and global_step % cfg.checkpoint_every == 0:
                    _save(os.path.join(run_dir, "last.pt"), epoch + 1)

        if num_trees > 0 and stale < cfg.patience:
            console.print(
                f"  [epoch]Epoch {epoch+1}/{cfg.max_epochs}[/epoch] "
                f"[dim]({time.monotonic() - epoch_start:.1f}s)[/dim]  "
                f"loss=[loss]{total_loss / num_trees:.4f}[/loss]"
            )
            if not cfg.validate_every:
                _validate(epoch + 1)
            _save(os.path.join(run_dir, "last.pt"), epoch + 1)
            if stale >= cfg.patience:
                warn(f"\nEarly stopping after {cfg.patience} validations without improvement")
                break
        elif stale >= cfg.patience:
            warn(f"\nEarly stopping at step {global_step}")
            break

    rule("Final Evaluation")
    best_path = os.path.join(run_dir, "best_model.pt")
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        dev_m = evaluate_dmrst(
            model, dev_pairs,
            output_dir=os.path.join(run_dir, "dev_predictions", "final"),
        )
        console.print(dmrst_metrics_table(dev_m, title="Final Dev Results"))
        if test_pairs is not None:
            test_m = evaluate_dmrst(
                model, test_pairs,
                output_dir=os.path.join(run_dir, "test_predictions", "final"),
            )
            console.print(dmrst_metrics_table(test_m, title="Final Test Results"))
    else:
        success(f"Training complete. Best {cfg.val_metric_name}: {best_val:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train the dmrst parser")
    parser.add_argument("config", help="Path to a jsonnet config file")
    args = parser.parse_args()
    cfg = DMRSTConfig.from_dict(Params.from_file(args.config).as_dict(quiet=True))
    train(cfg)


if __name__ == "__main__":
    main()
