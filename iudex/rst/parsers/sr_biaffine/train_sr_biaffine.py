import argparse
import dataclasses
import json
import logging
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
    edu_count_loss_weights,
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
from iudex.rst.data.metrics import evaluate_parseval, metrics_table
from iudex.rst.data.reader import infer_relation_types, read_rst_dir
from iudex.rst.data.tree import RstTree
from iudex.rst.parsers.sr_biaffine.configuration_sr_biaffine import SRBiaffineConfig
from iudex.rst.parsers.sr_biaffine.modeling_sr_biaffine import SRBiaffineParser

setup_logging()
logger = logging.getLogger(__name__)


def _write_rs4(tree: RstTree, output_dir: str, basename: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, basename), "w", encoding="utf-8") as f:
        f.write(tree.to_rs4_string())


@torch.no_grad()
def _evaluate_on_dev(
    model: SRBiaffineParser,
    dev_pairs: list[tuple[str, RstTree]],
    output_dir: str | None = None,
) -> dict[str, float]:
    """Run the model over `dev_pairs` and aggregate gold-EDU Parseval.
    No segmentation, this parser assumes gold EDUs."""
    gold_trees: list[RstTree] = []
    gold_preds: list[RstTree] = []
    for filepath, gold in dev_pairs:
        gold_trees.append(gold)
        pred = model.predict(gold)
        gold_preds.append(pred)
        if output_dir is not None:
            basename = os.path.splitext(os.path.basename(filepath))[0] + ".rs4"
            _write_rs4(pred, output_dir, basename)
    return evaluate_parseval(gold_trees, gold_preds)


def train(cfg: SRBiaffineConfig) -> None:
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
    model = SRBiaffineParser(cfg, compile_encoder=True).to(device)
    train_trees = [
        t for _, t in read_rst_dir(cfg.train_dir, relation_types=cfg.relation_types, relation_map=cfg.relation_map)
    ]
    dev_pairs = read_rst_dir(cfg.dev_dir, relation_types=cfg.relation_types, relation_map=cfg.relation_map)
    test_pairs = (
        read_rst_dir(cfg.test_dir, relation_types=cfg.relation_types, relation_map=cfg.relation_map)
        if cfg.test_dir is not None
        else None
    )

    phases = cfg.curriculum.plan()
    total_epochs = sum(p.epochs for p in phases)

    # Per-phase tree lists + EDU-count weight tables + step counts. total_steps
    # spans all phases so the LR schedule does not decay mid-curriculum. A
    # SimpleCurriculum yields a single full-document phase == prior behavior.
    phase_specs: list[tuple] = []  # (phase, trees, weight_table, phase_steps_per_epoch)
    total_steps = 0
    for phase in phases:
        phase_trees = cfg.curriculum.train_trees(train_trees, phase)
        wtab = (
            edu_count_loss_weights([len(t.edus) for t in phase_trees], exponent=cfg.edu_loss_weight_exponent)
            if cfg.edu_loss_weight_exponent
            else None
        )
        spe = max(1, len(phase_trees) // cfg.grad_accum)
        phase_specs.append((phase, phase_trees, wtab, spe))
        total_steps += spe * phase.epochs
    warmup = phase_specs[0][3] if cfg.num_warmup_steps is None else cfg.num_warmup_steps

    # Flatten phases to a per-absolute-epoch spec so the single epoch loop (and
    # resume by absolute epoch) stays unchanged.
    epoch_to_spec: list[tuple] = []
    for phase, phase_trees, wtab, spe in phase_specs:
        for _ in range(phase.epochs):
            epoch_to_spec.append((phase, phase_trees, wtab, spe))

    if len(phases) > 1:
        dim(
            "Curriculum phases (cap/epochs/trees): "
            + ", ".join(f"{p.cap if p.cap is not None else 'full'}/{p.epochs}/{len(tr)}" for p, tr, _, _ in phase_specs)
        )

    console.print(config_panel(cfg_dict))
    console.print(device_panel(device, seed=cfg.seed, checkpoint_dir=run_dir))
    console.print(model_panel(model, num_train_trees=len(train_trees), grad_accum=cfg.grad_accum))
    console.print(
        schedule_panel(
            steps_per_epoch=phase_specs[0][3],
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
    console.print(weight_decay_panel(model, optimizer))
    scheduler = make_scheduler(optimizer, warmup, total_steps)

    state = resume_or_init(run_dir, model=model, optimizer=optimizer, scheduler=scheduler, expected_hash=cfg_hash)
    global_step = state["global_step"]
    start_epoch = state["epoch"]
    best_val = state["best_val"]
    stale = state["stale_validations"]

    def _save(path: str, epoch: int) -> None:
        save_checkpoint(
            path,
            model,
            optimizer,
            scheduler,
            config=cfg_dict,
            config_hash=cfg_hash,
            parser_kind="sr_biaffine",
            global_step=global_step,
            epoch=epoch,
            best_val=best_val,
            stale_validations=stale,
        )

    def _validate(epoch: int, epoch_in_phase: int, dev_set: list) -> None:
        nonlocal best_val, stale
        # Empty dev_set => the curriculum suppresses validation for this phase
        # (e.g. subtree warmup phases). begin_validation_epoch counts epochs
        # WITHIN the phase, so it skips the first N slow early evals of a
        # validating phase even when the curriculum places that phase late in
        # the global epoch sequence (a global-epoch gate would be inert there).
        if epoch_in_phase < cfg.begin_validation_epoch or not dev_set:
            return
        # Cadence gate (validate_every); the final epoch always validates.
        if epoch % cfg.validate_every != 0 and epoch != total_epochs:
            return
        pred_dir = os.path.join(run_dir, "dev_predictions", f"epoch{epoch}_step{global_step}")
        model.eval()
        metrics = _evaluate_on_dev(model, dev_set, output_dir=pred_dir)
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
    training_complete = start_epoch >= total_epochs or stale >= cfg.patience
    if training_complete:
        reason = "all epochs completed" if start_epoch >= total_epochs else "patience exhausted"
        dim(f"Skipping training: {reason} on prior run; jumping to final evaluation.")

    recent_losses = deque(maxlen=200)
    rng = random.Random(cfg.seed)
    if not training_complete:
        rule("Training")
    training_start = time.monotonic()

    prev_phase = None
    for epoch in range(start_epoch, total_epochs):
        if stale >= cfg.patience or aborted.value:
            break
        phase, phase_trees, wtab, spe = epoch_to_spec[epoch]
        dev_set = cfg.curriculum.dev_pairs(dev_pairs, phase)
        if phase is not prev_phase:
            phase_idx = next(i for i, p in enumerate(phases) if p is phase)
            cap_desc = "full documents" if phase.cap is None else f"subtrees <= {phase.cap} EDUs"
            rule(
                f"Curriculum phase {phase_idx + 1}/{len(phases)}: {cap_desc} | "
                f"{len(phase_trees)} trees | {'validating' if dev_set else 'no dev (warmup)'}"
            )
            # First epoch of a validating phase (not a mid-phase resume): reset
            # best/patience so prior phases cannot block saves or trip early-stop.
            if dev_set and (epoch == 0 or epoch_to_spec[epoch - 1][0] is not phase):
                best_val, stale = -1.0, 0
            prev_phase = phase
        trees = list(phase_trees)
        rng.shuffle(trees)
        epoch_start = time.monotonic()
        model.train()
        total_loss = 0.0
        num_trees = 0
        epoch_step = 0

        with make_progress_bar() as progress:
            task = progress.add_task(
                "training",
                total=spe,
                epoch=f"{epoch + 1}/{total_epochs}",
                loss_str="loss=-.----",
                lr_str="",
                mem_str="",
                total_elapsed="0:00:00",
            )

            for tree_idx, tree in enumerate(trees):
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
                    loss = model(tree)["loss"]
                if wtab is not None:
                    loss = loss * wtab.get(len(tree.edus), 1.0)
                if cfg.grad_accum > 1:
                    loss = loss / cfg.grad_accum
                loss.backward()
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
                    tb.log_scalars("train", tb_train, global_step)
                    mem_log = f"  mem=[dim]{mem[1]:.1f}GB[/dim]" if mem else ""
                    progress.console.print(
                        f"  [step]step {epoch_step}/{spe}[/step]  "
                        f"loss=[loss]{avg_loss:.4f}[/loss]  "
                        f"grad=[dim]{grad_norm:.4f}[/dim]  "
                        f"lr=[dim]{lr_display}[/dim]{mem_log}"
                    )

        if num_trees > 0:
            console.print(
                f"  [epoch]Epoch {epoch + 1}/{total_epochs}[/epoch] "
                f"[dim]({time.monotonic() - epoch_start:.1f}s)[/dim]  "
                f"loss=[loss]{total_loss / num_trees:.4f}[/loss]"
            )
            # In-phase epoch: offset from the phase's first global epoch
            # (phase objects are shared by identity across epoch_to_spec).
            phase_first = next(i for i, s in enumerate(epoch_to_spec) if s[0] is phase)
            _validate(epoch + 1, epoch + 1 - phase_first, dev_set)
            _save(os.path.join(run_dir, "last.pt"), epoch + 1)
            if stale >= cfg.patience or aborted.value:
                warn(f"\nEarly stopping after {cfg.patience} validations without improvement")
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
    parser = argparse.ArgumentParser(description="Train the sr_biaffine parser")
    parser.add_argument("config", help="Path to a jsonnet config file")
    args = parser.parse_args()
    cfg = SRBiaffineConfig.from_dict(Params.from_file(args.config).as_dict(quiet=True))
    train(cfg)


if __name__ == "__main__":
    main()
