// seq2seq_sr trained on GUM 12.1
{
    relation_map: null,

    // Model
    model_name: 'google/t5gemma-2-1b-1b',
    // Memory-budget defaults for a 24 GB GPU. See seq2seq_sr_rstdt.jsonnet
    // for the rationale.
    max_input_length: 3072,
    max_output_length: 5120,
    gradient_checkpointing: false,

    // Data
    train_dir: 'data/gum_12.1.0_notok/train',
    dev_dir: 'data/gum_12.1.0_notok/dev',
    test_dir: 'data/gum_12.1.0_notok/test',

    // Training
    lr: 3e-4,
    // Lowered from 10.0: with the small action-head replacing the 262K-vocab
    // lm_head, copy CE is no longer a dominant gradient sink. w=10 was
    // biasing the previous run toward over-eager shifts (~10% over-segmentation).
    action_loss_weight: 3.0,
    label_smoothing: 0.1,
    // Bumped from 0.01: with ~150 GUM train docs and a 1B base, more
    // anti-overfit pressure than the standard default.
    weight_decay: 0.05,
    max_epochs: 200,
    dev_batch_size: 8,
    batch_size: 1,
    grad_accum: 8,
    optimizer: 'adafactor',
    num_warmup_steps: null,
    max_grad_norm: 1.0,
    amp: true,
    patience: 10,
    log_every: 5,
    validate_every: 131,
    checkpoint_every: 131,
    checkpoint_dir: 'checkpoints',
    run_name: null,
    seed: 42,
    val_metric_name: 'e2e_full_f1',

    // Inference-only constraint: require >=N <copy> emissions before
    // <shift> becomes legal. Suppresses over-segmentation patterns
    // ("Education" + "and early loves" out of one gold EDU). At end of
    // source the constraint relaxes so the final EDU can still commit.
    min_edu_length: 1,

    // LoRA on the seq2seq stack. The lm_head is replaced at parser init
    // with a small fresh head projecting to just the action vocab (~100
    // dims), so the old 'out_proj' tied-weight story is gone — only the
    // input embedding needs `modules_to_save`, and even that gets its
    // pretrained rows frozen via `train_only_new_embedding_rows` (only
    // the ~100 newly-added action-token rows accumulate gradient).
    peft: {
        r: 4,
        alpha: 8,
        // Bumped from 0.05: heavier LoRA dropout for the regularized run.
        dropout: 0.10,
        target_modules: 'all-linear',
        bias: 'none',
        dora: false,
        modules_to_save: ['embed_tokens'],
        train_only_new_embedding_rows: true,
    },

    // Decoding
    num_beams: 4,
    use_validity_constraints: true,
    eval_decode_greedy: true,
}
