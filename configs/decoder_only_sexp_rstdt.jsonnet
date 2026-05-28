// decoder_only_sexp trained on RST-DT with the 18 coarse Carlson & Marcu classes.
// Nested s-expression sibling of decoder_only_sr. Default variant is post-order
// + copy. Canonical config: every field on `DecoderOnlySexpConfig` is set
// explicitly.
{
    // Data
    train_dir: 'data/rstdt/train',
    dev_dir: 'data/rstdt/dev',
    test_dir: 'data/rstdt/test',
    relation_types: null,
    relation_map: import 'lib/rstdt_coarse_map.libsonnet',

    // Model
    model_name: 'google/gemma-3-1b-it',
    causal_mode: true,
    max_input_length: 3072,
    max_output_length: 5120,
    gradient_checkpointing: false,

    // S-expression knobs.
    traversal_order: 'postorder',
    use_copy: true,
    // Only meaningful when use_copy=false. true = hard-mask content
    // positions to source_ids[cursor] (COPY-via-constraint). false =
    // free content generation (Hu and Wan 2023's apparent setup).
    constrain_content: true,

    peft: {
        r: 8,
        alpha: 16,
        dropout: 0.10,
        target_modules: 'all-linear',
        bias: 'none',
        dora: false,
        modules_to_save: ['embed_tokens'],
        train_only_new_embedding_rows: true,
    },

    // Training
    lr: 3e-4,
    weight_decay: 0.05,
    max_epochs: 200,
    batch_size: 1,
    grad_accum: 8,
    optimizer: 'adafactor',
    num_warmup_steps: null,
    max_grad_norm: 1.0,
    amp: true,
    patience: 5,
    log_every: 5,
    validate_every: 131,
    checkpoint_every: 131,
    checkpoint_dir: 'checkpoints',
    run_name: null,
    seed: 42,
    val_metric_name: 'e2e_full_f1',
    action_loss_weight: 1.0,
    // Under use_copy=false this is auto-scaled by ~ACTION_HEAD_SIZE /
    // ~FULL_VOCAB_SIZE (~100/262000) in __post_init__ so the per-off-class
    // smoothing mass stays comparable across the two head sizes. Set to 0.0
    // to opt out of the auto-scale (idempotent zero).
    label_smoothing: 0.1,

    // Dev eval
    dev_max_docs: null,
    dev_batch_size: 1,

    // Decoding
    num_beams: 4,
    use_validity_constraints: true,
    eval_decode_greedy: true,
    min_edu_length: 1,
}
