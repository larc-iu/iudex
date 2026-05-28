// decoder_only_sr trained on RST-DT with the 18 coarse Carlson & Marcu classes.
// Single-stream sibling of seq2seq_sr. Canonical config: every field on
// `DecoderOnlySRConfig` is set explicitly.
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

    // LoRA on the causal LM. lm_head replaced with a small action-vocab
    // head; only new action-token rows of embed_tokens update.
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
