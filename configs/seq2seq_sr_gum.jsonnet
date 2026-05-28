// seq2seq_sr trained on GUM 12.1 (fine relations). Canonical config:
// every field on `Seq2SeqSRConfig` is set explicitly, even if at the
// dataclass default, so the file is self-documenting.
{
    // Data
    train_dir: 'data/gum_12.1.0_notok/train',
    dev_dir: 'data/gum_12.1.0_notok/dev',
    test_dir: 'data/gum_12.1.0_notok/test',
    relation_types: null,   // inferred at train time from train_dir+dev_dir
    relation_map: null,     // GUM uses its native fine relation set

    // Model
    model_name: 'google/t5gemma-2-1b-1b',
    // Memory-budget defaults for a 24 GB GPU. Bump for 40/80 GB cards.
    max_input_length: 3072,
    max_output_length: 5120,
    gradient_checkpointing: false,

    // LoRA on the seq2seq stack. The lm_head is replaced at parser init
    // with a small fresh head projecting to just the action vocab (~100
    // dims). Only `embed_tokens` needs `modules_to_save`, and even that
    // gets its pretrained rows frozen via `train_only_new_embedding_rows`
    // (only the ~100 newly-added action-token rows accumulate gradient).
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
    // With the small ~100-dim action head, copy and structural CE are
    // already same-scale, so w=1.0 (no rebalance) is the principled default.
    action_loss_weight: 1.0,
    label_smoothing: 0.1,

    // Dev eval
    dev_max_docs: null,    // null = full dev set every epoch
    dev_batch_size: 8,

    // Decoding
    num_beams: 4,
    use_validity_constraints: true,
    eval_decode_greedy: true,
    // Inference-only constraint: require >=N <copy> emissions before
    // <shift> becomes legal. 1 = no constraint. Bump to 2 or 3 to suppress
    // over-segmentation at decode time.
    min_edu_length: 1,
}
