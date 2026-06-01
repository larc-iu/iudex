// decoder_only_sr on GUM 12.1 (fine relations). Causal LM with source + shift-reduce
// actions in one stream (with a <copy> sentinel for source subwords).
{
    // Data
    train_dir: 'data/gum_12.1.0_notok/train',
    dev_dir: 'data/gum_12.1.0_notok/dev',
    test_dir: 'data/gum_12.1.0_notok/test',
    relation_types: null,                                  // inferred at train time
    relation_map: null,                                    // GUM uses its native fine set

    // Model
    model_name: 'google/gemma-3-1b-it',
    max_input_length: 3072,                                // source subwords (single-stream: source + actions share the budget)
    max_output_length: 5120,                               // action stream
    gradient_checkpointing: false,                         // set true to trade compute for memory
    causal_mode: true,                                     // marks this as a decoder-only (causal LM) parser config

    // LoRA. null = full fine-tuning. The lm_head is replaced with a small action-vocab
    // head and only the new action-token embedding rows train, so modules_to_save /
    // train_only_new_embedding_rows have no effect here (they belong to the LoRA
    // config shared across parsers, which is a superset of every parser's knobs).
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

    // Training schedule, and the total run length (there is no separate max_epochs):
    // this trains on full documents for `epochs` epochs. To warm up on small subtrees
    // first, use e.g. { type: 'subtree_size', size_schedule: [8, 20, 60, null], phase_epochs: 5 }.
    curriculum: { epochs: 200 },

    // Training
    lr: 3e-4,
    weight_decay: 0.05,
    batch_size: 1,
    grad_accum: 8,
    optimizer: 'adafactor',                                // or 'adamw' (more memory)
    num_warmup_steps: null,                                // null = 1-epoch warmup, 0 = none
    max_grad_norm: 1.0,
    amp: true,                                             // bf16 autocast (CUDA), inference stays fp32
    patience: 5,
    log_every: 5,
    begin_validation_epoch: 0,                             // skip slow dev decodes until this epoch
    checkpoint_dir: 'checkpoints',
    run_name: null,
    seed: 42,
    val_metric_name: 'e2e_full_f1',

    // Loss
    action_loss_weight: 1.0,                               // gradient multiplier on structural actions, 1.0 = no rebalance
    edu_loss_weight_exponent: 0.0,                         // weight docs by #EDUs**exp, 0 = equal
    label_smoothing: 0.1,

    // Dev eval
    dev_max_docs: null,                                    // null = full dev set each epoch (final eval always full)
    dev_batch_size: 1,                                     // bump to batch dev decodes

    // Decoding
    num_beams: 4,
    use_validity_constraints: true,
    eval_decode_greedy: true,
    min_edu_length: 1,                                     // require >=N <copy> before <shift> is legal (1 = off)
}
