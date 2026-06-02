// decoder_only_sexp on RST-DT (coarse). Causal LM with source + a nested
// s-expression linearization of the tree in one stream.
{
    // Data
    train_dir: 'data/rstdt/train',
    dev_dir: 'data/rstdt/dev',
    test_dir: 'data/rstdt/test',
    relation_types: null,                                  // inferred at train time
    relation_map: import 'lib/rstdt_coarse_map.libsonnet',

    // Model
    model_name: 'google/gemma-3-1b-it',
    max_input_length: 3072,                                // source subwords (single-stream: source + tree share the budget)
    max_output_length: 5120,                               // tree stream
    gradient_checkpointing: false,                         // set true to trade compute for memory
    causal_mode: true,                                     // marks this as a decoder-only (causal LM) parser config

    // LoRA. null = full fine-tuning. Under use_copy (default) the lm_head is replaced
    // with a small action-vocab head and only the new embedding rows train, so
    // modules_to_save / train_only_new_embedding_rows are honored only when use_copy=false.
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
    validate_every: 1,                                     // run dev validation every N epochs (1 = every epoch, final always validates)
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
    min_edu_length: 1,                                     // require >=N content tokens before a leaf can close (1 = off)

    // Sexp-specific
    traversal_order: 'postorder',                          // or 'preorder'
    use_copy: true,                                        // true: <copy> sentinel + small head. false: verbatim source + full head (Hu & Wan 2023)
    constrain_content: true,                               // use_copy=false only: hard-mask content positions to the source cursor
}
