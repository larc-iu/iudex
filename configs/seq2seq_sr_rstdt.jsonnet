// seq2seq_sr on RST-DT (coarse). Encoder-decoder LM generates a bottom-up
// shift-reduce action sequence (with a <copy> sentinel for source subwords).
{
    // Data
    train_dir: 'data/rstdt/train',
    dev_dir: 'data/rstdt/dev',
    test_dir: 'data/rstdt/test',
    relation_types: null,                                  // inferred at train time
    relation_map: import 'lib/rstdt_coarse_map.libsonnet',

    // Model
    model_name: 'google/t5gemma-2-1b-1b',
    max_input_length: 3072,                                // encoder subwords (no specials)
    max_output_length: 5120,                               // decoder target (source copies + shift/reduce actions)
    gradient_checkpointing: false,                         // set true to trade compute for memory

    // LoRA. null = full fine-tuning. The lm_head is replaced with a small action-vocab
    // head and only the new action-token embedding rows train, so modules_to_save /
    // train_only_new_embedding_rows are no-ops here (kept for the strict parse).
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

    // Curriculum (Registrable). SimpleCurriculum = cold full-document training and
    // owns the epoch budget. SubtreeSizeCurriculum warms up on small subtrees first.
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
