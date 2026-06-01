// sr_biaffine on GUM 12.1 (fine relations). Transition-based shift-reduce + biaffine, gold EDUs.
{
    // Data
    train_dir: 'data/gum_12.1.0/train',
    dev_dir: 'data/gum_12.1.0/dev',
    test_dir: 'data/gum_12.1.0/test',
    relation_types: null,                                  // inferred at train time
    relation_map: null,                                    // GUM uses its native fine set

    // Model
    model_name: 'jhu-clsp/ettin-encoder-400m',
    ffn_hidden_size: 512,                                  // biaffine reduce-label head width
    action_ffn_hidden_size: 512,                           // shift/reduce action head width
    dropout: 0.2,
    stride: 100,                                           // context tokens per side for long-doc striding
    peft: null,                                            // LoRA. null = full fine-tuning (else e.g. {r:16, alpha:32, dropout:0.05})

    // Curriculum (Registrable). SimpleCurriculum = cold full-document training and
    // owns the epoch budget. SubtreeSizeCurriculum warms up on small subtrees first.
    curriculum: { epochs: 50 },

    // Training
    lr: 2e-4,
    encoder_lr: 1e-5,                                      // null = use `lr` for the encoder too
    grad_accum: 1,
    amp: true,                                             // bf16 autocast (CUDA), inference stays fp32
    patience: 10,
    max_grad_norm: 1.0,
    weight_decay: 0.01,
    num_warmup_steps: 1000,                                // null = 1-epoch warmup, 0 = none
    log_every: 1,
    begin_validation_epoch: 0,                             // skip dev eval until this epoch
    edu_loss_weight_exponent: 0.0,                         // weight docs by #EDUs**exp, 0 = equal
    checkpoint_dir: 'checkpoints',
    run_name: null,
    seed: 42,
    val_metric_name: 'span_f1',
}
