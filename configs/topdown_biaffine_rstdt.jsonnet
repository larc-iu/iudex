// topdown_biaffine on RST-DT (coarse). Greedy top-down + biaffine, gold EDUs.
{
    // Data
    train_dir: 'data/rstdt/train',
    dev_dir: 'data/rstdt/dev',
    test_dir: 'data/rstdt/test',
    relation_types: null,                                  // inferred at train time
    relation_map: import 'lib/rstdt_coarse_map.libsonnet',

    // Model
    model_name: 'jhu-clsp/ettin-encoder-400m',
    ffn_hidden_size: 512,                                  // biaffine split + label head width
    dropout: 0.2,
    stride: 100,                                           // context tokens per side for long-doc striding
    peft: null,                                            // LoRA. null = full fine-tuning (else e.g. {r:16, alpha:32, dropout:0.05})

    // Training schedule, and the total run length (there is no separate max_epochs):
    // this trains on full documents for `epochs` epochs. To warm up on small subtrees
    // first, use e.g. { type: 'subtree_size', size_schedule: [8, 20, 60, null], phase_epochs: 5 }.
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
    validate_every: 1,                                     // run dev validation every N epochs (1 = every epoch, final always validates)
    edu_loss_weight_exponent: 0.0,                         // weight docs by #EDUs**exp, 0 = equal
    checkpoint_dir: 'checkpoints',
    run_name: null,
    seed: 42,
    val_metric_name: 'span_f1',
}
