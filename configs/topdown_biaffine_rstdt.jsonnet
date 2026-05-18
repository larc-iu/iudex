// topdown_biaffine trained on RST-DT with the 18 coarse Carlson & Marcu classes.
{
    relation_map: import 'lib/rstdt_coarse_map.libsonnet',

    // Model
    model_name: "SpanBERT/spanbert-base-cased",
    ffn_hidden_size: 512,
    dropout: 0.2,
    stride: 100,

    // Data
    train_dir: "data/rstdt/train",
    dev_dir: "data/rstdt/dev",
    test_dir: "data/rstdt/test",

    // Training
    lr: 2e-4,
    encoder_lr: 1e-5,
    max_epochs: 30,
    grad_accum: 1,
    patience: 10,
    max_grad_norm: 1.0,
    weight_decay: 0.01,
    num_warmup_steps: 1000,
    log_every: 50,
    validate_every: null,
    checkpoint_every: null,
    checkpoint_dir: "checkpoints",
    run_name: null,
    seed: 42,
    val_metric_name: "span_f1",
}
