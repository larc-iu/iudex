// topdown_biaffine trained on GUM 12.1 (English RST). Gold EDU segmentation —
// this parser does not train a segmenter.
{
    relation_map: null,

    // Model
    model_name: "jhu-clsp/ettin-encoder-150m",
    ffn_hidden_size: 512,
    dropout: 0.2,
    stride: 100,

    // Data
    train_dir: "data/gum_12.1.0/train",
    dev_dir: "data/gum_12.1.0/dev",
    test_dir: "data/gum_12.1.0/test",

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
