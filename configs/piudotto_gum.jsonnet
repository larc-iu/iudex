// piudotto trained on GUM 12.1 (English RST + EDU segmentation).
{
    relation_map: null,

    // Encoder (fully fine-tuned with a smaller encoder_lr; no layer freezing)
    model_name: "jhu-clsp/ettin-encoder-150m",
    stride: 100,
    encoder_dropout: 0.2,

    // Span representation + biaffine scorers
    span_pooling: "concat",
    classifier_hidden_size: 256,
    classifier_dropout: 0.2,
    classifier_use_bias: true,
    label_input_pooling: "mean",

    // Joint EDU segmentation. Linear-chain CRF over the BIE scheme by default;
    // set `loss: "ce"` for the lighter per-token CE + constrained Viterbi. Set
    // the whole block to `null` to train a gold-EDU-only parser.
    segmentation: {
        scheme: "BIE",
        loss: "crf",
        pos_weight: 10.0,  // "ce" loss only
        dropout: 0.2,
    },

    // Detokenize corpus EDU text to natural form so the segmenter trains on the
    // same kind of text `predict_from_text` sees. Only applied with segmentation.
    detokenizer: { type: "sacremoses", lang: "en" },

    // Tree decoding: greedy by default. Switch to "cky" for the global optimum.
    decoding: "greedy",

    // Training objective: per-node CE by default. Set `margin_training: {margin: 1.0}`
    // for Stern et al. 2017 max-margin (CKY at train time, ~2x slower).
    margin_training: null,

    // Data
    train_dir: "data/gum_12.1.0/train",
    dev_dir: "data/gum_12.1.0/dev",
    test_dir: "data/gum_12.1.0/test",

    // Training
    lr: 1e-4,
    encoder_lr: 2e-5,
    max_epochs: 30,
    grad_accum: 3,
    patience: 10,
    max_grad_norm: 5.0,
    weight_decay: 0.01,
    num_warmup_steps: 1000,
    log_every: 50,
    validate_every: null,
    checkpoint_every: null,
    checkpoint_dir: "checkpoints",
    run_name: null,
    seed: 42,
    val_metric_name: "e2e_full_f1",

    // EMA-based loss weighting (see `_EMAConfig`). Set to `null` for unweighted sum.
    ema: { momentum: 0.95, temperature: 2.0 },
}
