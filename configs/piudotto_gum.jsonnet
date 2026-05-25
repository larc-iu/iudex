// piudotto trained on GUM 12.1 (English RST + EDU segmentation).
{
    relation_map: null,

    // Encoder (fully fine-tuned with a smaller encoder_lr; no layer freezing)
    model_name: "jhu-clsp/ettin-encoder-150m",
    stride: 100,
    encoder_dropout: 0.2,

    // LoRA encoder fine-tuning (see _PeftConfig). Null = full fine-tuning.
    // Enable with e.g. peft: { r: 16, alpha: 32, dropout: 0.05 } (and bump encoder_lr).
    peft: null,

    // Span representation + biaffine scorers
    span_pooling: "attention",
    classifier_hidden_size: 256,
    classifier_dropout: 0.2,
    classifier_use_bias: true,
    label_input_pooling: "mean",

    // EDU-level Transformer over the pooled per-EDU vectors (random init), so EDUs
    // contextualize before span scoring. 0 disables. `edu_encoder_hidden_size` runs
    // it in a narrow bottleneck (down-project H->width, contextualize, up-project +
    // zero-init residual) to keep it low-capacity, since the full-width version
    // overfit. Narrower = more regularized; widen if it underfits.
    edu_encoder_layers: 2,
    edu_encoder_hidden_size: 384,
    edu_encoder_heads: 8,
    edu_encoder_dropout: 0.2,

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
    detokenizer: null,

    // Tree decoding: "greedy" by default. Switch to "cky" for the global optimum.
    decoding: "greedy",

    // Training objective: per-node CE by default. Set `margin_training: {margin: 1.0}`
    // for Stern et al. 2017 max-margin (CKY at train time, ~2x slower).
    margin_training: null,

    // Data
    train_dir: "data/gum_12.1.0_notok/train",
    dev_dir: "data/gum_12.1.0_notok/dev",
    test_dir: "data/gum_12.1.0_notok/test",

    // Training
    lr: 3e-4,
    encoder_lr: 2e-5,
    max_epochs: 100,
    grad_accum: 3,
    patience: 10,
    max_grad_norm: 10.0,
    weight_decay: 0.01,
    num_warmup_steps: 100,
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
