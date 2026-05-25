// DMRST trained on GUM 12.1 (English RST + EDU segmentation).
{
    relation_map: null,

    // Model
    model_name: "jhu-clsp/ettin-encoder-150m",
    stride: 100,
    attention_type: "dot_product",
    classifier_use_bias: true,
    num_rnn_layers: 1,
    encoder_dropout: 0.5,
    decoder_dropout: 0.5,
    labeler_dropout: 0.5,
    doc_gru_dropout: 0.2,
    label_input_pooling: "mean",
    // original DMRST: freeze embeddings and first 3 encoder layers
    freeze_encoder_layers: 0,
    freeze_embeddings: false,

    // LoRA encoder fine-tuning (see _PeftConfig). Null = full fine-tuning. To enable,
    // keep freeze_encoder_layers: 0 / freeze_embeddings: false (peft rejects combining
    // with them), then e.g. peft: { r: 16, alpha: 32, dropout: 0.05 } (and bump encoder_lr).
    // peft: {
    //     r: 16,
    //     alpha: 32,
    //     dropout: 0.05
    // },
    peft: null,

    // Joint EDU segmentation. Set to `null` to disable.
    segmentation: {
        scheme: 'BIE',
        loss: 'crf',
        dropout: 0.3,
    },
    // original DMRST: no CRF
    // segmentation: {
    //     pos_weight: 10.0,
    //     start_loss: false,
    // },

    // Detokenize corpus EDU text to natural form so the segmenter trains on the
    // same kind of text `predict_from_text` sees. Only applied with segmentation.
    detokenizer: null,

    // Data
    train_dir: "data/gum_12.1.0_notok/train",
    dev_dir: "data/gum_12.1.0_notok/dev",
    test_dir: "data/gum_12.1.0_notok/test",

    // Training
    lr: 2e-4,
    encoder_lr: 2e-5,
    max_epochs: 100,
    grad_accum: 3,
    patience: 10,
    max_grad_norm: 20.0,
    weight_decay: 0.01,
    num_warmup_steps: 100,
    log_every: 1,
    validate_every: null,
    checkpoint_every: null,
    checkpoint_dir: "checkpoints",
    run_name: null,
    seed: 42,
    val_metric_name: "e2e_full_f1",

    // Dynamic loss weighting. `window: 2` reproduces the paper's lagged
    // L(t-1)/L(t-2) ratio; larger windows give smoother ratios at the cost
    // of slower adaptation (recommended for noisy whole-tree training).
    // Set to `null` for unweighted sum.
    dlw: { temperature: 2.0, window: 30 },
}
