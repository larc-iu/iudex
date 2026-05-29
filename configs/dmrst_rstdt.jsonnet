// DMRST trained on RST-DT with the 18 coarse Carlson & Marcu classes.
{
    relation_map: import 'lib/rstdt_coarse_map.libsonnet',

    // Model
    model_name: "xlm-roberta-base",
    stride: 100,
    // DMRST's original fixed 300-token sliding window (reference EncoderRNN):
    // 300 content tokens per window with `stride` context discarded per side.
    encoder_window_size: 300,
    attention_type: "dot_product",
    classifier_use_bias: true,
    num_rnn_layers: 1,
    encoder_dropout: 0.5,
    decoder_dropout: 0.5,
    labeler_dropout: 0.5,
    doc_gru_dropout: 0.2,
    label_input_pooling: "mean",
    freeze_encoder_layers: 3,
    freeze_embeddings: true,

    // LoRA encoder fine-tuning (see _PeftConfig). Null = full fine-tuning. To enable,
    // also set freeze_encoder_layers: 0 and freeze_embeddings: false (peft rejects
    // combining with them), then e.g. peft: { r: 16, alpha: 32, dropout: 0.05 }.
    peft: null,

    // Joint EDU segmentation. Set to `null` to disable (lose `predict_from_text`).
    segmentation: {
        pos_weight: 10.0,
        start_loss: true,
    },

    // Detokenize corpus EDU text to natural form so the segmenter trains on the
    // same kind of text `predict_from_text` sees. Only applied with segmentation.
    detokenizer: { type: "sacremoses", lang: "en" },

    // Data
    train_dir: "data/rstdt/train",
    dev_dir: "data/rstdt/dev",
    test_dir: "data/rstdt/test",

    // Training
    lr: 1e-4,
    encoder_lr: 2e-5,
    max_epochs: 15,
    grad_accum: 3,
    patience: 10,
    max_grad_norm: 5.0,
    weight_decay: 0.01,
    num_warmup_steps: 1000,
    log_every: 1,
    validate_every: null,
    checkpoint_every: null,
    checkpoint_dir: "checkpoints",
    run_name: null,
    seed: 42,
    val_metric_name: "e2e_full_f1",

    // Dynamic loss weighting (paper §3.2). `window: 2` is the paper's lagged
    // L(t-1)/L(t-2) ratio. Larger windows smooth the ratio at the cost of
    // slower adaptation. Set to `null` for an unweighted sum.
    dlw: { temperature: 2.0, window: 2 },
}
