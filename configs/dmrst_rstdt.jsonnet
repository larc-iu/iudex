// DMRST trained on RST-DT with the 18 coarse Carlson & Marcu classes.
{
    relation_map: import 'lib/rstdt_coarse_map.libsonnet',

    // Model
    model_name: "xlm-roberta-base",
    stride: 100,
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

    // Joint EDU segmentation. Set to `null` to disable (lose `predict_from_text`).
    segmentation: {
        pos_weight: 10.0,
        start_loss: false,
    },

    // Data
    train_dir: "data/rstdt/train",
    dev_dir: "data/rstdt/dev",
    test_dir: "data/rstdt/test",

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

    // Dynamic loss weighting. `window: 2` reproduces the paper's lagged
    // L(t-1)/L(t-2) ratio; larger windows give smoother ratios at the cost
    // of slower adaptation (recommended for noisy whole-tree training).
    // Set to `null` for unweighted sum.
    dlw: { temperature: 2.0, window: 2 },
}
