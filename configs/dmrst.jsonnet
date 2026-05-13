{
    // Fold the ~110 fine-grained RST-DT relations to the 18 coarse classes
    // used by DMRST and most benchmark papers. Set to null to train on the
    // raw fine-grained inventory instead (use `lib/relations_rstdt.libsonnet`
    // for the matching explicit `relation_types`).
    relation_map: import 'lib/rstdt_coarse_map.libsonnet',

    // Let the trainer infer the 18 coarse (relation, kind) pairs from the
    // remapped data, so we don't have to keep an explicit list in sync.
    relation_types: null,

    // Model
    model_name: "xlm-roberta-base",
    attn_implementation: "eager",
    stride: 100,
    attention_type: "biaffine",
    classifier_use_bias: true,
    num_rnn_layers: 1,
    encoder_dropout: 0.5,
    decoder_dropout: 0.5,
    labeler_dropout: 0.5,
    doc_gru_dropout: 0.2,
    average_edu_level: true,
    freeze_encoder_layers: 3,

    // Joint EDU segmentation: train a per-subtoken EDU-end classifier alongside
    // the parser; enables `predict_from_text` for end-to-end inference.
    joint_segmentation: true,
    seg_pos_weight: 10.0,
    seg_start_loss: false,

    // Data
    train_dir: "data/rstdt/train",
    dev_dir: "data/rstdt/dev",
    test_dir: "data/rstdt/test",

    // Training
    lr: 1e-4,
    encoder_lr: 2e-5,
    max_epochs: 100,
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

    // Dynamic loss weighting (paper §3.2)
    dlw_enabled: true,
    dlw_temperature: 2.0,
}
