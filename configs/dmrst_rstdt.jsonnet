// DMRST on RST-DT (coarse), joint RST parsing + EDU segmentation.
// Modern showcase (ettin encoder, CRF segmentation). For the paper-faithful
// XLM-R setup see configs/repro/dmrst_rstdt.jsonnet.
{
    // Data
    train_dir: 'data/rstdt/train',
    dev_dir: 'data/rstdt/dev',
    test_dir: 'data/rstdt/test',
    relation_types: null,                                  // inferred at train time
    relation_map: import 'lib/rstdt_coarse_map.libsonnet',

    // Model
    model_name: 'jhu-clsp/ettin-encoder-150m',
    stride: 100,
    encoder_window_size: null,                             // null = default striding over the whole doc; set an int for a fixed-size window (see configs/repro/dmrst_rstdt.jsonnet)
    attention_type: 'dot_product',                         // or 'biaffine'
    classifier_use_bias: true,
    num_rnn_layers: 1,
    encoder_dropout: 0.5,
    decoder_dropout: 0.5,
    labeler_dropout: 0.5,
    doc_gru_dropout: 0.2,
    label_input_pooling: 'mean',                           // or 'last_edu'
    freeze_embeddings: false,
    freeze_encoder_layers: 0,
    peft: null,                                            // LoRA. null = full fine-tuning

    // Training schedule, and the total run length (there is no separate max_epochs):
    // this trains on full documents for `epochs` epochs. To warm up on small subtrees
    // first, use e.g. { type: 'subtree_size', size_schedule: [8, 20, 60, null], phase_epochs: 5 }.
    curriculum: { epochs: 100 },

    // Joint EDU segmentation. Scheme-based segmenter (BIE/BO/EO) with crf or ce
    // loss. null disables it (and `predict_from_text`).
    segmentation: {
        scheme: 'BIE',
        loss: 'crf',
        dropout: 0.3,
        pos_weight: 10.0,                                  // 'ce' loss only, ignored under crf
        start_loss: false,                                 // binary end-tagger only (scheme: null)
    },
    // Detokenize EDU text before segmentation training so it matches the raw text
    // `predict_from_text` sees. null = feed corpus text as-is.
    detokenizer: null,

    // Dynamic loss weighting (paper §3.2). window: 2 = the paper's lagged
    // L(t-1)/L(t-2) ratio, larger windows are smoother (better for whole-tree
    // training). null = unweighted sum.
    dlw: { temperature: 2.0, window: 30 },

    // Training
    lr: 2e-4,
    encoder_lr: 2e-5,                                      // null = use `lr` for the encoder too
    grad_accum: 3,
    amp: true,                                             // bf16 autocast (CUDA), inference stays fp32
    patience: 10,
    max_grad_norm: 20.0,
    weight_decay: 0.01,
    num_warmup_steps: 100,                                 // null = 1-epoch warmup, 0 = none
    log_every: 1,
    begin_validation_epoch: 0,                             // skip dev eval until this epoch
    validate_every: 1,                                     // run dev validation every N epochs (1 = every epoch, final always validates)
    edu_loss_weight_exponent: 0.0,                         // weight docs by #EDUs**exp, 0 = equal
    checkpoint_dir: 'checkpoints',
    run_name: null,
    seed: 42,
    val_metric_name: 'e2e_full_f1',
}
