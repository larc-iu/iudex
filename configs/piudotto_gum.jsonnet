// piudotto on GUM 12.1 (fine relations), joint RST parsing + EDU segmentation.
// iudex original: a non-recurrent sibling of DMRST leaning on the pretrained encoder.
{
    // Data
    train_dir: 'data/gum_12.1.0_notok/train',
    dev_dir: 'data/gum_12.1.0_notok/dev',
    test_dir: 'data/gum_12.1.0_notok/test',
    relation_types: null,                                  // inferred at train time
    relation_map: null,                                    // GUM uses its native fine set

    // Encoder
    model_name: 'jhu-clsp/ettin-encoder-150m',
    stride: 100,                                           // context tokens per side for long-doc striding
    encoder_dropout: 0.2,
    peft: null,                                            // LoRA. null = full fine-tuning (else e.g. {r:16, alpha:32, dropout:0.05})

    // Span representation + biaffine scorer
    span_pooling: 'attention',                             // or 'concat'
    classifier_hidden_size: 256,
    classifier_dropout: 0.2,
    classifier_use_bias: true,
    label_input_pooling: 'mean',                           // or 'last_edu'

    // EDU-level Transformer over pooled per-EDU vectors (contextualizes EDUs before
    // scoring). 0 layers disables it. hidden_size = bottleneck width, null = full width.
    edu_encoder_layers: 2,
    edu_encoder_hidden_size: 384,
    edu_encoder_heads: 8,
    edu_encoder_dropout: 0.2,

    // Autoregressive pointer decoder over the top-down split sequence (the non-RNN
    // replacement for DMRST's GRU). 0 layers = history-free per-node biaffine scoring.
    decoder_layers: 2,
    decoder_hidden_size: 768,                              // bottleneck width, null = full width
    decoder_heads: 8,
    decoder_dropout: 0.5,
    pointer_attention_type: 'biaffine',                    // or 'dot_product'
    decoder_order: 'dfs',                                  // 'dfs' (preorder) or 'bfs' (level order)

    // Curriculum (Registrable). SimpleCurriculum = cold full-document training and
    // owns the epoch budget. SubtreeSizeCurriculum warms up on small subtrees first.
    curriculum: { epochs: 100 },

    // Joint EDU segmentation. Scheme (BIE/BO/EO) with crf or ce loss. null disables
    // it (and `predict_from_text`).
    segmentation: {
        scheme: 'BIE',
        loss: 'crf',
        pos_weight: 10.0,                                  // 'ce' loss only, ignored under crf
        dropout: 0.2,
    },
    // Detokenize EDU text before segmentation training so it matches the raw text
    // `predict_from_text` sees. null = feed corpus text as-is.
    detokenizer: null,

    // EMA-based loss weighting (split/label/seg components). null = unweighted sum.
    ema: { momentum: 0.95, temperature: 2.0 },

    // Training
    lr: 3e-4,
    encoder_lr: 2e-5,                                      // null = use `lr` for the encoder too
    grad_accum: 3,
    amp: true,                                             // bf16 autocast (CUDA), inference stays fp32
    patience: 10,
    max_grad_norm: 10.0,
    weight_decay: 0.01,
    num_warmup_steps: 100,                                 // null = 1-epoch warmup, 0 = none
    log_every: 50,
    begin_validation_epoch: 0,                             // skip dev eval until this epoch
    edu_loss_weight_exponent: 0.0,                         // weight docs by #EDUs**exp, 0 = equal
    checkpoint_dir: 'checkpoints',
    run_name: null,
    seed: 42,
    val_metric_name: 'e2e_full_f1',
}
