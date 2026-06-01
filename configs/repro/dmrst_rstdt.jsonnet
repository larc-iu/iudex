// DMRST on RST-DT (coarse), faithful to Liu, Shi & Chen 2021.
// Reproduces the paper's setup: XLM-R encoder, fixed 300-token sliding window,
// binary end-tagger segmentation, lagged 2-step dynamic loss weighting.
{
    // Data
    train_dir: 'data/rstdt/train',
    dev_dir: 'data/rstdt/dev',
    test_dir: 'data/rstdt/test',
    relation_types: null,                                  // inferred at train time
    relation_map: import '../lib/rstdt_coarse_map.libsonnet',

    // Model
    model_name: 'xlm-roberta-base',
    stride: 100,
    // Paper's fixed sliding window: 300 content tokens per window, `stride` context
    // tokens dropped per interior side, no [CLS]/[SEP]. null = shared striding.
    encoder_window_size: 300,
    attention_type: 'dot_product',                         // or 'biaffine'
    classifier_use_bias: true,
    num_rnn_layers: 1,
    encoder_dropout: 0.5,
    decoder_dropout: 0.5,
    labeler_dropout: 0.5,
    doc_gru_dropout: 0.2,
    label_input_pooling: 'mean',                           // or 'last_edu'
    freeze_embeddings: true,                               // paper freezes embeddings
    freeze_encoder_layers: 3,                              // and the first 3 layers
    peft: null,                                            // LoRA. null = full fine-tuning

    // Curriculum (Registrable). SimpleCurriculum = cold full-document training and
    // owns the epoch budget. SubtreeSizeCurriculum warms up on small subtrees first.
    curriculum: { epochs: 15 },

    // Joint EDU segmentation. Paper's binary per-token end-tagger (scheme: null).
    // Set scheme to BIE/BO/EO to use the shared scheme-based segmenter instead
    // (then `loss` and `dropout` apply and `start_loss`/`pos_weight` are ignored).
    segmentation: {
        scheme: null,
        pos_weight: 10.0,                                  // EDU ends are rare, so upweight
        start_loss: true,
        loss: 'crf',
        dropout: 0.5,
    },
    // Detokenize EDU text before segmentation training. null matches the paper
    // (word-tokenized text fed straight in, leak-free under XLM-R SentencePiece).
    detokenizer: null,

    // Dynamic loss weighting (paper §3.2). window: 2 = the paper's lagged
    // L(t-1)/L(t-2) ratio. null = unweighted sum.
    dlw: { temperature: 2.0, window: 2 },

    // Training
    lr: 1e-4,
    encoder_lr: 2e-5,                                      // null = use `lr` for the encoder too
    grad_accum: 3,
    amp: true,                                             // bf16 autocast (CUDA), inference stays fp32
    patience: 10,
    max_grad_norm: 5.0,
    weight_decay: 0.01,
    num_warmup_steps: 1000,                                // null = 1-epoch warmup, 0 = none
    log_every: 1,
    begin_validation_epoch: 0,                             // skip dev eval until this epoch
    edu_loss_weight_exponent: 0.0,                         // weight docs by #EDUs**exp, 0 = equal
    checkpoint_dir: 'checkpoints',
    run_name: null,
    seed: 42,
    val_metric_name: 'e2e_full_f1',
}
