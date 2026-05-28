// decoder_only_sexp trained on GUM 12.1 (fine relations). Nested s-expression
// sibling of decoder_only_sr. Default variant is post-order + copy, matching
// the "richest" combination from the experimental design. Canonical config:
// every field on `DecoderOnlySexpConfig` is set explicitly.
{
    // Data
    train_dir: 'data/gum_12.1.0_notok/train',
    dev_dir: 'data/gum_12.1.0_notok/dev',
    test_dir: 'data/gum_12.1.0_notok/test',
    relation_types: null,
    relation_map: null,

    // Model
    model_name: 'google/gemma-3-1b-it',
    causal_mode: true,
    max_input_length: 3072,
    max_output_length: 5120,
    gradient_checkpointing: false,

    // S-expression knobs. `use_copy` is the registry's signature_field.
    traversal_order: 'postorder',
    use_copy: true,

    // LoRA on the causal LM. When use_copy=true the lm_head is replaced at
    // parser init with a small fresh head over the action vocab; when
    // use_copy=false the full pretrained head stays (source subwords need
    // to be scored).
    peft: {
        r: 8,
        alpha: 16,
        dropout: 0.10,
        target_modules: 'all-linear',
        bias: 'none',
        dora: false,
        modules_to_save: ['embed_tokens'],
        train_only_new_embedding_rows: true,
    },

    // Training
    lr: 3e-4,
    weight_decay: 0.05,
    max_epochs: 200,
    batch_size: 1,
    grad_accum: 8,
    optimizer: 'adafactor',
    num_warmup_steps: null,
    max_grad_norm: 1.0,
    amp: true,
    patience: 5,
    log_every: 5,
    validate_every: 131,
    checkpoint_every: 131,
    checkpoint_dir: 'checkpoints',
    run_name: null,
    seed: 42,
    val_metric_name: 'e2e_full_f1',
    action_loss_weight: 1.0,
    label_smoothing: 0.1,

    // Dev eval
    dev_max_docs: null,
    dev_batch_size: 1,

    // Decoding
    num_beams: 4,
    use_validity_constraints: true,
    eval_decode_greedy: true,
    min_edu_length: 1,
}
