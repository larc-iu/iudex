local relations = import 'lib/relations.libsonnet';

local model_name = "SpanBERT/spanbert-base-cased";
//local model_name = "bert-base-cased";
//local model_name = "roberta-base";
//local model_name = "microsoft/deberta-v3-base";

{
    relation_types: relations,

    // Model
    model_name: model_name,
    ffn_hidden_size: 512,
    dropout: 0.2,
    stride: 100,
    attn_implementation: "eager",

    // Data
    train_dir: "data/gum9/train",
    dev_dir: "data/gum9/dev",

    // Training
    lr: 2e-4,
    encoder_lr: 1e-5,  // optional; set to null to use `lr` for the encoder too
    max_epochs: 100,
    grad_accum: 1,
    patience: 10,
    max_grad_norm: 1.0,
    weight_decay: 0.01,
    num_warmup_steps: 1000,
    log_every: 50,
    validate_every: null,
    checkpoint_every: null,
    checkpoint_dir: "checkpoints",
    run_name: null,  // optional; final run dir = checkpoint_dir/<run_name>-<hash> or checkpoint_dir/<hash>
    seed: 42,
    val_metric_name: "span_f1",
}
