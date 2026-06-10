// Hu & Wan 2023 doc-level repro attempt (overnight 2026-06-10, local 4090).
// Faithful where cheap: ORIGINAL t5-base (their Raffel et al. cite, not the
// finicky v1.1), full-vocab labels (use_copy false), full fine-tune in fp32
// (the prior t5-base degeneracy is explained by AdamW on bf16 master weights,
// fixed in modeling), AdamW 3e-4, effective batch 2, EDU-count loss weighting
// (their Eq. 2), beam 6 at final eval. Known deviations, noted not fixed:
// subtree-size warmup instead of true sentence-level pretraining (uncapped,
// ~3.4k subtrees vs their 7.3k sentences), 12+45 epochs vs their 50+100,
// linear LR decay vs cosine, added-token relation labels vs natural words,
// no last-5-epoch weight averaging, single seed. test_dir null: dev-only
// policy tonight, test eval is a morning decision.
local base = import '../seq2seq_sexp_rstdt.jsonnet';
base + {
    run_name: 'huwan_t5base_sexp',
    model_name: 'google-t5/t5-base',
    use_copy: false,
    constrain_content: true,
    peft: null,
    optimizer: 'adamw',
    lr: 3e-4,
    num_warmup_steps: 2700,
    batch_size: 2,
    grad_accum: 1,
    gradient_checkpointing: false,
    max_input_length: 4096,
    curriculum: { type: 'subtree_size', size_schedule: [8, null], phase_epochs: [12, 45], max_epoch_expand_factor: null },
    edu_loss_weight_exponent: 1.0,
    validate_every: 5,
    dev_max_docs: 16,
    patience: 4,
    num_beams: 6,
    eval_decode_greedy: true,
    train_dir: 'data/rstdt_pinned/train',
    dev_dir: 'data/rstdt_pinned/dev',
    test_dir: null,
    seed: 42,
}
