// Width-band loss experiment (drafted overnight 2026-06-10, NOT yet run).
// Motivation: cascade probe showed the generative deficit is mid-width (5-16
// EDU) merge decisions that cascade upward. This upweights exactly those
// reduce positions. Mirrors the no-curric r16 baseline recipe otherwise.
local base = import '../seq2seq_sr_rstdt.jsonnet';
base + {
    run_name: 'widthband_s2s_sr_rstdt',
    gradient_checkpointing: true,
    batch_size: 2,
    grad_accum: 4,
    peft: base.peft + { r: 16, alpha: 32 },
    width_band_loss: { min_width: 5, max_width: 16, weight: 2.0 },
    validate_every: 3,
    patience: 4,
    begin_validation_epoch: 18,
    dev_batch_size: 8,
    seed: 42,
}
