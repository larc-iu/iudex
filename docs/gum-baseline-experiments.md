# GUM 12.1 baseline experiments

Reference numbers I should consult when launching cluster jobs on
`seq2seq_sr` / `decoder_only_sr`. All numbers are GUM dev (fine
relations) unless noted. Bar to beat: DMRST at **0.4181** dev /
**0.4531** test e2e_full_f1.

## Best dev e2e_full_f1, side by side

| Run | Parser | Config hash | Best e2e_full_f1 | seg_f1 | e2e_span_f1 | e2e_nuc_f1 | e2e_rel_f1 | Best step / epoch |
|---|---|---|---|---|---|---|---|---|
| **dmrst** | dmrst | `c1a66fffa4d1` | **0.4181** | 0.9437 | 0.6769 | 0.5347 | 0.4204 | 6106 / 86 |
| seq2seq baseline | seq2seq_sr | `b7e5fdd430e3` | 0.3456 | 0.9517 | 0.6198 | 0.4684 | 0.3492 | 1048 / 39 |
| seq2seq exp1 (action_w=1, r=8) | seq2seq_sr | `bdfb0d76ea6d` | **0.3692** | 0.9532 | 0.6330 | 0.4845 | 0.3736 | 1572 / 59 |
| seq2seq exp2 (action_w=1, r=16) | seq2seq_sr | `0da89fcaefb5` | 0.3534 | 0.9522 | 0.6278 | 0.4691 | 0.3570 | 655 / 25 |

DMRST's test e2e_full_f1 was 0.4531 (the seq2seq runs were killed before final test eval).

Segmentation is essentially solved across all runs (>0.94); the gap to DMRST
is in attachment + labeling.

## DMRST on GUM (best run, `c1a66fffa4d1`)

Reference config (matches `configs/dmrst_gum.jsonnet` at the time of training):

```
model_name: jhu-clsp/ettin-encoder-400m
stride: 100
attention_type: dot_product
classifier_use_bias: true
num_rnn_layers: 1
encoder_dropout: 0.5
decoder_dropout: 0.5
labeler_dropout: 0.5
doc_gru_dropout: 0.2
label_input_pooling: mean
freeze_embeddings: true
freeze_encoder_layers: 3
segmentation: { pos_weight: 10, start_loss: false, scheme: BIE, loss: crf, dropout: 0.5 }
detokenizer: null
dlw: { temperature: 2, window: 30 }
lr: 3e-4
encoder_lr: 2e-5
max_epochs: 100
grad_accum: 3
amp: true
patience: 10
max_grad_norm: 10
weight_decay: 0.01
num_warmup_steps: 100
val_metric_name: e2e_full_f1
seed: 42
```

Final metrics (from `checkpoints/c1a66fffa4d1/final_metrics.json`):

| Split | seg_f1 | e2e_span_f1 | e2e_nuc_f1 | e2e_rel_f1 | e2e_full_f1 |
|---|---|---|---|---|---|
| dev | 0.9437 | 0.6769 | 0.5347 | 0.4204 | **0.4181** |
| test | 0.9444 | 0.7161 | 0.5794 | 0.4561 | **0.4531** |

## seq2seq_sr baseline (`b7e5fdd430e3`)

Config: identical to the pre-cleanup `configs/seq2seq_sr_gum.jsonnet`
except for the action_loss_weight / peft.r values, since the canonical
configs now reflect exp1's improved settings.

Diff vs canonical `configs/seq2seq_sr_gum.jsonnet`:

```
action_loss_weight: 3.0     (canonical: 1.0)
peft.r:             4       (canonical: 8)
peft.alpha:         8       (canonical: 16)
```

All other knobs identical: `google/t5gemma-2-1b-1b`, lr=3e-4, weight_decay=0.05,
dropout=0.10, label_smoothing=0.1, max_input_length=3072, max_output_length=5120,
batch_size=1, grad_accum=8, adafactor, no warmup, amp=true, patience=5.

Dev trajectory (per validation, step interval=131):

```
step  131  -> 0.0106
step  262  -> 0.2560
step  393  -> 0.2924
step  524  -> 0.3249
step  655  -> 0.3120
step  786  -> 0.3451
step  917  -> 0.3144
step 1048  -> 0.3456  <- best
step 1179  -> 0.3342
```

Saturated around step ~1000 / epoch 39. Run was killed mid-training to swap configs.

## seq2seq_sr exp1: action_w=1.0 + r=8 (`bdfb0d76ea6d`)

Config: matches canonical `configs/seq2seq_sr_gum.jsonnet`. No diff. (exp1's
knobs are now the canonical default.)

Dev trajectory:

```
step  131  -> 0.0111
step  262  -> 0.1778
step  393  -> 0.3349
step  524  -> 0.3600
step  655  -> 0.3654
step  786  -> 0.3673
step  917  -> 0.3692  <- best
```

Was killed for exp2 around step ~1572 / epoch 59. Climb slowed but had not
clearly plateaued. **+0.0236 over baseline**; gap to DMRST closed to 0.0489.

Attribution caveat: exp1 changed two knobs at once (action_loss_weight and
peft.r). Panel B/C argued action_loss_weight was the load-bearing change;
exp2's underperformance (below) is consistent with that.

## seq2seq_sr exp2: action_w=1.0 + r=16 (`0da89fcaefb5`)

Diff vs canonical:

```
peft.r:     16    (canonical: 8)
peft.alpha: 32    (canonical: 16)
```

Dev trajectory:

```
step  131  -> 0.0190
step  262  -> 0.1294
step  393  -> 0.3082
step  524  -> 0.3382
step  655  -> 0.3534  <- best at time of kill
```

Killed at step 655 / epoch 25 because trajectory was clearly behind exp1's
at the same step (0.3534 vs exp1's 0.3654 at step 655). Larger LoRA rank
didn't help on top of action_loss_weight=1.0; suggests exp1's gain came
from the loss rebalance, not the capacity increase.

## Takeaways for cluster runs

- The single most impactful knob found overnight was **`action_loss_weight: 3.0 → 1.0`**. Now the canonical default.
- LoRA rank above 8 doesn't appear to help with this base model at GUM scale.
- All runs use `google/t5gemma-2-1b-1b`; results scale with a bigger base model are unknown.
- Segmentation is consistently 0.94-0.95 across all seq2seq runs. Don't optimize for seg_f1; the gap to DMRST is attachment + labeling.
- DMRST converged at epoch 86 / step 6106 with `patience: 10`; the seq2seq runs early-stopped (or were killed) much earlier with `patience: 5`. **Consider raising patience to 10+ on the cluster** where wall time is less constrained.
- For RST-DT, no seq2seq runs have been done yet. Use `configs/seq2seq_sr_rstdt.jsonnet` as the starting point and expect similar hparam sensitivity.
- The `decoder_only_sr` parser is implementation-complete and tested but has **no training data yet**. Same hparam recipe as seq2seq_sr is the v1 default.

## Provenance

Numbers extracted from each run's TensorBoard `tb/` event files
(`dev/{e2e_full_f1,seg_f1,e2e_span_f1,e2e_nuc_f1,e2e_rel_f1}` scalars).
DMRST final test numbers from `checkpoints/c1a66fffa4d1/final_metrics.json`.
