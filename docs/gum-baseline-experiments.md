# GUM 12.1 baseline experiments

Reference numbers I should consult when launching cluster jobs on
`seq2seq_sr` / `decoder_only_sr` / `seq2seq_sexp` / `decoder_only_sexp`.
All in-house numbers are GUM dev (fine relations) unless noted.

**Bars to beat (external)**:
- Maekawa et al. 2024 (Llama 2-70B + QLoRA, step-wise prompting): **0.552** full F1 (test) on GUM.
- Hu & Wan 2023 (T5 + linearized s-expression, no copy): RST-DT only in the published numbers; not directly comparable here.

**Bar to beat (in-house, prior parser)**: DMRST at **0.4181** dev / **0.4531** test e2e_full_f1.

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

## External baselines (from the literature)

For context when interpreting cluster-run numbers.

### Maekawa et al. 2024 (EACL)
- **Title**: "Can we obtain significant success in RST discourse parsing by using Large Language Models?"
- **Approach**: Llama 2 7B / 13B / 70B fine-tuned with QLoRA. Decoder-only LLM invoked once per parsing step in a bottom-up transition system. Stack2 / Stack1 / Queue1 spans presented as context; model emits a single token in {Shift, Reduce}, then re-prompted for nuclearity + label when reducing.
- **Headline (Llama 2-70B, bottom-up, full Parseval F1 on test)**:
  - RST-DT: span 79.8 / nuc 70.4 / rel 60.0 / full **0.581**
  - Instr-DT: span 79.1 / nuc 60.4 / rel 55.1 / full **0.473**
  - GUM: span 76.4 / nuc 64.7 / rel 56.4 / full **0.552**
- **Note**: ~3 F1 over prior DeBERTa SOTA on RST-DT and Instr-DT; ~7 F1 over prior on GUM. Inference cost is minutes per document at 70B (vs seconds for graph-based parsers).
- **Comparison framing**: the relevant axis is **inference paradigm**, not parameter count. Maekawa is *step-wise iterative prompting* (many forward passes per document, one per parser action). Our decoder_only_* parsers are *one-shot generation* (a single forward decode of the entire serialized tree). Reporting "X% of the parameters" is misleading because step-wise amortizes compute over many calls. Wall-clock comparisons depend strongly on the document's action count. The fair framing if our numbers approach Maekawa's: "competitive at one-shot decode with substantially lower per-document inference cost."
- **arXiv**: https://arxiv.org/abs/2403.05065 ; **code**: https://github.com/nttcslab-nlp/RSTParser_EACL24

### Hu & Wan 2023 (TASLP)
- **Title**: "RST Discourse Parsing as Text-to-Text Generation"
- **Approach**: T5 encoder-decoder fine-tune. Output is the **linearized s-expression of the whole RST tree** with input text reproduced verbatim inside the brackets. Constrained decoding for well-formedness, **no copy mechanism**.
- **Headline**: RST-DT, reported as outperforming existing methods on both parsing and segmentation. Exact F1 table not accessible without the IEEE paywall PDF.
- **Comparison framing**: our `seq2seq_sexp` with `use_copy=False` adopts the **same serialization choice** (sexp, words in-stream, no copy mechanism) but differs in the rest of the recipe: T5Gemma 2 vs T5, LoRA on the base vs full FT, no sentence→document curriculum, and (per the new `constrain_content` knob) at most decoded-content positions hard-masked to source IDs vs free generation. So claim "Hu & Wan-style serialization" rather than "we replicate Hu & Wan." A real replication would require the `constrain_content: false` setting AND full fine-tuning AND the matched curriculum.
- **IEEE**: https://ieeexplore.ieee.org/document/10224326/

### Mabona et al. 2019 (EMNLP)
- **Title**: "Neural Generative Rhetorical Structure Parsing"
- **Approach**: RNNG (pre-Transformer). Stack-LSTM joint LM over structural actions and `GEN(w)` word-emitting actions. Bottom-up traversal; words emitted verbatim from softmax over vocabulary, no copy.
- **Headline**: With fixed beam search, +6.8 unlabelled / +2.9 labelled F1 over a vanilla RNNG baseline on RST-DT. Comparable to non-additional-data SOTA at the time.
- **Closest comparison in our matrix**: `seq2seq_sr` with `use_copy=False` (if we ever ran it; currently not supported by design) — but more importantly, `seq2seq_sexp` post-order with `use_copy=False` since the RNNG's unrolled trace is isomorphic to a depth-first bracketed tree with words in-stream.
- **arXiv**: https://arxiv.org/abs/1909.11049 ; **ACL**: https://aclanthology.org/D19-1233/

## Known confounds

- **`label_smoothing` scale across `use_copy` modes**: when `use_copy=True` the action head projects to ~100 classes; when `use_copy=False` it projects to the full backbone vocabulary (~262K classes), so a fixed `label_smoothing=0.1` distributes very different per-off-class mass in the two regimes. We hold `label_smoothing` at 0.1 across both modes (it applies uniformly to all cells, full-vocab cells included). There is no auto-scaling.
- **Head architecture is bound to `use_copy`**: with `use_copy=True` we replace the lm_head with a small fresh `Linear(hidden, ~100)`. With `use_copy=False` we keep the full pretrained lm_head. So a `use_copy` ablation conflates "having COPY" with "tiny vs full head". This is a property of the COPY mechanism itself and not a confound we can remove without breaking one mode or the other (see the `use_copy` field docstring in both sexp configs).

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
