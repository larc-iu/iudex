# parsers/common

Code shared at train/predict time across the RST parsers. Per CLAUDE.md, only
reusable `nn.Module`s and byte-identical pure helpers live here; each parser's
training/predict loop stays in its own folder for self-contained reading.

## What's here

Shared by several parsers:
- `config.py` — config parsing helpers + the `PeftConfig` LoRA dataclass (the one
  config dataclass shared by every parser).
- `encoding.py` — `load_encoder_and_tokenizer` (the encoder-based parsers).
- `curriculum.py` — `Curriculum` strategies (`SimpleCurriculum` / `SubtreeSizeCurriculum`), every parser.
- `detokenization.py` — the `Detokenizer` abstraction.
- `inference.py` / `predict_cli.py` — source/checkpoint resolution + the shared predict CLI.

Encoder-based parsers (`dmrst`, `piudotto`, `topdown_biaffine`, `sr_biaffine`):
- `segmentation.py` — the EDU-boundary `Segmenter` (piudotto, dmrst).
- `biaffine.py` — the `DeepBiAffine` span scorer (topdown_biaffine, sr_biaffine).
- `pointer.py` — `PointerAttention` (dmrst).

Generative parsers (`seq2seq_sr`, `decoder_only_sr`, `seq2seq_sexp`, `decoder_only_sexp`):
- `seqgen.py` — EDU→token alignment, beam-search primitives, KV-cache reorder, the
  shift-reduce `ShiftReduceDecodeState`, embedding-gradient masking + head warm-init.
- `sexp_constraints.py` — the s-expression pushdown automaton (`SexpDecodingState`) and
  `GoldEduForcer` (the two sexp parsers).
- `generative_eval.py` — the shared dev/test eval orchestration (`evaluate_on_dev`),
  which talks to a parser only through the small `GenerativeParser` Protocol.

## Reading the generative parsers (start here)

The four generative parsers are a 2×2: backbone (encoder-decoder `seq2seq_*` vs causal
`decoder_only_*`) × serialization (shift-reduce `*_sr` vs s-expression `*_sexp`). They are
four self-contained parsers, not one branchy class (see CLAUDE.md, "generative parsers",
for why). A productive reading order:

1. `seq2seq_sr/` — the canonical/reference parser. Read its README and
   `modeling_seq2seq_sr.py` top-to-bottom first.
2. `seqgen.py` — the shared decode/alignment/beam machinery the SR parsers lean on.
3. `decoder_only_sr/` — the same parser on a causal backbone (read it as a delta:
   single-stream input layout, otherwise identical).
4. `seq2seq_sexp/` + `sexp_constraints.py` — swap the shift-reduce serialization for a
   constrained s-expression (the PDA lives in `sexp_constraints.py`).
5. `decoder_only_sexp/` — the s-expression parser on the causal backbone.
6. `generative_eval.py` — how all four are evaluated through one Protocol.
