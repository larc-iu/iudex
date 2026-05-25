# piudotto

*P*arser from *I*ndiana *U*niversity for *D*iscourse *O*rganization *T*hrough *T*ree *O*perations — an iudex-original span-based end-to-end RST parser.

A modern, minimal sibling of `dmrst`: it leans on the pretrained encoder and
has no GRUs. dmrst's recurrent pointer decoder is replaced by an optional
*Transformer* pointer decoder (off by default).

- **Segmentation.** Joint per-token EDU-boundary tagging over a configurable
  scheme (`BIE` / `BO` / `EO`). The default head is a linear-chain **CRF**
  (learned transitions + structural masks, trained by NLL); set
  `segmentation.loss = "ce"` for the lighter independent per-token
  cross-entropy + constrained-Viterbi variant. Set `segmentation: null` to
  train a gold-EDU-only parser (and lose `predict_from_text`).
- **Split scoring.** With `decoder_layers = 0` (default), each candidate split
  `(b, k, e)` is scored independently by a deep biaffine over pooled left/right
  sub-span representations. With `decoder_layers > 0`, an autoregressive
  Transformer decoder runs over the top-down decision sequence (causal
  self-attention over the DFS-ordered prior decisions, cross-attention over the
  EDU reprs) and a pointer head scores the split from the history-conditioned
  query, the non-RNN analog of dmrst's GRU decoder.
- **Label scoring.** A deep biaffine over the pooled left/right children scores
  the joint `(nuclearity, relation)` label, in both modes.
- **Decoding.** Greedy top-down. With the decoder on, decoding is autoregressive
  (each step re-runs the decoder over the committed prefix).
- **Training objective.** Per-node teacher-forced cross-entropy on splits and
  labels. With the decoder on, the gold spans are scored in one causal-masked
  decoder pass over the DFS sequence; the loss decomposition is unchanged.
  Component losses are combined with EMA-based dynamic weighting (`ema`).

See the top-level `CLAUDE.md` for the project layout and `configs/piudotto_*.jsonnet`
for runnable configurations.
