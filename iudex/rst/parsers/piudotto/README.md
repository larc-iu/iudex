# piudotto

*P*arser from *I*ndiana *U*niversity for *D*iscourse *O*rganization *T*hrough *T*ree *O*perations — an iudex-original span-based end-to-end RST parser.

A modern, minimal sibling of `dmrst`: it leans entirely on the pretrained
encoder (no GRUs) and scores every candidate constituent with span-based deep
biaffines, instead of a top-down pointer network.

- **Segmentation.** Joint per-token EDU-boundary tagging over a configurable
  scheme (`BIE` / `BO` / `EO`). The default head is a linear-chain **CRF**
  (learned transitions + structural masks, trained by NLL); set
  `segmentation.loss = "ce"` for the lighter independent per-token
  cross-entropy + constrained-Viterbi variant. Set `segmentation: null` to
  train a gold-EDU-only parser (and lose `predict_from_text`).
- **Tree scoring.** Each candidate split `(b, k, e)` is scored by a deep
  biaffine over pooled left/right sub-span representations; a second biaffine
  scores the joint `(nuclearity, relation)` label.
- **Decoding.** Greedy top-down by default (`decoding: "greedy"`); set
  `decoding: "cky"` for the globally optimal binary tree via an O(n³) chart.
- **Training objective.** Per-node teacher-forced cross-entropy by default; set
  `margin_training: {margin: 1.0}` for a Stern et al. 2017 max-margin objective
  against the cost-augmented CKY tree (runs CKY each step). Component losses are
  combined with EMA-based dynamic weighting (`ema`).

See the top-level `CLAUDE.md` for the project layout and `configs/piudotto_*.jsonnet`
for runnable configurations.
