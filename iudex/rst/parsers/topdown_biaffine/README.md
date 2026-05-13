# topdown_biaffine

Top-down RST parser with biaffine split and label scoring. Assumes gold EDU
segmentation. Implements the top-down "Kobayashi config" from:

- **Paper:** Naoki Kobayashi, Tsutomu Hirao, Hidetaka Kamigaito, Manabu Okumura,
  Masaaki Nagata. *A Simple and Strong Baseline for End-to-End Neural
  RST-style Discourse Parsing.* EMNLP Findings 2022.
  [arXiv:2210.08355](https://arxiv.org/abs/2210.08355)
- **Reference implementation:** <https://github.com/nttcslab-nlp/RSTParser_EMNLP22>
