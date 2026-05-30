# sr_biaffine

Transition-based (shift-reduce) RST parser, the bottom-up sibling of
`topdown_biaffine`. Same paper, same shared span representation, different
decoding strategy.

- **Paper:** [A Simple and Strong Baseline for End-to-End Neural
  RST-style Discourse Parsing](https://aclanthology.org/2022.findings-emnlp.501/)
- **Reference implementation:** <https://github.com/nttcslab-nlp/RSTParser_EMNLP22>
  (the `shift_reduce_parser_*` / `shift_reduce_classifier_*` models)

## Design

State is a stack of EDU spans plus a left-to-right EDU queue. At each step the
parser reads the top two stack spans (s1, s2) and the queue-front EDU (q1) and
chooses SHIFT or REDUCE. On REDUCE it labels the new node with a (nuclearity,
relation) pair. A span is represented as the mean of its first and last
subtoken embeddings, identical to `topdown_biaffine`.

Two heads, both fed the same span representations:
- **action head** (SHIFT vs REDUCE): an FFN over the concatenation of the s1,
  s2, q1 span representations. This is a 2-way structural decision, matching
  the reference's V1 standalone action head.
- **label head** ((nuc, rel) for a REDUCE): a deep biaffine over the two stack
  spans being merged, with s2 the left child and s1 the right child. This
  reuses the shared `common/biaffine.py::DeepBiAffine` (hence the `_biaffine`
  name), where the reference uses a plain FFN.

Training replays the gold shift-reduce oracle (`RstTree.to_shift_reduce`) and
teacher-forces both heads: the action head on every step, the label head only
on REDUCE steps. Loss is their unweighted average. Decoding is greedy with
legality masking (no SHIFT once the queue is empty, no REDUCE below two stack
items), and the action sequence is rebuilt into a tree via
`RstTree.from_shift_reduce`.

## Deviations from the reference

- No organization features (sentence/paragraph boundary indicators). Our
  `RstTree` does not carry sentence/paragraph metadata, so we omit them
  (equivalent to the reference's `disable_org_feat`), matching `topdown_biaffine`.
- No shift/reduce loss reweighting (the reference's optional Guz et al. 2020
  penalty). The two heads are averaged evenly.
- The reduce-label head is a deep biaffine rather than a plain FFN.
- Nuclearity is predicted by the label head (folded into the joint `{nuc}_{rel}`
  label space, as in `topdown_biaffine`), not by the action head. The
  reference's recommended variant instead uses a joint shift/reduce `act_nuc`
  action head with a separate relation head.
