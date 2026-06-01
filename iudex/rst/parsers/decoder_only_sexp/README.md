# decoder_only_sexp

End-to-end RST parser via a fine-tuned decoder-only causal LM (default
`google/gemma-3-1b-it`) that emits a nested s-expression serialization of the
tree. Single-stream sibling of `seq2seq_sexp`, structural sibling of
`decoder_only_sr`: instead of a separate encoder pass, the source subwords sit
in front of the s-expression stream under one causal mask, separated by a
learned `<|start_of_sexp|>` token (`[BOS] source [SEP] sexp_tokens [EOS]`, with
the prefix masked to -100 so only the s-expression portion is scored).

Two orthogonal knobs control the serialization: `traversal_order` ('preorder'
vs 'postorder') and `use_copy` (true uses a `<copy>` token at leaf positions,
false emits source subwords verbatim in stream). When `use_copy=True` the
lm_head is replaced with a small fresh head over just the action vocab. When
`use_copy=False` the full pretrained head stays so source subwords can be
scored.

Everything else is identical to `seq2seq_sexp`: a pushdown-automaton validity
mask (`iudex.rst.parsers.common.sexp_constraints`) enforces structural validity
during decoding so any surviving beam produces a parseable s-expression, and
tree reconstruction goes through `RstTree.from_sexp`.
