# decoder_only_sexp

End-to-end RST parser via a fine-tuned decoder-only causal LM (default
`google/gemma-3-1b-it`) that emits a nested s-expression serialization of the
tree. Single-stream sibling of `seq2seq_sexp`, structural sibling of
`decoder_only_sr`. Two orthogonal knobs control the serialization:
`traversal_order` ('preorder' vs 'postorder') and `use_copy` (true uses a
`<copy>` token at leaf positions, false emits source subwords verbatim in
stream). When `use_copy=True` the lm_head is replaced with a small fresh head
over just the action vocab. When `use_copy=False` the full pretrained head
stays so source subwords can be scored.
